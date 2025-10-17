"""Filter aging-theory reviews using an OpenAI relevance check.

The script loads the JSON file produced by the metadata collection steps and
asks an OpenAI chat model to decide whether each article is relevant to aging
theory based on its title and abstract. Records marked as irrelevant are
discarded.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for calling the chat completion endpoint.

Usage
-----
```bash
python scripts/step2_filter_reviews.py \
    --input data/pipeline/start_reviews.json \
    --output data/pipeline/filtered_reviews.json
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

from openai import AsyncOpenAI


OPENAI_SYSTEM_PROMPT = (
    "You are an expert gerontology curator. Respond with JSON containing `relevant` "
    "(true/false) and `explanation`."
)


def _coerce_bool(value: object) -> bool | None:
    """Interpret truthy strings/ints returned by the model."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


async def _call_openai(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
    delay: float,
) -> Dict:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - network error handling
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc
        finally:
            if delay > 0:
                await asyncio.sleep(delay)

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError) as err:  # pragma: no cover
        raise RuntimeError(f"Unexpected OpenAI response: {response!r}") from err

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as err:
        raise RuntimeError(
            "OpenAI returned invalid JSON payload: " + content
        ) from err

    return parsed


def build_prompt(batch: Sequence[Dict]) -> str:
    lines = [
        "You will receive multiple review articles about scientific topics.",
        "For each item, determine if the article is primarily about aging theory.",
        "Return a JSON array with the same number of elements as the items below.",
        "Each array element must correspond to the item at the same position.",
        "Every element must be an object with fields `relevant` (true/false) and `explanation`.",
        "Items:",
    ]

    for index, record in enumerate(batch, start=1):
        abstract = record.get("abstract") or ""
        sources = record.get("sources")
        source_line = ""
        if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
            joined = ", ".join(
                str(item) for item in sources if isinstance(item, str) and item.strip()
            )
            if joined:
                source_line = f"Sources: {joined}\n"
        elif isinstance(sources, str) and sources.strip():
            source_line = f"Sources: {sources.strip()}\n"
        elif isinstance(record.get("provenance"), str) and record["provenance"].strip():
            source_line = f"Sources: {record['provenance'].strip()}\n"

        lines.extend(
            [
                f"Item {index}:",
                f"Title: {record.get('title', '')}",
                source_line.rstrip(),
                f"Abstract: {abstract}",
                "",
            ]
        )

    return "\n".join(line for line in lines if line is not None)


def _parse_batch_response(
    batch: Sequence[Dict], response: object
) -> Tuple[Dict[int, Dict], List[int]]:
    processed: Dict[int, Dict] = {}
    fallback: List[int] = []

    if isinstance(response, list):
        elements = response
        if len(elements) != len(batch):
            # keep len(elements) so we don't index past range
            elements = list(elements)[: len(batch)]
            fallback.extend(range(len(elements), len(batch)))
        for idx, item in enumerate(elements):
            if not isinstance(item, dict):
                fallback.append(idx)
                continue

            relevant = _coerce_bool(item.get("relevant"))
            if relevant is None:
                fallback.append(idx)
                continue

            item_data = dict(item)
            item_data["relevant"] = relevant
            if "explanation" not in item_data:
                item_data["explanation"] = ""
            processed[idx] = item_data

        missing = set(range(len(batch))) - set(processed.keys()) - set(fallback)
        if missing:
            fallback.extend(sorted(missing))

        return processed, sorted(set(fallback))

    if not isinstance(response, dict):
        return processed, list(range(len(batch)))

    for idx in range(len(batch)):
        key = str(idx + 1)
        item = response.get(key)
        if not isinstance(item, dict):
            fallback.append(idx)
            continue

        relevant = _coerce_bool(item.get("relevant"))
        if relevant is None:
            fallback.append(idx)
            continue

        item_data = dict(item)
        item_data["relevant"] = relevant
        if "explanation" not in item_data:
            item_data["explanation"] = ""
        processed[idx] = item_data

    missing = set(range(len(batch))) - set(processed.keys()) - set(fallback)
    if missing:
        fallback.extend(sorted(missing))

    return processed, sorted(set(fallback))


async def async_filter_records(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    *,
    delay: float = 0.5,
    batch_size: int = 5,
    concurrency: int = 5,
) -> List[Dict]:
    records_list = list(records)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()
    processed_total = 0
    kept_total = 0

    async with AsyncOpenAI(api_key=api_key) as client:
        async def log_progress(relevant: bool) -> None:
            nonlocal processed_total, kept_total
            async with progress_lock:
                processed_total += 1
                if relevant:
                    kept_total += 1
                print(
                    f"Processed {processed_total} records — kept {kept_total}",
                    flush=True,
                )

        async def process_batch(start: int) -> None:
            batch = records_list[start : start + batch_size]
            prompt = build_prompt(batch)
            processed_items: Dict[int, Dict] = {}
            fallback_indices: List[int] = []
            try:
                response = await _call_openai(
                    client,
                    prompt,
                    model,
                    semaphore,
                    delay,
                )
            except RuntimeError as err:
                print(
                    f"Batch {start + 1}-{start + len(batch)} failed with error: {err}. "
                    "Retrying items individually.",
                    file=sys.stderr,
                )
                fallback_indices = list(range(len(batch)))
            else:
                processed_items, fallback_indices = _parse_batch_response(batch, response)

            for offset, decision in processed_items.items():
                record = batch[offset]
                record["llm_filter"] = decision
                await log_progress(decision.get("relevant") is True)

            for offset in fallback_indices:
                record = batch[offset]
                single_prompt = build_prompt([record])
                try:
                    single_response = await _call_openai(
                        client,
                        single_prompt,
                        model,
                        semaphore,
                        delay,
                    )
                except RuntimeError as err:
                    print(
                        f"Record {start + offset + 1} failed after retry: {err}",
                        file=sys.stderr,
                    )
                    await log_progress(False)
                    continue

                single_items, pending = _parse_batch_response([record], single_response)
                decision = single_items.get(0)
                relevant = False
                if decision is not None:
                    record["llm_filter"] = decision
                    relevant = decision.get("relevant") is True
                else:
                    print(
                        f"Record {start + offset + 1} returned invalid JSON even after retry.",
                        file=sys.stderr,
                    )
                if pending:
                    print(
                        f"Record {start + offset + 1} still pending after retry; giving up.",
                        file=sys.stderr,
                    )
                await log_progress(relevant)

        tasks = [
            asyncio.create_task(process_batch(start))
            for start in range(0, len(records_list), batch_size)
        ]
        if tasks:
            await asyncio.gather(*tasks)

    kept = [
        record
        for record in records_list
        if record.get("llm_filter", {}).get("relevant") is True
    ]
    return kept


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter aging-theory reviews with OpenAI")
    parser.add_argument("--input", default="data/pipeline/start_reviews.json")
    parser.add_argument("--output", default="data/pipeline/filtered_reviews.json")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat completion model identifier.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between OpenAI calls to avoid rate limits.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help=(
            "Number of records to include in each LLM request. Larger batches reduce "
            "API calls but consume more tokens; pick a value that keeps prompts "
            "within the model's context window (e.g., 5-10 abstracts for GPT-4o)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent OpenAI requests.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    kept = asyncio.run(
        async_filter_records(
            records,
            api_key,
            args.model,
            delay=args.delay,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
        )
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} relevant reviews to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

