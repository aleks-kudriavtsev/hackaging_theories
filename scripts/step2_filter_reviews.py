"""Filter aging-theory reviews using an OpenAI relevance check.

The script loads the JSON file produced by the metadata collection steps and
asks an OpenAI chat model to decide whether each article is relevant to aging
theory based on its title and abstract. Records marked as irrelevant are
discarded. Large batches are automatically processed in parallel worker
processes so long runs saturate the available CPU cores.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for calling the chat completion endpoint.

Usage
-----
```bash
python scripts/step2_filter_reviews.py \
    --input data/pipeline/start_reviews.json \
    --output data/pipeline/filtered_reviews.json \
    --processes 4
```
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from multiprocessing import Process, Queue
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
                temperature=1,
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


def call_openai(prompt: str, api_key: str, model: str) -> Dict:
    """Backward-compatible synchronous wrapper around :func:`_call_openai`."""

    async def _run() -> Dict:
        async with AsyncOpenAI(api_key=api_key) as client:  # type: ignore[call-arg]
            semaphore = asyncio.Semaphore(1)
            return await _call_openai(client, prompt, model, semaphore, delay=0)

    return asyncio.run(_run())


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


def _normalize_item(item: object) -> Dict | None:
    if not isinstance(item, dict):
        return None

    relevant = _coerce_bool(item.get("relevant"))
    if relevant is None:
        return None

    normalized = dict(item)
    normalized["relevant"] = relevant
    if "explanation" not in normalized or normalized["explanation"] is None:
        normalized["explanation"] = ""

    return normalized


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

    if len(batch) == 1:
        single_item = _normalize_item(response)
        if single_item is not None:
            processed[0] = single_item
            return processed, []

    for idx in range(len(batch)):
        key = str(idx + 1)
        item_data = _normalize_item(response.get(key))
        if item_data is None:
            fallback.append(idx)
            continue

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
    progress_prefix: str | None = None,
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
        prefix = f"[{progress_prefix}] " if progress_prefix else ""

        async def log_progress(relevant: bool) -> None:
            nonlocal processed_total, kept_total
            async with progress_lock:
                processed_total += 1
                if relevant:
                    kept_total += 1
                print(
                    f"{prefix}Processed {processed_total} records — kept {kept_total}",
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
                    f"{prefix}Batch {start + 1}-{start + len(batch)} failed with error: {err}. "
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
                        f"{prefix}Record {start + offset + 1} failed after retry: {err}",
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
                        f"{prefix}Record {start + offset + 1} returned invalid JSON even after retry.",
                        file=sys.stderr,
                    )
                if pending:
                    print(
                        f"{prefix}Record {start + offset + 1} still pending after retry; giving up.",
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


def filter_records(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    *,
    delay: float = 0.5,
    batch_size: int = 5,
    concurrency: int = 5,
) -> List[Dict]:
    """Synchronous helper mirroring :func:`async_filter_records`.

    The implementation swaps the internal coroutine transport so legacy callers
    that monkeypatch :func:`call_openai` continue to work. The heavy lifting is
    still delegated to :func:`async_filter_records` so behaviour stays aligned
    between both entry points.
    """

    async def _compat_call_openai(
        client: AsyncOpenAI,
        prompt: str,
        model_name: str,
        semaphore: asyncio.Semaphore,
        call_delay: float,
    ) -> Dict:
        async with semaphore:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, call_openai, prompt, api_key, model_name
            )
            if call_delay > 0:
                await asyncio.sleep(call_delay)
            return result

    original_call = _call_openai
    try:
        globals()["_call_openai"] = _compat_call_openai
        return asyncio.run(
            async_filter_records(
                records,
                api_key,
                model,
                delay=delay,
                batch_size=batch_size,
                concurrency=concurrency,
            )
        )
    finally:
        globals()["_call_openai"] = original_call


def _split_records(records: List[Dict], parts: int) -> List[List[Dict]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if not records:
        return [[] for _ in range(parts)]

    chunk_sizes = []
    base, remainder = divmod(len(records), parts)
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        chunk_sizes.append(size)

    chunks: List[List[Dict]] = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(list(records[start:end]))
        start = end

    return chunks


def _worker_filter(
    index: int,
    records: List[Dict],
    queue: "Queue[Tuple[int, List[Dict] | None, str | None]]",
    api_key: str,
    model: str,
    *,
    delay: float,
    batch_size: int,
    concurrency: int,
) -> None:
    prefix = f"worker-{index + 1}"
    try:
        kept = asyncio.run(
            async_filter_records(
                records,
                api_key,
                model,
                delay=delay,
                batch_size=batch_size,
                concurrency=concurrency,
                progress_prefix=prefix,
            )
        )
    except Exception as exc:  # pragma: no cover - inter-process error propagation
        queue.put((index, None, str(exc)))
        return

    queue.put((index, kept, None))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter aging-theory reviews with OpenAI")
    parser.add_argument("--input", default="data/pipeline/start_reviews.json")
    parser.add_argument("--output", default="data/pipeline/filtered_reviews.json")
    parser.add_argument(
        "--model",
        default="gpt-5-nano",
        help=(
            "OpenAI chat completion model identifier. The default gpt-5-nano tier "
            "keeps this filtering stage inside the ~$10 per million articles "
            "budget while preserving dependable relevance calls on batched "
            "abstracts."
        ),
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
            "within the model's context window (e.g., 5-10 abstracts for gpt-5-nano)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent OpenAI requests.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Number of OS processes used for filtering. When omitted the script "
            "auto-scales based on the record volume, aiming for roughly 25 reviews "
            "per worker while capping usage at the available CPU cores."
        ),
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    total_records = len(records)
    if args.processes is not None and args.processes <= 0:
        print("--processes must be a positive integer", file=sys.stderr)
        return 2

    cpu_total = os.cpu_count() or 1
    min_records_per_worker = 25
    if total_records:
        auto_processes = min(cpu_total, max(1, math.ceil(total_records / min_records_per_worker)))
    else:
        auto_processes = 1
    processes = args.processes or auto_processes
    if total_records > 0:
        processes = min(processes, total_records)
    else:
        processes = 1

    if total_records == 0:
        kept: List[Dict] = []
    elif processes <= 1 or total_records <= 1:
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
    else:
        queue: "Queue[Tuple[int, List[Dict] | None, str | None]]" = Queue()
        chunks = _split_records(list(records), processes)
        jobs = [
            Process(
                target=_worker_filter,
                args=(idx, chunk, queue, api_key, args.model),
                kwargs={
                    "delay": args.delay,
                    "batch_size": args.batch_size,
                    "concurrency": args.concurrency,
                },
            )
            for idx, chunk in enumerate(chunks)
            if chunk
        ]

        if not jobs:
            kept = []
        else:
            for job in jobs:
                job.start()

            results: List[Tuple[int, List[Dict]]] = []
            errors: List[Tuple[int, str]] = []
            for _ in range(len(jobs)):
                index, chunk_result, error = queue.get()
                if error:
                    errors.append((index, error))
                elif chunk_result is not None:
                    results.append((index, chunk_result))

            for job in jobs:
                job.join()

            if errors:
                for index, error in errors:
                    print(
                        f"worker-{index + 1} failed with error: {error}",
                        file=sys.stderr,
                    )
                return 3

            kept = [record for _, chunk in sorted(results, key=lambda item: item[0]) for record in chunk]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} relevant reviews to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

