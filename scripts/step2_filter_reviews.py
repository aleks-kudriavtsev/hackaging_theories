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
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple
import urllib.request
import urllib.error
import urllib.parse


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


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


def call_openai(prompt: str, api_key: str, model: str) -> Dict:
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert gerontology curator. Respond with JSON "
                        "containing `relevant` (true/false) and `explanation`."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        OPENAI_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network error handling
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI API error {exc.code}: {error_body}") from exc

    data = json.loads(body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as err:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unexpected OpenAI response: {data!r}") from err

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
        "Return a JSON object where each key is the item number (e.g., \"1\").",
        "Each value must be an object with fields `relevant` (true/false) and `explanation`.",
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


def filter_records(
    records: Iterable[Dict],
    api_key: str,
    model: str,
    delay: float = 0.5,
    batch_size: int = 5,
) -> List[Dict]:
    records_list = list(records)
    kept: List[Dict] = []
    processed_total = 0

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, len(records_list), batch_size):
        batch = records_list[start : start + batch_size]
        prompt = build_prompt(batch)
        processed_items: Dict[int, Dict] = {}
        fallback_indices: List[int] = []
        try:
            response = call_openai(prompt, api_key, model=model)
        except RuntimeError as err:
            print(
                f"Batch {start + 1}-{start + len(batch)} failed with error: {err}. "
                "Retrying items individually.",
                file=sys.stderr,
            )
            fallback_indices = list(range(len(batch)))
        else:
            processed_items, fallback_indices = _parse_batch_response(batch, response)
        finally:
            time.sleep(delay)

        for offset, decision in processed_items.items():
            record = batch[offset]
            record["llm_filter"] = decision
            if decision.get("relevant") is True:
                kept.append(record)
            processed_total += 1
            print(f"Processed {processed_total} records — kept {len(kept)}", flush=True)

        for offset in fallback_indices:
            record = batch[offset]
            single_prompt = build_prompt([record])
            try:
                single_response = call_openai(single_prompt, api_key, model=model)
            except RuntimeError as err:
                print(
                    f"Record {start + offset + 1} failed after retry: {err}",
                    file=sys.stderr,
                )
                processed_total += 1
                print(
                    f"Processed {processed_total} records — kept {len(kept)}",
                    flush=True,
                )
                time.sleep(delay)
                continue

            time.sleep(delay)
            single_items, pending = _parse_batch_response([record], single_response)
            decision = single_items.get(0)
            if decision is not None:
                record["llm_filter"] = decision
                if decision.get("relevant") is True:
                    kept.append(record)
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
            processed_total += 1
            print(f"Processed {processed_total} records — kept {len(kept)}", flush=True)

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
        help="Number of records to include in each LLM request.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    kept = filter_records(
        records,
        api_key,
        args.model,
        delay=args.delay,
        batch_size=args.batch_size,
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

