"""Derive aging theories mentioned across filtered full-text reviews.

This final step reads the enriched dataset created by
``step3_fetch_fulltext.py`` and prompts an OpenAI model to extract the names of
aging theories discussed in each review. The script aggregates unique theory
labels across all processed articles and stores both the per-article annotations
and the combined set for downstream ontology building.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for the OpenAI chat completions API.

Usage
-----
```bash
python scripts/step4_extract_theories.py \
    --input data/pipeline/filtered_reviews_fulltext.json \
    --output data/pipeline/aging_theories.json
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Dict, Iterable, List, Set
import urllib.error
import urllib.request


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _normalise_theory_list(payload: Dict) -> Dict:
    theories = payload.get("theories")
    if theories is None:
        return payload
    if isinstance(theories, list):
        cleaned = []
        for item in theories:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    cleaned.append(name)
        payload["theories"] = cleaned
    elif isinstance(theories, str):
        payload["theories"] = [theories.strip()] if theories.strip() else []
    else:
        payload["theories"] = []
    return payload


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
                        "Extract distinct aging theories from the supplied review "
                        "text. Return JSON with keys `theories` (list of strings) "
                        "and `notes` (optional explanatory text)."
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
    except urllib.error.HTTPError as exc:  # pragma: no cover - network fallback
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

    return _normalise_theory_list(parsed)


def build_prompt(record: Dict, max_chars: int) -> str:
    text = record.get("full_text") or record.get("abstract") or ""
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
        text = text.rsplit(" ", 1)[0]  # avoid cutting mid-word
        text += "..."
    title = record.get("title", "")
    citation = record.get("journal", "")
    return textwrap.dedent(
        f"""
        Review title: {title}
        Citation source: {citation}

        Extract every distinct theory of aging discussed in this review. Focus on
        conceptual theories (e.g., oxidative stress theory, disposable soma,
        network theory). Ignore unrelated topics.

        Review text:
        {text}
        """
    ).strip()


def extract_theories(records: Iterable[Dict], api_key: str, model: str, delay: float, max_chars: int) -> List[Dict]:
    records_list = list(records)
    annotated: List[Dict] = []
    for idx, record in enumerate(records_list, start=1):
        prompt = build_prompt(record, max_chars)
        response = call_openai(prompt, api_key, model)
        annotated_record = record.copy()
        annotated_record["theory_extraction"] = response
        annotated.append(annotated_record)
        print(f"Annotated {idx}/{len(records_list)} reviews", flush=True)
        time.sleep(delay)
    return annotated


def aggregate_theories(annotated: Iterable[Dict]) -> List[str]:
    theories: Set[str] = set()
    for record in annotated:
        extracted = record.get("theory_extraction", {}).get("theories") or []
        for name in extracted:
            if isinstance(name, str):
                cleaned = name.strip()
                if cleaned:
                    theories.add(cleaned)
    return sorted(theories)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract aging theories from filtered reviews")
    parser.add_argument("--input", default="data/pipeline/filtered_reviews_fulltext.json")
    parser.add_argument("--output", default="data/pipeline/aging_theories.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum number of characters from each review to send to the LLM.",
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    annotated = extract_theories(records, api_key, args.model, args.delay, args.max_chars)
    aggregated = aggregate_theories(annotated)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({"articles": annotated, "aggregated_theories": aggregated}, fh, ensure_ascii=False, indent=2)

    print(f"Saved theory catalogue with {len(aggregated)} unique theories to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

