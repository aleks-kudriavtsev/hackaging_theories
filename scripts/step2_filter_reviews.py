"""Filter PubMed reviews using an OpenAI relevance check.

The script loads the JSON file produced by ``step1_pubmed_search.py`` and asks
an OpenAI chat model to decide whether each article is relevant to aging theory
based on its title and abstract. Records marked as irrelevant are discarded.

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
from typing import Dict, Iterable, List
import urllib.request
import urllib.error
import urllib.parse


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


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
    return json.loads(content)


def build_prompt(record: Dict) -> str:
    abstract = record.get("abstract") or ""
    return (
        "Determine if the following PubMed review is primarily about aging "
        "theory. Reply with JSON.\n"
        f"Title: {record.get('title', '')}\n"
        f"Abstract: {abstract}"
    )


def filter_records(records: Iterable[Dict], api_key: str, model: str, delay: float = 0.5) -> List[Dict]:
    kept: List[Dict] = []
    for idx, record in enumerate(records, start=1):
        prompt = build_prompt(record)
        decision = call_openai(prompt, api_key, model=model)
        record["llm_filter"] = decision
        if decision.get("relevant") is True:
            kept.append(record)
        # Friendly progress indicator for longer runs.
        print(
            f"Processed {idx} records — kept {len(kept)}",
            flush=True,
        )
        time.sleep(delay)
    return kept


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter PubMed reviews with OpenAI")
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
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    kept = filter_records(records, api_key, args.model, delay=args.delay)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(kept)} relevant reviews to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

