"""Synthesize a hierarchical ontology from extracted aging theories.

This script consumes the output of ``step4_extract_theories.py`` (the list of
articles with their per-review theory annotations) and asks an OpenAI model to
bootstrap a multi-level ontology.  The model receives a condensed summary of
every discovered theory together with article counts and representative review
titles.  It responds with grouped theory clusters that may include nested
subtheories when a parent node accumulates substantial literature support.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for the OpenAI chat completions API.

Usage
-----
```bash
python scripts/step5_generate_ontology.py \
    --input data/pipeline/aging_theories.json \
    --output data/pipeline/aging_ontology.json
```
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Mapping, Sequence


OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _normalise_groups(payload: Mapping[str, object]) -> Dict[str, object]:
    """Ensure the OpenAI payload has a predictable ``groups`` list."""

    groups = payload.get("groups")
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        normalised_groups: List[object] = []
        for entry in groups:
            if isinstance(entry, Mapping):
                normalised_groups.append(dict(entry))
        payload = dict(payload)
        payload["groups"] = normalised_groups
        return payload
    payload = dict(payload)
    payload["groups"] = []
    return payload


def _call_openai(prompt: str, api_key: str, model: str) -> Dict[str, object]:
    payload = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ontology architect specialising in aging and "
                        "geroscience. Group related theories into parent clusters "
                        "and create subtheories for popular themes. Use the supplied "
                        "statistics to decide when to elevate a parent group. "
                        "Return JSON with a top-level 'groups' list. Each group "
                        "must contain 'name', optional 'description', and a "
                        "'theories' list. Theories may include nested 'children'."
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
        raise RuntimeError("OpenAI returned invalid JSON payload: " + content) from err

    return _normalise_groups(parsed)


def _summarise_theories(
    articles: Iterable[Mapping[str, object]],
    *,
    limit: int,
    max_examples: int,
) -> List[Dict[str, object]]:
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}

    for record in articles:
        title = record.get("title") if isinstance(record, Mapping) else None
        if not isinstance(title, str):
            title = ""
        theories = (
            record.get("theory_extraction", {})
            if isinstance(record, Mapping)
            else {}
        )
        if not isinstance(theories, Mapping):
            continue
        labels = theories.get("theories")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            continue
        seen: set[str] = set()
        for raw in labels:
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            counts[name] = counts.get(name, 0) + 1
            if name not in examples:
                examples[name] = []
            if title and len(examples[name]) < max_examples:
                examples[name].append(title)

    sorted_theories = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    if limit > 0:
        sorted_theories = sorted_theories[:limit]

    summary: List[Dict[str, object]] = []
    for name, count in sorted_theories:
        summary.append(
            {
                "name": name,
                "article_count": count,
                "sample_titles": examples.get(name, [])[:max_examples],
            }
        )
    return summary


def _build_prompt(summary: Sequence[Mapping[str, object]], total_unique: int) -> str:
    theories_block = json.dumps(summary, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        You are designing an ontology for theories of aging based on literature
        review evidence. There are {total_unique} unique theories overall. The
        JSON list below summarises the most frequently cited theories with their
        article counts and representative review titles.

        Instructions:
        - Group related theories under a higher-level "group" when they share a
          conceptual theme (e.g., damage accumulation, programmed aging,
          sociocultural perspectives).
        - Within each group, list individual theories ordered from most to least
          supported by article_count.
        - When a theory itself covers many articles or naturally decomposes into
          multiple variants, include a "children" list to capture the
          subtheories.
        - Provide concise "description" fields for groups and for any theory
          that has children so downstream analysts understand the distinctions.
        - Preserve the exact theory names as they appear in the summary whenever
          they are used as nodes.

        Respond with JSON containing a top-level key "groups".

        Summary of theories:
        {theories_block}
        """
    ).strip()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an ontology from extracted theories")
    parser.add_argument("--input", default="data/pipeline/aging_theories.json")
    parser.add_argument("--output", default="data/pipeline/aging_ontology.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Maximum number of theories to include in the ontology prompt (0 for all).",
    )
    parser.add_argument(
        "--examples-per-theory",
        type=int,
        default=3,
        help="How many article titles to include as context for each theory.",
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
        data = json.load(fh)

    articles = data.get("articles") if isinstance(data, Mapping) else None
    if not isinstance(articles, Sequence):
        print("Input JSON is missing the 'articles' array", file=sys.stderr)
        return 1

    total_unique = 0
    aggregated = data.get("aggregated_theories") if isinstance(data, Mapping) else None
    if isinstance(aggregated, Sequence) and not isinstance(aggregated, (str, bytes)):
        total_unique = sum(1 for item in aggregated if isinstance(item, str))

    summary = _summarise_theories(
        articles,
        limit=max(args.top_n, 0),
        max_examples=max(args.examples_per_theory, 0),
    )
    if total_unique == 0:
        total_unique = len(summary)

    prompt = _build_prompt(summary, total_unique)
    response = _call_openai(prompt, api_key, args.model)

    output_payload = {
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "input_file": args.input,
        "summary": summary,
        "total_unique_theories": total_unique,
        "ontology": response,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, ensure_ascii=False, indent=2)

    print(f"Saved ontology with {len(response.get('groups', []))} groups to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
