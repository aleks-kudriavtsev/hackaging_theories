"""Synthesize a hierarchical ontology from the canonical theory registry.

This script consumes the output of ``step4_extract_theories.py`` (the list of
articles with their per-review theory annotations) and asks an OpenAI model to
bootstrap a multi-level ontology.  Instead of prompting on raw theory strings
it now relies on the canonical theory registry produced during step 4.  Each
canonical theory is summarised with its identifier, preferred label, aliases
and the exact list of supporting article IDs.  The LLM is asked to organise the
registry into groups and subgroups while *preserving* theory identifiers and
the associated article references in its response.

Once the LLM proposes a hierarchical layout the script performs a
post-processing reconciliation step.  Every article→theory link is validated
against the registry, stray assignments are removed, missing articles are
reinstated and any theories the LLM omitted are appended to a dedicated
"Ungrouped" cluster.  The final payload therefore contains both the original
LLM structure (for transparency) and the reconciled ontology with explicit
group/subgroup/theory nodes embedding the verified supporting article IDs.
A reconciliation report summarises all adjustments for downstream audit.

Environment variables
---------------------
- ``OPENAI_API_KEY`` — required for the OpenAI chat completions API.

Usage
-----
```bash
python scripts/step5_generate_ontology.py \
    --input data/pipeline/aging_theories.json \
    --output data/pipeline/aging_ontology.json \
    [--processes 4] [--chunk-size 120]
```
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import multiprocessing
import os
import sys
import textwrap
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


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
                        "must contain 'name', optional 'description', optional "
                        "'subgroups', and a 'theories' list. Theories may include "
                        "nested 'children'."
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


def _chunk_summary(
    summary: Sequence[Mapping[str, object]],
    *,
    chunk_size: int,
) -> Iterable[List[Mapping[str, object]]]:
    chunk_size = max(1, chunk_size)
    for start in range(0, len(summary), chunk_size):
        yield [dict(item) for item in summary[start : start + chunk_size]]


def _summarise_registry(
    registry: Mapping[str, Mapping[str, object]],
    *,
    limit: int,
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for theory_id, payload in registry.items():
        if not isinstance(payload, Mapping):
            continue
        label = payload.get("label") if isinstance(payload.get("label"), str) else None
        aliases = [alias for alias in payload.get("aliases", []) if isinstance(alias, str)]
        articles = [
            str(article_id)
            for article_id in payload.get("supporting_articles", [])
            if isinstance(article_id, str)
        ]
        summary.append(
            {
                "theory_id": theory_id,
                "label": label or theory_id,
                "aliases": aliases,
                "article_count": len(articles),
                "supporting_articles": articles,
            }
        )

    summary.sort(key=lambda item: (-item["article_count"], item["label"].lower()))
    if limit > 0:
        summary = summary[:limit]
    return summary


def generate_grouping(
    summary_chunk: Sequence[Mapping[str, object]],
    total_unique: int,
    model: str,
    api_key: str,
) -> Dict[str, object]:
    """Generate ontology groups for a subset of the registry summary."""

    prompt = _build_prompt(summary_chunk, total_unique)
    return _call_openai(prompt, api_key, model)


def _build_prompt(summary: Sequence[Mapping[str, object]], total_unique: int) -> str:
    theories_block = json.dumps(summary, ensure_ascii=False, indent=2)
    return textwrap.dedent(
        f"""
        You are designing an ontology for theories of aging based on literature
        review evidence. There are {total_unique} canonical theories in total.
        The JSON list below summarises each theory with its identifier, label,
        aliases and supporting article IDs.

        Instructions:
        - Group related theories under a higher-level "group" when they share a
          conceptual theme (e.g., damage accumulation, programmed aging,
          sociocultural perspectives).
        - A group may optionally contain nested "subgroups" to provide a
          hierarchy deeper than two levels.
        - Within each group, include a "theories" array.  Each theory entry must
          reference a canonical ``theory_id`` provided in the summary and may
          optionally provide a "preferred_label" and "description".
        - You may include nested "children" arrays inside a theory to capture
          subtheories, but those child theories must also reference valid
          ``theory_id`` values.
        - **Do not invent new theory IDs or article IDs.** Preserve the supplied
          ``supporting_articles`` list for every theory you place.
        - Respond with JSON containing a top-level key "groups".

        Summary of canonical theories:
        {theories_block}
        """
    ).strip()


def _canonical_article_index(
    registry: Mapping[str, Mapping[str, object]]
) -> Dict[str, List[str]]:
    article_index: Dict[str, List[str]] = {}
    for theory_id, payload in registry.items():
        articles = payload.get("supporting_articles") if isinstance(payload, Mapping) else None
        if not isinstance(articles, Sequence) or isinstance(articles, (str, bytes)):
            continue
        for article_id in articles:
            if isinstance(article_id, str):
                article_index.setdefault(article_id, []).append(theory_id)
    for theory_ids in article_index.values():
        theory_ids.sort()
    return article_index


def _deduplicate_group(
    group_node: Mapping[str, Any],
    *,
    seen_theories: Set[str],
) -> Optional[Dict[str, Any]]:
    name = group_node.get("name") or group_node.get("label")
    if not isinstance(name, str) or not name.strip():
        name = "Unnamed group"

    payload: Dict[str, Any] = {"name": name.strip()}
    description = group_node.get("description")
    if isinstance(description, str) and description.strip():
        payload["description"] = description.strip()

    subgroups: List[Dict[str, Any]] = []
    for key in ("subgroups", "children"):
        for subgroup in group_node.get(key, []) or []:
            if isinstance(subgroup, Mapping):
                processed = _deduplicate_group(subgroup, seen_theories=seen_theories)
                if processed:
                    subgroups.append(processed)
    if subgroups:
        payload["subgroups"] = subgroups

    theories: List[Dict[str, Any]] = []
    for theory in group_node.get("theories", []) or []:
        if not isinstance(theory, Mapping):
            continue
        theory_id = theory.get("theory_id") or theory.get("id")
        if not isinstance(theory_id, str):
            continue
        if theory_id in seen_theories:
            continue
        seen_theories.add(theory_id)
        theories.append(dict(theory))
    if theories:
        payload["theories"] = theories

    if "subgroups" not in payload and "theories" not in payload:
        return None
    return payload


def _merge_groupings(responses: Sequence[Mapping[str, object]]) -> List[Dict[str, Any]]:
    seen_theories: Set[str] = set()
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for response in responses:
        normalised = _normalise_groups(response)
        for group in normalised.get("groups", []) or []:
            if not isinstance(group, Mapping):
                continue
            processed = _deduplicate_group(group, seen_theories=seen_theories)
            if not processed:
                continue
            key = processed["name"].lower()
            if key not in merged:
                merged[key] = {"name": processed["name"]}
                order.append(key)
            target = merged[key]
            if processed.get("description") and not target.get("description"):
                target["description"] = processed["description"]
            if processed.get("theories"):
                target.setdefault("theories", []).extend(processed["theories"])
            if processed.get("subgroups"):
                target.setdefault("subgroups", []).extend(processed["subgroups"])

    return [merged[key] for key in order]


def _reconcile_theory_node(
    node: Mapping[str, Any],
    registry: Mapping[str, Mapping[str, object]],
    *,
    article_index: Mapping[str, Sequence[str]],
    seen: MutableMapping[str, str],
    reconciliation: Dict[str, List[Dict[str, object]]],
) -> Optional[Dict[str, Any]]:
    theory_id = node.get("theory_id") or node.get("id")
    if not isinstance(theory_id, str):
        reconciliation.setdefault("dropped_theories", []).append(
            {
                "reason": "Missing canonical theory_id",
                "raw_node": node,
            }
        )
        return None

    if theory_id not in registry:
        reconciliation.setdefault("dropped_theories", []).append(
            {
                "reason": "Theory ID not present in canonical registry",
                "theory_id": theory_id,
            }
        )
        return None

    if theory_id in seen:
        reconciliation.setdefault("duplicate_theories", []).append(
            {
                "theory_id": theory_id,
                "first_location": seen[theory_id],
                "second_location": node.get("preferred_label") or node.get("label"),
            }
        )
        return None

    seen[theory_id] = node.get("preferred_label") or node.get("label") or registry[theory_id].get("label") or theory_id

    canonical_entry = registry[theory_id]
    canonical_articles = [
        article_id
        for article_id in canonical_entry.get("supporting_articles", [])
        if isinstance(article_id, str)
    ]

    llm_articles = [
        article_id
        for article_id in node.get("supporting_articles", [])
        if isinstance(article_id, str)
    ]

    canonical_set = set(canonical_articles)
    llm_set = set(llm_articles)

    removed_articles = sorted(llm_set - canonical_set)
    added_articles = sorted(canonical_set - llm_set)

    for article_id in removed_articles:
        reconciliation.setdefault("reassignments", []).append(
            {
                "article_id": article_id,
                "from_theory_id": theory_id,
                "to_theory_ids": list(article_index.get(article_id, [])),
                "action": "removed",
                "reason": "Article not linked to theory in canonical registry",
            }
        )

    for article_id in added_articles:
        reconciliation.setdefault("reassignments", []).append(
            {
                "article_id": article_id,
                "from_theory_id": None,
                "to_theory_ids": [theory_id],
                "action": "added",
                "reason": "Article linked to theory in canonical registry but missing from LLM output",
            }
        )

    processed_children: List[Dict[str, Any]] = []
    for child in node.get("children", []) or []:
        if isinstance(child, Mapping):
            reconciled = _reconcile_theory_node(
                child,
                registry,
                article_index=article_index,
                seen=seen,
                reconciliation=reconciliation,
            )
            if reconciled:
                processed_children.append(reconciled)

    final_node: Dict[str, Any] = {
        "theory_id": theory_id,
        "label": canonical_entry.get("label"),
        "preferred_label": node.get("preferred_label") or node.get("label") or canonical_entry.get("label"),
        "aliases": canonical_entry.get("aliases", []),
        "supporting_articles": canonical_articles,
    }
    if node.get("description"):
        final_node["description"] = node["description"]
    if processed_children:
        final_node["children"] = processed_children

    return final_node


def _reconcile_groups(
    groups: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Mapping[str, object]],
    *,
    article_index: Mapping[str, Sequence[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, object]]]]:
    reconciliation: Dict[str, List[Dict[str, object]]] = {}
    seen: Dict[str, str] = {}

    def process_group(group_node: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        name = group_node.get("name") or group_node.get("label")
        if not isinstance(name, str) or not name.strip():
            name = "Unnamed group"
        description = group_node.get("description") if isinstance(group_node.get("description"), str) else None

        subgroups_raw: List[Mapping[str, Any]] = []
        for child in group_node.get("subgroups", []) or []:
            if isinstance(child, Mapping):
                subgroups_raw.append(child)
        for child in group_node.get("children", []) or []:
            if isinstance(child, Mapping):
                subgroups_raw.append(child)

        theories_raw: List[Mapping[str, Any]] = []
        for theory in group_node.get("theories", []) or []:
            if isinstance(theory, Mapping):
                theories_raw.append(theory)

        processed_subgroups: List[Dict[str, Any]] = []
        for subgroup in subgroups_raw:
            processed = process_group(subgroup)
            if processed:
                processed_subgroups.append(processed)

        processed_theories: List[Dict[str, Any]] = []
        for theory_node in theories_raw:
            reconciled = _reconcile_theory_node(
                theory_node,
                registry,
                article_index=article_index,
                seen=seen,
                reconciliation=reconciliation,
            )
            if reconciled:
                processed_theories.append(reconciled)

        if not processed_subgroups and not processed_theories:
            return None

        group_payload: Dict[str, Any] = {
            "name": name.strip(),
        }
        if description:
            group_payload["description"] = description.strip()
        if processed_subgroups:
            group_payload["subgroups"] = processed_subgroups
        if processed_theories:
            group_payload["theories"] = processed_theories
        return group_payload

    processed_groups: List[Dict[str, Any]] = []
    for group_node in groups:
        if isinstance(group_node, Mapping):
            processed = process_group(group_node)
            if processed:
                processed_groups.append(processed)

    missing_theories = [theory_id for theory_id in registry.keys() if theory_id not in seen]
    if missing_theories:
        fallback_group: Dict[str, Any] = {
            "name": "Ungrouped canonical theories",
            "description": "Canonical theories not placed by the LLM grouping.",
            "theories": [],
        }
        for theory_id in missing_theories:
            entry = registry[theory_id]
            canonical_articles = [
                article_id
                for article_id in entry.get("supporting_articles", [])
                if isinstance(article_id, str)
            ]
            fallback_group["theories"].append(
                {
                    "theory_id": theory_id,
                    "label": entry.get("label"),
                    "preferred_label": entry.get("label"),
                    "aliases": entry.get("aliases", []),
                    "supporting_articles": canonical_articles,
                }
            )
        processed_groups.append(fallback_group)
        reconciliation.setdefault("unplaced_theories", []).append(
            {
                "theory_ids": missing_theories,
                "reason": "Added fallback group for theories omitted by LLM ontology",
            }
        )

    return processed_groups, reconciliation


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an ontology from extracted theories")
    parser.add_argument("--input", default="data/pipeline/aging_theories.json")
    parser.add_argument("--output", default="data/pipeline/aging_ontology.json")
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help=(
            "OpenAI chat completion model identifier. gpt-5-mini is the default "
            "for ontology assembly, providing broader synthesis quality than the "
            "nano tier while staying within the ~$10 per million articles budget."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=60,
        help="Maximum number of theories to include in the ontology prompt (0 for all).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Number of worker processes for ontology generation. If omitted, the "
            "script auto-scales to the available CPU count whenever the summary "
            "exceeds the chunk threshold, otherwise it runs in a single process."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help=(
            "Maximum number of theories per chunk when splitting the registry summary. "
            "Chunks are used when more than one process is requested or when the summary exceeds this threshold."
        ),
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

    registry = data.get("theory_registry") if isinstance(data, Mapping) else None
    if not isinstance(registry, Mapping):
        print("Input JSON is missing the 'theory_registry' mapping", file=sys.stderr)
        return 1

    total_unique = len(registry)
    summary = _summarise_registry(registry, limit=max(args.top_n, 0))

    if total_unique == 0:
        print("No canonical theories available in registry", file=sys.stderr)
        return 1

    chunk_threshold = max(args.chunk_size, 1)

    try:
        cpu_total = multiprocessing.cpu_count()
    except NotImplementedError:  # pragma: no cover - defensive guard
        cpu_total = 1

    auto_processes = cpu_total if len(summary) > chunk_threshold else 1

    if args.processes is None:
        processes = auto_processes
        requested_processes: Optional[int] = None
    else:
        processes = args.processes
        requested_processes = args.processes

    process_source = (
        "auto" if requested_processes is None or processes == auto_processes else "manual"
    )

    if processes < 1:
        print("--processes must be at least 1", file=sys.stderr)
        return 1

    should_chunk = processes > 1 or len(summary) > chunk_threshold

    print(
        f"Using {processes} worker process{'es' if processes != 1 else ''} "
        f"({process_source}; auto suggestion: {auto_processes})."
    )

    if should_chunk:
        summary_chunks = list(_chunk_summary(summary, chunk_size=chunk_threshold))
        worker_args = [
            (chunk, total_unique, args.model, api_key)
            for chunk in summary_chunks
        ]

        if processes > 1 and len(summary_chunks) > 1:
            with multiprocessing.Pool(processes=processes) as pool:
                chunk_responses = pool.starmap(generate_grouping, worker_args)
        else:
            chunk_responses = [
                generate_grouping(chunk, total_unique, args.model, api_key)
                for chunk in summary_chunks
            ]

        merged_groups = _merge_groupings(chunk_responses)
        response = {"groups": merged_groups, "chunks": chunk_responses}
        groups_for_reconciliation = merged_groups
    else:
        response = generate_grouping(summary, total_unique, args.model, api_key)
        groups_for_reconciliation = response.get("groups", [])

    article_index = _canonical_article_index(registry)
    reconciled_groups, reconciliation = _reconcile_groups(
        groups_for_reconciliation,
        registry,
        article_index=article_index,
    )

    output_payload = {
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "input_file": args.input,
        "total_unique_theories": total_unique,
        "worker_processes": processes,
        "auto_processes_suggestion": auto_processes,
        "requested_processes": requested_processes,
        "prompt_summary": summary,
        "ontology": {
            "raw": response,
            "final": {
                "groups": reconciled_groups,
            },
        },
        "reconciliation_report": {
            "total_groups": len(reconciled_groups),
            "total_reassignments": len(reconciliation.get("reassignments", [])),
            "worker_processes": processes,
            "process_configuration": process_source,
            "auto_processes_suggestion": auto_processes,
            "requested_processes": requested_processes,
            **reconciliation,
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, ensure_ascii=False, indent=2)

    print(
        "Saved reconciled ontology with "
        f"{len(reconciled_groups)} groups to {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
