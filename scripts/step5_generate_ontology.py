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
    [--processes 4] [--chunk-size 120] [--examples-per-theory 3]
```
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import datetime as _dt
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# The OpenAI ``gpt-5-mini`` model currently accepts up to ~272k tokens.  The
# prompt builder below therefore keeps a safety margin and attempts to stay
# within ~240k tokens (roughly 960k UTF-8 characters assuming 4 characters per
# token on average).  When the registry summary exceeds this budget the script
# progressively truncates long ``supporting_articles`` and
# ``representative_articles`` lists before calling the API.
PROMPT_TOKEN_BUDGET = 240_000
AVERAGE_CHARS_PER_TOKEN = 4
PROMPT_CHARACTER_BUDGET = PROMPT_TOKEN_BUDGET * AVERAGE_CHARS_PER_TOKEN

# Caps applied when shrinking large summaries.  ``supporting_articles`` is
# trimmed first, gradually reducing the number of article identifiers included
# in the prompt.  If the payload still exceeds the budget the representative
# article metadata is reduced as well, eventually falling back to an empty list
# when absolutely necessary.
SUPPORTING_ARTICLE_CAPS = (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 0)
REPRESENTATIVE_ARTICLE_CAPS = (40, 20, 10, 5, 2, 1, 0)

# Soft cap applied when reconciling oversized ontology groups. When more than
# this number of theories end up directly under the same group, the overflow is
# redistributed into auto-generated subgroups. The value mirrors the
# ``--max-theories-per-group`` CLI option default.
DEFAULT_MAX_THEORIES_PER_GROUP = 40


def _load_json_payload(path: str) -> Mapping[str, object] | Sequence[object]:
    """Load a JSON payload, tolerating concatenated JSON documents."""

    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as err:
            if "Extra data" not in err.msg:
                raise

            fh.seek(0)
            content = fh.read()

    decoder = json.JSONDecoder()
    index = 0
    documents: List[object] = []
    length = len(content)
    while index < length:
        while index < length and content[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            payload, next_index = decoder.raw_decode(content, index)
        except json.JSONDecodeError as fallback_err:
            raise RuntimeError(
                f"Failed to parse JSON payload from {path}: {fallback_err}"
            ) from fallback_err
        documents.append(payload)
        index = next_index

    if not documents:
        raise RuntimeError(f"No JSON documents found in {path}")

    if len(documents) == 1:
        payload = documents[0]
        if isinstance(payload, (Mapping, Sequence)):
            return payload
        raise RuntimeError(
            f"JSON document in {path} is not a mapping or sequence: {type(payload)!r}"
        )

    for payload in documents:
        if isinstance(payload, Mapping) and "theory_registry" in payload:
            print(
                "Warning: multiple JSON documents detected; using the first one "
                "that contains a 'theory_registry' key.",
                file=sys.stderr,
            )
            return payload

    return documents


def _coerce_articles_from_documents(
    documents: Sequence[Mapping[str, object]]
) -> List[Dict[str, object]]:
    """Extract article annotations from checkpoint-like JSON documents."""

    if not documents:
        return []

    by_index: Dict[int, Dict[str, object]] = {}
    collected_articles: List[Dict[str, object]] = []

    for entry in documents:
        raw_articles = entry.get("articles")
        if isinstance(raw_articles, Sequence) and not isinstance(raw_articles, (str, bytes)):
            items = [item for item in raw_articles if isinstance(item, Mapping)]
            if items:
                collected_articles.extend(dict(item) for item in items)
                continue

        idx = entry.get("index")
        record = entry.get("record")
        if isinstance(idx, int) and isinstance(record, Mapping):
            by_index[idx] = dict(record)
            continue

        if isinstance(entry.get("theory_extraction"), Mapping):
            collected_articles.append(dict(entry))

    if by_index:
        return [by_index[idx] for idx in sorted(by_index)]

    if collected_articles:
        return collected_articles

    return []


def _load_registry_builder() -> Optional[
    Callable[[Iterable[Dict[str, object]], str, str, Optional[float]], Dict[str, Dict[str, object]]]
]:
    """Return the ``build_theory_registry`` helper even when ``scripts`` isn't importable.

    When ``step5_generate_ontology.py`` is executed directly (``python scripts/...``)
    Python initialises ``sys.path`` with the *scripts* directory instead of the
    project root.  Importing ``scripts.step4_extract_theories`` therefore fails
    because the ``scripts`` package is not visible.  This loader first attempts
    the conventional import and then falls back to loading the module from the
    neighbouring ``step4_extract_theories.py`` file via ``importlib``.
    """

    module_name = "scripts.step4_extract_theories"
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        module = None

    if module is None:
        module_path = Path(__file__).resolve().with_name("step4_extract_theories.py")
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Ensure future imports reuse the loaded module.
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:  # pragma: no cover - defensive guard
                    # Remove the partially initialised module before falling back.
                    sys.modules.pop(module_name, None)
                    module = None

    if module is None:
        return None

    builder = getattr(module, "build_theory_registry", None)
    if callable(builder):
        return builder
    return None


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
            "temperature": 1,
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


def _compute_chunk_size(
    summary_count: int,
    base_chunk_size: int,
    processes: int,
    *,
    min_chunk_size: int,
) -> int:
    min_chunk_size = max(1, min(min_chunk_size, base_chunk_size))
    if summary_count <= 0:
        return max(1, min(base_chunk_size, min_chunk_size))

    divisor = max(processes, 1)
    per_worker = math.ceil(summary_count / divisor)
    chunk = max(min_chunk_size, per_worker)
    chunk = min(chunk, base_chunk_size)
    chunk = min(chunk, summary_count)
    return max(1, chunk)


def _build_article_map(
    records: Sequence[Mapping[str, object]]
) -> Dict[str, Dict[str, object]]:
    article_map: Dict[str, Dict[str, object]] = {}
    for idx, record in enumerate(records):
        article_id: Optional[str] = None
        for key in ("id", "uid", "doi", "openalex_id", "pmid"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                article_id = value.strip()
                break
            if isinstance(value, (int, float)) and value:
                article_id = str(value)
                break
        if not article_id:
            article_id = f"article-{idx + 1}"

        payload: Dict[str, object] = {"id": article_id}

        title = record.get("title")
        if isinstance(title, str) and title.strip():
            payload["title"] = title.strip()

        for key in ("doi", "pmid", "openalex_id", "journal"):
            value = record.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    payload[key] = cleaned

        publication_year = record.get("publication_year")
        if isinstance(publication_year, int):
            payload["publication_year"] = publication_year
        elif isinstance(publication_year, str):
            cleaned_year = publication_year.strip()
            if cleaned_year.isdigit():
                payload["publication_year"] = int(cleaned_year)

        authors = record.get("authors")
        if isinstance(authors, Sequence) and not isinstance(authors, (str, bytes)):
            cleaned_authors = [
                author.strip()
                for author in authors
                if isinstance(author, str) and author.strip()
            ]
            if cleaned_authors:
                payload["authors"] = cleaned_authors[:5]

        article_map[article_id] = payload

    return article_map


def _summarise_registry(
    registry: Mapping[str, Mapping[str, object]],
    *,
    limit: int,
    articles_map: Mapping[str, Mapping[str, object]] | None = None,
    examples_per_theory: int = 0,
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for theory_id, payload in registry.items():
        if not isinstance(payload, Mapping):
            continue
        label = payload.get("label") if isinstance(payload.get("label"), str) else None
        aliases = [alias for alias in payload.get("aliases", []) if isinstance(alias, str)]
        supporting_articles = [
            str(article_id)
            for article_id in payload.get("supporting_articles", [])
            if isinstance(article_id, str)
        ]
        summary.append(
            {
                "theory_id": theory_id,
                "label": label or theory_id,
                "aliases": aliases,
                "article_count": len(supporting_articles),
                "supporting_articles": supporting_articles,
            }
        )

        if examples_per_theory > 0 and supporting_articles:
            representatives: List[Dict[str, object]] = []
            for article_id in supporting_articles[: examples_per_theory]:
                metadata = articles_map.get(article_id) if isinstance(articles_map, Mapping) else None
                if metadata:
                    representatives.append(dict(metadata))
                else:
                    representatives.append({"id": article_id})
            if representatives:
                summary[-1]["representative_articles"] = representatives

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

    chunk, prompt_metadata, prompt = _prepare_summary_chunk_for_prompt(
        summary_chunk,
        total_unique,
        max_prompt_characters=PROMPT_CHARACTER_BUDGET,
    )

    if prompt_metadata.get("estimated_prompt_tokens", 0) > PROMPT_TOKEN_BUDGET:
        raise RuntimeError(
            "Ontology prompt remains above the token budget even after "
            "truncation. Consider reducing --top-n, --examples-per-theory or "
            "the number of worker processes."
        )

    response = _call_openai(prompt, api_key, model)
    if isinstance(response, dict):
        response.setdefault("_prompt_metadata", prompt_metadata)
        response["_prompt_metadata"].setdefault(
            "chunk_size",
            len(chunk),
        )
    return response


def _build_prompt(
    summary: Sequence[Mapping[str, object]],
    total_unique: int,
    *,
    truncated_supporting: bool = False,
    truncated_examples: bool = False,
) -> str:
    theories_block = json.dumps(summary, ensure_ascii=False, indent=2)

    extra_instruction_block = ""
    if truncated_supporting or truncated_examples:
        bullet_lines: List[str] = []
        if truncated_supporting:
            bullet_lines.append(
                "- Some ``supporting_articles`` arrays are truncated for brevity; "
                "treat the provided IDs as representative samples without "
                "inventing new ones."
            )
        if truncated_examples:
            bullet_lines.append(
                "- ``representative_articles`` lists may also be truncated; rely on "
                "the supplied examples as guidance without assuming they are "
                "exhaustive."
            )
        extra_instruction_block = "\n" + "\n".join(
            f"        {line}" for line in bullet_lines
        )

    return textwrap.dedent(
        f"""
        You are designing an ontology for theories of aging based on literature
        review evidence. There are {total_unique} canonical theories in total.
        The JSON list below summarises each theory with its identifier, label,
        aliases, supporting article IDs, and representative article metadata
        when available.

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
          ``supporting_articles`` list for every theory you place.{extra_instruction_block}
        - Respond with JSON containing a top-level key "groups".

        Summary of canonical theories:
        {theories_block}
        """
    ).strip()


def _estimate_prompt_tokens(prompt: str) -> int:
    return math.ceil(len(prompt) / max(1, AVERAGE_CHARS_PER_TOKEN))


def _cap_sequence_field(
    records: Sequence[MutableMapping[str, object]],
    field: str,
    cap: int,
) -> bool:
    if cap < 0:
        return False
    changed = False
    for entry in records:
        value = entry.get(field)
        if isinstance(value, list) and len(value) > cap:
            entry[field] = value[:cap]
            changed = True
    return changed


def _prepare_summary_chunk_for_prompt(
    summary_chunk: Sequence[Mapping[str, object]],
    total_unique: int,
    *,
    max_prompt_characters: int,
) -> Tuple[List[Dict[str, object]], Dict[str, Any], str]:
    chunk: List[Dict[str, object]] = []
    for item in summary_chunk:
        if isinstance(item, Mapping):
            chunk.append({key: value for key, value in item.items()})

    truncated_supporting = False
    truncated_examples = False

    prompt = _build_prompt(
        chunk,
        total_unique,
        truncated_supporting=truncated_supporting,
        truncated_examples=truncated_examples,
    )
    prompt_length = len(prompt)
    estimated_tokens = _estimate_prompt_tokens(prompt)

    metadata: Dict[str, Any] = {
        "max_prompt_characters": max_prompt_characters,
        "initial_prompt_characters": prompt_length,
        "initial_estimated_tokens": estimated_tokens,
        "truncated_supporting_articles": False,
        "truncated_representative_articles": False,
        "prompt_trimmed": False,
        "adjustments": [],
    }

    if prompt_length <= max_prompt_characters:
        metadata.update(
            {
                "final_prompt_characters": prompt_length,
                "estimated_prompt_tokens": estimated_tokens,
                "within_character_budget": True,
                "within_token_budget": estimated_tokens <= PROMPT_TOKEN_BUDGET,
            }
        )
        metadata["chunk_size"] = len(chunk)
        return chunk, metadata, prompt

    metadata["within_character_budget"] = False

    for cap in SUPPORTING_ARTICLE_CAPS:
        changed = _cap_sequence_field(chunk, "supporting_articles", cap)
        if not changed:
            continue
        truncated_supporting = True
        metadata["prompt_trimmed"] = True
        metadata["supporting_articles_cap"] = cap
        metadata["adjustments"].append({"supporting_articles_cap": cap})
        prompt = _build_prompt(
            chunk,
            total_unique,
            truncated_supporting=truncated_supporting,
            truncated_examples=truncated_examples,
        )
        prompt_length = len(prompt)
        estimated_tokens = _estimate_prompt_tokens(prompt)
        if prompt_length <= max_prompt_characters:
            metadata.update(
                {
                    "final_prompt_characters": prompt_length,
                    "estimated_prompt_tokens": estimated_tokens,
                    "within_character_budget": True,
                    "within_token_budget": estimated_tokens <= PROMPT_TOKEN_BUDGET,
                    "truncated_supporting_articles": True,
                    "truncated_representative_articles": truncated_examples,
                }
            )
            metadata["chunk_size"] = len(chunk)
            return chunk, metadata, prompt

    for cap in REPRESENTATIVE_ARTICLE_CAPS:
        changed = _cap_sequence_field(chunk, "representative_articles", cap)
        if not changed:
            continue
        truncated_examples = True
        metadata["prompt_trimmed"] = True
        metadata["representative_articles_cap"] = cap
        metadata["adjustments"].append({"representative_articles_cap": cap})
        prompt = _build_prompt(
            chunk,
            total_unique,
            truncated_supporting=truncated_supporting,
            truncated_examples=truncated_examples,
        )
        prompt_length = len(prompt)
        estimated_tokens = _estimate_prompt_tokens(prompt)
        if prompt_length <= max_prompt_characters:
            metadata.update(
                {
                    "final_prompt_characters": prompt_length,
                    "estimated_prompt_tokens": estimated_tokens,
                    "within_character_budget": True,
                    "within_token_budget": estimated_tokens <= PROMPT_TOKEN_BUDGET,
                    "truncated_supporting_articles": truncated_supporting,
                    "truncated_representative_articles": True,
                }
            )
            metadata["chunk_size"] = len(chunk)
            return chunk, metadata, prompt

    # Final fallback: drop supporting articles entirely if necessary.
    if _cap_sequence_field(chunk, "supporting_articles", 0):
        truncated_supporting = True
        metadata["prompt_trimmed"] = True
        metadata["supporting_articles_cap"] = 0
        metadata["adjustments"].append({"supporting_articles_cap": 0})
        prompt = _build_prompt(
            chunk,
            total_unique,
            truncated_supporting=truncated_supporting,
            truncated_examples=truncated_examples,
        )
        prompt_length = len(prompt)
        estimated_tokens = _estimate_prompt_tokens(prompt)

    metadata.update(
        {
            "final_prompt_characters": prompt_length,
            "estimated_prompt_tokens": estimated_tokens,
            "within_character_budget": prompt_length <= max_prompt_characters,
            "within_token_budget": estimated_tokens <= PROMPT_TOKEN_BUDGET,
            "truncated_supporting_articles": truncated_supporting,
            "truncated_representative_articles": truncated_examples,
        }
    )
    metadata["chunk_size"] = len(chunk)
    return chunk, metadata, prompt


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


def _limit_group_theories(
    groups: Sequence[Mapping[str, Any]],
    max_theories: int,
    *,
    reconciliation: MutableMapping[str, List[Dict[str, object]]],
) -> List[Dict[str, Any]]:
    """Split groups whose direct theory lists exceed ``max_theories``.

    The ontology reconciliation step occasionally yields top-level groups that
    collect a very large number of theories. Downstream collectors perform
    ontology-driven expansion per group, so oversized buckets can starve smaller
    theories of search coverage. This helper enforces a soft cap by
    redistributing overflow theories into auto-generated subgroups while keeping
    the canonical metadata intact.
    """

    max_theories = max(0, max_theories)
    if max_theories == 0:
        return [dict(group) for group in groups if isinstance(group, Mapping)]

    adjustments: List[Dict[str, object]] = []

    def _process_group(
        group: Mapping[str, Any],
        lineage: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(group, Mapping):
            return None

        group_name = group.get("name") if isinstance(group.get("name"), str) else ""
        current_lineage = list(lineage)
        if group_name:
            current_lineage.append(group_name)

        base: Dict[str, Any] = {}
        for key, value in group.items():
            if key in {"subgroups", "theories"}:
                continue
            base[key] = value

        processed_subgroups: List[Dict[str, Any]] = []
        raw_subgroups = group.get("subgroups")
        if isinstance(raw_subgroups, Sequence) and not isinstance(raw_subgroups, (str, bytes)):
            for child in raw_subgroups:
                processed_child = _process_group(child, current_lineage)
                if processed_child:
                    processed_subgroups.append(processed_child)

        theories: List[Dict[str, Any]] = []
        raw_theories = group.get("theories")
        if isinstance(raw_theories, Sequence) and not isinstance(raw_theories, (str, bytes)):
            for theory in raw_theories:
                if isinstance(theory, Mapping):
                    theories.append(dict(theory))

        if max_theories and len(theories) > max_theories:
            chunks: List[List[Dict[str, Any]]] = [
                theories[idx : idx + max_theories]
                for idx in range(0, len(theories), max_theories)
            ]

            base["theories"] = chunks[0]

            overflow_groups: List[Dict[str, Any]] = []
            label_seed = group_name or "Group"
            for chunk_index, chunk in enumerate(chunks[1:], start=2):
                overflow_groups.append(
                    {
                        "name": f"{label_seed} (auto-split {chunk_index})",
                        "description": "Automatically created to satisfy the max_theories_per_group limit.",
                        "theories": chunk,
                    }
                )

            if processed_subgroups or overflow_groups:
                base["subgroups"] = processed_subgroups + overflow_groups
            else:
                base.pop("subgroups", None)

            adjustments.append(
                {
                    "group_path": " > ".join(current_lineage) if current_lineage else label_seed,
                    "original_theory_count": len(theories),
                    "retained_theory_ids": [theory.get("theory_id") for theory in chunks[0]],
                    "overflow_groups": [
                        [theory.get("theory_id") for theory in chunk]
                        for chunk in chunks[1:]
                    ],
                    "limit": max_theories,
                }
            )
        else:
            if theories:
                base["theories"] = theories
            else:
                base.pop("theories", None)
            if processed_subgroups:
                base["subgroups"] = processed_subgroups
            else:
                base.pop("subgroups", None)

        return base

    processed_groups: List[Dict[str, Any]] = []
    for group in groups:
        processed = _process_group(group, [])
        if processed:
            processed_groups.append(processed)

    if adjustments:
        reconciliation.setdefault("group_splits", []).extend(adjustments)

    return processed_groups


def _normalise_term(term: str) -> str:
    return " ".join(term.split())


def _query_variants(term: str) -> List[str]:
    cleaned = _normalise_term(term)
    if not cleaned:
        return []
    variants: List[str] = []

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate:
            variants.append(candidate)

    lower = cleaned.lower()
    _add(cleaned)
    if " " in cleaned:
        _add(f'"{cleaned}"')
    if "aging" not in lower and "ageing" not in lower:
        quoted = f'"{cleaned}"' if " " in cleaned else cleaned
        _add(f"{quoted} aging")
    if "theory" not in lower:
        _add(f"{cleaned} theory")
    return variants


def _collect_theory_terms(theory: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    seen: Set[str] = set()

    def _push(value: Any) -> None:
        if isinstance(value, str):
            normalised = _normalise_term(value)
            if normalised and normalised.lower() not in seen:
                seen.add(normalised.lower())
                names.append(normalised)

    _push(theory.get("preferred_label"))
    _push(theory.get("label"))
    aliases = theory.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        for alias in aliases:
            _push(alias)
    return names


def _attach_suggested_queries(groups: Sequence[Dict[str, Any]]) -> None:
    def _process_group(group: Dict[str, Any]) -> List[str]:
        aggregated_terms: List[str] = []
        seen_terms: Set[str] = set()

        theories = group.get("theories")
        if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
            for theory in theories:
                if isinstance(theory, Mapping):
                    for term in _collect_theory_terms(theory):
                        if term.lower() not in seen_terms:
                            seen_terms.add(term.lower())
                            aggregated_terms.append(term)

        subgroups = group.get("subgroups")
        if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
            for child in subgroups:
                if isinstance(child, dict):
                    child_terms = _process_group(child)
                    for term in child_terms:
                        if term.lower() not in seen_terms:
                            seen_terms.add(term.lower())
                            aggregated_terms.append(term)

        suggestions: List[str] = []
        seen_suggestions: Set[str] = set()
        for term in aggregated_terms:
            for candidate in _query_variants(term):
                key = candidate.lower()
                if key not in seen_suggestions:
                    seen_suggestions.add(key)
                    suggestions.append(candidate)

        group["suggested_queries"] = suggestions
        return aggregated_terms

    for group in groups:
        if isinstance(group, dict):
            _process_group(group)


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
            "Number of worker processes for ontology generation. When omitted the "
            "script auto-scales using the available CPU cores and keeps roughly "
            "25 canonical theories per worker to balance throughput and prompt "
            "quality."
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
    parser.add_argument(
        "--examples-per-theory",
        type=int,
        default=3,
        help=(
            "Number of representative articles to include for each theory in the ontology prompt. "
            "Set to 0 to disable article examples."
        ),
    )
    parser.add_argument(
        "--llm-response",
        default=None,
        help=(
            "Path to a JSON file containing a precomputed ontology response. When provided "
            "the OpenAI API is not called and the supplied payload is used instead."
        ),
    )
    parser.add_argument(
        "--max-theories-per-group",
        type=int,
        default=DEFAULT_MAX_THEORIES_PER_GROUP,
        help=(
            "Maximum number of theories allowed per ontology group after reconciliation. "
            "Defaults to %(default)s, which automatically splits oversized groups into "
            "auto-generated subgroups. Values <= 0 disable the safeguard."
        ),
    )
    parser.add_argument(
        "--registry-model",
        default="gpt-5-nano",
        help=(
            "OpenAI chat completion model used when reconstructing the canonical "
            "theory registry from checkpoint records."
        ),
    )
    parser.add_argument(
        "--registry-request-timeout",
        type=float,
        default=60.0,
        help=(
            "Timeout (in seconds) for OpenAI calls performed during registry "
            "reconstruction when the input lacks a canonical registry."
        ),
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    using_precomputed_response = bool(args.llm_response)
    if not api_key and not using_precomputed_response:
        print("OPENAI_API_KEY is required", file=sys.stderr)
        return 1

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        return 1

    try:
        data = _load_json_payload(args.input)
    except (OSError, RuntimeError, json.JSONDecodeError) as err:
        print(f"Failed to load JSON from {args.input}: {err}", file=sys.stderr)
        return 1

    data_mapping: Optional[Mapping[str, object]] = None
    fallback_documents: List[Mapping[str, object]] = []

    if isinstance(data, Mapping):
        data_mapping = data
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        fallback_documents = [
            entry for entry in data if isinstance(entry, Mapping)
        ]
        for entry in fallback_documents:
            registry_candidate = entry.get("theory_registry")
            if isinstance(registry_candidate, Mapping):
                data_mapping = entry
                break

    registry = data_mapping.get("theory_registry") if isinstance(data_mapping, Mapping) else None
    article_records: List[Mapping[str, object]] = []
    registry_source = "embedded"

    if isinstance(data_mapping, Mapping):
        raw_articles = data_mapping.get("articles")
        if isinstance(raw_articles, Sequence) and not isinstance(raw_articles, (str, bytes)):
            article_records = [
                record for record in raw_articles if isinstance(record, Mapping)
            ]

    if registry is None and not article_records and fallback_documents:
        article_records = _coerce_articles_from_documents(fallback_documents)

    registry_reconstructed = False
    if registry is None and article_records:
        try:
            loader = _load_registry_builder  # type: ignore[name-defined]
        except NameError:  # pragma: no cover - defensive guard
            loader = None
        build_theory_registry = loader() if callable(loader) else None
        if build_theory_registry is None:
            print(
                "Input JSON is missing the 'theory_registry' mapping and the fallback "
                "reconstruction helpers are unavailable.",
                file=sys.stderr,
            )
            return 1

        print(
            "Input file is missing the canonical theory registry; attempting to "
            "reconstruct it from checkpoint annotations...",
            file=sys.stderr,
        )
        registry = build_theory_registry(
            article_records,
            api_key,
            args.registry_model,
            args.registry_request_timeout,
        )
        registry_reconstructed = True
        registry_source = "reconstructed"

    if not isinstance(registry, Mapping):
        print(
            "Input JSON is missing the 'theory_registry' mapping and no annotated "
            "records were available to rebuild it.",
            file=sys.stderr,
        )
        return 1

    article_map: Dict[str, Dict[str, object]] = {}
    if article_records:
        article_map = _build_article_map(article_records)

    total_unique = len(registry)
    examples_per_theory = max(args.examples_per_theory, 0)
    summary = _summarise_registry(
        registry,
        limit=max(args.top_n, 0),
        articles_map=article_map,
        examples_per_theory=examples_per_theory,
    )

    if total_unique == 0:
        print("No canonical theories available in registry", file=sys.stderr)
        return 1

    summary_count = len(summary)
    base_chunk_size = max(args.chunk_size, 1)

    try:
        cpu_total = multiprocessing.cpu_count()
    except NotImplementedError:  # pragma: no cover - defensive guard
        cpu_total = 1

    min_chunk = min(base_chunk_size, max(10, max(1, base_chunk_size // 4)))

    if summary_count:
        max_workers_by_records = max(1, math.ceil(summary_count / min_chunk))
        auto_processes = min(cpu_total, max_workers_by_records, summary_count)
        auto_processes = max(1, auto_processes)
    else:
        auto_processes = 1

    auto_chunk_size = _compute_chunk_size(
        summary_count,
        base_chunk_size,
        auto_processes,
        min_chunk_size=min_chunk,
    )

    if args.processes is None:
        processes = auto_processes
        requested_processes: Optional[int] = None
        chunk_floor = min_chunk
        chunk_size = auto_chunk_size
    else:
        processes = max(1, args.processes)
        if summary_count:
            processes = min(processes, summary_count)
        requested_processes = args.processes
        chunk_floor = 1
        chunk_size = _compute_chunk_size(
            summary_count,
            base_chunk_size,
            processes,
            min_chunk_size=chunk_floor,
        )

    estimated_chunks = max(1, math.ceil(summary_count / chunk_size)) if summary_count else 1
    if processes > estimated_chunks:
        processes = estimated_chunks
        chunk_size = _compute_chunk_size(
            summary_count,
            base_chunk_size,
            processes,
            min_chunk_size=chunk_floor,
        )
        estimated_chunks = max(1, math.ceil(summary_count / chunk_size)) if summary_count else 1

    process_source = (
        "auto" if requested_processes is None or processes == auto_processes else "manual"
    )

    if processes < 1:
        print("--processes must be at least 1", file=sys.stderr)
        return 1

    should_chunk = processes > 1 or summary_count > chunk_size

    print(
        f"Using {processes} worker process{'es' if processes != 1 else ''} "
        f"({process_source}; auto suggestion: {auto_processes}) "
        f"with chunk size {chunk_size} (auto suggestion: {auto_chunk_size})."
    )

    prompt_adjustments: List[Dict[str, Any]] = []

    def _collect_prompt_metadata(responses: Sequence[Mapping[str, object]]) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []
        for idx, resp in enumerate(responses):
            if not isinstance(resp, Mapping):
                continue
            metadata = resp.get("_prompt_metadata")
            if isinstance(metadata, Mapping):
                entry = dict(metadata)
                entry.setdefault("chunk_index", idx)
                collected.append(entry)
        return collected

    if using_precomputed_response:
        try:
            with open(args.llm_response, "r", encoding="utf-8") as fh:
                precomputed = json.load(fh)
        except (OSError, json.JSONDecodeError) as err:
            print(
                f"Failed to load precomputed LLM response from {args.llm_response}: {err}",
                file=sys.stderr,
            )
            return 1

        if isinstance(precomputed, Mapping):
            response = _normalise_groups(dict(precomputed))
        else:
            print(
                f"Precomputed LLM response in {args.llm_response} must be a JSON object",
                file=sys.stderr,
            )
            return 1

        groups_for_reconciliation = response.get("groups", [])
        prompt_adjustments = []
    elif should_chunk:
        summary_chunks = list(_chunk_summary(summary, chunk_size=chunk_size))
        worker_args = [
            (chunk, total_unique, args.model, api_key)
            for chunk in summary_chunks
        ]

        if processes > 1 and len(summary_chunks) > 1:
            worker_count = min(processes, len(summary_chunks))
            with multiprocessing.Pool(processes=worker_count) as pool:
                chunk_responses = pool.starmap(generate_grouping, worker_args)
        else:
            chunk_responses = [
                generate_grouping(chunk, total_unique, args.model, api_key)
                for chunk in summary_chunks
            ]

        prompt_adjustments = _collect_prompt_metadata(chunk_responses)
        merged_groups = _merge_groupings(chunk_responses)
        response = {"groups": merged_groups, "chunks": chunk_responses}
        groups_for_reconciliation = merged_groups
    else:
        response = generate_grouping(summary, total_unique, args.model, api_key)
        groups_for_reconciliation = response.get("groups", [])
        prompt_adjustments = _collect_prompt_metadata([response])

    article_index = _canonical_article_index(registry)
    reconciled_groups, reconciliation = _reconcile_groups(
        groups_for_reconciliation,
        registry,
        article_index=article_index,
    )

    max_theories_per_group = max(0, args.max_theories_per_group)
    if max_theories_per_group:
        reconciled_groups = _limit_group_theories(
            reconciled_groups,
            max_theories_per_group,
            reconciliation=reconciliation,
        )

    _attach_suggested_queries(reconciled_groups)

    reconciliation.setdefault("notes", []).append(
        {
            "event": "worker_processes",
            "process_count": processes,
            "auto_suggestion": auto_processes,
            "requested_processes": requested_processes,
            "configuration": process_source,
            "chunk_size": chunk_size,
            "auto_chunk_size_suggestion": auto_chunk_size,
        }
    )

    if prompt_adjustments:
        trimmed_chunks = [
            entry.get("chunk_index")
            for entry in prompt_adjustments
            if isinstance(entry, Mapping) and entry.get("prompt_trimmed")
        ]
        reconciliation.setdefault("notes", []).append(
            {
                "event": "prompt_truncation",
                "total_chunks": len(prompt_adjustments),
                "trimmed_chunks": [idx for idx in trimmed_chunks if idx is not None],
                "token_budget": PROMPT_TOKEN_BUDGET,
                "character_budget": PROMPT_CHARACTER_BUDGET,
            }
        )

    if registry_reconstructed:
        reconciliation.setdefault("notes", []).append(
            {
                "event": "registry_reconstructed",
                "record_count": len(article_records),
                "source": registry_source,
                "registry_model": args.registry_model,
                "registry_request_timeout": args.registry_request_timeout,
            }
        )

    output_payload = {
        "generated_at": _dt.datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "input_file": args.input,
        "total_unique_theories": total_unique,
        "examples_per_theory": examples_per_theory,
        "registry_source": registry_source,
        "registry_reconstructed": registry_reconstructed,
        "worker_processes": processes,
        "auto_processes_suggestion": auto_processes,
        "requested_processes": requested_processes,
        "chunk_size": chunk_size,
        "auto_chunk_size_suggestion": auto_chunk_size,
        "prompt_summary": summary,
        "prompt_adjustments": prompt_adjustments,
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
            "chunk_size": chunk_size,
            "auto_chunk_size_suggestion": auto_chunk_size,
            **reconciliation,
        },
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, ensure_ascii=False, indent=2)

    total_reassignments = len(reconciliation.get("reassignments", []))
    print(
        "Reconciliation logged "
        f"{total_reassignments} adjustment{'s' if total_reassignments != 1 else ''} "
        f"using {processes} worker process{'es' if processes != 1 else ''}."
    )
    print(
        "Saved reconciled ontology with "
        f"{len(reconciled_groups)} groups to {args.output}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
