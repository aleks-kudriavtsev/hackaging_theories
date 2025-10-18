"""Utilities for loading ontology-provided query suggestions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence


logger = logging.getLogger(__name__)


def _normalise_strings(values: Any) -> list[str]:
    if isinstance(values, Mapping):
        items = values.values()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        items = values
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        items = []

    seen: set[str] = set()
    results: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(text)
    return results


def _fallback_suggestions(name: str) -> list[str]:
    cleaned = name.strip()
    return [cleaned] if cleaned else []


def _theory_suggestions(theory: Mapping[str, Any]) -> list[str]:
    preferred = []
    for key in ("suggested_queries", "queries"):
        preferred = _normalise_strings(theory.get(key))
        if preferred:
            return preferred

    names: list[str] = []
    for key in ("preferred_label", "label", "name", "theory_id"):
        value = theory.get(key)
        if isinstance(value, str) and value.strip():
            names.append(value.strip())

    aliases = theory.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                names.append(alias.strip())

    for name in names:
        suggestions = _normalise_strings(_query_variants(name))
        if suggestions:
            return suggestions

    if names:
        return _fallback_suggestions(names[0])
    return []


def _query_variants(term: str) -> Sequence[str]:
    cleaned = term.strip()
    if not cleaned:
        return []
    variants = {cleaned}
    lowered = cleaned.lower()
    if "theory" not in lowered:
        variants.add(f"{cleaned} theory")
    if "aging" not in lowered:
        variants.add(f"{cleaned} aging")
    return list(variants)


def _convert_group_node(group: Mapping[str, Any]) -> tuple[str, Dict[str, Any]] | None:
    raw_name = None
    for key in ("name", "label"):
        value = group.get(key)
        if isinstance(value, str) and value.strip():
            raw_name = value.strip()
            break
    if not raw_name:
        return None

    entry: Dict[str, Any] = {}
    suggestions = _normalise_strings(group.get("suggested_queries"))
    if not suggestions:
        suggestions = _fallback_suggestions(raw_name)
    if suggestions:
        entry["suggested_queries"] = suggestions

    subtargets: Dict[str, Dict[str, Any]] = {}

    subgroups = group.get("subgroups")
    if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
        for child in subgroups:
            if not isinstance(child, Mapping):
                continue
            converted = _convert_group_node(child)
            if converted:
                child_name, child_entry = converted
                subtargets[child_name] = child_entry

    theories = group.get("theories")
    if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
        for theory in theories:
            if not isinstance(theory, Mapping):
                continue
            theory_name = None
            for key in ("preferred_label", "label", "name", "theory_id"):
                value = theory.get(key)
                if isinstance(value, str) and value.strip():
                    theory_name = value.strip()
                    break
            if not theory_name:
                continue
            theory_entry: Dict[str, Any] = {}
            theory_suggestions = _theory_suggestions(theory)
            if not theory_suggestions:
                theory_suggestions = _fallback_suggestions(theory_name)
            theory_entry["suggested_queries"] = theory_suggestions
            subtargets.setdefault(theory_name, theory_entry)

    if subtargets:
        entry["subtheories"] = subtargets

    return raw_name, entry


def load_ontology_query_suggestions(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load ontology query hints from ``aging_ontology.json`` style payloads."""

    json_path = Path(path)
    if not json_path.exists():
        logger.debug("Ontology suggestions file missing at %s", json_path)
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - malformed fixture guard
        logger.warning("Failed to parse ontology suggestions at %s: %s", json_path, exc)
        return {}

    ontology = payload.get("ontology") if isinstance(payload, Mapping) else None
    final = ontology.get("final") if isinstance(ontology, Mapping) else None
    groups = final.get("groups") if isinstance(final, Mapping) else None
    if not isinstance(groups, Sequence):
        return {}

    suggestions: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        converted = _convert_group_node(group)
        if not converted:
            continue
        name, entry = converted
        suggestions[name] = entry
    return suggestions


def _merge_suggestion_lists(existing: Sequence[str], new_values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for item in list(existing) + list(new_values):
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(text)
    return merged


def merge_query_suggestions(
    targets: MutableMapping[str, Any],
    suggestions: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge ontology suggestion hints into runtime target configuration."""

    applied: Dict[str, Any] = {}

    for name, config in list(targets.items()):
        if not isinstance(config, MutableMapping):
            config = {"target": config}
            targets[name] = config

        suggestion_entry = suggestions.get(name)
        if not isinstance(suggestion_entry, Mapping):
            suggestion_entry = {}

        loader_suggestions = _normalise_strings(suggestion_entry.get("suggested_queries"))
        existing_suggestions = _normalise_strings(config.get("suggested_queries"))
        final_suggestions = _merge_suggestion_lists(existing_suggestions, loader_suggestions)

        node_applied: Dict[str, Any] = {}
        if final_suggestions:
            config["suggested_queries"] = final_suggestions
            node_applied["suggested_queries"] = final_suggestions
        elif not existing_suggestions and loader_suggestions:
            config["suggested_queries"] = loader_suggestions
            node_applied["suggested_queries"] = loader_suggestions

        subtargets = config.get("subtheories")
        child_suggestions = suggestion_entry.get("subtheories")
        if isinstance(subtargets, MutableMapping) and isinstance(child_suggestions, Mapping):
            child_applied = merge_query_suggestions(subtargets, child_suggestions)
            if child_applied:
                node_applied["subtheories"] = child_applied

        if node_applied:
            applied[name] = node_applied

    return applied


__all__ = ["load_ontology_query_suggestions", "merge_query_suggestions"]

