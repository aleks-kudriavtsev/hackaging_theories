"""Utilities for post-processing ontology outputs.

This module implements the "optimization ontology" step requested for the
Hackaging pipeline.  The goal is to balance the number of supporting articles
assigned to each theory after the ontology has been generated:

* Prefer around three papers per theory (with a hard lower bound of one and an
  upper bound of four).
* When a theory collects more than the permitted number of papers, split it
  into additional theories so that every resulting node respects the
  constraints.
* Articles that appear in many theories are re-assigned to those with the
  lowest paper counts in order to even out coverage across the ontology.

The entry point :func:`optimise_ontology_payload` mutates the in-memory
representation of an ``aging_ontology.json`` payload to enforce those rules and
returns a structured summary of the adjustments performed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


logger = logging.getLogger(__name__)

_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    """Return a filesystem-friendly slug for ``value``."""

    normalized = value.strip().lower()
    candidate = _SLUG_TOKEN.sub("-", normalized).strip("-")
    return candidate or "theory"


def _normalise_articles(articles: Iterable[Any]) -> List[str]:
    """Return a deduplicated list of article identifiers preserving order."""

    seen: set[str] = set()
    cleaned: List[str] = []
    for entry in articles:
        text = str(entry).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _ensure_list(mapping: MutableMapping[str, Any], key: str) -> List[Any]:
    value = mapping.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        cleaned = list(value)
    elif value is None:
        cleaned = []
    else:
        cleaned = [value]
    mapping[key] = cleaned
    return cleaned


def _theory_label(theory: Mapping[str, Any]) -> str:
    for key in ("label", "preferred_label", "name", "theory_id"):
        value = theory.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Theory"


def _ensure_theory_id(theory: MutableMapping[str, Any], used: set[str]) -> str:
    """Guarantee that ``theory`` exposes a unique ``theory_id`` key."""

    for key in ("theory_id", "id", "identifier"):
        value = theory.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate in used:
                base = candidate
                counter = 2
                while candidate in used:
                    candidate = f"{base}-{counter}"
                    counter += 1
                theory[key] = candidate
            used.add(candidate)
            if key != "theory_id":
                theory.setdefault("theory_id", candidate)
            return theory["theory_id"]

    slug = _slugify(_theory_label(theory))
    candidate = slug
    counter = 2
    while candidate in used:
        candidate = f"{slug}-{counter}"
        counter += 1
    theory["theory_id"] = candidate
    used.add(candidate)
    return candidate


@dataclass
class TheoryRef:
    """Lightweight wrapper exposing convenience helpers for a theory node."""

    key: str
    label: str
    group: MutableMapping[str, Any]
    node: MutableMapping[str, Any]

    def articles(self) -> List[str]:
        current = self.node.get("supporting_articles")
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            cleaned = _normalise_articles(current)
            if cleaned != list(current):
                self.node["supporting_articles"] = cleaned
            return cleaned
        self.node["supporting_articles"] = []
        return []

    def update_articles(self, values: Iterable[Any]) -> None:
        self.node["supporting_articles"] = _normalise_articles(values)


@dataclass
class OptimizationSummary:
    """Structured report returned by :func:`optimise_ontology_payload`."""

    changed: bool
    iterations: int = 0
    created_theories: List[Dict[str, Any]] = field(default_factory=list)
    article_reassignments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "iterations": self.iterations,
            "created_theories": self.created_theories,
            "article_reassignments": self.article_reassignments,
        }


class OntologyOptimizer:
    """Encapsulate the optimization algorithm for ontology payloads."""

    def __init__(
        self,
        groups: Sequence[MutableMapping[str, Any]],
        *,
        target: int = 3,
        minimum: int = 1,
        maximum: int = 4,
    ) -> None:
        if minimum <= 0:
            raise ValueError("minimum must be greater than zero")
        if maximum < minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        if target < minimum:
            target = minimum
        if target > maximum:
            target = maximum
        self.groups = groups
        self.target = target
        self.minimum = minimum
        self.maximum = maximum
        self._used_ids: set[str] = set()
        self.summary = OptimizationSummary(changed=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> OptimizationSummary:
        self._refresh_used_ids()

        # Step 1: iteratively split theories that exceed the allowed maximum.
        iteration = 0
        while self._split_once():
            iteration += 1
            self.summary.changed = True

        self.summary.iterations = iteration

        # Step 2: rebalance articles across theories with multiple assignments.
        if self._rebalance_articles():
            self.summary.changed = True

        return self.summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_used_ids(self) -> None:
        self._used_ids.clear()
        for ref in self._collect_refs():
            self._used_ids.add(ref.key)

    def _collect_refs(self) -> List[TheoryRef]:
        refs: List[TheoryRef] = []

        def visit(group: MutableMapping[str, Any]) -> None:
            theories = group.get("theories")
            if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
                for node in theories:
                    if not isinstance(node, MutableMapping):
                        continue
                    key = _ensure_theory_id(node, self._used_ids)
                    label = _theory_label(node)
                    refs.append(TheoryRef(key=key, label=label, group=group, node=node))

            subgroups = group.get("subgroups")
            if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
                for child in subgroups:
                    if isinstance(child, MutableMapping):
                        visit(child)

        for group in self.groups:
            if isinstance(group, MutableMapping):
                visit(group)

        return refs

    def _split_once(self) -> bool:
        refs = self._collect_refs()
        oversize = [ref for ref in refs if len(ref.articles()) > self.maximum]
        if not oversize:
            return False

        # Prioritise the theory with the highest surplus.
        oversize.sort(key=lambda ref: len(ref.articles()), reverse=True)
        target_ref = oversize[0]
        articles = target_ref.articles()
        chunk_size = min(self.maximum, max(self.target, self.minimum))

        chunks: List[List[str]] = []
        start = 0
        while start < len(articles):
            end = min(start + chunk_size, len(articles))
            chunk = articles[start:end]
            if len(chunk) < self.minimum and chunks:
                chunks[-1].extend(chunk)
                break
            chunks.append(chunk)
            start = end

        if not chunks:
            return False

        base_id = target_ref.key
        base_label = target_ref.label
        target_ref.update_articles(chunks[0])

        for index, chunk in enumerate(chunks[1:], start=1):
            new_node = deepcopy(target_ref.node)
            new_id = self._unique_theory_id(f"{base_id}_opt{index}")
            new_label = f"{base_label} (Optimization {index})"
            new_node["theory_id"] = new_id
            new_node["label"] = new_label
            _ensure_list(new_node, "aliases")
            metadata = new_node.setdefault("metadata", {})
            metadata["optimization_origin"] = base_id
            metadata["optimization_variant"] = index
            metadata.setdefault("notes", "Generated during optimization step")
            new_node["supporting_articles"] = chunk
            theories = _ensure_list(target_ref.group, "theories")
            theories.append(new_node)
            self._used_ids.add(new_id)
            self.summary.created_theories.append(
                {
                    "theory_id": new_id,
                    "origin": base_id,
                    "articles": list(chunk),
                }
            )

        return True

    def _unique_theory_id(self, candidate: str) -> str:
        if candidate not in self._used_ids:
            return candidate
        base = candidate
        counter = 2
        while candidate in self._used_ids:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _rebalance_articles(self) -> bool:
        refs = self._collect_refs()
        if not refs:
            return False

        counts: Dict[str, int] = {ref.key: len(ref.articles()) for ref in refs}
        article_map: Dict[str, List[TheoryRef]] = {}
        for ref in refs:
            for article in ref.articles():
                article_map.setdefault(article, []).append(ref)

        changed = False
        for article, assigned in article_map.items():
            if len(assigned) <= 1:
                continue
            assigned.sort(key=lambda ref: (counts[ref.key], ref.label.lower()))
            keeper = assigned[0]
            for ref in assigned[1:]:
                if counts[ref.key] <= self.minimum:
                    continue
                articles = ref.articles()
                if article not in articles:
                    continue
                if counts[ref.key] - 1 < self.minimum:
                    continue
                new_articles = [item for item in articles if item != article]
                ref.update_articles(new_articles)
                counts[ref.key] -= 1
                changed = True
                self.summary.article_reassignments.append(
                    {"article": article, "from": ref.key, "to": keeper.key}
                )

        return changed


def optimise_ontology_payload(
    payload: MutableMapping[str, Any],
    *,
    target: int = 3,
    minimum: int = 1,
    maximum: int = 4,
) -> OptimizationSummary:
    """Apply optimisation rules to an ``aging_ontology.json`` style payload."""

    ontology = payload.get("ontology") if isinstance(payload, MutableMapping) else None
    final = ontology.get("final") if isinstance(ontology, MutableMapping) else None
    groups = final.get("groups") if isinstance(final, MutableMapping) else None

    if not isinstance(groups, list):
        logger.debug("Ontology payload missing 'final.groups'; skipping optimisation")
        return OptimizationSummary(changed=False)

    optimizer = OntologyOptimizer(groups, target=target, minimum=minimum, maximum=maximum)
    summary = optimizer.run()

    if summary.changed:
        optimisation_meta = payload.setdefault("optimization_ontology", {})
        optimisation_meta.update(summary.to_dict())

    return summary


def optimise_file(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    target: int = 3,
    minimum: int = 1,
    maximum: int = 4,
) -> OptimizationSummary:
    """Load ``input_path``, optimise it and persist the result to ``output_path``."""

    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, MutableMapping):
            raise RuntimeError(f"Expected mapping at {input_path}; found {type(payload)!r}")

    summary = optimise_ontology_payload(
        payload,
        target=target,
        minimum=minimum,
        maximum=maximum,
    )

    output = input_path if output_path is None else output_path
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary

