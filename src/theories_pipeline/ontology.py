"""Ontology helpers for hierarchical theory definitions.

This module centralizes parsing and inspection of the nested theory
configuration used throughout the pipeline.  The configuration is expected to
define a tree (or forest) of theories where each node may specify an optional
``target`` quota and any number of ``subtheories``.  The :class:`TheoryOntology`
class exposes convenience helpers for walking the hierarchy, looking up
parents/children, calculating depth, and summarising quota coverage based on
observed paper counts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass
class OntologyNode:
    """Represents a theory node inside the ontology hierarchy."""

    name: str
    target: Optional[int] = None
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: str) -> None:
        if child not in self.children:
            self.children.append(child)

    def merge_metadata(self, extra: Mapping[str, Any]) -> None:
        """Attach arbitrary metadata to the ontology node."""

        for key, value in extra.items():
            self.metadata[key] = deepcopy(value)


@dataclass(frozen=True)
class CoverageRecord:
    """Coverage statistics for a single ontology node."""

    name: str
    depth: int
    count: int
    target: Optional[int]

    @property
    def deficit(self) -> Optional[int]:
        if self.target is None:
            return None
        return max(0, self.target - self.count)

    @property
    def met(self) -> bool:
        deficit = self.deficit
        return deficit is None or deficit == 0


class TheoryOntology:
    """Container for a hierarchical theory/subtheory ontology."""

    def __init__(self, nodes: Mapping[str, OntologyNode]) -> None:
        self._nodes: Dict[str, OntologyNode] = {name: node for name, node in nodes.items()}
        self._depth_cache: Dict[str, int] = {}
        self._root_names: List[str] = [name for name, node in self._nodes.items() if node.parent is None]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_targets_config(
        cls,
        config: Mapping[str, Mapping[str, object]] | Sequence[Mapping[str, object]],
    ) -> "TheoryOntology":
        """Build an ontology from the nested ``corpus.targets`` configuration."""

        nodes: Dict[str, OntologyNode] = {}

        def ensure(name: str) -> OntologyNode:
            node = nodes.get(name)
            if node is None:
                node = OntologyNode(name=name)
                nodes[name] = node
            return node

        def extract_target(value: object) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            try:
                return int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        def iter_entries(
            payload: object,
        ) -> Sequence[tuple[str, Mapping[str, object]]]:
            entries: List[tuple[str, Mapping[str, object]]] = []
            if isinstance(payload, Mapping):
                name_field = payload.get("name")
                if isinstance(name_field, str):
                    entries.append(
                        (
                            name_field,
                            {
                                key: value
                                for key, value in payload.items()
                                if key != "name"
                            },
                        )
                    )
                else:
                    for key, value in payload.items():
                        if not isinstance(key, str):
                            continue
                        if isinstance(value, Mapping):
                            child_name = value.get("name")
                            if isinstance(child_name, str):
                                entries.append(
                                    (
                                        child_name,
                                        {
                                            sub_key: sub_value
                                            for sub_key, sub_value in value.items()
                                            if sub_key != "name"
                                        },
                                    )
                                )
                            else:
                                entries.append((key, value))
                        else:
                            entries.append((key, {}))
            elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                for item in payload:
                    if isinstance(item, Mapping):
                        name_field = item.get("name")
                        if isinstance(name_field, str):
                            entries.append(
                                (
                                    name_field,
                                    {
                                        key: value
                                        for key, value in item.items()
                                        if key != "name"
                                    },
                                )
                            )
            return entries

        def visit(name: str, data: Mapping[str, object], parent: Optional[str]) -> None:
            node = ensure(name)
            node_cfg = dict(data) if isinstance(data, Mapping) else {}
            node.target = extract_target(node_cfg.get("target"))
            node.parent = parent
            if parent:
                ensure(parent).add_child(name)
            metadata_payload: Dict[str, Any] = {}
            for key, value in node_cfg.items():
                if key in {"target", "subtheories"}:
                    continue
                if key == "metadata" and isinstance(value, Mapping):
                    for meta_key, meta_value in value.items():
                        metadata_payload[str(meta_key)] = deepcopy(meta_value)
                    continue
                metadata_payload[key] = deepcopy(value)
            if metadata_payload:
                node.merge_metadata(metadata_payload)
            sub_config = node_cfg.get("subtheories")
            for child_name, child_data in iter_entries(sub_config):
                visit(child_name, child_data, name)

        for theory_name, theory_cfg in iter_entries(config):
            visit(theory_name, theory_cfg, None)

        return cls(nodes)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def names(self) -> Sequence[str]:
        return tuple(self._nodes.keys())

    def roots(self) -> Sequence[str]:
        return tuple(self._root_names)

    def get(self, name: str) -> OntologyNode:
        return self._nodes[name]

    def parent(self, name: str) -> Optional[str]:
        return self._nodes[name].parent

    def children(self, name: str) -> Sequence[str]:
        return tuple(self._nodes[name].children)

    def target(self, name: str) -> Optional[int]:
        return self._nodes[name].target

    def depth(self, name: str) -> int:
        if name in self._depth_cache:
            return self._depth_cache[name]
        depth = 0
        node = self._nodes[name]
        while node.parent is not None:
            depth += 1
            node = self._nodes[node.parent]
        self._depth_cache[name] = depth
        return depth

    def levels(self) -> Dict[int, List[str]]:
        levels: Dict[int, List[str]] = {}
        for name in self._nodes:
            depth = self.depth(name)
            levels.setdefault(depth, []).append(name)
        for names in levels.values():
            names.sort()
        return dict(sorted(levels.items()))

    def post_order(self) -> List[str]:
        order: List[str] = []
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for child in self.children(name):
                visit(child)
            order.append(name)

        for root in self.roots():
            visit(root)
        return order

    # ------------------------------------------------------------------
    # Coverage helpers
    # ------------------------------------------------------------------
    def coverage(self, counts: Mapping[str, int]) -> Dict[str, CoverageRecord]:
        records: Dict[str, CoverageRecord] = {}
        for name in self._nodes:
            records[name] = CoverageRecord(
                name=name,
                depth=self.depth(name),
                count=int(counts.get(name, 0)),
                target=self.target(name),
            )
        return records

    def depth_deficits(self, counts: Mapping[str, int]) -> Dict[int, List[CoverageRecord]]:
        """Group deficit-bearing nodes by their depth in the ontology."""

        grouped: Dict[int, List[CoverageRecord]] = {}
        for record in self.coverage(counts).values():
            deficit = record.deficit
            if deficit is None or deficit <= 0:
                continue
            grouped.setdefault(record.depth, []).append(record)

        for entries in grouped.values():
            entries.sort(key=lambda item: item.name)

        return dict(sorted(grouped.items()))

    def deficit_summary_by_depth(
        self, counts: Mapping[str, int]
    ) -> Dict[int, Dict[str, object]]:
        """Summarise deficit information grouped by ontology depth."""

        summary: Dict[int, Dict[str, object]] = {}
        for depth, records in self.depth_deficits(counts).items():
            summary[depth] = {
                "total_deficit": sum(record.deficit or 0 for record in records),
                "nodes": [
                    {
                        "name": record.name,
                        "count": record.count,
                        "target": record.target,
                        "deficit": record.deficit,
                    }
                    for record in records
                ],
            }
        return summary

    def format_coverage_report(self, counts: Mapping[str, int]) -> str:
        """Return a human-readable quota coverage report."""

        coverage = self.coverage(counts)
        lines = ["Ontology quota coverage:"]
        for depth, names in self.levels().items():
            lines.append(f"  Depth {depth}:")
            for name in names:
                record = coverage[name]
                indent = "    " * depth
                target_text = record.target if record.target is not None else "n/a"
                deficit = record.deficit
                if deficit is None:
                    status = "no target"
                elif deficit == 0:
                    status = "target met"
                else:
                    status = f"deficit {deficit}"
                lines.append(f"{indent}    - {name}: {record.count} / {target_text} ({status})")

        depth_summary = self.deficit_summary_by_depth(counts)
        if depth_summary:
            lines.append("")
            lines.append("Deficit summary by depth:")
            for depth, payload in depth_summary.items():
                total_deficit = int(payload.get("total_deficit", 0))
                nodes = payload.get("nodes", [])
                label = ", ".join(
                    f"{entry['name']} (-{entry['deficit']})"
                    for entry in nodes
                    if entry.get("deficit")
                )
                lines.append(
                    f"  Depth {depth}: total deficit {total_deficit} across {len(nodes)} nodes ({label})"
                )
        return "\n".join(lines)


__all__ = ["OntologyNode", "CoverageRecord", "TheoryOntology"]

