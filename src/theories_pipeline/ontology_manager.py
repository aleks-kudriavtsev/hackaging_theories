"""Runtime management for the theory ontology hierarchy.

The static :mod:`theories_pipeline.ontology` module focuses on parsing a fixed
configuration into a :class:`~theories_pipeline.ontology.TheoryOntology`
instance.  During enrichment runs we want to expand that ontology on the fly,
persist those additions, and inform downstream components such as
classifiers.  The :class:`OntologyManager` defined here provides that runtime
layer: it tracks dynamically appended nodes, rebuilds the ontology whenever
the hierarchy changes, and broadcasts structured updates to interested
listeners (for example the :class:`~theories_pipeline.theories.TheoryClassifier`).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping

from .ontology import TheoryOntology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OntologyUpdate:
    """Structured payload describing a runtime ontology change."""

    added: Dict[str, Dict[str, Any]]
    removed: Iterable[str] = ()


class OntologyManager:
    """Manage runtime updates to a :class:`TheoryOntology` instance."""

    def __init__(
        self,
        base_config: Mapping[str, Mapping[str, Any]],
        *,
        storage_path: Path | None = None,
    ) -> None:
        self._base_config = json.loads(json.dumps(base_config))
        self._storage_path = Path(storage_path) if storage_path else Path(
            "data/cache/runtime_ontology.json"
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._listeners: list[Callable[[TheoryOntology, OntologyUpdate], None]] = []
        self._runtime_nodes: Dict[str, Dict[str, Any]] = {}
        self._load()
        self._ontology = TheoryOntology.from_targets_config(self._build_config())

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(
                "Failed to decode runtime ontology cache at %s; ignoring",
                self._storage_path,
            )
            return
        nodes = payload.get("nodes", [])
        if isinstance(nodes, list):
            for entry in nodes:
                if not isinstance(entry, Mapping):
                    continue
                name = entry.get("name")
                parent = entry.get("parent")
                if not isinstance(name, str):
                    continue
                config = entry.get("config") if isinstance(entry.get("config"), Mapping) else {}
                metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
                keywords = entry.get("keywords")
                self._runtime_nodes[name] = {
                    "parent": parent if isinstance(parent, str) else None,
                    "config": dict(config),
                    "keywords": list(keywords) if isinstance(keywords, list) else None,
                    "metadata": dict(metadata),
                }

    def _persist(self) -> None:
        payload = {
            "nodes": [
                {
                    "name": name,
                    "parent": data.get("parent"),
                    "config": data.get("config", {}),
                    "metadata": data.get("metadata", {}),
                    "keywords": data.get("keywords"),
                }
                for name, data in sorted(self._runtime_nodes.items())
            ]
        }
        self._storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def ontology(self) -> TheoryOntology:
        return self._ontology

    def has_node(self, name: str) -> bool:
        return name in self._runtime_nodes or name in self._ontology.names()

    def register_listener(
        self, callback: Callable[[TheoryOntology, OntologyUpdate], None]
    ) -> None:
        self._listeners.append(callback)

    def append_node(
        self,
        name: str,
        *,
        parent: str | None = None,
        config: Mapping[str, Any] | None = None,
        keywords: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Append a node to the ontology hierarchy at runtime."""

        if self.has_node(name):
            logger.debug("Ontology node '%s' already exists; skipping append", name)
            return
        if parent and not self.has_node(parent):
            raise ValueError(f"Parent node '{parent}' not found for runtime ontology append")

        stored_config = dict(config) if config else {}
        stored_metadata = dict(metadata) if metadata else {}
        normalized_keywords = [kw.lower() for kw in keywords] if keywords else None
        self._runtime_nodes[name] = {
            "parent": parent,
            "config": stored_config,
            "metadata": stored_metadata,
            "keywords": normalized_keywords,
        }
        self._rebuild_and_notify(added={name: {
            "parent": parent,
            "config": stored_config,
            "metadata": stored_metadata,
            "keywords": normalized_keywords,
        }})

    def rebuild(self) -> None:
        """Rebuild the ontology using the current runtime state."""

        self._rebuild_and_notify()

    def to_config(self) -> Dict[str, Any]:
        """Return the merged ontology configuration (base + runtime)."""

        return self._build_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_config(self) -> Dict[str, Any]:
        merged = json.loads(json.dumps(self._base_config))
        pending: Dict[str, Dict[str, Any]] = dict(self._runtime_nodes)

        def ensure_node(path: MutableMapping[str, Any], node_name: str) -> MutableMapping[str, Any]:
            node_cfg = path.get(node_name)
            if not isinstance(node_cfg, MutableMapping):
                node_cfg = {}
                path[node_name] = node_cfg
            sub_map = node_cfg.get("subtheories")
            if not isinstance(sub_map, MutableMapping):
                node_cfg["subtheories"] = {}
            return node_cfg

        while pending:
            progress = False
            for node_name, data in list(pending.items()):
                parent = data.get("parent")
                config_raw = data.get("config", {})
                config = dict(config_raw) if isinstance(config_raw, Mapping) else {}
                metadata_raw = data.get("metadata", {})
                metadata_payload = (
                    {str(k): deepcopy(v) for k, v in metadata_raw.items()}
                    if isinstance(metadata_raw, Mapping)
                    else {}
                )
                if parent is None:
                    node_cfg = ensure_node(merged, node_name)
                    for key, value in config.items():
                        if key == "subtheories":
                            continue
                        if key == "metadata" and isinstance(value, Mapping):
                            for meta_key, meta_value in value.items():
                                metadata_payload[str(meta_key)] = deepcopy(meta_value)
                            continue
                        node_cfg[key] = deepcopy(value)
                    if metadata_payload:
                        existing_meta = node_cfg.get("metadata")
                        if isinstance(existing_meta, MutableMapping):
                            existing_meta.update(metadata_payload)
                        elif metadata_payload:
                            node_cfg["metadata"] = dict(metadata_payload)
                    pending.pop(node_name)
                    progress = True
                    continue

                parent_cfg = self._find_node_config(merged, parent)
                if parent_cfg is None:
                    continue
                sub_map = parent_cfg.setdefault("subtheories", {})
                child_cfg = ensure_node(sub_map, node_name)
                for key, value in config.items():
                    if key == "subtheories":
                        continue
                    if key == "metadata" and isinstance(value, Mapping):
                        for meta_key, meta_value in value.items():
                            metadata_payload[str(meta_key)] = deepcopy(meta_value)
                        continue
                    child_cfg[key] = deepcopy(value)
                if metadata_payload:
                    existing_meta = child_cfg.get("metadata")
                    if isinstance(existing_meta, MutableMapping):
                        existing_meta.update(metadata_payload)
                    elif metadata_payload:
                        child_cfg["metadata"] = dict(metadata_payload)
                pending.pop(node_name)
                progress = True
            if not progress:
                unresolved = ", ".join(sorted(pending))
                raise ValueError(f"Could not resolve parents for runtime ontology nodes: {unresolved}")

        return merged

    def _find_node_config(
        self, config: MutableMapping[str, Any], name: str
    ) -> MutableMapping[str, Any] | None:
        queue: list[MutableMapping[str, Any]] = [config]
        while queue:
            current = queue.pop(0)
            for node_name, node_cfg in current.items():
                if node_name == name and isinstance(node_cfg, MutableMapping):
                    return node_cfg
                if isinstance(node_cfg, MutableMapping):
                    sub = node_cfg.get("subtheories")
                    if isinstance(sub, MutableMapping):
                        queue.append(sub)
        return None

    def _rebuild_and_notify(self, *, added: Dict[str, Dict[str, Any]] | None = None) -> None:
        config = self._build_config()
        self._ontology = TheoryOntology.from_targets_config(config)
        self._persist()
        update = OntologyUpdate(added=added or {}, removed=())
        for listener in list(self._listeners):
            try:
                listener(self._ontology, update)
            except Exception:  # pragma: no cover - defensive log only
                logger.exception("Ontology listener %r failed", listener)


__all__ = ["OntologyManager", "OntologyUpdate"]

