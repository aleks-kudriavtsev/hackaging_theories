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
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .llm import LLMClient, LLMClientError, LLMMessage
from .ontology import TheoryOntology
from .ontology_summaries import clean_summary, fallback_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OntologyUpdate:
    """Structured payload describing a runtime ontology change."""

    added: Dict[str, Dict[str, Any]]
    removed: Iterable[str] = ()
    keyword_updates: Dict[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeNodeSpec:
    """Description of a runtime ontology node append operation."""

    name: str
    parent: str | None
    config: Mapping[str, Any] = field(default_factory=dict)
    keywords: Iterable[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


class OntologyManager:
    """Manage runtime updates to a :class:`TheoryOntology` instance."""

    DEFAULT_SUMMARY_SYSTEM_PROMPT = (
        "You are an expert gerontology ontologist. "
        "Write a concise definition (max 35 words) for the provided theory."
    )

    DEFAULT_SUMMARY_USER_PROMPT = (
        "Theory name: {name}\n"
        "Parent: {parent}\n"
        "Keywords: {keywords}\n"
        "Bootstrap: {bootstrap}\n"
        "Additional metadata: {metadata}\n"
        "Respond with a single sentence summary only."
    )

    def __init__(
        self,
        base_config: Mapping[str, Mapping[str, Any]],
        *,
        storage_path: Path | None = None,
        llm_client: LLMClient | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        self._base_config = json.loads(json.dumps(base_config))
        self._storage_path = Path(storage_path) if storage_path else Path(
            "data/cache/runtime_ontology.json"
        )
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._listeners: list[Callable[[TheoryOntology, OntologyUpdate], None]] = []
        self._runtime_nodes: Dict[str, Dict[str, Any]] = {}
        self._llm_client = llm_client
        self._summary_system_prompt = self.DEFAULT_SUMMARY_SYSTEM_PROMPT
        self._summary_user_prompt = summary_prompt or self.DEFAULT_SUMMARY_USER_PROMPT
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
                provenance = (
                    entry.get("provenance")
                    if isinstance(entry.get("provenance"), Mapping)
                    else {}
                )
                keywords = entry.get("keywords")
                self._runtime_nodes[name] = {
                    "parent": parent if isinstance(parent, str) else None,
                    "config": dict(config),
                    "keywords": list(keywords) if isinstance(keywords, list) else None,
                    "metadata": dict(metadata),
                    "provenance": dict(provenance),
                }

    def _persist(self) -> None:
        payload = {
            "nodes": [
                {
                    "name": name,
                    "parent": data.get("parent"),
                    "config": data.get("config", {}),
                    "metadata": data.get("metadata", {}),
                    "provenance": data.get("provenance", {}),
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
        provenance: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append a node to the ontology hierarchy at runtime.

        Returns ``True`` when the node was added, ``False`` when it already
        existed.
        """

        if self.has_node(name):
            logger.debug("Ontology node '%s' already exists; skipping append", name)
            return False
        if parent and not self.has_node(parent):
            raise ValueError(f"Parent node '{parent}' not found for runtime ontology append")

        stored_config = dict(config) if config else {}
        stored_metadata = dict(metadata) if metadata else {}
        stored_provenance = dict(provenance) if provenance else {}
        normalized_keywords = [kw.lower() for kw in keywords] if keywords else None
        self._ensure_summary(name, parent, stored_metadata, keywords)
        self._runtime_nodes[name] = {
            "parent": parent,
            "config": stored_config,
            "metadata": stored_metadata,
            "provenance": stored_provenance,
            "keywords": normalized_keywords,
        }
        keyword_updates = {name: normalized_keywords} if normalized_keywords else {}
        self._rebuild_and_notify(
            added={
                name: {
                    "parent": parent,
                    "config": stored_config,
                    "metadata": stored_metadata,
                    "provenance": stored_provenance,
                    "keywords": normalized_keywords,
                }
            },
            keyword_updates=keyword_updates,
        )
        return True

    def update_keywords(
        self,
        name: str,
        keywords: Iterable[str] | None,
        *,
        merge: bool = False,
    ) -> bool:
        """Update the keyword list associated with ``name``.

        Returns ``True`` when the keyword set changed.
        """

        if not self.has_node(name):
            raise KeyError(f"Ontology node '{name}' not found")

        normalized: List[str] = []
        if keywords:
            seen: set[str] = set()
            for raw in keywords:
                if not isinstance(raw, str):
                    continue
                token = raw.strip().lower()
                if not token or token in seen:
                    continue
                seen.add(token)
                normalized.append(token)

        entry = self._runtime_nodes.get(name)
        if entry is None:
            parent: str | None = None
            try:
                parent = self._ontology.parent(name)
            except KeyError:
                parent = None
            entry = {
                "parent": parent,
                "config": {},
                "metadata": {},
                "provenance": {},
                "keywords": None,
            }
            self._runtime_nodes[name] = entry

        existing: List[str] = list(entry.get("keywords") or [])
        if merge and existing:
            seen_existing = set(existing)
            combined = list(existing)
            for token in normalized:
                if token not in seen_existing:
                    seen_existing.add(token)
                    combined.append(token)
            target = combined
        else:
            target = normalized

        target_value = target or None
        if (entry.get("keywords") or None) == target_value:
            return False

        entry["keywords"] = target_value
        keyword_updates = {name: target} if target else {name: []}
        self._rebuild_and_notify(keyword_updates=keyword_updates)
        return True

    def append_child(
        self,
        parent: str,
        spec: RuntimeNodeSpec,
    ) -> bool:
        """Convenience wrapper to append a child node relative to ``parent``."""

        parent_override = spec.parent if spec.parent is not None else parent
        return self.append_node(
            spec.name,
            parent=parent_override,
            config=spec.config,
            keywords=spec.keywords,
            metadata=spec.metadata,
            provenance=spec.provenance,
        )

    def append_sibling(
        self,
        sibling: str,
        spec: RuntimeNodeSpec,
    ) -> bool:
        """Append ``spec`` as a sibling of ``sibling`` if possible."""

        parent_override = spec.parent
        parent = None
        try:
            parent = self._ontology.parent(sibling)
        except KeyError:
            parent = None
        if parent is None:
            runtime_parent = self._runtime_nodes.get(sibling, {}).get("parent")
            if isinstance(runtime_parent, str):
                parent = runtime_parent
        if parent_override is not None:
            parent = parent_override
        return self.append_node(
            spec.name,
            parent=parent,
            config=spec.config,
            keywords=spec.keywords,
            metadata=spec.metadata,
            provenance=spec.provenance,
        )

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
                provenance_raw = data.get("provenance", {})
                metadata_payload = (
                    {str(k): deepcopy(v) for k, v in metadata_raw.items()}
                    if isinstance(metadata_raw, Mapping)
                    else {}
                )
                if isinstance(provenance_raw, Mapping) and provenance_raw:
                    metadata_payload.setdefault("runtime_provenance", {})
                    provenance_container = metadata_payload["runtime_provenance"]
                    if isinstance(provenance_container, MutableMapping):
                        for prov_key, prov_value in provenance_raw.items():
                            provenance_container[str(prov_key)] = deepcopy(prov_value)
                    else:
                        metadata_payload["runtime_provenance"] = {
                            str(prov_key): deepcopy(prov_value)
                            for prov_key, prov_value in provenance_raw.items()
                        }
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

    def _rebuild_and_notify(
        self,
        *,
        added: Dict[str, Dict[str, Any]] | None = None,
        keyword_updates: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        config = self._build_config()
        self._ontology = TheoryOntology.from_targets_config(config)
        self._persist()
        keyword_payload = {
            name: list(values)
            for name, values in (keyword_updates or {}).items()
        }
        update = OntologyUpdate(
            added=added or {},
            removed=(),
            keyword_updates=keyword_payload,
        )
        for listener in list(self._listeners):
            try:
                listener(self._ontology, update)
            except Exception:  # pragma: no cover - defensive log only
                logger.exception("Ontology listener %r failed", listener)

    def _ensure_summary(
        self,
        name: str,
        parent: str | None,
        metadata: Dict[str, Any],
        keywords: Iterable[str] | None,
    ) -> None:
        existing = clean_summary(metadata.get("summary"))
        if existing:
            metadata["summary"] = existing
            return
        summary_text = self._request_llm_summary(name, parent, metadata, keywords)
        if not summary_text:
            bootstrap_meta = metadata.get("bootstrap")
            summary_text = fallback_summary(
                name,
                keywords=keywords,
                bootstrap=bootstrap_meta if isinstance(bootstrap_meta, Mapping) else None,
            )
        cleaned = clean_summary(summary_text)
        if cleaned:
            metadata["summary"] = cleaned

    def _request_llm_summary(
        self,
        name: str,
        parent: str | None,
        metadata: Mapping[str, Any],
        keywords: Iterable[str] | None,
    ) -> str | None:
        if not self._llm_client:
            return None
        parent_value = parent or "<root>"
        keyword_text = ", ".join(str(item) for item in keywords) if keywords else "<none>"
        bootstrap = metadata.get("bootstrap") if isinstance(metadata, Mapping) else None
        other_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"summary", "bootstrap"}
        }
        try:
            bootstrap_text = json.dumps(bootstrap or {}, ensure_ascii=False, default=str)
        except TypeError:
            bootstrap_text = str(bootstrap)
        try:
            metadata_text = json.dumps(other_metadata, ensure_ascii=False, default=str)
        except TypeError:
            metadata_text = str(other_metadata)
        user_prompt = self._summary_user_prompt.format(
            name=name,
            parent=parent_value,
            keywords=keyword_text or "<none>",
            bootstrap=bootstrap_text,
            metadata=metadata_text,
        )
        messages = [
            LLMMessage(role="system", content=self._summary_system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        try:
            response = self._llm_client.generate([messages])[0]
        except LLMClientError as exc:  # pragma: no cover - exercised via tests without LLM
            logger.warning("Ontology summary LLM failed for %s: %s", name, exc)
            return None
        content = response.content.strip()
        if not content:
            return None
        return content.splitlines()[0]


__all__ = ["OntologyManager", "OntologyUpdate", "RuntimeNodeSpec"]

