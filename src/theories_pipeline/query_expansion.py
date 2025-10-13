"""Adaptive query expansion utilities for the theories pipeline.

This module provides helpers that analyse retrieved literature for an ontology
node and propose additional search queries.  Two strategies are supported:

* **LLM-generated queries** powered by :mod:`.llm` which synthesise new search
  phrases from snippets of the already collected papers.
* **Embedding neighbours** which mine high-value terms from the retrieved
  corpus via TF-IDF cosine similarity.

Every expansion attempt is stored on disk under ``data/cache/query_expansion``
so that subsequent runs can audit which prompts and candidates were evaluated
and whether they improved recall.  The cache is intentionally JSON-serialisable
and stable to facilitate reproducible experiments.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .literature import PaperMetadata
from .llm import LLMClient, LLMMessage
from .ontology import OntologyNode

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]

DEFAULT_CACHE_DIR = Path("data/cache/query_expansion")
DEFAULT_GPT_PROMPT = (
    "You are assisting with literature retrieval for the Hackaging theories pipeline. "
    "We are exploring the ontology node '{node_name}' which currently has {current_total} "
    "papers collected out of a target of {target}. Existing query templates are: {queries}. "
    "Below are snippets from representative papers (title and abstract excerpts):\n" 
    "{snippets}\n"
    "Suggest up to {max_queries} new search queries that could retrieve additional relevant "
    "papers. Respond with a JSON array of strings containing only the raw query text."
)


@dataclass
class QueryExpansionSettings:
    """Configuration controlling query expansion behaviour."""

    enabled: bool = True
    use_gpt: bool = True
    use_embeddings: bool = True
    max_new_queries: int = 5
    max_snippets: int = 12
    max_gpt_queries: int = 5
    embedding_neighbors: int = 5
    embedding_ngram_min: int = 1
    embedding_ngram_max: int = 2
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    gpt_prompt: str = DEFAULT_GPT_PROMPT

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cache_dir"] = str(self.cache_dir)
        return data

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any] | None,
        *,
        base: "QueryExpansionSettings" | None = None,
    ) -> "QueryExpansionSettings":
        if base is None:
            kwargs: Dict[str, Any] = {}
        else:
            kwargs = dict(base.to_dict())
            kwargs["cache_dir"] = base.cache_dir
        if data:
            for key, value in data.items():
                if key not in cls.__dataclass_fields__:  # type: ignore[attr-defined]
                    continue
                if key == "cache_dir" and value is not None:
                    kwargs[key] = Path(value)
                elif key in {"embedding_ngram_min", "embedding_ngram_max", "max_new_queries", "max_snippets", "max_gpt_queries", "embedding_neighbors"}:
                    try:
                        kwargs[key] = int(value)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue
                elif key in {"enabled", "use_gpt", "use_embeddings"}:
                    kwargs[key] = bool(value)
                elif key == "gpt_prompt" and isinstance(value, str):
                    kwargs[key] = value
                else:
                    kwargs[key] = value
        return cls(**kwargs)


@dataclass
class QueryCandidate:
    """Generated query candidate with provenance information."""

    query: str
    source: str
    score: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {"query": self.query, "source": self.source}
        if self.score is not None:
            payload["score"] = self.score
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class QueryExpansionSession:
    """Represents one expansion attempt for an ontology node."""

    session_id: str
    node_name: str
    base_queries: Sequence[str]
    candidates: List[QueryCandidate]

    def selected_queries(self) -> List[str]:
        return [candidate.query for candidate in self.candidates]


class QueryExpansionCache:
    """JSON backed cache for storing expansion candidates and performance."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_CACHE_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def _slugify(self, text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower()).strip("-")
        return cleaned or "untitled"

    def _path(self, node_name: str) -> Path:
        return self.directory / f"{self._slugify(node_name)}.json"

    def _load(self, node_name: str) -> Dict[str, Any]:
        path = self._path(node_name)
        if not path.exists():
            return {"node": node_name, "history": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            logger.warning("Failed to decode query expansion cache for %s; resetting", node_name)
            return {"node": node_name, "history": []}

    def _store(self, node_name: str, payload: Mapping[str, Any]) -> None:
        path = self._path(node_name)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def begin_session(
        self,
        session: QueryExpansionSession,
        settings: QueryExpansionSettings,
        snippets: Sequence[str],
    ) -> None:
        record = self._load(session.node_name)
        history: List[MutableMapping[str, Any]] = record.get("history", [])  # type: ignore[assignment]
        entry: Dict[str, Any] = {
            "session_id": session.session_id,
            "timestamp": time.time(),
            "base_queries": list(session.base_queries),
            "candidates": [candidate.to_dict() for candidate in session.candidates],
            "settings": settings.to_dict(),
            "snippets": list(snippets),
            "performance": None,
        }
        history.append(entry)
        record["history"] = history
        self._store(session.node_name, record)

    def record_performance(
        self,
        session: QueryExpansionSession,
        performance: Mapping[str, Any],
    ) -> None:
        record = self._load(session.node_name)
        updated = False
        for entry in record.get("history", []):
            if entry.get("session_id") == session.session_id:
                entry["performance"] = dict(performance)
                updated = True
                break
        if not updated:  # pragma: no cover - defensive
            record.setdefault("history", []).append(
                {
                    "session_id": session.session_id,
                    "timestamp": time.time(),
                    "base_queries": list(session.base_queries),
                    "candidates": [candidate.to_dict() for candidate in session.candidates],
                    "settings": performance.get("settings"),
                    "snippets": [],
                    "performance": dict(performance),
                }
            )
        self._store(session.node_name, record)


class QueryExpander:
    """Generate and persist adaptive query candidates for ontology nodes."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None,
        default_settings: QueryExpansionSettings | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.default_settings = default_settings or QueryExpansionSettings()
        self.cache = QueryExpansionCache(self.default_settings.cache_dir)
        self._session_cache: Dict[str, QueryExpansionCache] = {}

    def settings_for(self, override: Mapping[str, Any] | None) -> QueryExpansionSettings:
        return QueryExpansionSettings.from_mapping(override, base=self.default_settings)

    def expand(
        self,
        node: OntologyNode,
        *,
        base_queries: Sequence[str],
        papers: Sequence[PaperMetadata],
        settings: QueryExpansionSettings,
        context: Mapping[str, Any] | None = None,
    ) -> QueryExpansionSession | None:
        if not settings.enabled:
            return None

        snippets = list(self._prepare_snippets(papers, max_snippets=settings.max_snippets))
        candidates: List[QueryCandidate] = []

        if settings.use_gpt and self.llm_client:
            candidates.extend(
                self._generate_with_gpt(
                    node,
                    base_queries=base_queries,
                    snippets=snippets,
                    settings=settings,
                    context=context or {},
                )
            )
        elif settings.use_gpt:
            logger.debug("Skipping GPT query expansion for %s: no LLM client configured", node.name)

        if settings.use_embeddings:
            candidates.extend(
                self._generate_with_embeddings(
                    node,
                    base_queries=base_queries,
                    snippets=snippets,
                    settings=settings,
                )
            )

        # Normalise queries, remove duplicates, and limit the count.
        seen_queries = set()
        unique_candidates: List[QueryCandidate] = []
        for candidate in candidates:
            normalized = candidate.query.strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            candidate.query = normalized
            unique_candidates.append(candidate)
            if len(unique_candidates) >= settings.max_new_queries:
                break

        if not unique_candidates:
            return None

        session = QueryExpansionSession(
            session_id=f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}",
            node_name=node.name,
            base_queries=list(base_queries),
            candidates=unique_candidates,
        )
        cache = self.cache
        if settings.cache_dir != self.cache.directory:
            cache = QueryExpansionCache(settings.cache_dir)
        cache.begin_session(session, settings, snippets)
        self._session_cache[session.session_id] = cache
        return session

    def record_performance(
        self,
        session: QueryExpansionSession,
        *,
        before_total: int,
        after_total: int,
        new_unique: int,
    ) -> None:
        performance = {
            "before_total": before_total,
            "after_total": after_total,
            "delta": after_total - before_total,
            "new_unique": new_unique,
        }
        cache = self._session_cache.pop(session.session_id, self.cache)
        cache.record_performance(session, performance)

    # ------------------------------------------------------------------
    # Candidate generators
    # ------------------------------------------------------------------
    def _prepare_snippets(
        self,
        papers: Sequence[PaperMetadata],
        *,
        max_snippets: int,
    ) -> Iterable[str]:
        for paper in papers[:max_snippets]:
            title = paper.title.strip() if paper.title else ""
            abstract = paper.abstract.strip() if paper.abstract else ""
            snippet_parts = [part for part in (title, abstract) if part]
            if not snippet_parts:
                continue
            snippet = " - ".join(snippet_parts)
            if len(snippet) > 400:
                snippet = f"{snippet[:397]}..."
            yield snippet

    def _generate_with_gpt(
        self,
        node: OntologyNode,
        *,
        base_queries: Sequence[str],
        snippets: Sequence[str],
        settings: QueryExpansionSettings,
        context: Mapping[str, Any],
    ) -> List[QueryCandidate]:
        if not self.llm_client:
            return []
        prompt = settings.gpt_prompt.format(
            node_name=node.name,
            target=node.target or "unknown",
            queries=", ".join(base_queries),
            snippets="\n".join(f"- {snippet}" for snippet in snippets) or "(no snippets)",
            max_queries=settings.max_gpt_queries,
            current_total=context.get("current_total", 0),
        )
        messages = [
            [
                LLMMessage(role="system", content="You generate search queries for academic literature."),
                LLMMessage(role="user", content=prompt),
            ]
        ]
        response = self.llm_client.generate(messages)[0]
        content = response.content.strip()
        try:
            queries = json.loads(content)
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON list of queries")
            parsed = [str(item).strip() for item in queries if str(item).strip()]
        except (json.JSONDecodeError, ValueError):  # pragma: no cover - defensive
            parsed = [line.strip("- ") for line in content.splitlines() if line.strip()]
        candidates = [
            QueryCandidate(query=query, source="gpt", metadata={"cached": response.cached})
            for query in parsed[: settings.max_gpt_queries]
        ]
        if not candidates:
            logger.debug("LLM returned no usable query suggestions for %s", node.name)
        return candidates

    def _generate_with_embeddings(
        self,
        node: OntologyNode,
        *,
        base_queries: Sequence[str],
        snippets: Sequence[str],
        settings: QueryExpansionSettings,
    ) -> List[QueryCandidate]:
        if not snippets:
            return []
        if TfidfVectorizer is None or cosine_similarity is None:
            logger.debug("Skipping embedding-based expansion for %s: scikit-learn not available", node.name)
            return []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(settings.embedding_ngram_min, settings.embedding_ngram_max),
        )
        matrix = vectorizer.fit_transform(snippets)
        centroid = matrix.mean(axis=0)
        similarities = cosine_similarity(matrix, centroid)
        ranked_indices = similarities.reshape(-1).argsort()[::-1]

        feature_array = vectorizer.get_feature_names_out()
        term_scores = matrix.sum(axis=0).A1
        ranked_terms = sorted(
            ((feature_array[idx], term_scores[idx]) for idx in range(len(feature_array))),
            key=lambda item: item[1],
            reverse=True,
        )

        candidates: List[QueryCandidate] = []
        for term, score in ranked_terms:
            if not term.strip():
                continue
            query_text = f"{node.name} {term}".strip()
            candidates.append(
                QueryCandidate(
                    query=query_text,
                    source="embedding",
                    score=float(score),
                    metadata={"top_snippet_index": int(ranked_indices[0]) if ranked_indices.size else 0},
                )
            )
            if len(candidates) >= settings.embedding_neighbors:
                break
        return candidates


__all__ = [
    "QueryExpander",
    "QueryExpansionSettings",
    "QueryExpansionSession",
    "QueryCandidate",
]
