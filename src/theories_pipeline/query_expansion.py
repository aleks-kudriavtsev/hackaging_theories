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
import re
import shlex
import string
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .literature import PaperMetadata
from .llm import LLMClient, LLMMessage
from .ontology import OntologyNode
from .ontology_manager import RuntimeNodeSpec

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
    bootstrap_new_theories: bool = False
    bootstrap_candidate_limit: int | None = None
    bootstrap_mode: str = "child"
    bootstrap_max_labels: int | None = None
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
                elif key in {
                    "embedding_ngram_min",
                    "embedding_ngram_max",
                    "max_new_queries",
                    "max_snippets",
                    "max_gpt_queries",
                    "embedding_neighbors",
                    "bootstrap_candidate_limit",
                    "bootstrap_max_labels",
                }:
                    try:
                        kwargs[key] = int(value)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue
                elif key in {"enabled", "use_gpt", "use_embeddings", "bootstrap_new_theories"}:
                    kwargs[key] = bool(value)
                elif key == "bootstrap_mode" and isinstance(value, str):
                    kwargs[key] = value
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


_PUNCT_TRIM = string.punctuation.replace("-", "")


def queries_to_keywords(queries: Sequence[str]) -> List[str]:
    """Normalise raw query strings into keyword tokens suitable for storage."""

    if not queries:
        return []
    keywords: List[str] = []
    seen: set[str] = set()
    for raw in queries:
        if not isinstance(raw, str):
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        cleaned = cleaned.replace("/", " ").replace("|", " ").replace("+", " ")
        cleaned = cleaned.replace("(", " ").replace(")", " ")
        cleaned = re.sub(r"\b(?:AND|OR|NOT)\b", " ", cleaned, flags=re.IGNORECASE)
        for delimiter in ",;":
            cleaned = cleaned.replace(delimiter, " ")
        try:
            parts = shlex.split(cleaned)
        except ValueError:
            parts = cleaned.split()
        for part in parts:
            token = part.strip()
            if not token:
                continue
            token = token.strip(_PUNCT_TRIM)
            token = " ".join(token.split())
            if not token:
                continue
            lowered = token.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(lowered)
    return keywords


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
        *,
        new_papers: Sequence[PaperMetadata] | None = None,
        per_query_new_unique: Mapping[str, int] | None = None,
        per_query_new_papers: Mapping[str, Sequence[PaperMetadata]] | None = None,
    ) -> None:
        record = self._load(session.node_name)
        updated = False
        for entry in record.get("history", []):
            if entry.get("session_id") == session.session_id:
                entry["performance"] = dict(performance)
                entry["new_papers"] = [paper.to_dict() for paper in new_papers or []]
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
                    "new_papers": [paper.to_dict() for paper in new_papers or []],
                }
            )
        self._update_query_metrics(
            record,
            session,
            performance,
            new_papers or [],
            per_query_new_unique=per_query_new_unique,
            per_query_new_papers=per_query_new_papers,
        )
        self._store(session.node_name, record)

    def _update_query_metrics(
        self,
        record: MutableMapping[str, Any],
        session: QueryExpansionSession,
        performance: Mapping[str, Any],
        new_papers: Sequence[PaperMetadata],
        *,
        per_query_new_unique: Mapping[str, int] | None = None,
        per_query_new_papers: Mapping[str, Sequence[PaperMetadata]] | None = None,
    ) -> None:
        metrics = record.setdefault("metrics", {})
        query_metrics: MutableMapping[str, MutableMapping[str, Any]] = metrics.setdefault(
            "queries", {}
        )
        timestamp = time.time()
        per_query_unique_map: Dict[str, int] = {}
        if per_query_new_unique:
            per_query_unique_map = {
                str(query): int(value or 0)
                for query, value in per_query_new_unique.items()
            }
        else:
            raw_unique = performance.get("per_query_new_unique")
            if isinstance(raw_unique, Mapping):
                per_query_unique_map = {
                    str(query): int(value or 0)
                    for query, value in raw_unique.items()
                }

        def _build_payload(papers: Sequence[PaperMetadata]) -> List[Dict[str, Any]]:
            payload: List[Dict[str, Any]] = []
            seen_identifiers: set[str] = set()
            for paper in papers:
                identifier = paper.identifier
                if not identifier or identifier in seen_identifiers:
                    continue
                seen_identifiers.add(identifier)
                payload.append(
                    {
                        "identifier": identifier,
                        "title": paper.title,
                        "source": paper.source,
                    }
                )
            return payload

        per_query_payloads: Dict[str, List[Dict[str, Any]]] = {}
        if per_query_new_papers:
            for query, papers in per_query_new_papers.items():
                per_query_payloads[str(query)] = _build_payload(papers)
        elif new_papers:
            fallback_payload = _build_payload(new_papers)
            if fallback_payload:
                per_query_payloads = {candidate.query: list(fallback_payload) for candidate in session.candidates}
        for candidate in session.candidates:
            query_key = candidate.query
            payload = query_metrics.setdefault(query_key, {})
            payload["attempts"] = int(payload.get("attempts", 0)) + 1
            payload["successes"] = int(payload.get("successes", 0))
            payload["new_unique_total"] = int(payload.get("new_unique_total", 0))
            payload.setdefault("sources", [])
            payload.setdefault("keywords", queries_to_keywords([query_key]))
            payload.setdefault("supporting_papers", [])
            payload.setdefault("promoted_nodes", [])
            payload.setdefault("metadata", {})
            payload["metadata"]["last_session_id"] = session.session_id
            payload["metadata"]["last_updated"] = timestamp
            payload["sources"] = list({*payload["sources"], candidate.source})
            query_unique = int(per_query_unique_map.get(query_key, 0) or 0)
            if query_unique > 0:
                payload["successes"] = int(payload.get("successes", 0)) + 1
                payload["new_unique_total"] = int(payload.get("new_unique_total", 0)) + query_unique
                existing_support = payload["supporting_papers"]
                seen_identifiers = {entry.get("identifier") for entry in existing_support if isinstance(entry, Mapping)}
                for paper_entry in per_query_payloads.get(query_key, []):
                    identifier = paper_entry.get("identifier")
                    if identifier and identifier not in seen_identifiers:
                        existing_support.append(paper_entry)
                        seen_identifiers.add(identifier)

    def get_query_metrics(self, node_name: str) -> Dict[str, Mapping[str, Any]]:
        record = self._load(node_name)
        metrics = record.get("metrics", {})
        if not isinstance(metrics, Mapping):
            return {}
        queries = metrics.get("queries")
        if not isinstance(queries, Mapping):
            return {}
        sanitized: Dict[str, Mapping[str, Any]] = {}
        for query, payload in queries.items():
            if not isinstance(query, str) or not isinstance(payload, Mapping):
                continue
            sanitized[query] = payload
        return sanitized

    def mark_query_promoted(self, node_name: str, query: str, node_label: str) -> None:
        record = self._load(node_name)
        metrics = record.setdefault("metrics", {})
        query_metrics: MutableMapping[str, MutableMapping[str, Any]] = metrics.setdefault(
            "queries", {}
        )
        payload = query_metrics.setdefault(query, {})
        promoted = payload.setdefault("promoted_nodes", [])
        if isinstance(promoted, list) and node_label not in promoted:
            promoted.append(node_label)
        elif not isinstance(promoted, list):
            payload["promoted_nodes"] = [node_label]
        self._store(node_name, record)


@dataclass(frozen=True)
class QueryPerformanceRecord:
    """Aggregate statistics for a query evaluated during expansion."""

    node_name: str
    query: str
    attempts: int
    successes: int
    new_unique_total: int
    keywords: Sequence[str]
    sources: Sequence[str]
    supporting_papers: Sequence[Mapping[str, Any]]
    promoted_nodes: Sequence[str]
    last_session_id: str | None = None


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
        new_papers: Sequence[PaperMetadata] | None = None,
        per_query_new_unique: Mapping[str, int] | None = None,
        per_query_new_papers: Mapping[str, Sequence[PaperMetadata]] | None = None,
    ) -> None:
        performance = {
            "before_total": before_total,
            "after_total": after_total,
            "delta": after_total - before_total,
            "new_unique": new_unique,
        }
        if per_query_new_unique is not None:
            performance["per_query_new_unique"] = {
                str(query): int(value or 0)
                for query, value in per_query_new_unique.items()
            }
        cache = self._session_cache.pop(session.session_id, self.cache)
        cache.record_performance(
            session,
            performance,
            new_papers=new_papers or [],
            per_query_new_unique=per_query_new_unique,
            per_query_new_papers=per_query_new_papers,
        )

    def consistent_queries(
        self,
        node_name: str,
        *,
        selected_queries: Sequence[str],
        min_successes: int = 2,
        min_new_unique: int = 3,
        min_success_rate: float = 0.6,
    ) -> List[QueryPerformanceRecord]:
        metrics = self.cache.get_query_metrics(node_name)
        consistent: List[QueryPerformanceRecord] = []
        for raw_query in selected_queries:
            if not isinstance(raw_query, str):
                continue
            payload = metrics.get(raw_query)
            if not payload:
                continue
            attempts = int(payload.get("attempts", 0) or 0)
            successes = int(payload.get("successes", 0) or 0)
            new_unique_total = int(payload.get("new_unique_total", 0) or 0)
            if attempts <= 0:
                continue
            success_rate = successes / attempts if attempts else 0.0
            if successes < min_successes or new_unique_total < min_new_unique:
                continue
            if success_rate < min_success_rate:
                continue
            keywords = payload.get("keywords")
            if not isinstance(keywords, Sequence):
                keywords = queries_to_keywords([raw_query])
            sources = payload.get("sources")
            if not isinstance(sources, Sequence):
                sources = []
            supporting = payload.get("supporting_papers")
            if not isinstance(supporting, Sequence):
                supporting = []
            promoted = payload.get("promoted_nodes")
            if not isinstance(promoted, Sequence):
                promoted = []
            metadata = payload.get("metadata")
            last_session_id = None
            if isinstance(metadata, Mapping):
                last_session_id = metadata.get("last_session_id")
            consistent.append(
                QueryPerformanceRecord(
                    node_name=node_name,
                    query=raw_query,
                    attempts=attempts,
                    successes=successes,
                    new_unique_total=new_unique_total,
                    keywords=list(keywords),
                    sources=list(sources),
                    supporting_papers=list(supporting),
                    promoted_nodes=list(promoted),
                    last_session_id=str(last_session_id) if last_session_id else None,
                )
            )
        return consistent

    def mark_query_promoted(self, node_name: str, query: str, node_label: str) -> None:
        self.cache.mark_query_promoted(node_name, query, node_label)

    def build_runtime_spec(
        self,
        record: QueryPerformanceRecord,
        *,
        parent: str | None,
        new_papers: Sequence[PaperMetadata],
        existing_names: Iterable[str] | None = None,
    ) -> RuntimeNodeSpec | None:
        candidate_keywords = self._theme_keywords(record, new_papers)
        if not candidate_keywords:
            return None
        candidate_name = self._theme_name(candidate_keywords, existing_names)
        if not candidate_name:
            return None
        provenance_payload: Dict[str, Any] = {
            "source": "adaptive_query_theme",
            "query": record.query,
            "metrics": {
                "attempts": record.attempts,
                "successes": record.successes,
                "new_unique_total": record.new_unique_total,
            },
            "supporting_papers": [dict(item) for item in record.supporting_papers],
        }
        if record.last_session_id:
            provenance_payload["last_session_id"] = record.last_session_id
        if new_papers:
            provenance_payload["latest_supporting_papers"] = [
                {
                    "identifier": paper.identifier,
                    "title": paper.title,
                    "source": paper.source,
                }
                for paper in new_papers
            ]
        metadata = {
            "bootstrap": {
                "source": "adaptive_query_theme",
                "query": record.query,
            }
        }
        return RuntimeNodeSpec(
            name=candidate_name,
            parent=parent,
            config={},
            keywords=candidate_keywords,
            metadata=metadata,
            provenance=provenance_payload,
        )

    def _theme_keywords(
        self, record: QueryPerformanceRecord, new_papers: Sequence[PaperMetadata]
    ) -> List[str]:
        base_keywords = [token for token in record.keywords if isinstance(token, str) and token]
        base_keywords = [token.lower() for token in base_keywords]
        theme_terms = _extract_theme_terms(new_papers)
        combined: List[str] = []
        seen: set[str] = set()
        for token in base_keywords + theme_terms:
            normalized = token.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            combined.append(normalized)
            if len(combined) >= 10:
                break
        return combined

    def _theme_name(
        self, keywords: Sequence[str], existing_names: Iterable[str] | None
    ) -> str | None:
        if not keywords:
            return None
        existing = {name.lower() for name in (existing_names or [])}
        base_tokens = [token for token in keywords[:3] if token]
        if not base_tokens:
            return None
        candidate = " ".join(token.title() for token in base_tokens).strip()
        if not candidate:
            return None
        if candidate.lower() not in existing:
            return candidate
        suffix = 2
        while suffix < 10:
            renamed = f"{candidate} {suffix}"
            if renamed.lower() not in existing:
                return renamed
            suffix += 1
        return None

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
        if TfidfVectorizer is None or cosine_similarity is None:  # pragma: no cover - optional dependency
            logger.debug("Skipping embedding-based expansion for %s: scikit-learn not available", node.name)
            return []

        if not snippets:
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


_THEME_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_STOPWORDS = {
    "and",
    "the",
    "with",
    "from",
    "into",
    "using",
    "for",
    "into",
    "that",
    "this",
    "over",
    "between",
    "after",
    "before",
    "during",
    "among",
    "their",
    "within",
    "study",
    "analysis",
    "effects",
    "effect",
    "aging",
    "ageing",
    "older",
    "adults",
}


def _extract_theme_terms(papers: Sequence[PaperMetadata]) -> List[str]:
    if not papers:
        return []
    counter: Counter[str] = Counter()
    for paper in papers:
        text = " ".join(
            part
            for part in (
                paper.title or "",
                paper.abstract or "",
            )
            if part
        ).lower()
        if not text:
            continue
        for match in _THEME_TOKEN_PATTERN.findall(text):
            token = match.lower().strip("- ")
            if not token or token in _STOPWORDS:
                continue
            counter[token] += 1
    if not counter:
        return []
    ranked = [token for token, _count in counter.most_common(10)]
    return ranked


__all__ = [
    "QueryExpander",
    "QueryExpansionSettings",
    "QueryExpansionSession",
    "QueryCandidate",
    "QueryPerformanceRecord",
    "queries_to_keywords",
]
