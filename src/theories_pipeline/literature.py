"""Utilities for retrieving literature metadata for the Hackaging theories pipeline.

This module implements a pluggable retrieval backend capable of streaming
metadata from multiple public scholarly APIs (OpenAlex, CrossRef, PubMed) while
supporting pagination, rate limiting, and persistent incremental state. The
primary entry point is :class:`LiteratureRetriever`, which coordinates provider
requests, deduplicates records, and stores progress under ``data/`` so repeated
collection runs resume from the previous cursor positions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperMetadata:
    """Minimal metadata required for downstream classification and extraction."""

    identifier: str
    title: str
    authors: Sequence[str]
    abstract: str
    source: str
    year: int | None = None
    doi: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "source": self.source,
            "year": self.year,
            "doi": self.doi,
        }

    @staticmethod
    def from_dict(data: MutableMapping[str, Any]) -> "PaperMetadata":
        return PaperMetadata(
            identifier=data["identifier"],
            title=data["title"],
            authors=tuple(data.get("authors", ())),
            abstract=data.get("abstract", ""),
            source=data.get("source", "unknown"),
            year=data.get("year"),
            doi=data.get("doi"),
        )

    @property
    def dedupe_key(self) -> str:
        if self.doi:
            return self.doi.lower()
        return self.identifier.lower()


@dataclass
class ProviderConfig:
    """Configuration for a literature provider."""

    name: str
    type: str
    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    query_shards: Sequence[str] | None = None
    batch_size: int = 200
    rate_limit_per_sec: float | None = None
    timeout: float | None = 30.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderPage:
    """Response page returned by a provider request."""

    papers: List[PaperMetadata]
    next_cursor: str | None
    exhausted: bool


class RateLimiter:
    """Simple time-based rate limiter (per-provider)."""

    def __init__(self, per_second: float | None) -> None:
        if per_second and per_second > 0:
            self.interval = 1.0 / per_second
        else:
            self.interval = 0.0
        self._last_ts: float | None = None

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        if self._last_ts is None:
            self._last_ts = now
            return
        elapsed = now - self._last_ts
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_ts = time.monotonic()


class StateStore:
    """Persisted retrieval state stored as JSON under ``data/``."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "literature_state.json"
        if self.state_path.exists():
            self._data = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self._data = {}

    def get(self, key: str) -> Dict[str, Any]:
        data = self._data.get(key, {})
        return json.loads(json.dumps(data))  # deep copy

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self._data[key] = value
        self._flush()

    def clear(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
            self._flush()

    def write_summary(self, summary: Dict[str, Any]) -> None:
        summary_path = self.directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def _flush(self) -> None:
        self.state_path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")


class BaseProvider:
    """Base class for pluggable literature providers."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        if requests is None:
            raise RuntimeError(
                "The 'requests' package is required for HTTP-based literature providers"
            )
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(config.rate_limit_per_sec)

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def query_shards(self) -> Sequence[str]:
        if self.config.query_shards:
            return list(self.config.query_shards)
        return ("{query}",)

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        raise NotImplementedError


class OpenAlexProvider(BaseProvider):
    """Provider implementation for the OpenAlex Works API."""

    DEFAULT_URL = "https://api.openalex.org/works"

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:  # pragma: no cover - network IO
        params = {
            "search": query,
            "per-page": self.config.batch_size,
            "cursor": cursor or "*",
        }
        headers: Dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        url = self.config.base_url or self.DEFAULT_URL
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        papers: List[PaperMetadata] = []
        for item in payload.get("results", []):
            abstract = decode_openalex_abstract(item.get("abstract_inverted_index"))
            authors = [auth.get("author", {}).get("display_name", "") for auth in item.get("authorships", [])]
            doi = item.get("doi")
            year = item.get("publication_year")
            identifier = item.get("id") or (f"openalex:{item.get('doi')}" if item.get("doi") else "")
            if not identifier:
                continue
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=item.get("title", ""),
                    authors=tuple(a for a in authors if a),
                    abstract=abstract,
                    source="openalex",
                    year=year,
                    doi=doi,
                )
            )
        next_cursor = payload.get("meta", {}).get("next_cursor")
        exhausted = next_cursor is None
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class CrossRefProvider(BaseProvider):
    """Provider implementation for the CrossRef Works API."""

    DEFAULT_URL = "https://api.crossref.org/works"

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:  # pragma: no cover - network IO
        params = {
            "query": query,
            "rows": self.config.batch_size,
            "cursor": cursor or "*",
            "cursor_max": 10000,
        }
        if self.config.api_key:
            params["mailto"] = self.config.api_key
        params.update(self.config.extra)
        url = self.config.base_url or self.DEFAULT_URL
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("message", {}).get("items", [])
        papers: List[PaperMetadata] = []
        for item in items:
            doi = item.get("DOI")
            identifier = f"crossref:{doi}" if doi else item.get("URL", "")
            if not identifier:
                continue
            authors = []
            for author in item.get("author", []) or []:
                given = author.get("given", "").strip()
                family = author.get("family", "").strip()
                if given and family:
                    authors.append(f"{given} {family}")
                elif family:
                    authors.append(family)
                elif given:
                    authors.append(given)
            abstract = item.get("abstract", "")
            year = None
            if "issued" in item and item["issued"].get("date-parts"):
                year = item["issued"]["date-parts"][0][0]
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=item.get("title", [""])[0],
                    authors=tuple(authors),
                    abstract=abstract,
                    source="crossref",
                    year=year,
                    doi=doi,
                )
            )
        next_cursor = payload.get("message", {}).get("next-cursor")
        exhausted = not next_cursor
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class PubMedProvider(BaseProvider):
    """Provider implementation for the PubMed E-Utilities API."""

    SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:  # pragma: no cover - network IO
        # PubMed uses an integer offset cursor.
        start = int(cursor or 0)
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retstart": start,
            "retmax": self.config.batch_size,
        }
        if self.config.api_key:
            search_params["api_key"] = self.config.api_key
        self.rate_limiter.wait()
        response = self.session.get(self.SEARCH_URL, params=search_params, timeout=self.config.timeout)
        response.raise_for_status()
        search_payload = response.json()
        result = search_payload.get("esearchresult", {})
        id_list = result.get("idlist", [])
        total_count = int(result.get("count", 0))
        if not id_list:
            return ProviderPage(papers=[], next_cursor=None, exhausted=True)

        summary_params = {
            "db": "pubmed",
            "retmode": "json",
            "id": ",".join(id_list),
        }
        if self.config.api_key:
            summary_params["api_key"] = self.config.api_key
        self.rate_limiter.wait()
        summary_response = self.session.get(self.SUMMARY_URL, params=summary_params, timeout=self.config.timeout)
        summary_response.raise_for_status()
        summary_payload = summary_response.json().get("result", {})
        papers: List[PaperMetadata] = []
        for pmid in id_list:
            record = summary_payload.get(pmid)
            if not record:
                continue
            authors = [
                " ".join(filter(None, (author.get("name", ""), author.get("authtype", "")))).strip()
                for author in record.get("authors", [])
            ]
            pubdate = record.get("pubdate", "")
            year = None
            if pubdate:
                try:
                    year = int(pubdate.split()[0])
                except (ValueError, IndexError):
                    year = None
            doi = None
            for article_id in record.get("articleids", []) or []:
                if article_id.get("idtype") == "doi":
                    doi = article_id.get("value")
                    break
            papers.append(
                PaperMetadata(
                    identifier=f"pubmed:{pmid}",
                    title=record.get("title", ""),
                    authors=tuple(a for a in authors if a),
                    abstract=record.get("elocationid", ""),
                    source="pubmed",
                    year=year,
                    doi=doi,
                )
            )

        next_start = start + len(id_list)
        exhausted = next_start >= total_count
        next_cursor = None if exhausted else str(next_start)
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


PROVIDER_REGISTRY: Dict[str, type[BaseProvider]] = {
    "openalex": OpenAlexProvider,
    "crossref": CrossRefProvider,
    "pubmed": PubMedProvider,
}


@dataclass
class RetrievalResult:
    papers: List[PaperMetadata]
    newly_added: int
    summary: Dict[str, Any]


class LiteratureRetriever:
    """Retrieves literature metadata from configured data sources."""

    def __init__(
        self,
        seed_data_path: Path | None = None,
        provider_configs: Sequence[ProviderConfig] | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self.seed_data_path = Path(seed_data_path) if seed_data_path else None
        if self.seed_data_path and not self.seed_data_path.exists():
            raise FileNotFoundError(f"Seed data path {self.seed_data_path} does not exist")
        self._seed_cache: List[PaperMetadata] | None = None
        self.providers: List[BaseProvider] = []
        for config in provider_configs or []:
            if not config.enabled:
                logger.debug("Skipping disabled provider %s", config.name)
                continue
            provider_cls = PROVIDER_REGISTRY.get(config.type.lower())
            if not provider_cls:
                raise ValueError(f"Unknown provider type '{config.type}' for {config.name}")
            self.providers.append(provider_cls(config))
        state_directory = state_dir or Path("data/cache/literature")
        self.state_store = StateStore(state_directory)

    def _load_seed_papers(self) -> List[PaperMetadata]:
        if self.seed_data_path is None:
            return []
        if self._seed_cache is None:
            if not self.seed_data_path.exists():
                raise FileNotFoundError(f"Seed data path {self.seed_data_path} does not exist")
            with self.seed_data_path.open("r", encoding="utf-8") as handle:
                raw_items = json.load(handle)
            papers: List[PaperMetadata] = []
            for item in raw_items:
                papers.append(
                    PaperMetadata(
                        identifier=item["identifier"],
                        title=item["title"],
                        authors=item.get("authors", []),
                        abstract=item.get("abstract", ""),
                        source=item.get("source", "seed"),
                        year=item.get("year"),
                        doi=item.get("doi"),
                    )
                )
            self._seed_cache = papers
        return list(self._seed_cache)

    def search(
        self,
        query: str,
        limit: int | None = None,
        providers: Iterable[
            Callable[[str, int | None], Iterable[PaperMetadata]]
        ]
        | None = None,
    ) -> List[PaperMetadata]:
        """Backward-compatible search returning seed data and optional providers."""

        results: List[PaperMetadata] = []
        seen_keys = set()

        for paper in self._load_seed_papers():
            if _matches_query(paper, query):
                results.append(paper)
                seen_keys.add(paper.dedupe_key)
                if limit is not None and len(results) >= limit:
                    return results[:limit]

        if providers is not None:
            for provider in providers:
                for paper in provider(query, limit):
                    key = paper.dedupe_key
                    if key not in seen_keys:
                        results.append(paper)
                        seen_keys.add(key)
                        if limit is not None and len(results) >= limit:
                            return results[:limit]
            return results[:limit] if limit is not None else results

        if self.providers:
            provider_names = [provider.name for provider in self.providers]
            retrieval = self.collect_queries(
                [query],
                target=limit,
                providers=provider_names,
                state_key=None,
                resume=False,
            )
            for paper in retrieval.papers:
                key = paper.dedupe_key
                if key not in seen_keys and _matches_query(paper, query):
                    results.append(paper)
                    seen_keys.add(key)
                    if limit is not None and len(results) >= limit:
                        break

        return results[:limit] if limit is not None else results

    def collect_queries(
        self,
        queries: Sequence[str],
        *,
        target: int | None = None,
        providers: Sequence[str] | None = None,
        state_key: str | None,
        resume: bool = True,
    ) -> RetrievalResult:
        """Collect papers for a set of query templates until ``target`` is met."""

        if not queries:
            return RetrievalResult(papers=[], newly_added=0, summary={"target": target or 0})

        provider_lookup = {provider.name: provider for provider in self.providers}
        selected_providers: List[BaseProvider]
        if providers:
            missing = [name for name in providers if name not in provider_lookup]
            if missing:
                raise ValueError(f"Unknown providers requested: {', '.join(missing)}")
            selected_providers = [provider_lookup[name] for name in providers]
        else:
            selected_providers = list(self.providers)

        state: Dict[str, Any] = {}
        if state_key and resume:
            state = self.state_store.get(state_key)
        seen_identifiers = set(state.get("seen_identifiers", []))
        stored_papers = [PaperMetadata.from_dict(item) for item in state.get("papers", [])]
        prior_total = len(seen_identifiers) or len(stored_papers)

        collected: List[PaperMetadata] = list(stored_papers)
        newly_added = 0
        provider_totals = state.get("provider_totals", {})
        provider_totals = {k: int(v) for k, v in provider_totals.items()}
        query_state = state.get("queries", {})

        # Always include seed papers once per request.
        if not state.get("seed_consumed"):
            for paper in self._load_seed_papers():
                key = paper.dedupe_key
                if key not in seen_identifiers:
                    collected.append(paper)
                    seen_identifiers.add(key)
                    newly_added += 1
            state["seed_consumed"] = True

        for query in queries:
            query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()
            per_query_state = query_state.get(query_hash, {})
            for provider in selected_providers:
                provider_state = per_query_state.get(provider.name, {})
                shard_states = provider_state.get("shards", {})
                for index, shard_template in enumerate(provider.query_shards):
                    shard_key = str(index)
                    shard_state = shard_states.get(shard_key, {})
                    if shard_state.get("exhausted"):
                        continue
                    cursor = shard_state.get("cursor")
                    final_query = shard_template.format(query=query)
                    while True:
                        if target is not None and len(seen_identifiers) >= target:
                            break
                        try:
                            page = provider.fetch_page(final_query, cursor=cursor)
                        except requests.RequestException as exc:  # pragma: no cover - network failure
                            logger.warning(
                                "Provider %s failed to fetch page for query '%s': %s",
                                provider.name,
                                final_query,
                                exc,
                            )
                            shard_state["exhausted"] = True
                            break

                        if not page.papers:
                            shard_state["exhausted"] = True
                            break

                        for paper in page.papers:
                            key = paper.dedupe_key
                            if key not in seen_identifiers:
                                seen_identifiers.add(key)
                                collected.append(paper)
                                newly_added += 1
                                provider_totals[provider.name] = provider_totals.get(provider.name, 0) + 1
                            else:
                                provider_totals.setdefault(provider.name, 0)

                        cursor = page.next_cursor
                        shard_state["cursor"] = cursor
                        if page.exhausted or cursor is None:
                            shard_state["exhausted"] = True
                            break
                        if target is not None and len(seen_identifiers) >= target:
                            break

                    shard_states[shard_key] = shard_state
                    if target is not None and len(seen_identifiers) >= target:
                        break
                provider_state["shards"] = shard_states
                per_query_state[provider.name] = provider_state
                if target is not None and len(seen_identifiers) >= target:
                    break
            query_state[query_hash] = per_query_state
            if target is not None and len(seen_identifiers) >= target:
                break

        state.update(
            {
                "seen_identifiers": sorted(seen_identifiers),
                "papers": [paper.to_dict() for paper in collected],
                "provider_totals": provider_totals,
                "queries": query_state,
            }
        )
        if state_key:
            self.state_store.set(state_key, state)

        summary = {
            "target": target,
            "total_unique": len(collected),
            "newly_retrieved": newly_added,
            "providers": provider_totals,
            "queries": list(queries),
            "met_target": target is None or len(collected) >= target,
            "prior_total": prior_total,
        }

        return RetrievalResult(papers=collected, newly_added=newly_added, summary=summary)


def decode_openalex_abstract(index: Dict[str, List[int]] | None) -> str:
    """Convert the OpenAlex inverted abstract index to plain text."""

    if not index:
        return ""
    words = sorted(((position, word) for word, positions in index.items() for position in positions))
    ordered_words = [word for _position, word in sorted(words, key=lambda item: item[0])]
    return " ".join(ordered_words)


def _matches_query(paper: PaperMetadata, query: str) -> bool:
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    if not terms:
        return True
    haystack = " ".join([paper.title or "", paper.abstract or ""]).lower()
    return all(term in haystack for term in terms)


__all__ = [
    "PaperMetadata",
    "ProviderConfig",
    "RetrievalResult",
    "LiteratureRetriever",
]

