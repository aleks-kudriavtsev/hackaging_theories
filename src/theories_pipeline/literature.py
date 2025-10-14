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
import html
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Sequence, Tuple, Mapping

try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pdfminer.high_level import extract_text as pdf_extract_text
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pdf_extract_text = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperSection:
    """Structured subsection of a paper's full text."""

    title: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return {"title": self.title, "text": self.text}

    @staticmethod
    def from_dict(data: MutableMapping[str, Any]) -> "PaperSection":
        return PaperSection(title=str(data.get("title", "")), text=str(data.get("text", "")))


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
    full_text: str = ""
    sections: Tuple[PaperSection, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "authors": list(self.authors),
            "abstract": self.abstract,
            "source": self.source,
            "year": self.year,
            "doi": self.doi,
            "full_text": self.full_text,
            "sections": [section.to_dict() for section in self.sections],
        }

    @staticmethod
    def from_dict(data: MutableMapping[str, Any]) -> "PaperMetadata":
        sections_data = data.get("sections") or []
        sections: Tuple[PaperSection, ...] = tuple(
            PaperSection.from_dict(item) for item in sections_data if isinstance(item, MutableMapping)
        )
        return PaperMetadata(
            identifier=data["identifier"],
            title=str(data.get("title", "")),
            authors=tuple(data.get("authors", ())),
            abstract=str(data.get("abstract", "") or ""),
            source=str(data.get("source", "unknown") or "unknown"),
            year=data.get("year"),
            doi=data.get("doi"),
            full_text=str(data.get("full_text") or ""),
            sections=sections,
        )

    @property
    def dedupe_key(self) -> str:
        if self.doi:
            return self.doi.lower()
        return self.identifier.lower()

    @property
    def analysis_text(self) -> str:
        if self.full_text and self.full_text.strip():
            return self.full_text
        if self.sections:
            joined = "\n\n".join(
                "\n".join(
                    part
                    for part in (section.title.strip(), section.text.strip())
                    if part
                )
                for section in self.sections
                if (section.title and section.title.strip()) or (section.text and section.text.strip())
            )
            if joined:
                return joined
        return self.abstract


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
        self._lock = Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
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
        cache_override = self.config.extra.get("fulltext_cache_dir") if self.config.extra else None
        cache_dir = Path(cache_override) if cache_override else Path("data/cache/fulltext")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._fulltext_cache_dir = cache_dir

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

    # ------------------------------------------------------------------
    # Full-text helpers
    # ------------------------------------------------------------------
    def _fulltext_cache_path(self, identifier: str, doi: str | None) -> Path:
        key = (doi or identifier).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return self._fulltext_cache_dir / f"{digest}.txt"

    def _read_cached_full_text(self, identifier: str, doi: str | None) -> str:
        path = self._fulltext_cache_path(identifier, doi)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem edge case
            logger.debug("Failed to read cached full text for %s: %s", identifier, exc)
            return ""

    def _write_full_text_cache(self, identifier: str, doi: str | None, text: str) -> None:
        if not text:
            return
        path = self._fulltext_cache_path(identifier, doi)
        try:
            path.write_text(text, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem edge case
            logger.debug("Failed to write cached full text for %s: %s", identifier, exc)

    def _download_text(self, url: str) -> str:
        headers = {
            "Accept": "text/plain, text/html;q=0.9, application/json;q=0.1",
            "User-Agent": "hackaging-theories-pipeline/1.0",
        }
        try:
            self.rate_limiter.wait()
            response = self.session.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            logger.debug("Failed to download full text from %s: %s", url, exc)
            return ""

        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            if pdf_extract_text is None:
                logger.debug(
                    "PDF response received from %s but pdfminer.six is not installed", url
                )
                return ""
            try:
                text = pdf_extract_text(io.BytesIO(response.content))
            except Exception as exc:  # pragma: no cover - optional dependency failure
                logger.debug("Failed to extract text from PDF %s: %s", url, exc)
                return ""
            normalized = _normalize_full_text(text)
            if len(normalized) > 500_000:
                normalized = normalized[:500_000]
            return normalized

        if "json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if payload is None:
                return ""
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            try:
                text = response.text
            except UnicodeDecodeError:
                logger.debug("Failed to decode text response from %s", url)
                return ""

        normalized = _normalize_full_text(text)
        if len(normalized) > 500_000:
            normalized = normalized[:500_000]
        return normalized

    def _resolve_full_text(
        self, identifier: str, doi: str | None, candidates: Iterable[str]
    ) -> str:
        cached = self._read_cached_full_text(identifier, doi)
        if cached:
            return cached
        for url in candidates:
            if not url:
                continue
            text = self._download_text(url)
            if text:
                self._write_full_text_cache(identifier, doi, text)
                return text
        return ""


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
            candidates: List[str] = []
            for location in (
                item.get("best_oa_location"),
                item.get("primary_location"),
            ):
                if not location:
                    continue
                for key in ("url_for_text", "url", "landing_page_url"):
                    value = location.get(key)
                    if value and value not in candidates:
                        candidates.append(value)
            for location in item.get("locations", []) or []:
                for key in ("url_for_text", "url", "landing_page_url"):
                    value = location.get(key)
                    if value and value not in candidates:
                        candidates.append(value)
            full_text = self._resolve_full_text(identifier, doi, candidates)
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=item.get("title", ""),
                    authors=tuple(a for a in authors if a),
                    abstract=abstract,
                    source="openalex",
                    year=year,
                    doi=doi,
                    full_text=full_text,
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
            candidates = []
            for link in item.get("link", []) or []:
                url = link.get("URL")
                if url and url not in candidates:
                    candidates.append(url)
            if item.get("URL") and item.get("URL") not in candidates:
                candidates.append(item["URL"])
            full_text = self._resolve_full_text(identifier, doi, candidates)
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=item.get("title", [""])[0],
                    authors=tuple(authors),
                    abstract=abstract,
                    source="crossref",
                    year=year,
                    doi=doi,
                    full_text=full_text,
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
            pmc_id = None
            for article_id in record.get("articleids", []) or []:
                id_type = article_id.get("idtype")
                value = article_id.get("value")
                if id_type == "doi" and value and not doi:
                    doi = value
                if id_type == "pmc" and value and not pmc_id:
                    pmc_id = value
            identifier = f"pubmed:{pmid}"
            candidates = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"]
            if pmc_id:
                normalized_pmc = pmc_id if pmc_id.lower().startswith("pmc") else f"PMC{pmc_id}"
                candidates.append(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{normalized_pmc}/")
            full_text = self._resolve_full_text(identifier, doi, candidates)
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=record.get("title", ""),
                    authors=tuple(a for a in authors if a),
                    abstract=record.get("elocationid", ""),
                    source="pubmed",
                    year=year,
                    doi=doi,
                    full_text=full_text,
                )
            )

        next_start = start + len(id_list)
        exhausted = next_start >= total_count
        next_cursor = None if exhausted else str(next_start)
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class SerpAPIScholarProvider(BaseProvider):
    """Provider that queries Google Scholar via the SerpAPI service."""

    DEFAULT_URL = "https://serpapi.com/search.json"

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:  # pragma: no cover - network IO
        if not self.config.api_key:
            raise RuntimeError("SerpAPI provider requires an API key")
        start = int(cursor or self.config.extra.get("start", 0))
        params = {
            "engine": "google_scholar",
            "q": query,
            "start": start,
            "api_key": self.config.api_key,
        }
        params.update({k: v for k, v in (self.config.extra or {}).items() if k not in {"start"}})
        url = self.config.base_url or self.DEFAULT_URL
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("organic_results", []) or []
        papers: List[PaperMetadata] = []
        for item in results:
            title = item.get("title", "")
            if not title:
                continue
            identifier = item.get("link") or f"serpapi:{hashlib.sha1(title.encode('utf-8')).hexdigest()}"
            snippet = item.get("snippet", "")
            publication = item.get("publication_info", {}) or {}
            year = None
            if publication.get("year"):
                try:
                    year = int(publication["year"])
                except (TypeError, ValueError):
                    year = None
            authors = []
            if publication.get("authors"):
                for author in publication["authors"]:
                    if isinstance(author, Mapping) and author.get("name"):
                        authors.append(author["name"])
                    elif isinstance(author, str):
                        authors.append(author)
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=title,
                    authors=tuple(authors),
                    abstract=snippet,
                    source="serpapi_scholar",
                    year=year,
                    doi=None,
                )
            )
        next_start = start + len(results)
        pagination = payload.get("serpapi_pagination", {}) or {}
        has_next = bool(pagination.get("next"))
        next_cursor = str(next_start) if has_next else None
        exhausted = not has_next or not results
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class SemanticScholarProvider(BaseProvider):
    """Provider using the Semantic Scholar search API."""

    DEFAULT_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:  # pragma: no cover - network IO
        offset = int(cursor or 0)
        params = {
            "query": query,
            "limit": self.config.batch_size,
            "offset": offset,
            "fields": "title,abstract,year,authors,url,externalIds",
        }
        headers: Dict[str, str] = {}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        url = self.config.base_url or self.DEFAULT_URL
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", []) or []
        papers: List[PaperMetadata] = []
        for item in data:
            identifier = item.get("paperId") or item.get("url")
            if not identifier:
                continue
            abstract = item.get("abstract", "")
            year = item.get("year")
            external_ids = item.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            authors = []
            for author in item.get("authors", []) or []:
                name = author.get("name") if isinstance(author, Mapping) else None
                if name:
                    authors.append(name)
            candidates = []
            url_field = item.get("url")
            if url_field:
                candidates.append(url_field)
            full_text = self._resolve_full_text(identifier, doi, candidates)
            papers.append(
                PaperMetadata(
                    identifier=identifier,
                    title=item.get("title", ""),
                    authors=tuple(authors),
                    abstract=abstract,
                    source="semantic_scholar",
                    year=year,
                    doi=doi,
                    full_text=full_text,
                )
            )
        total = int(payload.get("total", 0))
        next_offset = offset + len(data)
        exhausted = next_offset >= total or not data
        next_cursor = None if exhausted else str(next_offset)
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


PROVIDER_REGISTRY: Dict[str, type[BaseProvider]] = {
    "openalex": OpenAlexProvider,
    "crossref": CrossRefProvider,
    "pubmed": PubMedProvider,
    "serpapi_scholar": SerpAPIScholarProvider,
    "semantic_scholar": SemanticScholarProvider,
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
        *,
        parallel_fetch: int | None = None,
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
        workers = 1
        if parallel_fetch is not None:
            try:
                workers = int(parallel_fetch)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                workers = 1
        self.parallel_fetch = max(1, workers)

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
                        title=str(item.get("title", "")),
                        authors=item.get("authors", []),
                        abstract=str(item.get("abstract", "") or ""),
                        source=str(item.get("source", "seed") or "seed"),
                        year=item.get("year"),
                        doi=item.get("doi"),
                        full_text=str(item.get("full_text") or ""),
                        sections=tuple(
                            PaperSection.from_dict(section)
                            for section in item.get("sections", []) or []
                            if isinstance(section, MutableMapping)
                        ),
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

        stop_event = Event()
        state_lock = Lock()

        def _target_met() -> bool:
            return target is not None and len(seen_identifiers) >= target

        def _process_shard(
            provider: BaseProvider,
            *,
            final_query: str,
            shard_state: Dict[str, Any],
        ) -> None:
            nonlocal newly_added
            cursor = shard_state.get("cursor")
            while not stop_event.is_set():
                if _target_met():
                    stop_event.set()
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
                    with state_lock:
                        shard_state["exhausted"] = True
                    break

                if not page.papers:
                    with state_lock:
                        shard_state["exhausted"] = True
                        shard_state["cursor"] = page.next_cursor
                    break

                with state_lock:
                    for paper in page.papers:
                        key = paper.dedupe_key
                        if key not in seen_identifiers:
                            seen_identifiers.add(key)
                            collected.append(paper)
                            newly_added += 1
                            provider_totals[provider.name] = provider_totals.get(provider.name, 0) + 1
                        else:
                            provider_totals.setdefault(provider.name, 0)
                    next_cursor = page.next_cursor
                    shard_state["cursor"] = next_cursor
                    reached = target is not None and len(seen_identifiers) >= target
                    exhausted = page.exhausted or next_cursor is None
                    if exhausted:
                        shard_state["exhausted"] = True
                if page.exhausted or next_cursor is None:
                    break
                if reached:
                    stop_event.set()
                    break
                cursor = next_cursor

        max_workers = max(1, int(self.parallel_fetch))

        for query in queries:
            if _target_met():
                break
            query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()
            existing_query_state = query_state.get(query_hash, {})
            per_query_state: Dict[str, Any] = (
                dict(existing_query_state) if isinstance(existing_query_state, MutableMapping) else {}
            )

            if max_workers > 1:
                executor_cm = ThreadPoolExecutor(max_workers=max_workers)
            else:
                executor_cm = None

            futures = []
            try:
                for provider in selected_providers:
                    if _target_met():
                        break
                    provider_state_raw = per_query_state.get(provider.name, {})
                    provider_state: Dict[str, Any] = (
                        dict(provider_state_raw)
                        if isinstance(provider_state_raw, MutableMapping)
                        else {}
                    )
                    raw_shards = provider_state.get("shards", {})
                    shard_states: Dict[str, Dict[str, Any]] = {}
                    if isinstance(raw_shards, MutableMapping):
                        for shard_key, shard_value in raw_shards.items():
                            if isinstance(shard_value, MutableMapping):
                                shard_states[str(shard_key)] = dict(shard_value)
                            else:
                                shard_states[str(shard_key)] = {}
                    provider_state["shards"] = shard_states

                    for index, shard_template in enumerate(provider.query_shards):
                        if _target_met():
                            break
                        shard_key = str(index)
                        shard_state = shard_states.setdefault(shard_key, {})
                        if shard_state.get("exhausted"):
                            continue
                        final_query = shard_template.format(query=query)
                        if max_workers > 1:
                            futures.append(
                                executor_cm.submit(
                                    _process_shard,
                                    provider,
                                    final_query=final_query,
                                    shard_state=shard_state,
                                )
                            )
                        else:
                            _process_shard(
                                provider,
                                final_query=final_query,
                                shard_state=shard_state,
                            )
                            if _target_met():
                                stop_event.set()
                                break
                    per_query_state[provider.name] = provider_state
            finally:
                if futures:
                    for future in futures:
                        future.result()
                if executor_cm is not None:
                    executor_cm.shutdown(wait=True)

            query_state[query_hash] = per_query_state
            if _target_met():
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


def _matches_query(paper: PaperMetadata, query: str) -> bool:
    terms = [term.strip().lower() for term in query.split() if term.strip()]
    if not terms:
        return True
    haystack = " ".join(
        part
        for part in (paper.title or "", paper.analysis_text or "", paper.abstract or "")
        if part
    ).lower()
    return all(term in haystack for term in terms)


def _normalize_full_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</p>", "\n", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"[\t\f\r]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def decode_openalex_abstract(index: Dict[str, List[int]] | None) -> str:
    """Convert the OpenAlex inverted abstract index to plain text."""

    if not index:
        return ""
    words = sorted(((position, word) for word, positions in index.items() for position in positions))
    ordered_words = [word for _position, word in sorted(words, key=lambda item: item[0])]
    return " ".join(ordered_words)


__all__ = [
    "PaperMetadata",
    "PaperSection",
    "ProviderConfig",
    "RetrievalResult",
    "LiteratureRetriever",
]

