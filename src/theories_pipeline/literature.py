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
import unicodedata
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Sequence, Tuple
from urllib.parse import parse_qs, urlparse

try:  # pragma: no cover - optional dependency
    import requests
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from scihub import SciHub as _SciHubClient  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _SciHubClient = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pdfminer.high_level import extract_text as _pdf_extract_text
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _pdf_extract_text = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_CANONICAL_TOKEN_SPLIT = re.compile(r"[^\w]+", flags=re.UNICODE)


def _canonical_tokens(value: str) -> List[str]:
    if not value:
        return []
    normalized = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = stripped.casefold()
    return [token for token in _CANONICAL_TOKEN_SPLIT.split(lowered) if token]


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
    citation_count: int | None = None
    is_review: bool | None = None
    influential_citations: Tuple[str, ...] = ()

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
            "citation_count": self.citation_count,
            "is_review": self.is_review,
            "influential_citations": list(self.influential_citations),
        }

    @staticmethod
    def from_dict(data: MutableMapping[str, Any]) -> "PaperMetadata":
        sections_data = data.get("sections") or []
        sections: Tuple[PaperSection, ...] = tuple(
            PaperSection.from_dict(item) for item in sections_data if isinstance(item, MutableMapping)
        )
        def _parse_citation(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        influential = data.get("influential_citations") or []
        if isinstance(influential, (str, bytes)):
            influential_list = [str(influential)]
        else:
            influential_list = [str(item) for item in influential if item is not None]

        raw_is_review = data.get("is_review")
        if isinstance(raw_is_review, bool):
            parsed_is_review: bool | None = raw_is_review
        elif raw_is_review is None:
            parsed_is_review = None
        else:
            parsed_is_review = str(raw_is_review).strip().lower() in {"true", "1", "yes"}

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
            citation_count=_parse_citation(data.get("citation_count")),
            is_review=parsed_is_review,
            influential_citations=tuple(influential_list),
        )

    @property
    def dedupe_key(self) -> str:
        if self.doi:
            return self.doi.lower()
        return self.identifier.lower()

    def canonical_key(self) -> str | None:
        if self.doi:
            return self.doi.lower()
        title_tokens = _canonical_tokens(self.title)
        if not title_tokens:
            return None
        author_tokens: List[str] = []
        for author in self.authors:
            author_tokens.extend(_canonical_tokens(author))
        canonical_parts = ["title:" + "-".join(title_tokens)]
        if author_tokens:
            canonical_parts.append("authors:" + "-".join(sorted(author_tokens)))
        return "|".join(canonical_parts)

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

    def _download_text(self, url: str) -> tuple[str, bytes | None]:
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
            return "", None

        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                data = response.content
            except OSError as exc:  # pragma: no cover - requests edge case
                logger.debug("Failed to read PDF content from %s: %s", url, exc)
                return "", None
            return "", data

        if "json" in content_type:
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if payload is None:
                return "", None
            text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            try:
                text = response.text
            except UnicodeDecodeError:
                logger.debug("Failed to decode text response from %s", url)
                return "", None

        normalized = _normalize_full_text(text)
        if len(normalized) > 500_000:
            normalized = normalized[:500_000]
        return normalized, None

    def _extract_pdf_text(self, data: bytes, source: str) -> str:
        if not data:
            return ""
        if _pdf_extract_text is None:
            logger.debug(
                "Skipping PDF text extraction for %s because pdfminer.six is not installed",
                source,
            )
            return ""
        try:
            with io.BytesIO(data) as buffer:
                text = _pdf_extract_text(buffer)
        except Exception as exc:  # pragma: no cover - pdf parsing edge case
            logger.debug("Failed to extract text from PDF %s: %s", source, exc)
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
            text, pdf_payload = self._download_text(url)
            if text:
                self._write_full_text_cache(identifier, doi, text)
                return text
            if pdf_payload:
                pdf_text = self._extract_pdf_text(pdf_payload, url)
                if pdf_text:
                    self._write_full_text_cache(identifier, doi, pdf_text)
                    return pdf_text
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
            citation_count = item.get("cited_by_count")
            try:
                parsed_citations = int(citation_count) if citation_count is not None else None
            except (TypeError, ValueError):
                parsed_citations = None
            type_fields = [str(item.get("type") or ""), str(item.get("type_crossref") or "")]
            is_review = any("review" in value.lower() for value in type_fields if value)
            related = item.get("related_works") or []
            if isinstance(related, (str, bytes)):
                influential = (str(related),)
            else:
                influential = tuple(str(entry) for entry in related if entry)
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
                    citation_count=parsed_citations,
                    is_review=is_review,
                    influential_citations=influential,
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
            citation_count = item.get("is-referenced-by-count")
            try:
                parsed_citations = int(citation_count) if citation_count is not None else None
            except (TypeError, ValueError):
                parsed_citations = None
            type_value = str(item.get("type") or "")
            is_review = "review" in type_value.lower()
            references = item.get("reference") or []
            influential: Tuple[str, ...]
            if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
                influential_list = []
                for ref in references:
                    if not isinstance(ref, Mapping):
                        continue
                    candidate = ref.get("DOI") or ref.get("doi") or ref.get("key")
                    if candidate:
                        influential_list.append(str(candidate))
                influential = tuple(influential_list)
            else:
                influential = ()
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
                    citation_count=parsed_citations,
                    is_review=is_review,
                    influential_citations=influential,
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
            citation_count_raw = record.get("pmcrefcount")
            try:
                citation_count = int(citation_count_raw) if citation_count_raw not in {"", None} else None
            except (TypeError, ValueError):
                citation_count = None
            pub_types = record.get("pubtype") or []
            if isinstance(pub_types, (str, bytes)):
                pub_type_values = [pub_types]
            else:
                pub_type_values = [str(value) for value in pub_types]
            is_review = any("review" in value.lower() for value in pub_type_values)
            references = record.get("references") or []
            if isinstance(references, Sequence) and not isinstance(references, (str, bytes)):
                influential_list = []
                for ref in references:
                    if not isinstance(ref, Mapping):
                        continue
                    candidate = (
                        ref.get("sourceid")
                        or ref.get("uid")
                        or ref.get("pmid")
                        or ref.get("doi")
                    )
                    if candidate:
                        influential_list.append(str(candidate))
                influential = tuple(influential_list)
            else:
                influential = ()
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
                    citation_count=citation_count,
                    is_review=is_review,
                    influential_citations=influential,
                )
            )

        next_start = start + len(id_list)
        exhausted = next_start >= total_count
        next_cursor = None if exhausted else str(next_start)
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class SerpApiScholarProvider(BaseProvider):
    """Provider backed by SerpApi's Google Scholar engine."""

    DEFAULT_BASE_URL = "https://serpapi.com/search.json"
    SOURCE_NAME = "serpapi_scholar"
    DOI_PATTERN = re.compile(r"10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        self.engine = str(extra.get("engine", "google_scholar"))
        self.query_param = str(extra.get("query_param", "q"))
        self.start_param = str(extra.get("start_param", "start"))
        self.limit_param = str(extra.get("limit_param", "num"))
        self.result_key = str(extra.get("result_key", "organic_results"))
        max_page_size = extra.get("max_page_size", 10)
        try:
            self.page_size = max(1, min(int(max_page_size), int(config.batch_size)))
        except (TypeError, ValueError):
            self.page_size = min(10, config.batch_size)

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        if not self.config.api_key:
            raise RuntimeError("SerpApi requires an API key to be provided")

        url = self.config.base_url or self.DEFAULT_BASE_URL
        headers = {
            "Accept": "application/json",
            "User-Agent": "hackaging-theories-pipeline/1.0",
        }
        params: Dict[str, Any] = {
            "engine": self.engine,
            self.query_param: query,
            self.limit_param: self.page_size,
            "api_key": self.config.api_key,
        }
        try:
            start_value = int(cursor) if cursor is not None else 0
        except (TypeError, ValueError):
            start_value = 0
        params[self.start_param] = max(0, start_value)

        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()

        items = payload.get(self.result_key) or []
        papers: List[PaperMetadata] = []
        for item in items:
            metadata = self._record_to_metadata(item)
            if metadata:
                papers.append(metadata)

        next_cursor = self._parse_next_cursor(payload.get("serpapi_pagination"))
        exhausted = not next_cursor
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)

    def _record_to_metadata(self, record: Any) -> PaperMetadata | None:
        if not isinstance(record, Mapping):
            return None
        title = str(record.get("title") or "").strip()
        if not title:
            return None

        abstract = str(
            record.get("snippet")
            or record.get("publication_info", {}).get("summary")
            or ""
        ).strip()
        pub_info = record.get("publication_info") or {}
        authors = self._parse_authors(pub_info.get("authors"))
        year = self._parse_year(
            pub_info.get("year")
            or record.get("publication_date")
            or record.get("year")
            or pub_info.get("summary")
        )
        raw_identifier = record.get("result_id") or record.get("link") or record.get("id")
        if raw_identifier:
            identifier = f"serpapi:{str(raw_identifier)}"
        else:
            digest_source = f"{title}|{record.get('link', '')}|{year or ''}"
            identifier = f"serpapi:{hashlib.sha1(digest_source.encode('utf-8')).hexdigest()}"
        candidates = self._fulltext_candidates(record)
        doi = self._find_doi(candidates)
        citation_count = self._parse_int(
            record.get("inline_links", {}).get("cited_by", {}).get("total")
        )
        full_text = self._resolve_full_text(identifier, doi, candidates)
        return PaperMetadata(
            identifier=identifier,
            title=title,
            authors=authors,
            abstract=abstract,
            source=self.SOURCE_NAME,
            year=year,
            doi=doi,
            full_text=full_text,
            citation_count=citation_count,
            is_review=None,
            influential_citations=(),
        )

    def _parse_next_cursor(self, pagination: Any) -> str | None:
        if not isinstance(pagination, Mapping):
            return None
        next_url = pagination.get("next")
        if not next_url:
            return None
        try:
            parsed = urlparse(str(next_url))
        except Exception:  # pragma: no cover - defensive parsing
            return None
        params = parse_qs(parsed.query)
        start_values = params.get(self.start_param)
        if not start_values:
            return None
        return start_values[0]

    def _parse_authors(self, value: Any) -> Tuple[str, ...]:
        if not value:
            return ()
        authors: List[str] = []
        if isinstance(value, Mapping):
            for entry in value.values():
                authors.extend(self._parse_authors(entry))
            return tuple(authors)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for entry in value:
                if isinstance(entry, Mapping):
                    name = entry.get("name") or entry.get("author")
                else:
                    name = entry
                if not name:
                    continue
                name_str = str(name).strip()
                if name_str:
                    authors.append(name_str)
            return tuple(authors)
        parts = str(value).split(",")
        return tuple(part.strip() for part in parts if part.strip())

    def _parse_year(self, value: Any) -> int | None:
        if isinstance(value, int):
            return value
        text = str(value or "").strip()
        if not text:
            return None
        match = re.search(r"(19|20)[0-9]{2}", text)
        if not match:
            return None
        try:
            return int(match.group(0))
        except (TypeError, ValueError):
            return None

    def _parse_int(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _fulltext_candidates(self, record: Mapping[str, Any]) -> List[str]:
        candidates: List[str] = []
        link = record.get("link")
        if link:
            candidates.append(str(link))
        inline_links = record.get("inline_links") or {}
        if isinstance(inline_links, Mapping):
            for key in ("resources", "serpapi_scholar_fulltext", "versions", "serpapi_links"):
                value = inline_links.get(key)
                candidates.extend(self._extract_urls(value))
        for key in ("resources", "versions"):
            value = record.get(key)
            candidates.extend(self._extract_urls(value))
        seen: set[str] = set()
        unique: List[str] = []
        for url in candidates:
            normalized = str(url).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique

    def _extract_urls(self, value: Any) -> List[str]:
        urls: List[str] = []
        if not value:
            return urls
        if isinstance(value, (list, tuple, set)):
            for item in value:
                urls.extend(self._extract_urls(item))
            return urls
        if isinstance(value, Mapping):
            for key in ("link", "url", "href"):
                if key in value:
                    urls.extend(self._extract_urls(value[key]))
            return urls
        text = str(value).strip()
        if text:
            urls.append(text)
        return urls

    def _find_doi(self, candidates: Sequence[str]) -> str | None:
        for url in candidates:
            if not url:
                continue
            match = self.DOI_PATTERN.search(url)
            if match:
                return match.group(0).lower()
        return None


class SemanticScholarProvider(BaseProvider):
    """Provider for the Semantic Scholar Graph API."""

    DEFAULT_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    SOURCE_NAME = "semantic_scholar"
    DEFAULT_FIELDS = (
        "title,abstract,authors,year,externalIds,openAccessPdf,url,citationCount,"
        "influentialCitationIds"
    )

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        fields = extra.get("fields")
        if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
            field_values = [str(field).strip() for field in fields if str(field).strip()]
            self.fields = ",".join(field_values) if field_values else self.DEFAULT_FIELDS
        elif isinstance(fields, str) and fields.strip():
            self.fields = fields.strip()
        else:
            self.fields = self.DEFAULT_FIELDS
        self.query_param = str(extra.get("query_param", "query"))
        self.offset_param = str(extra.get("offset_param", "offset"))
        self.limit_param = str(extra.get("limit_param", "limit"))
        self.data_key = str(extra.get("data_key", "data"))
        self.next_key = str(extra.get("next_key", "next"))

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        url = self.config.base_url or self.DEFAULT_BASE_URL
        headers = {
            "Accept": "application/json",
            "User-Agent": "hackaging-theories-pipeline/1.0",
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        params: Dict[str, Any] = {
            self.query_param: query,
            self.limit_param: self.config.batch_size,
            "fields": self.fields,
        }
        if cursor is not None:
            params[self.offset_param] = cursor

        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get(self.data_key) or []
        papers: List[PaperMetadata] = []
        for record in records:
            metadata = self._record_to_metadata(record)
            if metadata:
                papers.append(metadata)

        next_cursor = payload.get(self.next_key)
        next_value = str(next_cursor) if next_cursor not in {None, ""} else None
        exhausted = not next_value
        return ProviderPage(papers=papers, next_cursor=next_value, exhausted=exhausted)

    def _record_to_metadata(self, record: Any) -> PaperMetadata | None:
        if not isinstance(record, Mapping):
            return None
        title = str(record.get("title") or "").strip()
        if not title:
            return None
        abstract = str(record.get("abstract") or record.get("tldr", {}).get("text") or "").strip()
        paper_id = record.get("paperId")
        if paper_id:
            identifier = f"semanticscholar:{paper_id}"
        else:
            external_ids = record.get("externalIds") if isinstance(record.get("externalIds"), Mapping) else {}
            fallback = (
                (external_ids or {}).get("CorpusId")
                or (external_ids or {}).get("DOI")
                or record.get("url")
            )
            if fallback:
                identifier = f"semanticscholar:{fallback}"
            else:
                digest_source = f"{title}|{record.get('year', '')}|{abstract[:80]}"
                identifier = f"semanticscholar:{hashlib.sha1(digest_source.encode('utf-8')).hexdigest()}"
        authors = self._parse_authors(record.get("authors"))
        year = self._parse_year(record.get("year"))
        external_ids = record.get("externalIds") if isinstance(record.get("externalIds"), Mapping) else {}
        doi = external_ids.get("DOI") if isinstance(external_ids, Mapping) else None
        if isinstance(doi, str):
            doi = doi.strip() or None
        candidates = self._fulltext_candidates(record, doi)
        citation_count = self._parse_int(record.get("citationCount"))
        influential_ids = record.get("influentialCitationIds")
        if isinstance(influential_ids, Sequence) and not isinstance(influential_ids, (str, bytes)):
            influential = tuple(str(value) for value in influential_ids if value)
        else:
            influential = ()
        full_text = self._resolve_full_text(identifier, doi, candidates)
        return PaperMetadata(
            identifier=identifier,
            title=title,
            authors=authors,
            abstract=abstract,
            source=self.SOURCE_NAME,
            year=year,
            doi=doi,
            full_text=full_text,
            citation_count=citation_count,
            is_review=None,
            influential_citations=influential,
        )

    def _parse_authors(self, value: Any) -> Tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            authors: List[str] = []
            for entry in value:
                if isinstance(entry, Mapping):
                    name = entry.get("name") or entry.get("authorId")
                else:
                    name = entry
                if not name:
                    continue
                name_str = str(name).strip()
                if name_str:
                    authors.append(name_str)
            return tuple(authors)
        parts = str(value).split(",")
        return tuple(part.strip() for part in parts if part.strip())

    def _parse_year(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_int(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _fulltext_candidates(self, record: Mapping[str, Any], doi: str | None) -> List[str]:
        candidates: List[str] = []
        open_access = record.get("openAccessPdf")
        if isinstance(open_access, Mapping):
            pdf_url = open_access.get("url") or open_access.get("pdfUrl")
            if pdf_url:
                candidates.append(str(pdf_url))
        url_value = record.get("url")
        if url_value:
            candidates.append(str(url_value))
        if doi:
            candidates.append(f"https://doi.org/{doi}")
        seen: set[str] = set()
        unique: List[str] = []
        for candidate in candidates:
            normalized = str(candidate).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(normalized)
        return unique


class SciHubProvider(BaseProvider):
    """Provider for retrieving full-text access metadata from Sci-Hub services."""

    DEFAULT_BASE_URL = "https://sci-hub2.p.rapidapi.com/search"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        self.mode = str(extra.get("mode") or "rapidapi").lower()
        self.rapidapi_host = extra.get("rapidapi_host")
        self.query_param = str(extra.get("query_param", "query"))
        self.cursor_param = str(extra.get("cursor_param", "cursor"))
        self.limit_param = str(extra.get("limit_param", "limit"))
        self.result_key = str(extra.get("result_key", "result"))
        self.cursor_key = str(extra.get("cursor_key", "next"))
        self.id_prefix = str(extra.get("id_prefix", "scihub"))
        params = extra.get("params")
        self.additional_params = dict(params) if isinstance(params, Mapping) else {}
        fulltext_keys = extra.get("fulltext_keys") or (
            "fullTextUrl",
            "fulltext",
            "pdfUrl",
            "pdf",
            "url",
        )
        if isinstance(fulltext_keys, (str, bytes)):
            fulltext_keys = [fulltext_keys]
        self.fulltext_keys = tuple(str(key) for key in fulltext_keys if str(key).strip())
        self.library_domain = extra.get("domain")
        self.library_email = extra.get("email") or config.api_key

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        if self.mode == "rapidapi":
            return self._fetch_via_rapidapi(query, cursor)
        return self._fetch_via_library(query, cursor)

    def _fetch_via_rapidapi(self, query: str, cursor: str | None) -> ProviderPage:
        url = self.config.base_url or self.DEFAULT_BASE_URL
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "hackaging-theories-pipeline/1.0",
        }
        if self.config.api_key:
            headers["X-RapidAPI-Key"] = self.config.api_key
        if self.rapidapi_host:
            headers["X-RapidAPI-Host"] = str(self.rapidapi_host)
        params: Dict[str, Any] = {self.query_param: query, self.limit_param: self.config.batch_size}
        params.update(self.additional_params)
        if cursor:
            params[self.cursor_param] = cursor
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get(self.result_key) or []
        papers: List[PaperMetadata] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            metadata = self._record_from_mapping(record)
            if metadata:
                papers.append(metadata)
        next_cursor = payload.get(self.cursor_key)
        if isinstance(next_cursor, Mapping):  # pragma: no cover - defensive
            next_cursor = next_cursor.get("cursor")
        next_value = str(next_cursor) if next_cursor else None
        exhausted = not next_value
        return ProviderPage(papers=papers, next_cursor=next_value, exhausted=exhausted)

    def _fetch_via_library(self, query: str, cursor: str | None) -> ProviderPage:  # pragma: no cover - optional dependency
        if _SciHubClient is None:
            raise RuntimeError("The 'scihub' package is required for library mode")
        if cursor:
            raise ValueError("SciHub library mode does not support cursor-based pagination")
        client = _SciHubClient()
        if self.library_email and hasattr(client, "email"):
            try:
                setattr(client, "email", str(self.library_email))
            except Exception:
                logger.debug("Unable to set Sci-Hub client email override")
        if self.library_domain and hasattr(client, "base_url"):
            try:
                setattr(client, "base_url", str(self.library_domain))
            except Exception:
                logger.debug("Unable to set Sci-Hub client domain override")
        self.rate_limiter.wait()
        try:
            results = client.search(query)
        except Exception as exc:
            raise RuntimeError(f"Sci-Hub library search failed: {exc}") from exc
        papers: List[PaperMetadata] = []
        for record in results or []:
            if not isinstance(record, Mapping):
                continue
            metadata = self._record_from_mapping(record)
            if metadata:
                papers.append(metadata)
        return ProviderPage(papers=papers, next_cursor=None, exhausted=True)

    def _record_from_mapping(self, record: Mapping[str, Any]) -> PaperMetadata | None:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or record.get("description") or "")
        doi = str(record.get("doi") or "").strip() or None
        identifier = str(
            record.get("identifier")
            or record.get("id")
            or record.get("url")
            or (f"{self.id_prefix}:{doi}" if doi else "")
        ).strip()
        if not identifier:
            digest_source = f"{title}|{doi}|{record.get('year', '')}"
            identifier = f"{self.id_prefix}:{hashlib.sha1(digest_source.encode('utf-8')).hexdigest()}"
        authors_raw = record.get("authors") or record.get("author")
        authors = self._parse_authors(authors_raw)
        year = self._parse_year(record.get("year") or record.get("publication_year"))
        candidates: List[str] = []
        for key in self.fulltext_keys:
            value = record.get(key)
            candidates.extend(self._extract_urls(value))
        if doi:
            doi_url = f"https://doi.org/{doi}"
            if doi_url not in candidates:
                candidates.append(doi_url)
        full_text = self._resolve_full_text(identifier, doi, candidates)
        return PaperMetadata(
            identifier=identifier,
            title=title,
            authors=authors,
            abstract=abstract,
            source="scihub",
            year=year,
            doi=doi,
            full_text=full_text,
            citation_count=None,
            is_review=None,
            influential_citations=(),
        )

    def _parse_authors(self, value: Any) -> Tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        text = str(value)
        if ";" in text:
            parts = text.split(";")
        elif "|" in text:
            parts = text.split("|")
        else:
            parts = text.split(",")
        return tuple(part.strip() for part in parts if part.strip())

    def _parse_year(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(str(value).strip()[:4])
        except (TypeError, ValueError):
            return None

    def _extract_urls(self, value: Any) -> List[str]:
        urls: List[str] = []
        if not value:
            return urls
        if isinstance(value, (list, tuple, set)):
            for item in value:
                urls.extend(self._extract_urls(item))
            return urls
        if isinstance(value, Mapping):
            for key in ("url", "href", "link"):
                if key in value:
                    urls.extend(self._extract_urls(value[key]))
            return urls
        url = str(value).strip()
        if url:
            urls.append(url)
        return urls


class AnnasArchiveProvider(BaseProvider):
    """Provider that queries Anna's Archive mirrors for open-access full texts."""

    DEFAULT_BASE_URL = "https://annas-archive-api.p.rapidapi.com/search"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        self.rapidapi_host = extra.get("rapidapi_host")
        self.query_param = str(extra.get("query_param", "q"))
        self.cursor_param = str(extra.get("cursor_param", "cursor"))
        self.limit_param = str(extra.get("limit_param", "limit"))
        self.result_key = str(extra.get("result_key", "results"))
        self.cursor_key = str(extra.get("cursor_key", "cursor"))
        params = extra.get("params")
        self.additional_params = dict(params) if isinstance(params, Mapping) else {}
        link_keys = extra.get("link_keys") or ("mirrors", "files", "urls")
        if isinstance(link_keys, (str, bytes)):
            link_keys = [link_keys]
        self.link_keys = tuple(str(key) for key in link_keys if str(key).strip())
        self.id_prefix = str(extra.get("id_prefix", "annas"))

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        url = self.config.base_url or self.DEFAULT_BASE_URL
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "hackaging-theories-pipeline/1.0",
        }
        if self.config.api_key:
            headers["X-RapidAPI-Key"] = self.config.api_key
        if self.rapidapi_host:
            headers["X-RapidAPI-Host"] = str(self.rapidapi_host)
        params: Dict[str, Any] = {self.query_param: query, self.limit_param: self.config.batch_size}
        params.update(self.additional_params)
        if cursor:
            params[self.cursor_param] = cursor
        self.rate_limiter.wait()
        response = self.session.get(url, params=params, headers=headers, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get(self.result_key) or []
        papers: List[PaperMetadata] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            metadata = self._record_from_mapping(record)
            if metadata:
                papers.append(metadata)
        next_cursor = payload.get(self.cursor_key)
        next_value = str(next_cursor) if next_cursor else None
        exhausted = not next_value
        return ProviderPage(papers=papers, next_cursor=next_value, exhausted=exhausted)

    def _record_from_mapping(self, record: Mapping[str, Any]) -> PaperMetadata | None:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("description") or record.get("abstract") or "")
        identifiers = record.get("identifiers") if isinstance(record.get("identifiers"), Mapping) else {}
        doi = str(
            record.get("doi")
            or (identifiers or {}).get("doi")
            or (identifiers or {}).get("doi_plain")
            or ""
        ).strip() or None
        identifier = str(
            record.get("md5")
            or record.get("id")
            or (identifiers or {}).get("md5")
            or (identifiers or {}).get("isbn")
            or ""
        ).strip()
        if not identifier:
            digest_source = f"{title}|{doi}|{record.get('year', '')}"
            identifier = f"{self.id_prefix}:{hashlib.sha1(digest_source.encode('utf-8')).hexdigest()}"
        authors = self._parse_authors(record.get("authors") or record.get("author"))
        year = self._parse_year(record.get("year"))
        candidates: List[str] = []
        for key in self.link_keys:
            value = record.get(key)
            candidates.extend(self._extract_urls(value))
        if doi:
            doi_url = f"https://doi.org/{doi}"
            if doi_url not in candidates:
                candidates.append(doi_url)
        full_text = self._resolve_full_text(identifier, doi, candidates)
        return PaperMetadata(
            identifier=identifier,
            title=title,
            authors=authors,
            abstract=abstract,
            source="annas_archive",
            year=year,
            doi=doi,
            full_text=full_text,
            citation_count=None,
            is_review=None,
            influential_citations=(),
        )

    def _parse_authors(self, value: Any) -> Tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        text = str(value)
        if ";" in text:
            parts = text.split(";")
        elif "|" in text:
            parts = text.split("|")
        else:
            parts = text.split(",")
        return tuple(part.strip() for part in parts if part.strip())

    def _parse_year(self, value: Any) -> int | None:
        if value in {None, ""}:
            return None
        try:
            return int(str(value).strip()[:4])
        except (TypeError, ValueError):
            return None

    def _extract_urls(self, value: Any) -> List[str]:
        urls: List[str] = []
        if not value:
            return urls
        if isinstance(value, (list, tuple, set)):
            for item in value:
                urls.extend(self._extract_urls(item))
            return urls
        if isinstance(value, Mapping):
            for key in ("url", "href", "mirror", "link"):
                if key in value:
                    urls.extend(self._extract_urls(value[key]))
            return urls
        url = str(value).strip()
        if url:
            urls.append(url)
        return urls


class _RxivProvider(BaseProvider):
    """Shared implementation for the bioRxiv and medRxiv JSON APIs."""

    DEFAULT_BASE_URL = "https://api.biorxiv.org/details"
    DEFAULT_SERVER = "biorxiv"
    SOURCE_NAME = "rxiv"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        base_url = config.base_url or f"{self.DEFAULT_BASE_URL}/{self.DEFAULT_SERVER}"
        self.base_url = base_url.rstrip("/")
        server = extra.get("server") or self.DEFAULT_SERVER
        self.server = str(server)
        categories = extra.get("categories") or []
        if isinstance(categories, str):
            categories = [categories]
        self._categories = {str(category).strip().lower() for category in categories if str(category).strip()}
        window = extra.get("date_window") if isinstance(extra.get("date_window"), MutableMapping) else {}
        date_from = (
            (window or {}).get("from")
            or (window or {}).get("start")
            or extra.get("date_from")
            or extra.get("from")
        )
        date_to = (
            (window or {}).get("to")
            or (window or {}).get("end")
            or extra.get("date_to")
            or extra.get("to")
        )
        try:
            window_days = int(extra.get("window_days", 30))
        except (TypeError, ValueError):
            window_days = 30
        today = date.today()
        default_to = today.isoformat()
        default_from = (today - timedelta(days=max(1, window_days))).isoformat()
        self.date_from = str(date_from or default_from)
        self.date_to = str(date_to or default_to)

    def _build_url(self, offset: int) -> str:
        return f"{self.base_url}/{self.date_from}/{self.date_to}/{offset}"

    def _record_matches(self, record: MutableMapping[str, Any], query: str) -> bool:
        terms = [term.strip().lower() for term in query.split() if term.strip()]
        if not terms:
            return True
        haystack_parts = [
            str(record.get("title", "")),
            str(record.get("abstract", "")),
            str(record.get("authors", "")),
            str(record.get("category", "")),
        ]
        haystack = " ".join(part for part in haystack_parts if part).lower()
        return all(term in haystack for term in terms)

    def _parse_authors(self, value: str | None) -> Tuple[str, ...]:
        if not value:
            return ()
        authors = [part.strip() for part in value.split(";") if part.strip()]
        return tuple(authors)

    def _to_metadata(self, record: MutableMapping[str, Any]) -> PaperMetadata:
        doi = str(record.get("doi") or "") or None
        identifier: str
        if doi:
            identifier = f"{self.SOURCE_NAME}:{doi}"
        else:
            key_source = f"{record.get('title', '')}|{record.get('date', '')}|{record.get('version', '')}"
            identifier = f"{self.SOURCE_NAME}:{hashlib.sha1(key_source.encode('utf-8')).hexdigest()}"
        authors = self._parse_authors(record.get("authors"))
        abstract = record.get("abstract") or ""
        if isinstance(abstract, str) and abstract.strip().lower() == "na":
            abstract = ""
        date_str = str(record.get("date") or "")
        year = None
        if date_str:
            try:
                year = int(date_str.split("-")[0])
            except (ValueError, IndexError):
                year = None
        candidates: List[str] = []
        for key in ("jatsxml", "rel_pdf", "pdf_url", "full_text"):
            value = record.get(key)
            if isinstance(value, str) and value and value not in candidates:
                candidates.append(value)
        if doi:
            doi_url = f"https://doi.org/{doi}"
            if doi_url not in candidates:
                candidates.append(doi_url)
        full_text = self._resolve_full_text(identifier, doi, candidates)
        return PaperMetadata(
            identifier=identifier,
            title=str(record.get("title", "")),
            authors=authors,
            abstract=str(abstract or ""),
            source=self.SOURCE_NAME,
            year=year,
            doi=doi,
            full_text=full_text,
            citation_count=None,
            is_review=None,
            influential_citations=(),
        )

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        offset = int(cursor or 0)
        url = self._build_url(offset)
        self.rate_limiter.wait()
        response = self.session.get(url, timeout=self.config.timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("collection", []) or []
        messages = payload.get("messages", []) or []
        message = messages[0] if messages else {}
        try:
            count = int(message.get("count", len(records)))
        except (TypeError, ValueError):
            count = len(records)
        try:
            total = int(message.get("total", offset + count))
        except (TypeError, ValueError):
            total = offset + count
        papers: List[PaperMetadata] = []
        for record in records:
            if not isinstance(record, MutableMapping):
                continue
            category = str(record.get("category", "")).strip().lower()
            if self._categories and category not in self._categories:
                continue
            if not self._record_matches(record, query):
                continue
            papers.append(self._to_metadata(record))
        next_offset = offset + count
        exhausted = next_offset >= total or not records
        next_cursor = None if exhausted else str(next_offset)
        return ProviderPage(papers=papers, next_cursor=next_cursor, exhausted=exhausted)


class BioRxivProvider(_RxivProvider):
    """Provider for the bioRxiv preprint server."""

    DEFAULT_SERVER = "biorxiv"
    SOURCE_NAME = "biorxiv"


class MedRxivProvider(_RxivProvider):
    """Provider for the medRxiv preprint server."""

    DEFAULT_SERVER = "medrxiv"
    SOURCE_NAME = "medrxiv"


PROVIDER_REGISTRY: Dict[str, type[BaseProvider]] = {
    "openalex": OpenAlexProvider,
    "crossref": CrossRefProvider,
    "pubmed": PubMedProvider,
    "serpapi_scholar": SerpApiScholarProvider,
    "semantic_scholar": SemanticScholarProvider,
    "scihub": SciHubProvider,
    "annas_archive": AnnasArchiveProvider,
    "biorxiv": BioRxivProvider,
    "medrxiv": MedRxivProvider,
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
                citation_raw = item.get("citation_count")
                try:
                    citation_count = int(citation_raw) if citation_raw is not None else None
                except (TypeError, ValueError):
                    citation_count = None
                raw_influential = item.get("influential_citations") or []
                if isinstance(raw_influential, (str, bytes)):
                    influential = (str(raw_influential),)
                else:
                    influential = tuple(str(entry) for entry in raw_influential if entry is not None)
                raw_is_review = item.get("is_review")
                if isinstance(raw_is_review, bool):
                    is_review = raw_is_review
                elif raw_is_review is None:
                    is_review = None
                else:
                    is_review = str(raw_is_review).strip().lower() in {"true", "1", "yes"}
                papers.append(
                    PaperMetadata(
                        identifier=item["identifier"],
                        title=str(item.get("title", "")),
                        authors=tuple(item.get("authors", ())),
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
                        citation_count=citation_count,
                        is_review=is_review,
                        influential_citations=influential,
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
        min_citation_count: int | None = None,
        prefer_reviews: bool = False,
        sort_by_citations: bool = False,
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
        seen_canonical_keys = set(state.get("seen_canonical_keys", []))
        stored_papers_raw = [PaperMetadata.from_dict(item) for item in state.get("papers", [])]

        def _passes_filters(paper: PaperMetadata) -> bool:
            if min_citation_count is not None:
                citations = paper.citation_count if paper.citation_count is not None else 0
                if citations < min_citation_count:
                    return False
            return True

        collected: List[PaperMetadata] = []
        accepted_identifiers: set[str] = set()
        for paper in stored_papers_raw:
            if _passes_filters(paper):
                collected.append(paper)
                key = paper.dedupe_key
                accepted_identifiers.add(key)
                seen_identifiers.add(key)
                canonical = paper.canonical_key()
                if canonical:
                    seen_canonical_keys.add(canonical)
        prior_total = len(accepted_identifiers)

        newly_added = 0
        provider_totals = state.get("provider_totals", {})
        provider_totals = {k: int(v) for k, v in provider_totals.items()}
        query_state = state.get("queries", {})

        # Always include seed papers once per request.
        if not state.get("seed_consumed"):
            for paper in self._load_seed_papers():
                key = paper.dedupe_key
                if key not in seen_identifiers:
                    seen_identifiers.add(key)
                    canonical = paper.canonical_key()
                    if canonical:
                        seen_canonical_keys.add(canonical)
                    if _passes_filters(paper):
                        collected.append(paper)
                        accepted_identifiers.add(key)
                        newly_added += 1
            state["seed_consumed"] = True

        stop_event = Event()
        state_lock = Lock()

        def _target_met() -> bool:
            return target is not None and len(accepted_identifiers) >= target

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
                        canonical = paper.canonical_key()
                        duplicate = key in seen_identifiers
                        if not duplicate and canonical:
                            duplicate = canonical in seen_canonical_keys
                        if duplicate:
                            seen_identifiers.add(key)
                            if canonical:
                                seen_canonical_keys.add(canonical)
                            provider_totals.setdefault(provider.name, 0)
                            continue

                        seen_identifiers.add(key)
                        if canonical:
                            seen_canonical_keys.add(canonical)
                        if _passes_filters(paper):
                            collected.append(paper)
                            accepted_identifiers.add(key)
                            newly_added += 1
                            provider_totals[provider.name] = provider_totals.get(provider.name, 0) + 1
                        else:
                            provider_totals.setdefault(provider.name, 0)
                    next_cursor = page.next_cursor
                    shard_state["cursor"] = next_cursor
                    reached = target is not None and len(accepted_identifiers) >= target
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

        sorted_info: Dict[str, bool] | None = None
        if prefer_reviews or sort_by_citations:

            def _sort_key(paper: PaperMetadata) -> Tuple[int, int, str, str]:
                review_score = 0
                if prefer_reviews:
                    if paper.is_review is True:
                        review_score = 0
                    elif paper.is_review is False:
                        review_score = 1
                    else:
                        review_score = 2
                citation_score = 0
                if sort_by_citations:
                    citation_value = paper.citation_count if paper.citation_count is not None else -1
                    citation_score = -citation_value
                return (
                    review_score if prefer_reviews else 0,
                    citation_score if sort_by_citations else 0,
                    paper.title.lower() if paper.title else "",
                    paper.identifier.lower(),
                )

            collected.sort(key=_sort_key)
            sorted_info = {
                "prefer_reviews": prefer_reviews,
                "sort_by_citations": sort_by_citations,
            }

        state.update(
            {
                "seen_identifiers": sorted(seen_identifiers),
                "seen_canonical_keys": sorted(seen_canonical_keys),
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

        if sorted_info is not None:
            summary["sorted"] = sorted_info

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

