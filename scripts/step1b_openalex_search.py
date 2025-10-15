"""Collect review articles on aging theories from OpenAlex.

The script queries the OpenAlex works API for review-style publications that
match a set of aging theory keywords. Results are normalised to the same schema
as the PubMed fetcher with additional OpenAlex identifiers so downstream
pipeline steps can treat both sources uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping


OPENALEX_WORKS_URL = "https://api.openalex.org/works"

DEFAULT_TERMS = [
    '"aging theory"',
    '"ageing theory"',
    '"theories of aging"',
]


def decode_inverted_index(index: Mapping[str, Iterable[int]] | None) -> str:
    """Reconstruct the abstract text from OpenAlex's inverted index format."""

    if not index:
        return ""
    positions: Dict[int, str] = {}
    max_idx = -1
    for word, occurrences in index.items():
        if not isinstance(word, str):
            continue
        for pos in occurrences:
            try:
                idx = int(pos)
            except (TypeError, ValueError):
                continue
            positions[idx] = word
            if idx > max_idx:
                max_idx = idx
    if max_idx < 0:
        return ""
    tokens: List[str] = []
    for idx in range(max_idx + 1):
        token = positions.get(idx)
        if token:
            tokens.append(token)
    return " ".join(tokens)


def normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = doi.strip()
    if not cleaned:
        return None
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "doi:",
    )
    lower = cleaned.lower()
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.strip()
    return cleaned or None


def extract_pmid(ids: Mapping[str, object] | None) -> str | None:
    if not ids:
        return None
    pmid_url = ids.get("pmid") if isinstance(ids, Mapping) else None
    if not isinstance(pmid_url, str):
        return None
    pmid_url = pmid_url.strip().rstrip("/")
    if not pmid_url:
        return None
    return pmid_url.split("/")[-1]


def extract_fulltext_links(record: Mapping[str, object]) -> Dict[str, str] | None:
    links: Dict[str, str] = {}
    primary = record.get("primary_location")
    if isinstance(primary, Mapping):
        pdf_url = primary.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            links.setdefault("pdf", pdf_url.strip())
        landing = primary.get("landing_page_url")
        if isinstance(landing, str) and landing.strip():
            links.setdefault("landing_page", landing.strip())
    open_access = record.get("open_access")
    if isinstance(open_access, Mapping):
        oa_url = open_access.get("oa_url")
        if isinstance(oa_url, str) and oa_url.strip():
            links.setdefault("oa_url", oa_url.strip())
    return links or None


@dataclass
class OpenAlexRecord:
    pmid: str | None
    title: str
    abstract: str
    publication_types: List[str]
    authors: List[str]
    journal: str | None
    publication_year: str | None
    doi: str | None
    openalex_id: str | None
    openalex_url: str | None
    open_access: Dict[str, object] | None
    fulltext_links: Dict[str, str] | None
    sources: List[str]

    @classmethod
    def from_api(cls, payload: Mapping[str, object]) -> "OpenAlexRecord":
        pmid = extract_pmid(payload.get("ids") if isinstance(payload, Mapping) else None)
        title = (payload.get("display_name") or "").strip()
        abstract = decode_inverted_index(payload.get("abstract_inverted_index"))
        pub_types: List[str] = []
        for key in ("type", "type_crossref"):
            value = payload.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned and cleaned not in pub_types:
                    pub_types.append(cleaned)
        authors: List[str] = []
        for authorship in payload.get("authorships") or []:
            if not isinstance(authorship, Mapping):
                continue
            author = authorship.get("author")
            if isinstance(author, Mapping):
                name = author.get("display_name")
                if isinstance(name, str):
                    name = name.strip()
                    if name:
                        authors.append(name)
        journal = None
        primary = payload.get("primary_location")
        if isinstance(primary, Mapping):
            source = primary.get("source")
            if isinstance(source, Mapping):
                display = source.get("display_name")
                if isinstance(display, str) and display.strip():
                    journal = display.strip()
        publication_year = payload.get("publication_year")
        if publication_year is not None:
            publication_year = str(publication_year).strip()
        doi = payload.get("doi")
        if isinstance(doi, str):
            doi = normalise_doi(doi)
        ids = payload.get("ids") if isinstance(payload, Mapping) else None
        if isinstance(ids, Mapping) and not doi:
            doi_id = ids.get("doi")
            if isinstance(doi_id, str):
                doi = normalise_doi(doi_id)
        openalex_id = payload.get("id")
        if isinstance(openalex_id, str):
            openalex_id = openalex_id.strip() or None
        openalex_url = None
        if openalex_id:
            openalex_url = openalex_id
        open_access = payload.get("open_access") if isinstance(payload, Mapping) else None
        if isinstance(open_access, Mapping):
            open_access = dict(open_access)
        else:
            open_access = None
        fulltext_links = extract_fulltext_links(payload)
        return cls(
            pmid=pmid,
            title=title,
            abstract=abstract,
            publication_types=pub_types,
            authors=authors,
            journal=journal,
            publication_year=publication_year if publication_year else None,
            doi=doi,
            openalex_id=openalex_id,
            openalex_url=openalex_url,
            open_access=open_access,
            fulltext_links=fulltext_links,
            sources=["openalex"],
        )


def ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def fetch_openalex(term: str, per_page: int, delay: float) -> List[Mapping[str, object]]:
    cursor = "*"
    results: List[Mapping[str, object]] = []
    while cursor:
        params = {
            "search": term,
            "filter": "type:journal-article,type_crossref:review",
            "per-page": per_page,
            "cursor": cursor,
        }
        url = f"{OPENALEX_WORKS_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request) as response:  # nosec - OpenAlex API
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network guard
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAlex request failed ({exc.code}): {body or exc.reason}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network guard
            raise RuntimeError(f"OpenAlex request failed: {exc.reason}") from exc
        items = payload.get("results") if isinstance(payload, Mapping) else None
        if isinstance(items, list):
            results.extend(item for item in items if isinstance(item, Mapping))
        meta = payload.get("meta") if isinstance(payload, Mapping) else None
        next_cursor = None
        if isinstance(meta, Mapping):
            next_cursor = meta.get("next_cursor")
        cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
        if cursor:
            time.sleep(delay)
    return results


def collect_records(terms: Iterable[str], per_page: int, delay: float) -> List[OpenAlexRecord]:
    seen: Dict[str, Mapping[str, object]] = {}
    for term in terms:
        fetched = fetch_openalex(term, per_page=per_page, delay=delay)
        for item in fetched:
            identifier = item.get("id") if isinstance(item, Mapping) else None
            if not isinstance(identifier, str):
                continue
            if identifier not in seen:
                seen[identifier] = item
    return [OpenAlexRecord.from_api(payload) for payload in seen.values()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect aging theory reviews from OpenAlex")
    parser.add_argument(
        "--output",
        default="data/pipeline/start_reviews_openalex.json",
        help="Where to write the OpenAlex metadata JSON.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="Number of results to request per page (max 200).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between paginated API requests (seconds).",
    )
    parser.add_argument(
        "--terms",
        nargs="*",
        default=DEFAULT_TERMS,
        help="Search terms to query (defaults to common aging theory phrases).",
    )
    args = parser.parse_args(argv)

    try:
        records = collect_records(args.terms, per_page=args.per_page, delay=args.delay)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not records:
        print("No OpenAlex records found for supplied terms", file=sys.stderr)
        return 1

    ensure_directory(args.output)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in records], handle, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} OpenAlex records to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

