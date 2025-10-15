"""Collect review articles on aging theories from PubMed.

This script performs an exhaustive PubMed search for review articles that match
the query "Aging Theory". The resulting records are stored as a JSON array with
metadata (PMID, title, abstract, authors, etc.) so downstream steps can reuse
the same structured payload.

Environment variables
---------------------
- ``PUBMED_API_KEY`` — optional, increases the request quota and rate limits.
- ``PUBMED_TOOL`` and ``PUBMED_EMAIL`` — recommended by NCBI for contact info.

Usage
-----
```bash
python scripts/step1_pubmed_search.py \
    --output data/pipeline/start_reviews.json
```

The file path defaults to ``data/pipeline/start_reviews.json``.
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
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, field
from typing import Iterable, List


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

DEFAULT_QUERY = (
    '(("aging theory"[TIAB] OR "ageing theory"[TIAB] '
    'OR "theories of aging"[TIAB]) AND review[PTYP])'
)


@dataclass
class PubMedRecord:
    """Lightweight container for the PubMed metadata we care about."""

    pmid: str
    title: str
    abstract: str
    publication_types: List[str]
    authors: List[str]
    journal: str | None
    publication_year: str | None
    doi: str | None
    sources: List[str] = field(default_factory=list)

    @classmethod
    def from_xml(cls, article: ET.Element) -> "PubMedRecord":
        pmid = article.findtext(".//MedlineCitation/PMID")
        title = (article.findtext(".//ArticleTitle") or "").strip()

        abstract_texts: List[str] = []
        for node in article.findall(".//AbstractText"):
            label = node.attrib.get("Label")
            text = ("".join(node.itertext()) or "").strip()
            if not text:
                continue
            if label:
                abstract_texts.append(f"{label}: {text}")
            else:
                abstract_texts.append(text)
        abstract = "\n\n".join(abstract_texts)

        publication_types = [
            (pt.text or "").strip()
            for pt in article.findall(".//PublicationTypeList/PublicationType")
            if (pt.text or "").strip()
        ]

        authors: List[str] = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName") or ""
            fore = author.findtext("ForeName") or ""
            collective = author.findtext("CollectiveName")
            if collective:
                authors.append(collective.strip())
                continue
            name = (fore + " " + last).strip()
            if name:
                authors.append(name)

        journal = article.findtext(
            ".//Journal/Title"
        ) or article.findtext(".//Journal/ISOAbbreviation")

        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        publication_year = None
        if pub_date is not None:
            year = pub_date.findtext("Year")
            medline_date = pub_date.findtext("MedlineDate")
            publication_year = (year or medline_date or "").strip() or None

        doi = None
        for article_id in article.findall(".//ArticleIdList/ArticleId"):
            id_type = (article_id.attrib.get("IdType") or "").lower()
            if id_type == "doi" and article_id.text:
                candidate = article_id.text.strip()
                if candidate:
                    doi = candidate
                break

        return cls(
            pmid=pmid or "",
            title=title,
            abstract=abstract,
            publication_types=publication_types,
            authors=authors,
            journal=(journal or "").strip() or None,
            publication_year=publication_year,
            doi=doi,
            sources=["pubmed"],
        )


def entrez_request(path: str, params: dict[str, str | int]) -> bytes:
    """Execute an HTTP GET against the NCBI E-utilities API."""

    api_key = os.environ.get("PUBMED_API_KEY")
    tool = os.environ.get("PUBMED_TOOL")
    email = os.environ.get("PUBMED_EMAIL")
    query = params.copy()
    if api_key:
        query.setdefault("api_key", api_key)
    if tool:
        query.setdefault("tool", tool)
    if email:
        query.setdefault("email", email)
    url = f"{EUTILS_BASE}/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request) as response:  # nosec: trusted host
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network guard
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Entrez request failed ({exc.code}): {body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network guard
        raise RuntimeError(f"Entrez request failed: {exc.reason}") from exc


def esearch_ids(term: str, batch_size: int = 500) -> List[str]:
    """Retrieve all PubMed IDs that match the provided search term."""

    ids: List[str] = []
    total_count = None

    while total_count is None or len(ids) < total_count:
        payload = {
            "db": "pubmed",
            "term": term,
            "retmax": batch_size,
            "retstart": len(ids),
            "retmode": "xml",
        }
        data = entrez_request("esearch.fcgi", payload)
        try:
            root = ET.fromstring(data)
        except ET.ParseError as err:  # pragma: no cover - defensive guard
            raise RuntimeError("Unable to parse ESearch XML response") from err
        # Surface any query issues flagged by the ESearch service.
        errors = [
            err.text.strip()
            for err in root.findall(".//ErrorList/*")
            if (err.text or "").strip()
        ]
        if errors:
            raise ValueError(
                "PubMed rejected the search term: " + "; ".join(errors)
            )

        if total_count is None:
            total_count_text = root.findtext(".//Count")
            total_count = int(total_count_text or 0)
        batch_ids = [elem.text for elem in root.findall(".//IdList/Id") if elem.text]
        ids.extend(batch_ids)
        if not batch_ids:
            break
        # Be polite with a tiny delay to avoid hammering the API.
        time.sleep(0.34)

    return ids


def chunked(iterable: Iterable[str], size: int) -> Iterable[List[str]]:
    """Yield fixed-size chunks from *iterable*."""

    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_records(pmids: Iterable[str]) -> List[PubMedRecord]:
    """Fetch full metadata for a list of PubMed IDs."""

    records: List[PubMedRecord] = []
    for batch in chunked(pmids, 200):
        payload = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        data = entrez_request("efetch.fcgi", payload)
        try:
            root = ET.fromstring(data)
        except ET.ParseError as err:  # pragma: no cover - defensive guard
            raise RuntimeError("Unable to parse EFetch XML response") from err
        for article in root.findall(".//PubmedArticle"):
            records.append(PubMedRecord.from_xml(article))
        time.sleep(0.34)
    return records


def ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect PubMed reviews on aging theory")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help=(
            "PubMed search query to execute (defaults to a Title/Abstract search for "
            "aging theories limited to review publication type)."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/pipeline/start_reviews.json",
        help="Where to store the resulting metadata JSON.",
    )
    args = parser.parse_args(argv)

    try:
        ids = esearch_ids(args.query)
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if not ids:
        print("No PubMed records found for query", file=sys.stderr)
        return 1

    try:
        records = fetch_records(ids)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    ensure_directory(args.output)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump([asdict(record) for record in records], fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

