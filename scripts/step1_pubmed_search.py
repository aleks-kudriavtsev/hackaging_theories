"""Collect review articles on aging theories from PubMed.

This script performs an exhaustive PubMed search for review articles that match
the query "Aging Theory". The resulting records are stored as a JSON array with
metadata (PMID, title, abstract, authors, etc.) so downstream steps can reuse
the same structured payload.

Environment variables
---------------------
- ``PUBMED_API_KEY`` — optional, increases the request quota and rate limits.

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
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Iterable, List


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


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

        return cls(
            pmid=pmid or "",
            title=title,
            abstract=abstract,
            publication_types=publication_types,
            authors=authors,
            journal=(journal or "").strip() or None,
            publication_year=publication_year,
        )


def entrez_request(path: str, params: dict[str, str | int]) -> bytes:
    """Execute an HTTP GET against the NCBI E-utilities API."""

    api_key = os.environ.get("PUBMED_API_KEY")
    query = params.copy()
    if api_key:
        query.setdefault("api_key", api_key)
    url = f"{EUTILS_BASE}/{path}?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url) as response:  # nosec: trusted host
        return response.read()


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
        root = ET.fromstring(data)
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
    for batch in chunked(list(pmids), 200):
        payload = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        data = entrez_request("efetch.fcgi", payload)
        root = ET.fromstring(data)
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
        default='"Aging Theory" AND review[Publication Type]',
        help="PubMed search query to execute (defaults to review articles on aging theory).",
    )
    parser.add_argument(
        "--output",
        default="data/pipeline/start_reviews.json",
        help="Where to store the resulting metadata JSON.",
    )
    args = parser.parse_args(argv)

    ids = esearch_ids(args.query)
    if not ids:
        print("No PubMed records found for query", file=sys.stderr)
        return 1

    records = fetch_records(ids)
    ensure_directory(args.output)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump([asdict(record) for record in records], fh, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

