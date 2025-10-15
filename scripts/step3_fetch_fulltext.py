"""Retrieve full texts for filtered PubMed reviews when available.

For each review retained after the OpenAI filtering stage the script attempts to
download an open-access full text from PubMed Central (PMC). If the article does
not have a PMC entry, the script preserves the record but sets ``full_text`` to
``null`` so downstream consumers can decide how to handle missing content.

Environment variables
---------------------
- ``PUBMED_API_KEY`` — optional, raises the E-utilities quota.

Usage
-----
```bash
python scripts/step3_fetch_fulltext.py \
    --input data/pipeline/filtered_reviews.json \
    --output data/pipeline/filtered_reviews_fulltext.json
```
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
from typing import Dict, List, Optional


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def entrez_request(path: str, params: Dict[str, str]) -> bytes:
    query = params.copy()
    api_key = os.environ.get("PUBMED_API_KEY")
    tool = os.environ.get("PUBMED_TOOL")
    email = os.environ.get("PUBMED_EMAIL")
    if api_key:
        query.setdefault("api_key", api_key)
    if tool:
        query.setdefault("tool", tool)
    if email:
        query.setdefault("email", email)
    url = f"{EUTILS_BASE}/{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request) as response:  # nosec: trusted endpoint
            return response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - network edge case
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Entrez request failed ({exc.code}): {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network edge case
        raise RuntimeError(f"Entrez request failed: {exc.reason}") from exc


def extract_pmcid(pubmed_xml: ET.Element) -> Optional[str]:
    for article_id in pubmed_xml.findall(
        ".//ArticleIdList/ArticleId[@IdType='pmc']"
    ):
        if article_id.text:
            return article_id.text.strip()
    return None


def fetch_pubmed_xml(pmid: str) -> Optional[ET.Element]:
    data = entrez_request(
        "efetch.fcgi",
        {"db": "pubmed", "id": pmid, "retmode": "xml"},
    )
    try:
        root = ET.fromstring(data)
    except ET.ParseError as err:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to parse PubMed XML response") from err
    article = root.find(".//PubmedArticle")
    return article


def fetch_pmc_fulltext(pmcid: str) -> Optional[str]:
    data = entrez_request(
        "efetch.fcgi",
        {"db": "pmc", "id": pmcid, "retmode": "xml"},
    )
    try:
        root = ET.fromstring(data)
    except ET.ParseError as err:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to parse PMC XML response") from err
    body = root.find(".//body")
    if body is None:
        return None
    text_chunks: List[str] = []

    def strip_namespace(tag: str) -> str:
        return tag.split("}", 1)[1] if "}" in tag else tag

    block_tags = {
        "p",
        "sec",
        "title",
        "abstract",
        "list",
        "list-item",
        "def-item",
        "caption",
        "fig",
        "table-wrap",
    }

    for elem in body.iter():
        pieces: List[str] = []
        if elem.text:
            text = elem.text.strip()
            if text:
                pieces.append(text)
        if elem.tail:
            tail = elem.tail.strip()
            if tail:
                pieces.append(tail)
        if not pieces:
            continue

        tag = strip_namespace(elem.tag)
        content = " ".join(pieces)
        if tag in block_tags:
            text_chunks.append(content)
        elif text_chunks:
            text_chunks[-1] = f"{text_chunks[-1]} {content}".strip()
        else:
            text_chunks.append(content)

    return "\n\n".join(text_chunks) if text_chunks else None


def enrich_records(records: Iterable[Dict]) -> List[Dict]:
    records_list = list(records)
    total = len(records_list)
    enriched: List[Dict] = []
    for idx, record in enumerate(records_list, start=1):
        pmid = record.get("pmid")
        full_text = None
        pmcid = None
        if pmid:
            article_xml = fetch_pubmed_xml(pmid)
            if article_xml is not None:
                pmcid = extract_pmcid(article_xml)
                record.setdefault("pubmed_xml", ET.tostring(article_xml, encoding="unicode"))
        if pmcid:
            full_text = fetch_pmc_fulltext(pmcid)
        record["pmcid"] = pmcid
        record["full_text"] = full_text
        enriched.append(record)
        print(
            f"Processed {idx}/{total} records — PMC {'found' if full_text else 'missing'}",
            flush=True,
        )
        time.sleep(0.34)
    return enriched


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch PMC full texts for filtered reviews")
    parser.add_argument("--input", default="data/pipeline/filtered_reviews.json")
    parser.add_argument("--output", default="data/pipeline/filtered_reviews_fulltext.json")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    enriched = enrich_records(records)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(enriched, fh, ensure_ascii=False, indent=2)

    print(f"Saved enriched metadata to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

