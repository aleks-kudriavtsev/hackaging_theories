"""Retrieve full texts for filtered reviews when available.

For each review retained after the OpenAI filtering stage the script attempts to
download an open-access full text from PubMed Central (PMC). When PMC content is
unavailable the script inspects OpenAlex metadata for hosted PDFs and downloads
them, extracting plain text via ``pdfminer.six`` (digital PDFs) or OCR using
``pdf2image`` + ``pytesseract`` for scanned documents. Fallback attempts and
failures are captured in a companion log for easier troubleshooting.

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
import io
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

    block_tags = {
        "p",
        "sec",
        "title",
        "abstract",
        "list-item",
        "caption",
        "fig",
        "table-wrap",
    }

    def collect_text(node: ET.Element, chunks: List[str]) -> None:
        if node.text:
            chunks.append(node.text)
        for child in node:
            collect_text(child, chunks)
        if node.tag in block_tags:
            chunks.append("\n\n")
        if node.tail:
            chunks.append(node.tail)

    text_chunks: List[str] = []
    collect_text(body, text_chunks)
    raw_text = "".join(text_chunks)
    normalized = re.sub(r"[ \t\r\f\v]+", " ", raw_text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" ?\n", "\n", normalized)
    normalized = normalized.strip()
    return normalized or None


try:  # pragma: no cover - optional dependency
    from pdfminer.high_level import extract_text as _pdf_extract_text
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _pdf_extract_text = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pdf2image import convert_from_bytes as _convert_from_bytes
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _convert_from_bytes = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import pytesseract as _pytesseract
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _pytesseract = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" ?\n", "\n", normalized)
    return normalized.strip()


def _find_openalex_pdf_url(record: Dict) -> Optional[str]:
    links = record.get("fulltext_links")
    candidates: List[str] = []
    if isinstance(links, dict):
        for key in ("pdf", "oa_url", "landing_page"):
            value = links.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    open_access = record.get("open_access")
    if isinstance(open_access, dict):
        for key in ("oa_url", "pdf_url"):
            value = open_access.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    if isinstance(record.get("open_access_pdf"), str):
        candidates.append(record["open_access_pdf"].strip())

    for url in candidates:
        if url.lower().endswith(".pdf"):
            return url
    if candidates:
        return candidates[0]
    return None


def _download_binary(url: str, timeout: float = 30.0) -> Optional[bytes]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "hackaging-theories-fulltext/1.0",
            "Accept": "application/pdf, */*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - trusted endpoint
            return response.read()
    except urllib.error.URLError as exc:  # pragma: no cover - network failure
        logger.debug("Failed to download %s: %s", url, exc)
        return None


def _extract_pdf_text(pdf_bytes: bytes) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    if not pdf_bytes:
        return None, {
            "digital_status": "empty_payload",
            "ocr_status": "not_attempted",
            "failure_reason": "empty_payload",
            "method": None,
        }

    metadata: Dict[str, Optional[str]] = {
        "digital_status": None,
        "ocr_status": "not_attempted",
        "failure_reason": None,
        "method": None,
    }

    if _pdf_extract_text is not None:
        try:
            with io.BytesIO(pdf_bytes) as buffer:
                text = _pdf_extract_text(buffer)
        except Exception as exc:  # pragma: no cover - pdf parsing edge case
            metadata["digital_status"] = "error"
            metadata["failure_reason"] = "pdfminer_error"
            logger.debug("pdfminer failed: %s", exc)
        else:
            normalized = _normalize_text(text)
            if normalized:
                metadata["digital_status"] = "success"
                metadata["ocr_status"] = "not_required"
                metadata["method"] = "pdfminer"
                return normalized, metadata
            metadata["digital_status"] = "empty"
    else:
        metadata["digital_status"] = "unavailable"
        logger.debug("pdfminer.six is not installed; skipping native PDF extraction")

    if _convert_from_bytes is None or _pytesseract is None:
        metadata["ocr_status"] = "missing_dependency"
        metadata["failure_reason"] = "ocr_missing_dependency"
        return None, metadata

    metadata["ocr_status"] = "performed"
    try:
        images = _convert_from_bytes(pdf_bytes)
    except Exception as exc:  # pragma: no cover - pdf rendering edge case
        logger.debug("pdf2image conversion failed: %s", exc)
        metadata["ocr_status"] = "performed_failed"
        metadata["failure_reason"] = "ocr_conversion_failed"
        return None, metadata

    text_chunks: List[str] = []
    for image in images:
        try:
            extracted = _pytesseract.image_to_string(image)
        except Exception as exc:  # pragma: no cover - ocr edge case
            logger.debug("pytesseract failed: %s", exc)
            continue
        if extracted:
            text_chunks.append(extracted)
    combined = _normalize_text("\n".join(text_chunks))
    if combined:
        metadata["ocr_status"] = "performed_success"
        metadata["method"] = "ocr"
        return combined, metadata

    metadata["ocr_status"] = "performed_no_text"
    metadata["failure_reason"] = "ocr_no_text"
    return None, metadata


def _process_record(
    record: Dict,
    index: int,
    total: int,
    worker_prefix: str,
) -> Tuple[Dict, Optional[Dict]]:
    pmid = record.get("pmid")
    full_text = None
    pmcid = None
    ocr_status = "not_applicable"
    full_text_source = None
    openalex_pdf_url = None
    record.pop("pdf_processing", None)

    if pmid:
        article_xml = fetch_pubmed_xml(pmid)
        if article_xml is not None:
            pmcid = extract_pmcid(article_xml)
            record.setdefault("pubmed_xml", ET.tostring(article_xml, encoding="unicode"))
    if pmcid:
        full_text = fetch_pmc_fulltext(pmcid)
        if full_text:
            full_text_source = "pmc"
    record["pmcid"] = pmcid

    extraction_meta: Dict[str, Optional[str]] | None = None
    if not full_text:
        openalex_pdf_url = _find_openalex_pdf_url(record) or None
        record["openalex_pdf_url"] = openalex_pdf_url
        if openalex_pdf_url:
            pdf_bytes = _download_binary(openalex_pdf_url)
            if pdf_bytes:
                pdf_text, extraction_meta = _extract_pdf_text(pdf_bytes)
                if extraction_meta:
                    record["pdf_processing"] = extraction_meta
                    ocr_status = extraction_meta.get("ocr_status") or ocr_status
                if pdf_text:
                    full_text = pdf_text
                    method = extraction_meta.get("method") if extraction_meta else None
                    if method == "pdfminer":
                        full_text_source = "openalex_pdf_digital"
                    elif method == "ocr":
                        full_text_source = "openalex_pdf_ocr"
                    else:
                        full_text_source = "openalex_pdf"
            else:
                ocr_status = "pdf_download_failed"
                extraction_meta = {
                    "digital_status": None,
                    "ocr_status": "not_attempted",
                    "failure_reason": "pdf_download_failed",
                    "method": None,
                }
                record["pdf_processing"] = extraction_meta
        else:
            ocr_status = "no_pdf_url"
            extraction_meta = {
                "digital_status": None,
                "ocr_status": "not_attempted",
                "failure_reason": "no_pdf_url",
                "method": None,
            }
            record["pdf_processing"] = extraction_meta
    else:
        record["openalex_pdf_url"] = None

    failure_reason = None
    if extraction_meta and extraction_meta.get("failure_reason"):
        failure_reason = extraction_meta["failure_reason"]
    elif extraction_meta and extraction_meta.get("digital_status"):
        failure_reason = extraction_meta["digital_status"]
    elif ocr_status not in {"not_applicable", "not_required", "performed_success"}:
        failure_reason = ocr_status

    failure_payload = None
    if not full_text and openalex_pdf_url and failure_reason:
        failure_payload = {
            "pmid": pmid,
            "pmcid": pmcid,
            "reason": failure_reason,
            "url": openalex_pdf_url,
            "metadata": extraction_meta,
        }

    record["ocr_status"] = ocr_status
    record["full_text_source"] = full_text_source
    record["full_text"] = full_text
    source_label = full_text_source or "missing"
    print(
        f"{worker_prefix} Processed {index + 1}/{total} records — full text {source_label}",
        flush=True,
    )
    return record, failure_payload


def _process_batch(
    worker_id: int,
    batch: List[Tuple[int, Dict]],
    result_queue: Any,
    total: int,
) -> None:
    prefix = f"[worker-{worker_id}]"
    enriched_batch: List[Tuple[int, Dict]] = []
    failures_batch: List[Dict] = []
    error_message: Optional[str] = None
    try:
        for index, record in batch:
            enriched_record, failure = _process_record(record, index, total, prefix)
            enriched_batch.append((index, enriched_record))
            if failure:
                failures_batch.append(failure)
    except Exception as exc:  # pragma: no cover - worker protection
        error_message = f"{type(exc).__name__}: {exc}"
        logger.exception("%s Worker encountered an error", prefix)
    finally:
        result_queue.put((worker_id, enriched_batch, failures_batch, error_message))


def _build_batches(records_list: List[Dict], worker_count: int) -> List[List[Tuple[int, Dict]]]:
    total = len(records_list)
    if worker_count <= 0:
        return []
    batch_size = math.ceil(total / worker_count)
    batches: List[List[Tuple[int, Dict]]] = []
    for worker_index in range(worker_count):
        start = worker_index * batch_size
        if start >= total:
            break
        end = min(start + batch_size, total)
        batch = [(idx, records_list[idx]) for idx in range(start, end)]
        batches.append(batch)
    return batches


def _collect_results(result_queue: Any, expected: int) -> Tuple[Dict[int, Dict], List[Dict]]:
    results: Dict[int, Dict] = {}
    failures: List[Dict] = []
    errors: List[Tuple[int, str]] = []
    for _ in range(expected):
        worker_id, enriched_batch, failures_batch, error_message = result_queue.get()
        for index, record in enriched_batch:
            results[index] = record
        failures.extend(failures_batch)
        if error_message:
            errors.append((worker_id, error_message))
    if errors:
        formatted = ", ".join(f"worker {wid}: {msg}" for wid, msg in errors)
        raise RuntimeError(f"Worker failures: {formatted}")
    return results, failures


def enrich_records(
    records: Iterable[Dict],
    *,
    processes: int = 1,
    concurrency: str = "process",
) -> Tuple[List[Dict], List[Dict]]:
    records_list = list(records)
    total = len(records_list)
    if total == 0:
        return [], []

    worker_count = max(1, min(processes, total))
    if worker_count == 1:
        enriched: Dict[int, Dict] = {}
        failures: List[Dict] = []
        for index, record in enumerate(records_list):
            enriched_record, failure = _process_record(record, index, total, "[worker-1]")
            enriched[index] = enriched_record
            if failure:
                failures.append(failure)
        ordered = [enriched[idx] for idx in range(total)]
        return ordered, failures

    batches = _build_batches(records_list, worker_count)
    if not batches:
        return records_list, []

    if concurrency == "thread":
        result_queue: queue.Queue = queue.Queue()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_batch,
                    worker_id,
                    batch,
                    result_queue,
                    total,
                )
                for worker_id, batch in enumerate(batches, start=1)
            ]
            results_map, failures = _collect_results(result_queue, len(batches))
            for future in futures:
                future.result()
    else:
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()
        processes_list: List[multiprocessing.Process] = []
        for worker_id, batch in enumerate(batches, start=1):
            process = ctx.Process(
                target=_process_batch,
                args=(worker_id, batch, result_queue, total),
            )
            process.start()
            processes_list.append(process)
        try:
            results_map, failures = _collect_results(result_queue, len(processes_list))
        finally:
            for process in processes_list:
                process.join()
            result_queue.close()
            result_queue.join_thread()

    ordered_results = [results_map[idx] for idx in range(total)]
    return ordered_results, failures


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch full texts for filtered reviews from PMC and OpenAlex"
    )
    parser.add_argument("--input", default="data/pipeline/filtered_reviews.json")
    parser.add_argument("--output", default="data/pipeline/filtered_reviews_fulltext.json")
    parser.add_argument(
        "--failures",
        default=None,
        help=(
            "Optional path for logging failed PDF/Full text retrieval attempts. "
            "Defaults to <output>.failures.json when omitted."
        ),
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Number of worker processes (default: os.cpu_count() when more than "
            "100 records, otherwise 1)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        choices=("process", "thread"),
        default="process",
        help="Concurrency model for workers (default: process).",
    )
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    total_records = len(records)
    requested_processes = args.processes
    if requested_processes is None:
        cpu_count = os.cpu_count() or 1
        requested_processes = cpu_count if total_records > 100 else 1
    enriched, failures = enrich_records(
        records,
        processes=requested_processes,
        concurrency=args.concurrency,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(enriched, fh, ensure_ascii=False, indent=2)

    print(f"Saved enriched metadata to {args.output}")

    failure_path = args.failures or f"{args.output}.failures.json"
    if failures:
        fail_dir = os.path.dirname(failure_path)
        if fail_dir:
            os.makedirs(fail_dir, exist_ok=True)
        with open(failure_path, "w", encoding="utf-8") as fh:
            json.dump(failures, fh, ensure_ascii=False, indent=2)
        print(f"Logged {len(failures)} retrieval failures to {failure_path}")
    else:
        print("No retrieval failures recorded")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

