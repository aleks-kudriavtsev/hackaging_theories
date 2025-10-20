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
import http.client
import io
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class RateLimiter:
    """Process-safe rate limiter for Entrez API requests."""

    def __init__(
        self,
        interval: float,
        *,
        state: multiprocessing.Value | None = None,
        lock: multiprocessing.synchronize.Lock | None = None,
    ) -> None:
        self.interval = max(0.0, interval)
        self._state = state or multiprocessing.Value("d", 0.0)
        self._lock = lock or multiprocessing.Lock()

    def wait(self) -> None:
        """Pause until the shared interval since the last request has elapsed."""

        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            last_call = self._state.value
            elapsed = now - last_call
            if elapsed < self.interval:
                sleep_for = self.interval - elapsed
                time.sleep(sleep_for)
                now = time.monotonic()
            self._state.value = now


class EntrezRequestError(RuntimeError):
    """Raised when an Entrez request fails with HTTP/URL errors."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
        message = f"Entrez request failed ({exc.code}): {error_body or exc.reason}"
        raise EntrezRequestError(message, status=exc.code) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network edge case
        raise EntrezRequestError(f"Entrez request failed: {exc.reason}") from exc


def _entrez_with_retries(
    path: str,
    params: Dict[str, str],
    *,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
    context: str,
) -> bytes:
    attempts = max(1, max_attempts)
    delay = max(0.1, retry_wait)
    attempt = 1
    while True:
        if rate_limiter is not None:
            rate_limiter.wait()
        try:
            return entrez_request(path, params)
        except EntrezRequestError as exc:
            status = exc.status or 0
            retriable = status == 429 or 500 <= status < 600
            if not retriable:
                raise
            if attempt >= attempts:
                message = (
                    f"{context}: Entrez rate limit triggered after {attempts} attempts "
                    f"(HTTP {status}). Increase --entrez-interval or retry later."
                )
                logger.warning(message)
                raise RuntimeError(message) from exc
            logger.warning(
                "%s: Entrez returned HTTP %s, retrying in %.2f seconds (%d/%d)",
                context,
                status,
                delay,
                attempt,
                attempts,
            )
            time.sleep(delay)
            delay *= 2
            attempt += 1


def extract_pmcid(pubmed_xml: ET.Element) -> Optional[str]:
    for article_id in pubmed_xml.findall(
        ".//ArticleIdList/ArticleId[@IdType='pmc']"
    ):
        if article_id.text:
            return article_id.text.strip()
    return None


def fetch_pubmed_xml(
    pmid: str,
    *,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
) -> Optional[ET.Element]:
    data = _entrez_with_retries(
        "efetch.fcgi",
        {"db": "pubmed", "id": pmid, "retmode": "xml"},
        rate_limiter=rate_limiter,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
        context=f"PMID {pmid}",
    )
    try:
        root = ET.fromstring(data)
    except ET.ParseError as err:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to parse PubMed XML response") from err
    article = root.find(".//PubmedArticle")
    return article


_PMC_BLOCK_TAGS = {
    "p",
    "sec",
    "title",
    "abstract",
    "list-item",
    "caption",
    "fig",
    "table-wrap",
}


def _collect_pmc_text(node: ET.Element, chunks: List[str]) -> None:
    if node.text:
        chunks.append(node.text)
    for child in node:
        _collect_pmc_text(child, chunks)
    if node.tag in _PMC_BLOCK_TAGS:
        chunks.append("\n\n")
    if node.tail:
        chunks.append(node.tail)


def _extract_pmc_body_text(body: ET.Element) -> Optional[str]:
    text_chunks: List[str] = []
    _collect_pmc_text(body, text_chunks)
    raw_text = "".join(text_chunks)
    normalized = _normalize_text(raw_text)
    return normalized or None


def fetch_pmc_fulltexts_batch(
    pmc_ids: Iterable[str],
    *,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
    chunk_size: int = 200,
) -> Tuple[Dict[str, str], Set[str]]:
    unique_ids: List[str] = []
    seen: Set[str] = set()
    for pmcid in pmc_ids:
        if not pmcid:
            continue
        if pmcid not in seen:
            seen.add(pmcid)
            unique_ids.append(pmcid)

    if not unique_ids:
        return {}, set()

    resolved_texts: Dict[str, str] = {}
    missing_bodies: Set[str] = set()
    chunk_size = max(1, chunk_size)

    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start : start + chunk_size]
        context = (
            f"PMCID {chunk[0]}"
            if len(chunk) == 1
            else f"PMCID batch {chunk[0]}-{chunk[-1]}"
        )
        data = _entrez_with_retries(
            "efetch.fcgi",
            {"db": "pmc", "id": ",".join(chunk), "retmode": "xml"},
            rate_limiter=rate_limiter,
            max_attempts=max_attempts,
            retry_wait=retry_wait,
            context=context,
        )
        try:
            root = ET.fromstring(data)
        except ET.ParseError as err:  # pragma: no cover - defensive guard
            raise RuntimeError("Unable to parse PMC XML response") from err

        articles = root.findall(".//article")
        for article in articles:
            pmcid_node = article.find(".//article-id[@pub-id-type='pmcid']")
            if pmcid_node is None:
                pmcid_node = article.find(".//article-id[@pub-id-type='pmc']")
            pmcid_value = pmcid_node.text.strip() if pmcid_node is not None and pmcid_node.text else None
            if not pmcid_value:
                continue
            body = article.find("./body")
            if body is None:
                missing_bodies.add(pmcid_value)
                continue
            text = _extract_pmc_body_text(body)
            if text:
                resolved_texts[pmcid_value] = text
            else:
                missing_bodies.add(pmcid_value)

        for requested in chunk:
            if requested not in resolved_texts and requested not in missing_bodies:
                missing_bodies.add(requested)

    return resolved_texts, missing_bodies


def fetch_pmc_fulltext(
    pmcid: str,
    *,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
) -> Optional[str]:
    resolved, missing = fetch_pmc_fulltexts_batch(
        [pmcid],
        rate_limiter=rate_limiter,
        max_attempts=max_attempts,
        retry_wait=retry_wait,
        chunk_size=1,
    )
    if pmcid in missing:
        return None
    return resolved.get(pmcid)


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
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
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
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "hackaging-theories-fulltext/1.0",
                "Accept": "application/pdf, */*",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - trusted endpoint
            return response.read()
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        http.client.HTTPException,
        socket.timeout,
        TimeoutError,
        ConnectionError,
        ssl.SSLError,
        OSError,
        ValueError,
    ) as exc:  # pragma: no cover - network failure
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


def _merge_pdf_processing(
    existing: Dict[str, Any], update: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Merge ``pdf_processing`` metadata without clobbering fetch failures."""

    if not update:
        return existing

    if not existing:
        return update.copy()

    merged = existing.copy()
    for key, value in update.items():
        if key == "failure_reason" and merged.get("failure_reason"):
            if value and merged["failure_reason"] != value:
                merged.setdefault("secondary_failure_reason", value)
            continue
        merged[key] = value
    return merged


def _process_record(
    record: Dict,
    index: int,
    total: int,
    worker_prefix: str,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
    *,
    resolved_pmcid: Optional[str] = None,
    resolved_pubmed_xml: Optional[str] = None,
    pubmed_fetch_failed: bool = False,
    pmc_full_text: Optional[str] = None,
    pmc_missing_body: bool = False,
    pmc_prefetched: bool = False,
) -> Tuple[Dict, Optional[Dict]]:
    pmid = record.get("pmid")
    full_text = None
    pmcid = resolved_pmcid
    ocr_status = "not_applicable"
    full_text_source = None
    openalex_pdf_url = None
    record.pop("pdf_processing", None)
    pdf_processing_meta: Dict[str, Any] = {}

    if pubmed_fetch_failed:
        pdf_processing_meta.setdefault("failure_reason", "pubmed_fetch_failed")

    article_xml = None
    if pmid and pmcid is None and not pubmed_fetch_failed:
        try:
            article_xml = fetch_pubmed_xml(
                pmid,
                rate_limiter=rate_limiter,
                max_attempts=max_attempts,
                retry_wait=retry_wait,
            )
        except (EntrezRequestError, RuntimeError) as exc:
            logger.warning("Failed to fetch PubMed XML for %s: %s", pmid, exc)
            pdf_processing_meta.setdefault("failure_reason", "pubmed_fetch_failed")
        else:
            if article_xml is not None:
                pmcid = extract_pmcid(article_xml)
                resolved_pubmed_xml = ET.tostring(article_xml, encoding="unicode")

    if resolved_pubmed_xml:
        record.setdefault("pubmed_xml", resolved_pubmed_xml)

    if pmcid and full_text is None:
        if pmc_prefetched:
            if pmc_full_text:
                full_text = pmc_full_text
                full_text_source = "pmc"
        else:
            try:
                full_text = fetch_pmc_fulltext(
                    pmcid,
                    rate_limiter=rate_limiter,
                    max_attempts=max_attempts,
                    retry_wait=retry_wait,
                )
            except (EntrezRequestError, RuntimeError) as exc:
                logger.warning("Failed to fetch PMC full text for %s: %s", pmcid, exc)
                pdf_processing_meta.setdefault("failure_reason", "pmc_fetch_failed")
                full_text = None
            if full_text:
                full_text_source = "pmc"

    if pmcid and pmc_missing_body and not full_text:
        pdf_processing_meta.setdefault("failure_reason", "pmc_missing_body")
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
                    pdf_processing_meta = _merge_pdf_processing(
                        pdf_processing_meta, extraction_meta
                    )
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
                pdf_processing_meta = _merge_pdf_processing(
                    pdf_processing_meta, extraction_meta
                )
        else:
            ocr_status = "no_pdf_url"
            extraction_meta = {
                "digital_status": None,
                "ocr_status": "not_attempted",
                "failure_reason": "no_pdf_url",
                "method": None,
            }
            pdf_processing_meta = _merge_pdf_processing(
                pdf_processing_meta, extraction_meta
            )
    else:
        record["openalex_pdf_url"] = None

    if pdf_processing_meta:
        record["pdf_processing"] = pdf_processing_meta

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


def _process_records_inner(
    batch: List[Tuple[int, Dict]],
    total: int,
    prefix: str,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
    entrez_batch_size: int,
) -> Tuple[List[Tuple[int, Dict]], List[Dict]]:
    prepared: List[Tuple[int, Dict, Optional[str], Optional[str], bool]] = []
    requested_pmcids: List[str] = []
    for index, record in batch:
        pmid = record.get("pmid")
        pmcid: Optional[str] = None
        pubmed_xml_str: Optional[str] = None
        pubmed_failed = False
        if pmid:
            try:
                article_xml = fetch_pubmed_xml(
                    pmid,
                    rate_limiter=rate_limiter,
                    max_attempts=max_attempts,
                    retry_wait=retry_wait,
                )
            except (EntrezRequestError, RuntimeError) as exc:
                logger.warning("Failed to fetch PubMed XML for %s: %s", pmid, exc)
                pubmed_failed = True
            else:
                if article_xml is not None:
                    pmcid = extract_pmcid(article_xml)
                    pubmed_xml_str = ET.tostring(article_xml, encoding="unicode")
        prepared.append((index, record, pmcid, pubmed_xml_str, pubmed_failed))
        if pmcid:
            requested_pmcids.append(pmcid)

    pmc_texts: Dict[str, str] = {}
    missing_bodies: Set[str] = set()
    if requested_pmcids:
        pmc_texts, missing_bodies = fetch_pmc_fulltexts_batch(
            requested_pmcids,
            rate_limiter=rate_limiter,
            max_attempts=max_attempts,
            retry_wait=retry_wait,
            chunk_size=max(1, entrez_batch_size),
        )
        for pmcid in requested_pmcids:
            if pmcid not in pmc_texts and pmcid not in missing_bodies:
                missing_bodies.add(pmcid)

    enriched_batch: List[Tuple[int, Dict]] = []
    failures_batch: List[Dict] = []
    for index, record, pmcid, pubmed_xml_str, pubmed_failed in prepared:
        pmc_text = pmc_texts.get(pmcid) if pmcid else None
        missing_body = pmcid in missing_bodies if pmcid else False
        enriched_record, failure = _process_record(
            record,
            index,
            total,
            prefix,
            rate_limiter,
            max_attempts,
            retry_wait,
            resolved_pmcid=pmcid,
            resolved_pubmed_xml=pubmed_xml_str,
            pubmed_fetch_failed=pubmed_failed,
            pmc_full_text=pmc_text,
            pmc_missing_body=missing_body,
            pmc_prefetched=pmcid is not None,
        )
        enriched_batch.append((index, enriched_record))
        if failure:
            failures_batch.append(failure)
        if missing_body:
            failures_batch.append(
                {
                    "pmid": record.get("pmid"),
                    "pmcid": pmcid,
                    "reason": "pmc_missing_body",
                    "url": None,
                    "metadata": None,
                }
            )
    return enriched_batch, failures_batch


def _process_batch(
    worker_id: int,
    batch: List[Tuple[int, Dict]],
    result_queue: Any,
    total: int,
    rate_limiter: RateLimiter | None,
    max_attempts: int,
    retry_wait: float,
    entrez_batch_size: int,
) -> None:
    prefix = f"[worker-{worker_id}]"
    enriched_batch: List[Tuple[int, Dict]] = []
    failures_batch: List[Dict] = []
    error_message: Optional[str] = None
    try:
        enriched_batch, failures_batch = _process_records_inner(
            batch,
            total,
            prefix,
            rate_limiter,
            max_attempts,
            retry_wait,
            entrez_batch_size,
        )
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
    rate_limiter: RateLimiter | None = None,
    entrez_interval: float = 0.34,
    entrez_max_attempts: int = 5,
    entrez_retry_wait: float | None = None,
    entrez_batch_size: int = 200,
) -> Tuple[List[Dict], List[Dict]]:
    records_list = list(records)
    total = len(records_list)
    if total == 0:
        return [], []

    if rate_limiter is not None:
        limiter = rate_limiter
    else:
        if concurrency == "process":
            ctx = multiprocessing.get_context("spawn")
            limiter = RateLimiter(
                entrez_interval,
                state=ctx.Value("d", 0.0),
                lock=ctx.Lock(),
            )
        else:
            limiter = RateLimiter(entrez_interval)
    retry_wait = entrez_retry_wait if entrez_retry_wait is not None else max(1.0, limiter.interval)
    attempts = max(1, entrez_max_attempts)

    worker_count = max(1, min(processes, total))
    if worker_count == 1:
        single_batch = [(idx, records_list[idx]) for idx in range(total)]
        enriched_pairs, failures = _process_records_inner(
            single_batch,
            total,
            "[worker-1]",
            limiter,
            attempts,
            retry_wait,
            entrez_batch_size,
        )
        enriched_map = {idx: record for idx, record in enriched_pairs}
        ordered = [enriched_map[idx] for idx in range(total)]
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
                    limiter,
                    attempts,
                    retry_wait,
                    entrez_batch_size,
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
        shared_limiter = limiter
        for worker_id, batch in enumerate(batches, start=1):
            process = ctx.Process(
                target=_process_batch,
                args=(
                    worker_id,
                    batch,
                    result_queue,
                    total,
                    shared_limiter,
                    attempts,
                    retry_wait,
                    entrez_batch_size,
                ),
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
    parser.add_argument(
        "--entrez-interval",
        type=float,
        default=None,
        help=(
            "Minimum delay in seconds between Entrez requests. "
            "Defaults to PUBMED_RATE_INTERVAL or 0.34 when unset."
        ),
    )
    parser.add_argument(
        "--entrez-max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum retry attempts for Entrez HTTP 429/5xx responses. "
            "Defaults to PUBMED_MAX_ATTEMPTS or 5 when unset."
        ),
    )
    parser.add_argument(
        "--entrez-retry-wait",
        type=float,
        default=None,
        help=(
            "Initial wait time for Entrez retry backoff. Defaults to the "
            "effective Entrez interval or PUBMED_RETRY_WAIT when provided."
        ),
    )
    parser.add_argument(
        "--entrez-batch-size",
        type=int,
        default=None,
        help=(
            "Number of PMCIDs to request per Entrez efetch call. "
            "Defaults to PUBMED_BATCH_SIZE or 200 when unset."
        ),
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
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            logger.warning("Invalid value for %s=%s; falling back to %.2f", name, raw, default)
            return default

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            logger.warning("Invalid value for %s=%s; falling back to %d", name, raw, default)
            return default

    entrez_interval = (
        args.entrez_interval
        if args.entrez_interval is not None
        else _env_float("PUBMED_RATE_INTERVAL", 0.34)
    )
    entrez_max_attempts = (
        args.entrez_max_attempts
        if args.entrez_max_attempts is not None
        else _env_int("PUBMED_MAX_ATTEMPTS", 5)
    )
    entrez_retry_wait = (
        args.entrez_retry_wait
        if args.entrez_retry_wait is not None
        else os.environ.get("PUBMED_RETRY_WAIT")
    )
    entrez_batch_size = (
        args.entrez_batch_size
        if args.entrez_batch_size is not None
        else _env_int("PUBMED_BATCH_SIZE", 200)
    )
    retry_wait_value = None
    if entrez_retry_wait is not None:
        try:
            retry_wait_value = float(entrez_retry_wait)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid value for PUBMED_RETRY_WAIT=%s; using interval %.2f",
                entrez_retry_wait,
                entrez_interval,
            )
            retry_wait_value = None

    enriched, failures = enrich_records(
        records,
        processes=requested_processes,
        concurrency=args.concurrency,
        rate_limiter=None,
        entrez_interval=entrez_interval,
        entrez_max_attempts=entrez_max_attempts,
        entrez_retry_wait=retry_wait_value,
        entrez_batch_size=max(1, entrez_batch_size),
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

