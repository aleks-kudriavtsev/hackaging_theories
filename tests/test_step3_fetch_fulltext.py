import http.client
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import step3_fetch_fulltext as step3


def _raise_remote_disconnected(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - helper
    raise http.client.RemoteDisconnected("Remote end closed connection")


def test_download_binary_handles_remote_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step3.urllib.request, "urlopen", _raise_remote_disconnected)

    result = step3._download_binary("https://example.test/paper.pdf")

    assert result is None


def _raise_timeout(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - helper
    raise TimeoutError("timed out")


def test_download_binary_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step3.urllib.request, "urlopen", _raise_timeout)

    result = step3._download_binary("https://example.test/paper.pdf")

    assert result is None


def test_process_record_records_pdf_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step3.urllib.request, "urlopen", _raise_remote_disconnected)

    record = {
        "id": "test-record",
        "title": "Example",
        "fulltext_links": {"pdf": "https://example.test/paper.pdf"},
    }

    enriched, failure = step3._process_record(
        record,
        index=0,
        total=1,
        worker_prefix="[worker-test]",
        rate_limiter=None,
        max_attempts=1,
        retry_wait=1.0,
    )

    assert enriched["full_text"] is None
    assert enriched["ocr_status"] == "pdf_download_failed"
    assert failure is not None
    assert failure["reason"] == "pdf_download_failed"


def test_process_record_handles_pubmed_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_entrez(*args: Any, **kwargs: Any) -> Any:
        raise step3.EntrezRequestError("boom")

    monkeypatch.setattr(step3, "fetch_pubmed_xml", _raise_entrez)

    record = {
        "id": "test-record",
        "pmid": "123456",
        "title": "Example",
    }

    enriched, failure = step3._process_record(
        record,
        index=0,
        total=1,
        worker_prefix="[worker-test]",
        rate_limiter=None,
        max_attempts=1,
        retry_wait=1.0,
    )

    assert enriched["full_text"] is None
    assert enriched["pdf_processing"]["failure_reason"] == "pubmed_fetch_failed"
    assert failure is None
