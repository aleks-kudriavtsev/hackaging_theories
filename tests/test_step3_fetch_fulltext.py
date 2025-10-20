import http.client
import ssl
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
import xml.etree.ElementTree as ET

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import step3_fetch_fulltext as step3


def _raise_remote_disconnected(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - helper
    raise http.client.RemoteDisconnected("Remote end closed connection")


def test_download_binary_handles_remote_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step3.urllib.request, "urlopen", _raise_remote_disconnected)

    result = step3._download_binary("https://example.test/paper.pdf")

    assert result is None


@pytest.mark.parametrize("exception", [TimeoutError("timed out"), ssl.SSLError("boom")])
def test_download_binary_handles_read_errors(
    monkeypatch: pytest.MonkeyPatch, exception: BaseException
) -> None:
    class _ErrorResponse:
        def __enter__(self) -> "_ErrorResponse":
            return self

        def __exit__(self, *exc_info: Any) -> None:
            return None

        def read(self) -> bytes:
            raise exception

    monkeypatch.setattr(step3.urllib.request, "urlopen", lambda *args, **kwargs: _ErrorResponse())

    result = step3._download_binary("https://example.test/paper.pdf")

    assert result is None


def test_download_binary_handles_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyRequest:
        def __init__(self, url: str, headers: Dict[str, str]) -> None:
            assert " " in url  # ensure malformed URL propagated
            self.url = url
            self.headers = headers

    def _raise_value_error(_request: _DummyRequest, *args: Any, **kwargs: Any) -> None:
        raise ValueError("invalid URL")

    monkeypatch.setattr(step3.urllib.request, "Request", _DummyRequest)
    monkeypatch.setattr(step3.urllib.request, "urlopen", _raise_value_error)

    malformed_url = "https://example.test/has space.pdf"

    result = step3._download_binary(malformed_url)

    assert result is None

    record = {
        "id": "value-error-record",
        "title": "Example",
        "fulltext_links": {"pdf": malformed_url},
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
    assert enriched["pdf_processing"]["failure_reason"] == "pdf_download_failed"
    assert failure is not None
    assert failure["reason"] == "pdf_download_failed"


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


def test_process_record_handles_pmc_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_entrez(*args: Any, **kwargs: Any) -> Any:
        raise step3.EntrezRequestError("boom")

    monkeypatch.setattr(step3, "fetch_pmc_fulltext", _raise_entrez)

    record = {
        "id": "test-record",
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
        resolved_pmcid="PMC123",
    )

    assert enriched["full_text"] is None
    assert enriched["pdf_processing"]["failure_reason"] == "pmc_fetch_failed"
    assert failure is None


def test_fetch_pmc_fulltexts_batch_handles_missing_body(monkeypatch: pytest.MonkeyPatch) -> None:
    response_xml = """
    <pmc-articleset>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmcid">PMC123</article-id>
          </article-meta>
        </front>
        <body>
          <sec>
            <title>Intro</title>
            <p>Hello world.</p>
          </sec>
        </body>
      </article>
      <article>
        <front>
          <article-meta>
            <article-id pub-id-type="pmcid">PMC456</article-id>
          </article-meta>
        </front>
      </article>
    </pmc-articleset>
    """

    requested: List[str] = []

    def _fake_entrez(path: str, params: Dict[str, str], **_: Any) -> bytes:
        requested.append(params["id"])
        return response_xml.encode("utf-8")

    monkeypatch.setattr(step3, "_entrez_with_retries", _fake_entrez)

    texts, missing = step3.fetch_pmc_fulltexts_batch(
        ["PMC123", "PMC456"],
        rate_limiter=None,
        max_attempts=1,
        retry_wait=0.1,
        chunk_size=2,
    )

    assert requested == ["PMC123,PMC456"]
    assert texts["PMC123"] == "Intro\n\nHello world."
    assert "PMC456" in missing


def test_process_records_inner_prefetches_pmc(monkeypatch: pytest.MonkeyPatch) -> None:
    pmid_to_pmcid = {"1": "PMC111", "2": "PMC222"}

    def _fake_fetch_pubmed(pmid: str, **_: Any) -> ET.Element:
        pmcid = pmid_to_pmcid[pmid]
        return ET.fromstring(
            f"""
            <PubmedArticle>
              <MedlineCitation>
                <ArticleIdList>
                  <ArticleId IdType=\"pmc\">{pmcid}</ArticleId>
                </ArticleIdList>
              </MedlineCitation>
            </PubmedArticle>
            """
        )

    monkeypatch.setattr(step3, "fetch_pubmed_xml", _fake_fetch_pubmed)

    batch_calls: List[List[str]] = []

    def _fake_fetch_pmc_batch(pmc_ids: Iterable[str], **_: Any) -> Tuple[Dict[str, str], Set[str]]:
        batch_calls.append(list(pmc_ids))
        return {"PMC111": "Text one"}, {"PMC222"}

    monkeypatch.setattr(step3, "fetch_pmc_fulltexts_batch", _fake_fetch_pmc_batch)

    def _fail_single(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("fetch_pmc_fulltext should not be called when batch fetching")

    monkeypatch.setattr(step3, "fetch_pmc_fulltext", _fail_single)

    records: List[Dict[str, Any]] = [
        {"pmid": "1", "id": "record-1", "title": "One"},
        {"pmid": "2", "id": "record-2", "title": "Two"},
    ]
    batch = list(enumerate(records))

    enriched_pairs, failures = step3._process_records_inner(
        batch,
        total=2,
        prefix="[worker-test]",
        rate_limiter=None,
        max_attempts=1,
        retry_wait=0.5,
        entrez_batch_size=5,
    )

    assert batch_calls == [["PMC111", "PMC222"]]

    enriched_map = {index: rec for index, rec in enriched_pairs}
    assert enriched_map[0]["full_text"] == "Text one"
    assert enriched_map[0]["full_text_source"] == "pmc"
    assert enriched_map[1]["full_text"] is None
    assert enriched_map[1]["pdf_processing"]["failure_reason"] == "pmc_missing_body"

    pmc_failures = [failure for failure in failures if failure["reason"] == "pmc_missing_body"]
    assert any(failure["pmcid"] == "PMC222" for failure in pmc_failures)
