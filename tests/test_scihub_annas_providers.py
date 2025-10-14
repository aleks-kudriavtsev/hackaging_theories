import pytest
from types import SimpleNamespace
from typing import Any, Dict

from theories_pipeline import literature
from theories_pipeline.literature import (
    AnnasArchiveProvider,
    ProviderConfig,
    SciHubProvider,
)


class DummyResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.headers: Dict[str, str] = {"content-type": "application/json"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(self, payloads: Dict[Any, Dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[Dict[str, Any]] = []

    def get(
        self,
        url: str,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> DummyResponse:
        params = dict(params or {})
        cursor = params.get("cursor")
        payload = self.payloads.get(cursor)
        if payload is None:  # pragma: no cover - defensive guard
            raise AssertionError(f"Unexpected cursor {cursor!r}")
        self.calls.append({"url": url, "params": params, "headers": dict(headers or {})})
        return DummyResponse(payload)


def _patch_requests(
    monkeypatch: pytest.MonkeyPatch,
    payloads: Dict[Any, Dict[str, Any]],
):
    dummy_session = DummySession(payloads)
    stub = SimpleNamespace(Session=lambda: dummy_session, RequestException=Exception)
    monkeypatch.setattr(literature, "requests", stub)

    recorder: Dict[str, Any] = {}

    def fake_full_text(self, identifier: str, doi: str | None, candidates: Any) -> str:
        recorder["last"] = (identifier, doi, tuple(candidates))
        return "resolved text"

    monkeypatch.setattr(literature.BaseProvider, "_resolve_full_text", fake_full_text)
    return dummy_session, recorder


def test_scihub_provider_parses_rapidapi_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        None: {
            "result": [
                {
                    "title": "Aging biomarkers from Sci-Hub",
                    "doi": "10.1000/demo",
                    "authors": ["Doe, J.", "Smith, A."],
                    "year": 2021,
                    "abstract": "Test abstract",
                    "fullTextUrl": "https://example.org/text.txt",
                    "pdfUrl": "https://example.org/paper.pdf",
                    "url": "https://sci-hub.se/10.1000/demo",
                }
            ],
            "next": "cursor123",
        },
        "cursor123": {"result": [], "next": None},
    }
    _session, recorder = _patch_requests(monkeypatch, payloads)
    config = ProviderConfig(
        name="scihub",
        type="scihub",
        api_key="rapid-key",
        base_url="https://example-rapidapi.test/search",
        batch_size=20,
        rate_limit_per_sec=None,
        extra={
            "mode": "rapidapi",
            "rapidapi_host": "example-rapidapi.test",
            "fulltext_keys": ["fullTextUrl", "pdfUrl", "url"],
        },
    )
    provider = SciHubProvider(config)
    first_page = provider.fetch_page("aging", cursor=None)
    assert len(first_page.papers) == 1
    paper = first_page.papers[0]
    assert paper.identifier.endswith("10.1000/demo")
    assert paper.doi == "10.1000/demo"
    assert paper.full_text == "resolved text"
    assert recorder["last"][1] == "10.1000/demo"
    assert recorder["last"][2][0] == "https://example.org/text.txt"
    assert first_page.next_cursor == "cursor123"

    second_page = provider.fetch_page("aging", cursor=first_page.next_cursor)
    assert second_page.papers == []
    assert second_page.exhausted


def test_annas_archive_provider_collects_links(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        None: {
            "results": [
                {
                    "title": "Longevity library entry",
                    "md5": "abc123",
                    "doi": "10.2000/longevity",
                    "authors": "Doe;Roe",
                    "year": "2019",
                    "description": "A description",
                    "mirrors": ["https://example.org/mirror"],
                    "files": [{"url": "https://example.org/file"}],
                }
            ],
            "cursor": "next-cursor",
        },
        "next-cursor": {"results": [], "cursor": None},
    }
    _session, recorder = _patch_requests(monkeypatch, payloads)
    config = ProviderConfig(
        name="annas",
        type="annas_archive",
        api_key="rapid-key",
        base_url="https://annas.test/search",
        batch_size=25,
        rate_limit_per_sec=None,
        extra={
            "rapidapi_host": "annas.test",
            "link_keys": ["mirrors", "files"],
        },
    )
    provider = AnnasArchiveProvider(config)
    first_page = provider.fetch_page("longevity", cursor=None)
    assert len(first_page.papers) == 1
    paper = first_page.papers[0]
    assert paper.identifier == "abc123"
    assert paper.authors == ("Doe", "Roe")
    assert paper.doi == "10.2000/longevity"
    assert paper.full_text == "resolved text"
    assert "https://example.org/mirror" in recorder["last"][2]
    assert "https://example.org/file" in recorder["last"][2]
    assert first_page.next_cursor == "next-cursor"

    second_page = provider.fetch_page("longevity", cursor=first_page.next_cursor)
    assert second_page.papers == []
    assert second_page.exhausted
