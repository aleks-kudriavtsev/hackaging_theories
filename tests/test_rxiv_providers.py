from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from theories_pipeline import literature
from theories_pipeline.literature import BioRxivProvider, MedRxivProvider, ProviderConfig


class DummyResponse:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
        self.headers: Dict[str, str] = {"content-type": "application/json"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummySession:
    def __init__(self, payloads: Dict[int, Dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None, **_: Any) -> DummyResponse:
        self.calls.append(url)
        offset = int(url.rstrip("/").split("/")[-1])
        payload = self.payloads[offset]
        return DummyResponse(payload)


def _patch_requests(monkeypatch: pytest.MonkeyPatch, payloads: Dict[int, Dict[str, Any]]) -> None:
    dummy_session = DummySession(payloads)
    stub = SimpleNamespace(Session=lambda: dummy_session, RequestException=Exception)
    monkeypatch.setattr(literature, "requests", stub)
    monkeypatch.setattr(literature.BaseProvider, "_resolve_full_text", lambda *args, **kwargs: "full text")


def test_biorxiv_provider_filters_categories_and_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        0: {
            "messages": [{"count": 2, "total": 3}],
            "collection": [
                {
                    "title": "Aging biomarkers in neuroscience",
                    "authors": "Doe, J.; Roe, J.",
                    "abstract": "We discuss aging.",
                    "doi": "10.1101/2023.01.01.123456",
                    "date": "2023-01-02",
                    "version": "1",
                    "category": "neuroscience",
                    "jatsxml": "https://example.org/paper1.xml",
                },
                {
                    "title": "Unrelated genetics study",
                    "authors": "Smith, A.",
                    "abstract": "Genetics only.",
                    "doi": "10.1101/2023.01.02.987654",
                    "date": "2023-01-02",
                    "version": "1",
                    "category": "genetics",
                },
            ],
        },
        2: {
            "messages": [{"count": 1, "total": 3}],
            "collection": [
                {
                    "title": "Neuroscience pilot study",
                    "authors": "Lee, A.",
                    "abstract": "No mention of the query string here.",
                    "date": "2023-01-03",
                    "version": "1",
                    "category": "neuroscience",
                }
            ],
        },
    }
    _patch_requests(monkeypatch, payloads)
    config = ProviderConfig(
        name="biorxiv",
        type="biorxiv",
        base_url="https://api.biorxiv.org/details/biorxiv",
        batch_size=100,
        rate_limit_per_sec=None,
        extra={
            "date_window": {"from": "2023-01-01", "to": "2023-01-05"},
            "categories": ["neuroscience"],
        },
    )
    provider = BioRxivProvider(config)
    first_page = provider.fetch_page("aging", cursor=None)
    assert len(first_page.papers) == 1
    assert first_page.papers[0].title.startswith("Aging biomarkers")
    assert first_page.next_cursor == "2"
    assert not first_page.exhausted

    second_page = provider.fetch_page("aging", cursor=first_page.next_cursor)
    assert second_page.papers == []
    assert second_page.exhausted


def test_medrxiv_provider_generates_identifiers_without_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        0: {
            "messages": [{"count": 1, "total": 1}],
            "collection": [
                {
                    "title": "Clinical trial in geriatrics",
                    "authors": "Doe, J.; Smith, A.",
                    "abstract": "",
                    "date": "2023-01-04",
                    "version": "2",
                    "category": "geriatrics",
                }
            ],
        }
    }
    _patch_requests(monkeypatch, payloads)
    config = ProviderConfig(
        name="medrxiv",
        type="medrxiv",
        extra={
            "date_window": {"from": "2023-01-01", "to": "2023-01-05"},
            "categories": ["geriatrics"],
        },
    )
    provider = MedRxivProvider(config)
    page = provider.fetch_page("", cursor=None)
    assert len(page.papers) == 1
    paper = page.papers[0]
    assert paper.identifier.startswith("medrxiv:")
    assert paper.doi is None
    assert paper.authors == ("Doe, J.", "Smith, A.")
