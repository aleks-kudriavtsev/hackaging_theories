from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from theories_pipeline.literature import (
    BaseProvider,
    LiteratureRetriever,
    PaperMetadata,
    ProviderConfig,
    ProviderPage,
    SemanticScholarProvider,
    SerpApiScholarProvider,
)


@pytest.fixture(autouse=True)
def _stub_requests(monkeypatch):
    dummy_requests = SimpleNamespace(Session=lambda: SimpleNamespace())
    monkeypatch.setattr("theories_pipeline.literature.requests", dummy_requests, raising=False)


class _DummyResponse:
    def __init__(self, payload: dict, *, headers: dict | None = None):
        self._payload = payload
        self.headers = headers or {"content-type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):  # pragma: no cover - nothing to raise in tests
        return None


def test_serpapi_provider_parses_response(monkeypatch):
    config = ProviderConfig(
        name="serpapi_scholar",
        type="serpapi_scholar",
        api_key="test-key",
        batch_size=10,
    )
    provider = SerpApiScholarProvider(config)

    payload = {
        "organic_results": [
            {
                "title": "Cognitive aging trends",
                "result_id": "abc123",
                "link": "https://example.org/paper",
                "snippet": "We discuss new findings.",
                "publication_info": {
                    "authors": [{"name": "Alice"}, {"name": "Bob"}],
                    "year": "2019",
                    "summary": "Journal of Aging, 2019",
                },
                "inline_links": {
                    "cited_by": {"total": 42},
                    "serpapi_scholar_fulltext": [{"link": "https://example.org/fulltext"}],
                    "resources": [{"link": "https://doi.org/10.1000/xyz"}],
                },
            }
        ],
        "serpapi_pagination": {
            "next": "https://serpapi.com/search.json?engine=google_scholar&start=10",
        },
    }

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return _DummyResponse(payload)

    monkeypatch.setattr(provider, "session", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(BaseProvider, "_resolve_full_text", lambda *args, **kwargs: "cached text")

    page = provider.fetch_page("cognitive aging")
    assert captured["params"]["api_key"] == "test-key"
    assert page.next_cursor == "10"
    assert page.exhausted is False
    assert len(page.papers) == 1
    paper = page.papers[0]
    assert paper.identifier.startswith("serpapi:")
    assert paper.title == "Cognitive aging trends"
    assert paper.authors == ("Alice", "Bob")
    assert paper.citation_count == 42
    assert paper.doi == "10.1000/xyz"
    assert paper.full_text == "cached text"


def test_semantic_scholar_provider_parses_response(monkeypatch):
    config = ProviderConfig(
        name="semantic_scholar",
        type="semantic_scholar",
        api_key="sem-key",
        batch_size=50,
    )
    provider = SemanticScholarProvider(config)

    payload = {
        "data": [
            {
                "paperId": "SS123",
                "title": "Neural correlates of aging",
                "abstract": "Detailed abstract text.",
                "authors": [{"name": "Carol"}, {"name": "Dan"}],
                "year": 2020,
                "citationCount": 17,
                "externalIds": {"DOI": "10.2000/test"},
                "openAccessPdf": {"url": "https://example.org/pdf"},
                "influentialCitationIds": ["X1", "X2"],
                "url": "https://www.semanticscholar.org/paper/SS123",
            }
        ],
        "next": 100,
    }

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return _DummyResponse(payload)

    monkeypatch.setattr(provider, "session", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(BaseProvider, "_resolve_full_text", lambda *args, **kwargs: "oa text")

    page = provider.fetch_page("neural aging")
    assert captured["headers"]["x-api-key"] == "sem-key"
    assert page.next_cursor == "100"
    paper = page.papers[0]
    assert paper.identifier == "semanticscholar:SS123"
    assert paper.doi == "10.2000/test"
    assert paper.full_text == "oa text"
    assert paper.citation_count == 17
    assert paper.influential_citations == ("X1", "X2")


def test_scholar_providers_deduplicate_with_state(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(BaseProvider, "_resolve_full_text", lambda *args, **kwargs: "")

    serp_config = ProviderConfig(
        name="serpapi_scholar",
        type="serpapi_scholar",
        api_key="key",
        batch_size=10,
    )
    semantic_config = ProviderConfig(
        name="semantic_scholar",
        type="semantic_scholar",
        api_key="key",
        batch_size=50,
    )

    serp_metadata = PaperMetadata(
        identifier="serpapi:dup",
        title="Shared DOI paper",
        authors=("Author One",),
        abstract="Serp result",
        source="serpapi_scholar",
        doi="10.5555/shared",
    )
    semantic_metadata = PaperMetadata(
        identifier="semanticscholar:dup",
        title="Shared DOI paper",
        authors=("Author One",),
        abstract="Semantic result",
        source="semantic_scholar",
        doi="10.5555/shared",
    )

    def serp_fetch(self, query: str, cursor: str | None = None):
        if cursor:
            return ProviderPage(papers=[], next_cursor=None, exhausted=True)
        return ProviderPage(papers=[serp_metadata], next_cursor=None, exhausted=True)

    def semantic_fetch(self, query: str, cursor: str | None = None):
        if cursor:
            return ProviderPage(papers=[], next_cursor=None, exhausted=True)
        return ProviderPage(papers=[semantic_metadata], next_cursor=None, exhausted=True)

    monkeypatch.setattr(SerpApiScholarProvider, "fetch_page", serp_fetch)
    monkeypatch.setattr(SemanticScholarProvider, "fetch_page", semantic_fetch)

    retriever = LiteratureRetriever(
        None,
        provider_configs=[serp_config, semantic_config],
        state_dir=tmp_path,
    )

    result = retriever.collect_queries(
        ["aging"],
        target=None,
        providers=None,
        state_key="scholar",
        resume=True,
    )
    assert result.newly_added == 1
    assert len([paper for paper in result.papers if paper.doi == "10.5555/shared"]) == 1

    stored_state = retriever.state_store.get("scholar")
    assert stored_state["seen_canonical_keys"]
    assert set(stored_state["provider_totals"].keys()) == {"serpapi_scholar", "semantic_scholar"}

    repeat = retriever.collect_queries(
        ["aging"],
        target=None,
        providers=None,
        state_key="scholar",
        resume=True,
    )
    assert repeat.newly_added == 0
    assert len([paper for paper in repeat.papers if paper.doi == "10.5555/shared"]) == 1
