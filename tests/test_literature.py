from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

import theories_pipeline.literature as literature
from theories_pipeline.literature import (
    BaseProvider,
    LiteratureRetriever,
    PaperMetadata,
    ProviderPage,
    ProviderConfig,
    OpenAlexProvider,
    PubMedProvider,
)


class _StaticProvider:
    def __init__(self, name: str, papers: list[PaperMetadata]):
        self.name = name
        self._papers = list(papers)
        self.query_shards = ("{query}",)

    def fetch_page(self, query: str, cursor: str | None = None) -> ProviderPage:
        return ProviderPage(papers=list(self._papers), next_cursor=None, exhausted=True)


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to raise in tests
        return None


def _write_seed(tmp_path: Path) -> Path:
    data = [
        {
            "identifier": "p1",
            "title": "Activity engagement study",
            "authors": ["Jane Doe"],
            "abstract": "We discuss activity theory in older adults.",
            "source": "Seed",
            "year": 2022,
            "doi": "10.1000/test",
            "citation_count": 25,
            "is_review": True,
            "influential_citations": ["W1"],
        },
        {
            "identifier": "p2",
            "title": "Other topic",
            "authors": ["John Doe"],
            "abstract": "Unrelated abstract.",
            "source": "Seed",
            "year": 2021,
            "citation_count": 2,
            "is_review": False,
        },
        {
            "identifier": "p3",
            "title": "Engagement trends",
            "authors": ["Alex Smith"],
            "abstract": "Primary research report.",
            "source": "Seed",
            "year": 2020,
            "citation_count": 12,
            "is_review": True,
        },
    ]
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_openalex_provider_uses_query_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_requests = SimpleNamespace(Session=lambda: SimpleNamespace())
    monkeypatch.setattr(literature, "requests", dummy_requests, raising=False)
    config = ProviderConfig(
        name="openalex",
        type="openalex",
        api_key="test-openalex",
        batch_size=25,
        extra={"mailto": "contact@example.com"},
    )
    provider = OpenAlexProvider(config)

    captured: dict[str, Any] = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return _DummyResponse({"results": [], "meta": {"next_cursor": None}})

    monkeypatch.setattr(provider, "session", SimpleNamespace(get=fake_get))

    page = provider.fetch_page("aging populations")
    assert captured["params"]["api_key"] == "test-openalex"
    assert captured["params"]["mailto"] == "contact@example.com"
    assert captured["headers"] in (None, {})
    assert page.exhausted is True
    assert page.papers == []


@pytest.fixture
def duplicate_no_doi_papers() -> list[PaperMetadata]:
    return [
        PaperMetadata(
            identifier="prov-a:001",
            title="Understanding Cognitive Aging",
            authors=["Alice B. Smith", "John Doe"],
            abstract="",
            source="prov-a",
        ),
        PaperMetadata(
            identifier="prov-b:xyz",
            title="Understanding Cognitive Aging",
            authors=["john doe", "Alice B Smith"],
            abstract="",
            source="prov-b",
        ),
    ]


@pytest.fixture
def pubmed_sample_payload() -> dict[str, Any]:
    path = Path("tests/fixtures/pubmed_sample.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def pubmed_efetch_xml() -> str:
    path = Path("tests/fixtures/pubmed_efetch.xml")
    return path.read_text(encoding="utf-8")


def test_search_returns_matching_papers(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(_write_seed(tmp_path))
    results = retriever.search("activity theory")
    assert len(results) == 1
    assert results[0].identifier == "p1"


def test_search_respects_limit(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(_write_seed(tmp_path))
    results = retriever.search("", limit=1)
    assert len(results) == 1


def test_search_uses_providers(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(_write_seed(tmp_path))

    def provider(query: str, limit: int | None) -> list[PaperMetadata]:
        return [
            PaperMetadata(
                identifier="p3",
                title="Digital engagement",
                authors=["Someone"],
                abstract="",
                full_text="Discussing engagement theory in detail",
                source="Provider",
            )
        ]

    results = retriever.search("engagement", providers=[provider])
    identifiers = {paper.identifier for paper in results}
    assert {"p1", "p3"}.issubset(identifiers)


def test_missing_seed_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        LiteratureRetriever(missing)


def test_collect_queries_filters_by_citation_count(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(_write_seed(tmp_path))
    result = retriever.collect_queries(
        ["activity"],
        target=None,
        providers=None,
        state_key=None,
        resume=False,
        min_citation_count=10,
    )
    identifiers = {paper.identifier for paper in result.papers}
    assert identifiers == {"p1", "p3"}


def test_collect_queries_sorts_reviews_first(tmp_path: Path) -> None:
    retriever = LiteratureRetriever(_write_seed(tmp_path))
    result = retriever.collect_queries(
        [""],
        target=None,
        providers=None,
        state_key=None,
        resume=False,
        prefer_reviews=True,
        sort_by_citations=True,
    )
    ordered = [paper.identifier for paper in result.papers[:3]]
    assert ordered == ["p1", "p3", "p2"]


def test_collect_queries_deduplicates_no_doi_records(
    tmp_path: Path, duplicate_no_doi_papers: list[PaperMetadata]
) -> None:
    state_dir = tmp_path / "state"
    retriever = LiteratureRetriever(None, state_dir=state_dir)
    retriever.providers = [
        _StaticProvider("prov-a", [duplicate_no_doi_papers[0]]),
        _StaticProvider("prov-b", [duplicate_no_doi_papers[1]]),
    ]

    result = retriever.collect_queries(
        ["cognitive aging"],
        target=None,
        providers=None,
        state_key="dedupe",  # persist across runs
        resume=True,
    )
    matching_titles = [
        paper for paper in result.papers if paper.title == "Understanding Cognitive Aging"
    ]
    assert len(matching_titles) == 1
    assert result.newly_added == 1

    stored_state = retriever.state_store.get("dedupe")
    assert stored_state["seen_canonical_keys"]

    repeat_result = retriever.collect_queries(
        ["cognitive aging"],
        target=None,
        providers=None,
        state_key="dedupe",
        resume=True,
    )
    repeat_titles = [
        paper for paper in repeat_result.papers if paper.title == "Understanding Cognitive Aging"
    ]
    assert len(repeat_titles) == 1
    assert repeat_result.newly_added == 0


def test_pubmed_fetch_page_parses_true_abstract(
    monkeypatch: pytest.MonkeyPatch,
    pubmed_sample_payload: dict[str, Any],
    pubmed_efetch_xml: str,
) -> None:
    class _Session:
        def __init__(self) -> None:
            self.get = lambda *args, **kwargs: None  # patched below

    session = _Session()

    class _RequestsStub:
        RequestException = Exception

        def Session(self) -> _Session:
            return session

    monkeypatch.setattr(literature, "requests", _RequestsStub())

    config = ProviderConfig(name="pubmed", type="pubmed", batch_size=10)
    provider = PubMedProvider(config)
    monkeypatch.setattr(provider.rate_limiter, "wait", lambda: None)

    class _FakeResponse:
        def __init__(
            self,
            *,
            json_data: dict[str, Any] | None = None,
            text_data: str = "",
            status_code: int = 200,
        ) -> None:
            self._json = json_data
            self._text = text_data
            self.status_code = status_code

        def json(self) -> dict[str, Any]:
            if self._json is None:
                raise ValueError("No JSON payload available")
            return self._json

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"status={self.status_code}")

        @property
        def text(self) -> str:
            return self._text

    responses = [
        _FakeResponse(json_data=pubmed_sample_payload["search"]),
        _FakeResponse(json_data=pubmed_sample_payload["summary"]),
        _FakeResponse(text_data=pubmed_efetch_xml),
    ]

    def fake_get(
        url: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> _FakeResponse:
        assert responses, "Unexpected PubMed request"
        return responses.pop(0)

    monkeypatch.setattr(provider.session, "get", fake_get)

    page = provider.fetch_page("activity engagement")

    assert len(page.papers) == 2
    first, second = page.papers
    assert (
        first.abstract
        == "BACKGROUND: Older adults benefit from activity engagement.\n"
        "We evaluated multiple community-based interventions."
    )
    assert second.abstract == ""
    assert (
        second.abstract
        != pubmed_sample_payload["summary"]["result"]["23456789"]["elocationid"]
    )


def test_resolve_full_text_extracts_pdf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("pdfminer.high_level")

    pdf_bytes = (Path("tests/fixtures/hello.pdf")).read_bytes()

    class _Response:
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:  # pragma: no cover - always succeeds
            return None

        @property
        def content(self) -> bytes:
            return pdf_bytes

    class _Session:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url: str, headers: dict[str, str], timeout: float | None) -> _Response:
            self.calls += 1
            return _Response()

    cache_dir = tmp_path / "cache"
    session = _Session()

    class _RequestsStub:
        def Session(self) -> _Session:
            return session

    monkeypatch.setattr(literature, "requests", _RequestsStub())
    config = ProviderConfig(name="test", type="static", extra={"fulltext_cache_dir": str(cache_dir)})
    provider = BaseProvider(config)
    provider.session = session  # type: ignore[assignment]

    url = "https://example.test/paper.pdf"
    text = provider._resolve_full_text("paper-1", None, [url])
    assert "Hello from PDF fixture" in text
    assert "Second line of text" in text
    cache_file = provider._fulltext_cache_path("paper-1", None)
    assert cache_file.exists()
    cached_text = cache_file.read_text(encoding="utf-8")
    assert cached_text == text

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("cache should satisfy second call")

    provider.session.get = _fail  # type: ignore[assignment]
    cached_again = provider._resolve_full_text("paper-1", None, [url])
    assert cached_again == text
