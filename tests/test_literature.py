from __future__ import annotations

import json
from pathlib import Path

import pytest

from theories_pipeline.literature import LiteratureRetriever, PaperMetadata


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
