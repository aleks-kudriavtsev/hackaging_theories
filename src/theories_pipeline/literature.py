"""Utilities for retrieving literature metadata for the Hackaging theories pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class PaperMetadata:
    """Minimal metadata required for downstream classification and extraction."""

    identifier: str
    title: str
    authors: Sequence[str]
    abstract: str
    source: str
    year: int | None = None
    doi: str | None = None


class LiteratureRetriever:
    """Retrieves literature metadata from configured data sources.

    The retriever is intentionally file-based to keep the Hackaging challenge
    reproducible. External API clients can be swapped in by providing a callable
    via ``providers`` in ``search`` during tests or production use.
    """

    def __init__(self, seed_data_path: Path) -> None:
        self.seed_data_path = Path(seed_data_path)
        if not self.seed_data_path.exists():
            raise FileNotFoundError(f"Seed data path {self.seed_data_path} does not exist")

    def _load_seed_papers(self) -> List[PaperMetadata]:
        with self.seed_data_path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
        papers: List[PaperMetadata] = []
        for item in raw_items:
            papers.append(
                PaperMetadata(
                    identifier=item["identifier"],
                    title=item["title"],
                    authors=item.get("authors", []),
                    abstract=item.get("abstract", ""),
                    source=item.get("source", "seed"),
                    year=item.get("year"),
                    doi=item.get("doi"),
                )
            )
        return papers

    def search(
        self,
        query: str,
        limit: int | None = None,
        providers: Iterable[
            Callable[[str, int | None], Iterable[PaperMetadata]]
        ]
        | None = None,
    ) -> List[PaperMetadata]:
        """Search for papers matching the given query.

        The method first yields results from the configured seed dataset to
        guarantee deterministic behavior. Additional providers can be supplied
        to extend the retrieval process (e.g., mocked API responses). Providers
        must be callables accepting ``query`` and ``limit`` and returning an
        iterable of :class:`PaperMetadata`.
        """

        results: List[PaperMetadata] = []
        for paper in self._load_seed_papers():
            if _matches_query(paper, query):
                results.append(paper)
                if limit is not None and len(results) >= limit:
                    return results[:limit]

        if providers:
            for provider in providers:
                for paper in provider(query, limit):
                    if paper not in results:
                        results.append(paper)
                        if limit is not None and len(results) >= limit:
                            return results[:limit]

        return results[:limit] if limit is not None else results


def _matches_query(paper: PaperMetadata, query: str) -> bool:
    terms = [t.strip().lower() for t in query.split() if t.strip()]
    haystacks = " ".join([paper.title, paper.abstract]).lower()
    return all(term in haystacks for term in terms) if terms else True
