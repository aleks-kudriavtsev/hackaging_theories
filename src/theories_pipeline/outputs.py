"""CSV export helpers for the Hackaging theories pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

from .extraction import QuestionAnswer
from .literature import PaperMetadata
from .theories import AggregatedTheory, TheoryAggregationResult


QUESTION_COLUMNS: Sequence[str] = tuple(f"Q{i}" for i in range(1, 10))


def export_papers(papers: Iterable[PaperMetadata], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "identifier",
                "title",
                "authors",
                "abstract",
                "full_text",
                "sections",
                "source",
                "year",
                "doi",
                "citation_count",
                "is_review",
                "influential_citations",
            ],
        )
        writer.writeheader()
        for paper in papers:
            writer.writerow(
                {
                    "identifier": paper.identifier,
                    "title": paper.title,
                    "authors": "; ".join(paper.authors),
                    "abstract": paper.abstract,
                    "full_text": paper.full_text,
                    "sections": json.dumps(
                        [section.to_dict() for section in paper.sections], ensure_ascii=False
                    )
                    if paper.sections
                    else "",
                    "source": paper.source,
                    "year": paper.year if paper.year is not None else "",
                    "doi": paper.doi if paper.doi is not None else "",
                    "citation_count": (
                        str(paper.citation_count) if paper.citation_count is not None else ""
                    ),
                    "is_review": (
                        "true"
                        if paper.is_review is True
                        else "false" if paper.is_review is False else ""
                    ),
                    "influential_citations": "; ".join(paper.influential_citations),
                }
            )
    return path


def export_theories(
    aggregation: TheoryAggregationResult,
    path: Path,
    *,
    sort_by_count: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "theory_id",
                "theory_name",
                "number_of_collected_papers",
            ],
        )
        writer.writeheader()
        theories: Sequence[AggregatedTheory]
        if sort_by_count:
            theories = sorted(
                aggregation.theories,
                key=lambda item: (
                    -item.number_of_collected_papers,
                    item.theory_name.lower(),
                ),
            )
        else:
            theories = aggregation.theories
        for theory in theories:
            writer.writerow(
                {
                    "theory_id": theory.theory_id,
                    "theory_name": theory.theory_name,
                    "number_of_collected_papers": theory.number_of_collected_papers,
                }
            )
    return path


def _paper_lookup(papers: Mapping[str, PaperMetadata] | Iterable[PaperMetadata]) -> Mapping[str, PaperMetadata]:
    if isinstance(papers, Mapping):
        return papers
    return {paper.identifier: paper for paper in papers}


def export_theory_papers(
    aggregation: TheoryAggregationResult,
    papers: Mapping[str, PaperMetadata] | Iterable[PaperMetadata],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _paper_lookup(papers)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["theory_id", "paper_url", "paper_name", "paper_year"],
        )
        writer.writeheader()
        for theory in aggregation.theories:
            for paper_id in theory.paper_ids:
                paper = lookup.get(paper_id)
                if paper is None:
                    continue
                writer.writerow(
                    {
                        "theory_id": theory.theory_id,
                        "paper_url": paper.identifier,
                        "paper_name": paper.title,
                        "paper_year": paper.year if paper.year is not None else "",
                    }
                )
    return path


def export_question_answers(
    answers: Iterable[QuestionAnswer],
    papers: Mapping[str, PaperMetadata] | Iterable[PaperMetadata],
    aggregation: TheoryAggregationResult,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _paper_lookup(papers)
    answers_by_paper: Dict[str, Dict[str, tuple[float, str]]] = {}
    for answer in answers:
        paper_answers = answers_by_paper.setdefault(answer.paper_id, {})
        current = paper_answers.get(answer.question_id)
        if current is None or answer.confidence > current[0]:
            paper_answers[answer.question_id] = (answer.confidence, answer.answer)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "theory_id",
                "paper_url",
                "paper_name",
                "paper_year",
                *QUESTION_COLUMNS,
            ],
        )
        writer.writeheader()
        for paper_id, theory_ids in aggregation.paper_to_theory_ids.items():
            paper = lookup.get(paper_id)
            if paper is None:
                continue
            paper_answers = answers_by_paper.get(paper_id, {})
            for theory_id in theory_ids:
                row = {
                    "theory_id": theory_id,
                    "paper_url": paper.identifier,
                    "paper_name": paper.title,
                    "paper_year": paper.year if paper.year is not None else "",
                }
                for question_id in QUESTION_COLUMNS:
                    answer_entry = paper_answers.get(question_id)
                    row[question_id] = answer_entry[1] if answer_entry else ""
                writer.writerow(row)
    return path
