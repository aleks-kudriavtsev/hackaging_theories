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
QUESTION_CONFIDENCE_COLUMNS: Sequence[str] = tuple(f"{column}_confidence" for column in QUESTION_COLUMNS)

COMPETITION_PAPER_COLUMNS: Sequence[str] = (
    "paper_id",
    "paper_title",
    "paper_authors",
    "paper_abstract",
    "paper_source",
    "paper_year",
    "paper_doi",
)
COMPETITION_THEORY_COLUMNS: Sequence[str] = (
    "theory_id",
    "theory_name",
    "paper_count",
)
COMPETITION_THEORY_PAPER_COLUMNS: Sequence[str] = (
    "theory_id",
    "paper_id",
    "paper_year",
)
COMPETITION_QUESTION_COLUMNS: Sequence[str] = (
    "theory_id",
    "paper_id",
    "question_id",
    "answer",
    "confidence",
    "evidence",
)


def _format_confidence(value: float | None) -> str:
    if value is None:
        return ""
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".") or "0"


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
    *,
    include_confidence: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _paper_lookup(papers)
    answers_by_paper: Dict[str, Dict[str, tuple[float, str, str]]] = {}
    for answer in answers:
        paper_answers = answers_by_paper.setdefault(answer.paper_id, {})
        current = paper_answers.get(answer.question_id)
        if current is None or answer.confidence > current[0]:
            evidence = answer.evidence or ""
            paper_answers[answer.question_id] = (answer.confidence, answer.answer, evidence)
    fieldnames = [
        "theory_id",
        "paper_url",
        "paper_name",
        "paper_year",
        *QUESTION_COLUMNS,
    ]
    if include_confidence:
        fieldnames.extend(QUESTION_CONFIDENCE_COLUMNS)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                    if answer_entry:
                        confidence, answer_text, _ = answer_entry
                        row[question_id] = answer_text
                        if include_confidence:
                            row[f"{question_id}_confidence"] = _format_confidence(confidence)
                    else:
                        row[question_id] = ""
                        if include_confidence:
                            row[f"{question_id}_confidence"] = ""
                writer.writerow(row)
    return path


def export_competition_papers(
    papers: Iterable[PaperMetadata],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPETITION_PAPER_COLUMNS))
        writer.writeheader()
        for paper in sorted(papers, key=lambda item: item.identifier):
            writer.writerow(
                {
                    "paper_id": paper.identifier,
                    "paper_title": paper.title,
                    "paper_authors": "; ".join(paper.authors),
                    "paper_abstract": paper.abstract,
                    "paper_source": paper.source,
                    "paper_year": paper.year if paper.year is not None else "",
                    "paper_doi": paper.doi if paper.doi is not None else "",
                }
            )
    return path


def export_competition_theories(
    aggregation: TheoryAggregationResult,
    path: Path,
    *,
    sort_by_count: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPETITION_THEORY_COLUMNS))
        writer.writeheader()
        theories: Sequence[AggregatedTheory]
        if sort_by_count:
            theories = sorted(
                aggregation.theories,
                key=lambda item: (-item.number_of_collected_papers, item.theory_name.lower()),
            )
        else:
            theories = aggregation.theories
        for theory in theories:
            writer.writerow(
                {
                    "theory_id": theory.theory_id,
                    "theory_name": theory.theory_name,
                    "paper_count": theory.number_of_collected_papers,
                }
            )
    return path


def export_competition_theory_papers(
    aggregation: TheoryAggregationResult,
    papers: Mapping[str, PaperMetadata] | Iterable[PaperMetadata],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _paper_lookup(papers)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPETITION_THEORY_PAPER_COLUMNS))
        writer.writeheader()
        for theory in aggregation.theories:
            for paper_id in sorted(theory.paper_ids):
                paper = lookup.get(paper_id)
                if paper is None:
                    continue
                writer.writerow(
                    {
                        "theory_id": theory.theory_id,
                        "paper_id": paper.identifier,
                        "paper_year": paper.year if paper.year is not None else "",
                    }
                )
    return path


def export_competition_question_answers(
    answers: Iterable[QuestionAnswer],
    papers: Mapping[str, PaperMetadata] | Iterable[PaperMetadata],
    aggregation: TheoryAggregationResult,
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = _paper_lookup(papers)
    answers_by_paper: Dict[str, Dict[str, tuple[float, str, str]]] = {}
    for answer in answers:
        paper_answers = answers_by_paper.setdefault(answer.paper_id, {})
        current = paper_answers.get(answer.question_id)
        if current is None or answer.confidence > current[0]:
            evidence = answer.evidence or ""
            paper_answers[answer.question_id] = (answer.confidence, answer.answer, evidence)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPETITION_QUESTION_COLUMNS))
        writer.writeheader()
        for paper_id in sorted(aggregation.paper_to_theory_ids):
            paper = lookup.get(paper_id)
            if paper is None:
                continue
            theory_ids = sorted(aggregation.paper_to_theory_ids[paper_id])
            paper_answers = answers_by_paper.get(paper_id, {})
            for theory_id in theory_ids:
                for question_id in QUESTION_COLUMNS:
                    answer_entry = paper_answers.get(question_id)
                    if not answer_entry:
                        continue
                    confidence, answer_text, evidence = answer_entry
                    writer.writerow(
                        {
                            "theory_id": theory_id,
                            "paper_id": paper.identifier,
                            "question_id": question_id,
                            "answer": answer_text,
                            "confidence": _format_confidence(confidence),
                            "evidence": evidence,
                        }
                    )
    return path
