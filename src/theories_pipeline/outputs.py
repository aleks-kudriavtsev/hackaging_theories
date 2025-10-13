"""CSV export helpers for the Hackaging theories pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .extraction import QuestionAnswer
from .literature import PaperMetadata
from .theories import TheoryAssignment


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
                }
            )
    return path


def export_theories(assignments: Iterable[TheoryAssignment], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["paper_id", "theory", "score"])
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(
                {
                    "paper_id": assignment.paper_id,
                    "theory": assignment.theory,
                    "score": f"{assignment.score:.3f}",
                }
            )
    return path


def export_question_answers(answers: Iterable[QuestionAnswer], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "question_id",
                "question",
                "answer",
                "confidence",
                "evidence",
            ],
        )
        writer.writeheader()
        for answer in answers:
            writer.writerow(
                {
                    "paper_id": answer.paper_id,
                    "question_id": answer.question_id,
                    "question": answer.question,
                    "answer": answer.answer,
                    "confidence": f"{answer.confidence:.2f}",
                    "evidence": answer.evidence or "",
                }
            )
    return path
