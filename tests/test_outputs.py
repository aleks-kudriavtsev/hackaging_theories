from __future__ import annotations

import csv
from pathlib import Path

from theories_pipeline.extraction import QuestionAnswer
from theories_pipeline.literature import PaperMetadata, PaperSection
from theories_pipeline.outputs import (
    export_papers,
    export_question_answers,
    export_theories,
)
from theories_pipeline.theories import TheoryAssignment


def test_export_functions_create_csv(tmp_path: Path) -> None:
    papers = [
        PaperMetadata(
            identifier="p1",
            title="Sample",
            authors=["Author"],
            abstract="Abstract text",
            full_text="Full text body",
            sections=(PaperSection("Intro", "Section text"),),
            source="Seed",
            year=2020,
            doi="10.0/doi",
            citation_count=42,
            is_review=True,
            influential_citations=("W1", "W2"),
        )
    ]
    assignments = [TheoryAssignment("p1", "Activity Theory", 0.75)]
    answers = [QuestionAnswer("p1", "Q1", "Question", "Answer", 0.75, "Evidence")]

    paper_path = export_papers(papers, tmp_path / "papers.csv")
    theory_path = export_theories(assignments, tmp_path / "theories.csv")
    questions_path = export_question_answers(answers, tmp_path / "questions.csv")

    for path in [paper_path, theory_path, questions_path]:
        assert path.exists()
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
            assert len(rows) >= 2
    with paper_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        assert {"full_text", "sections", "citation_count", "is_review", "influential_citations"}.issubset(header)
        first = next(reader)
        assert first["full_text"] == "Full text body"
        assert first["citation_count"] == "42"
        assert first["is_review"] == "true"
        assert first["influential_citations"] == "W1; W2"
