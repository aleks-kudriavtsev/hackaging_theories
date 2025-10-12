from __future__ import annotations

import csv
from pathlib import Path

from theories_pipeline.extraction import QuestionAnswer
from theories_pipeline.literature import PaperMetadata
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
            source="Seed",
            year=2020,
            doi="10.0/doi",
        )
    ]
    assignments = [TheoryAssignment("p1", "Activity Theory", 0.75)]
    answers = [QuestionAnswer("p1", "Q1", "Question", "Answer")]

    paper_path = export_papers(papers, tmp_path / "papers.csv")
    theory_path = export_theories(assignments, tmp_path / "theories.csv")
    questions_path = export_question_answers(answers, tmp_path / "questions.csv")

    for path in [paper_path, theory_path, questions_path]:
        assert path.exists()
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
            assert len(rows) >= 2
