from __future__ import annotations

import csv
from pathlib import Path

from theories_pipeline.extraction import QuestionAnswer
from theories_pipeline.literature import PaperMetadata, PaperSection
from theories_pipeline.outputs import (
    COMPETITION_PAPER_COLUMNS,
    COMPETITION_QUESTION_COLUMNS,
    COMPETITION_THEORY_COLUMNS,
    COMPETITION_THEORY_PAPER_COLUMNS,
    QUESTION_COLUMNS,
    QUESTION_CONFIDENCE_COLUMNS,
    export_competition_papers,
    export_competition_question_answers,
    export_competition_theories,
    export_competition_theory_papers,
    export_papers,
    export_question_answers,
    export_theories,
    export_theory_papers,
)
from theories_pipeline.ontology import TheoryOntology
from theories_pipeline.theories import TheoryAssignment, aggregate_theory_assignments


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
    ontology = TheoryOntology.from_targets_config({"Activity Theory": {}})
    aggregation = aggregate_theory_assignments(assignments, ontology)

    paper_path = export_papers(papers, tmp_path / "papers.csv")
    theory_path = export_theories(aggregation, tmp_path / "theories.csv")
    theory_papers_path = export_theory_papers(aggregation, papers, tmp_path / "theory_papers.csv")
    questions_path = export_question_answers(
        answers,
        papers,
        aggregation,
        tmp_path / "questions.csv",
    )
    competition_papers_path = export_competition_papers(papers, tmp_path / "papers_competition.csv")
    competition_theories_path = export_competition_theories(
        aggregation, tmp_path / "theories_competition.csv"
    )
    competition_theory_papers_path = export_competition_theory_papers(
        aggregation, papers, tmp_path / "theory_papers_competition.csv"
    )
    competition_questions_path = export_competition_question_answers(
        answers,
        papers,
        aggregation,
        tmp_path / "questions_competition.csv",
    )

    for path in [
        paper_path,
        theory_path,
        theory_papers_path,
        questions_path,
        competition_papers_path,
        competition_theories_path,
        competition_theory_papers_path,
        competition_questions_path,
    ]:
        assert path.exists()
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
            assert len(rows) >= 2
    with paper_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
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
        ]
        first = next(reader)
        assert first["full_text"] == "Full text body"
        assert first["citation_count"] == "42"
        assert first["is_review"] == "true"
        assert first["influential_citations"] == "W1; W2"

    with theory_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "theory_id",
            "theory_name",
            "number_of_collected_papers",
        ]
        row = next(reader)
        assert row["theory_id"].startswith("activity-theory")
        assert row["number_of_collected_papers"] == "1"

    with theory_papers_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["theory_id", "paper_url", "paper_name", "paper_year"]
        row = next(reader)
        assert row["paper_url"] == "p1"
        assert row["paper_name"] == "Sample"

    with questions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "theory_id",
            "paper_url",
            "paper_name",
            "paper_year",
            *QUESTION_COLUMNS,
            *QUESTION_CONFIDENCE_COLUMNS,
        ]
        row = next(reader)
        assert row["Q1"] == "Answer"
        assert row["Q1_confidence"] == "0.75"

    with competition_papers_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_PAPER_COLUMNS)
        row = next(reader)
        assert row["paper_id"] == "p1"
        assert row["paper_title"] == "Sample"

    with competition_theories_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_COLUMNS)
        row = next(reader)
        assert row["paper_count"] == "1"

    with competition_theory_papers_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_PAPER_COLUMNS)
        row = next(reader)
        assert row["paper_id"] == "p1"

    with competition_questions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_QUESTION_COLUMNS)
        row = next(reader)
        assert row["question_id"] == "Q1"
        assert row["answer"] == "Answer"
        assert row["confidence"] == "0.75"
        fake_ground_truth = [
            {
                "theory_id": row["theory_id"],
                "paper_id": row["paper_id"],
                "question_id": row["question_id"],
                "expected_answer": row["answer"],
            }
        ]
        assert fake_ground_truth[0]["expected_answer"] == "Answer"


def test_aggregate_theory_assignments_selects_highest_confidence() -> None:
    ontology = TheoryOntology.from_targets_config(
        {
            "Root Theory": {
                "subtheories": {
                    "Branch": {
                        "subtheories": {
                            "Deep Leaf": {},
                        }
                    },
                    "Shallow Leaf": {},
                }
            }
        }
    )
    assignments = [
        TheoryAssignment("p1", "Shallow Leaf", 0.8, depth=1),
        TheoryAssignment("p1", "Deep Leaf", 0.9, depth=2),
        TheoryAssignment("p1", "Root Theory", 0.9, depth=0),
        TheoryAssignment("p2", "Shallow Leaf", 0.5, depth=1),
        TheoryAssignment("p2", "Deep Leaf", 0.5, depth=2),
    ]

    aggregation = aggregate_theory_assignments(assignments, ontology)

    deep_leaf_id = aggregation.theory_ids_by_name["Deep Leaf"]
    assert aggregation.paper_to_theory_ids["p1"] == (deep_leaf_id,)
    assert aggregation.paper_to_theory_ids["p2"] == (deep_leaf_id,)

    assigned_pairs = {
        (theory_id, paper_id)
        for theory_id, paper_ids in ((entry.theory_id, entry.paper_ids) for entry in aggregation.theories)
        for paper_id in paper_ids
    }
    assert (deep_leaf_id, "p1") in assigned_pairs
    assert (deep_leaf_id, "p2") in assigned_pairs
    # Ensure that no paper appears in more than one theory.
    papers_with_multiple = [
        paper_id for paper_id, ids in aggregation.paper_to_theory_ids.items() if len(ids) > 1
    ]
    assert papers_with_multiple == []


def test_aggregate_theory_assignments_filters_pre_grouped_ids() -> None:
    ontology = TheoryOntology.from_targets_config(
        {
            "Root Theory": {
                "subtheories": {
                    "Branch": {
                        "subtheories": {
                            "Deep Leaf": {},
                        }
                    },
                    "Shallow Leaf": {},
                }
            }
        }
    )

    assignments = [
        TheoryAssignment("p1", "Shallow Leaf", 0.8, depth=1),
        TheoryAssignment("p1", "Deep Leaf", 0.9, depth=2),
        TheoryAssignment("p2", "Shallow Leaf", 0.7, depth=1),
        TheoryAssignment("p2", "Deep Leaf", 0.6, depth=2),
    ]

    pre_grouped = {
        "Deep Leaf": ["p1", "p2"],
        "Shallow Leaf": ["p1", "p2"],
    }

    aggregation = aggregate_theory_assignments(
        assignments,
        ontology,
        paper_ids_by_theory=pre_grouped,
    )

    deep_leaf_id = aggregation.theory_ids_by_name["Deep Leaf"]
    shallow_leaf_id = aggregation.theory_ids_by_name["Shallow Leaf"]

    assert aggregation.paper_to_theory_ids == {
        "p1": (deep_leaf_id,),
        "p2": (shallow_leaf_id,),
    }

    assert aggregation.theory_index[deep_leaf_id].paper_ids == ("p1",)
    assert aggregation.theory_index[shallow_leaf_id].paper_ids == ("p2",)
