"""Tests for the question validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from theories_pipeline import question_validation


EXAMPLE_QUESTIONS = Path("data/examples/questions.csv")
GROUND_TRUTH = Path("tests/data/questions_ground_truth.csv")


def test_question_validation_matches_examples() -> None:
    report = question_validation.validate_from_paths(EXAMPLE_QUESTIONS, GROUND_TRUTH)

    assert not report.has_failures
    assert report.overall_expected == 6
    assert report.overall_found == 6
    assert report.overall_correct == 6
    assert pytest.approx(report.overall_recall) == 1.0
    assert pytest.approx(report.overall_accuracy) == 1.0

    free_radical = report.per_theory["free-radical-theory"]
    assert free_radical.expected == 2
    assert free_radical.correct == 2
    assert pytest.approx(free_radical.recall) == 1.0
    assert pytest.approx(free_radical.accuracy) == 1.0


def test_question_validation_detects_missing_and_mismatched(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.csv"
    ground_truth.write_text(
        "theory_id,paper_url,question_id,expected_answer\n"
        'free-radical-theory,missing-paper,Q1,"Yes, quantitatively shown"\n'
        "free-radical-theory,paper-fr-001,Q2,Contradicted by evidence\n",
        encoding="utf-8",
    )

    report = question_validation.validate_from_paths(EXAMPLE_QUESTIONS, ground_truth)

    assert report.has_failures
    assert len(report.missing_entries) == 1
    assert report.missing_entries[0].paper_id == "missing-paper"
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.paper_id == "paper-fr-001"
    assert mismatch.question_id == "Q2"
    assert mismatch.expected_label == "Contradicted by evidence"
    assert mismatch.actual_label == "Mechanism supported by experiments"

    metrics = report.per_theory["free-radical-theory"]
    assert metrics.expected == 2
    assert metrics.found == 1
    assert metrics.correct == 0
    assert metrics.missing == 1
    assert metrics.incorrect == 1
