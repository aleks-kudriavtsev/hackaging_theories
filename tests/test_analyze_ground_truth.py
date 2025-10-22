from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import analyze_ground_truth

EXAMPLE_QUESTIONS = PROJECT_ROOT / "data/examples/questions.csv"
GROUND_TRUTH = PROJECT_ROOT / "tests/data/questions_ground_truth.csv"


def test_run_analysis_matches_ground_truth(tmp_path: Path) -> None:
    report, rows = analyze_ground_truth.run_analysis(EXAMPLE_QUESTIONS, GROUND_TRUTH)

    assert not report.has_failures
    assert len(rows) == 6
    assert {row["status"] for row in rows} == {analyze_ground_truth.STATUS_OK}

    output = tmp_path / "answers.csv"
    analyze_ground_truth.write_answer_rows(rows, output)

    with output.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        statuses = {row["status"] for row in reader}
    assert statuses == {analyze_ground_truth.STATUS_OK}


def test_run_analysis_marks_missing_entries(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.csv"
    ground_truth.write_text(
        "theory_id,paper_url,question_id,expected_answer\n"
        "free-radical-theory,missing-paper,Q1,Yes\n",
        encoding="utf-8",
    )

    report, rows = analyze_ground_truth.run_analysis(EXAMPLE_QUESTIONS, ground_truth)

    assert report.has_failures
    assert report.missing_entries
    assert any(row["status"] == analyze_ground_truth.STATUS_MISSING for row in rows)
