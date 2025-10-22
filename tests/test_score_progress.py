from __future__ import annotations

import csv
import json
from math import isclose, log10
from pathlib import Path

from scripts import score_progress
from theories_pipeline.outputs import QUESTION_COLUMNS, QUESTION_CONFIDENCE_COLUMNS


def _write_theories_csv(path: Path) -> None:
    rows = [
        {
            "theory_id": "t1",
            "theory_name": "Theory One",
            "number_of_collected_papers": "10",
            "target": "20",
            "deficit": "10",
        },
        {
            "theory_id": "t2",
            "theory_name": "Theory Two",
            "number_of_collected_papers": "3",
            "target": "",
            "deficit": "",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["theory_id", "theory_name", "number_of_collected_papers", "target", "deficit"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_questions_csv(path: Path) -> None:
    fieldnames = [
        "theory_id",
        "paper_url",
        "paper_name",
        "paper_year",
        *QUESTION_COLUMNS,
        *QUESTION_CONFIDENCE_COLUMNS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "theory_id": "t1",
            "paper_url": "paper-1",
            "paper_name": "Sample Paper",
            "paper_year": "2024",
        }
        for column, confidence in zip(QUESTION_COLUMNS, QUESTION_CONFIDENCE_COLUMNS):
            row[column] = "Yes, supported"
            row[confidence] = "0.80"
        writer.writerow(row)


def test_generate_progress_report(tmp_path: Path) -> None:
    theories_path = tmp_path / "theories.csv"
    questions_path = tmp_path / "questions.csv"
    reports_dir = tmp_path / "reports"

    _write_theories_csv(theories_path)
    _write_questions_csv(questions_path)

    report = score_progress.generate_progress_report(
        theories_path,
        questions_path,
        reports_dir,
        confidence_threshold=0.5,
    )

    expected_log_score = log10(10) + log10(3)
    assert isclose(report["log_score"], expected_log_score)
    assert report["theory_count"] == 2
    assert report["question_row_count"] == 1
    breakdown = report["log_score_breakdown"]
    assert len(breakdown) == 2
    assert breakdown[0]["theory_id"] == "t1"
    assert isclose(breakdown[0]["log10_share"], log10(10) / expected_log_score)
    assert report["deficits"] == [
        {
            "theory_id": "t1",
            "theory_name": "Theory One",
            "target": 20,
            "count": 10,
            "deficit": 10,
        }
    ]
    assert not report["alerts"], "No alerts expected with healthy confidences"

    json_path = reports_dir / "progress_report.json"
    markdown_path = reports_dir / "progress_report.md"
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["deficits"][0]["theory_id"] == "t1"
    assert payload["log_score_breakdown"][0]["theory_id"] == "t1"
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Σ log₁₀ Contributions" in markdown_text
    assert "Target Deficits" in markdown_text
