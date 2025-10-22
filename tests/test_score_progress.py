from __future__ import annotations

import csv
import json
import sys
from math import isclose, log10
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import score_progress
from theories_pipeline.outputs import QUESTION_COLUMNS


def _write_theories_csv(path: Path, *, include_targets: bool = False) -> None:
    rows = [
        {
            "theory_id": "t1",
            "theory_name": "Theory One",
            "number_of_collected_papers": "10",
            "target": "20" if include_targets else None,
            "deficit": "10" if include_targets else None,
        },
        {
            "theory_id": "t2",
            "theory_name": "Theory Two",
            "number_of_collected_papers": "3",
            "target": "" if include_targets else None,
            "deficit": "" if include_targets else None,
        },
    ]
    fieldnames = ["theory_id", "theory_name", "number_of_collected_papers"]
    if include_targets:
        fieldnames.extend(["target", "deficit"])
    else:
        for row in rows:
            row.pop("target", None)
            row.pop("deficit", None)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_questions_csv(path: Path, *, include_confidence: bool = False) -> None:
    fieldnames = [
        "theory_id",
        "paper_url",
        "paper_name",
        "paper_year",
        *QUESTION_COLUMNS,
    ]
    if include_confidence:
        fieldnames.extend(f"{column}_confidence" for column in QUESTION_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            "theory_id": "t1",
            "paper_url": "paper-1",
            "paper_name": "Sample Paper",
            "paper_year": "2024",
        }
        for column in QUESTION_COLUMNS:
            row[column] = "Yes, supported"
            if include_confidence:
                row[f"{column}_confidence"] = "0.80"
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
    assert report["deficits"] == []
    assert report["deficit_data_available"] is False
    assert not report["alerts"], "No alerts expected with healthy confidences"

    json_path = reports_dir / "progress_report.json"
    markdown_path = reports_dir / "progress_report.md"
    assert json_path.exists()
    assert markdown_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["deficits"] == []
    assert payload["deficit_data_available"] is False
    assert payload["log_score_breakdown"][0]["theory_id"] == "t1"
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Σ log₁₀ Contributions" in markdown_text
    assert "Target Deficits" in markdown_text
    assert "Target/deficit columns not present in export." in markdown_text


def test_generate_progress_report_with_targets_and_confidence(tmp_path: Path) -> None:
    theories_path = tmp_path / "theories.csv"
    questions_path = tmp_path / "questions.csv"
    reports_dir = tmp_path / "reports"

    _write_theories_csv(theories_path, include_targets=True)
    _write_questions_csv(questions_path, include_confidence=True)

    report = score_progress.generate_progress_report(
        theories_path,
        questions_path,
        reports_dir,
        confidence_threshold=0.5,
    )

    assert report["deficit_data_available"] is True
    assert report["deficits"] == [
        {
            "theory_id": "t1",
            "theory_name": "Theory One",
            "target": 20,
            "count": 10,
            "deficit": 10,
        }
    ]
    metrics = report["question_metrics"]["Q1"]
    assert isclose(metrics["average_confidence"], 0.8)  # type: ignore[index]
    assert isclose(metrics["min_confidence"], 0.8)  # type: ignore[index]

    markdown_text = (reports_dir / "progress_report.md").read_text(encoding="utf-8")
    assert "| Theory One | 10 | 20 | 10 |" in markdown_text
