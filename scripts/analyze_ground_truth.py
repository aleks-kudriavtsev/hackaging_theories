"""Validate paper answers against a ground truth dataset and emit a CSV report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from theories_pipeline import question_validation


STATUS_OK = "ok"
STATUS_MISMATCH = "mismatch"
STATUS_MISSING = "missing"


def _build_answer_row(
    entry: question_validation.GroundTruthEntry,
    *,
    row: question_validation.QuestionRow | None,
    status: str,
) -> Mapping[str, str | None]:
    expected_label = question_validation._normalise_answer_label(entry.expected_answer)  # type: ignore[attr-defined]
    actual_text = ""
    actual_label = ""
    paper_name: str | None = None
    theory_id = entry.theory_id

    if row is not None:
        actual_text = row.answers.get(entry.question_id, "")
        actual_label = question_validation._normalise_answer_label(actual_text)  # type: ignore[attr-defined]
        paper_name = row.paper_name
        theory_id = row.theory_id or theory_id

    return {
        "theory_id": theory_id or question_validation.UNSPECIFIED_THEORY,
        "paper_id": entry.paper_id,
        "paper_name": paper_name,
        "question_id": entry.question_id,
        "expected_answer": entry.expected_answer,
        "expected_label": expected_label,
        "actual_answer": actual_text,
        "actual_label": actual_label,
        "status": status,
    }


def _find_matching_row(
    entry: question_validation.GroundTruthEntry,
    *,
    direct_index: Mapping[tuple[str, str], question_validation.QuestionRow],
    paper_index: Mapping[str, Sequence[question_validation.QuestionRow]],
) -> question_validation.QuestionRow | None:
    if entry.theory_id:
        row = direct_index.get((entry.theory_id, entry.paper_id))
        if row is not None:
            return row

    candidates = paper_index.get(entry.paper_id, [])
    if entry.theory_id:
        for candidate in candidates:
            if candidate.theory_id == entry.theory_id:
                return candidate
    if candidates:
        return candidates[0]
    return None


def generate_answer_rows(
    questions: Sequence[question_validation.QuestionRow],
    ground_truth: Sequence[question_validation.GroundTruthEntry],
    report: question_validation.ValidationReport,
) -> List[Mapping[str, str | None]]:
    direct_index, paper_index = question_validation._index_questions(questions)  # type: ignore[attr-defined]

    missing_keys = {
        (entry.theory_id, entry.paper_id, entry.question_id)
        for entry in report.missing_entries
    }
    mismatch_map = {
        (mismatch.theory_id, mismatch.paper_id, mismatch.question_id): mismatch
        for mismatch in report.mismatches
    }

    rows: List[Mapping[str, str | None]] = []
    for entry in ground_truth:
        status = STATUS_OK
        key = (entry.theory_id, entry.paper_id, entry.question_id)
        row = _find_matching_row(entry, direct_index=direct_index, paper_index=paper_index)

        if key in missing_keys:
            status = STATUS_MISSING
            row = None
        else:
            mismatch = mismatch_map.get(
                (
                    row.theory_id if row else entry.theory_id,
                    entry.paper_id,
                    entry.question_id,
                )
            )
            if mismatch is not None:
                status = STATUS_MISMATCH
        rows.append(_build_answer_row(entry, row=row, status=status))
    return rows


def run_analysis(
    questions_path: Path,
    ground_truth_path: Path,
) -> tuple[question_validation.ValidationReport, List[Mapping[str, str | None]]]:
    questions = question_validation.load_questions(questions_path)
    ground_truth = question_validation.load_ground_truth(ground_truth_path)
    report = question_validation.validate(questions, ground_truth)
    rows = generate_answer_rows(questions, ground_truth, report)
    return report, rows


def write_answer_rows(rows: Iterable[Mapping[str, str | None]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theory_id",
        "paper_id",
        "paper_name",
        "question_id",
        "expected_answer",
        "expected_label",
        "actual_answer",
        "actual_label",
        "status",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to the CSV containing paper answers for Q1–Q9.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="CSV or JSON file with ground-truth answers for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the generated analysis CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report, rows = run_analysis(args.questions, args.ground_truth)
    write_answer_rows(rows, args.output)

    print(question_validation.format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
