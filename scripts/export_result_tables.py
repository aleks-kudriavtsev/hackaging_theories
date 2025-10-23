#!/usr/bin/env python3
"""Generate Hackaging deliverable tables from pipeline CSV outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

import sys

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline.outputs import QUESTION_COLUMNS
from theories_pipeline.result_tables import (
    prepare_collected_papers,
    prepare_normalised_answers,
    prepare_theory_table,
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theories",
        type=Path,
        required=True,
        help="Path to the theories.csv file produced by the pipeline.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to the questions.csv file produced by the pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results"),
        help="Directory where the formatted tables should be written.",
    )
    parser.add_argument(
        "--theories-output",
        type=str,
        default="aging_theories_table.csv",
        help="Filename for the theories summary table.",
    )
    parser.add_argument(
        "--papers-output",
        type=str,
        default="collected_papers_table.csv",
        help="Filename for the collected papers table.",
    )
    parser.add_argument(
        "--answers-output",
        type=str,
        default="question_answers_table.csv",
        help="Filename for the detailed question answers table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    theories_rows = _load_csv(args.theories)
    questions_rows = _load_csv(args.questions)

    theories_table = prepare_theory_table(theories_rows)
    papers_table = prepare_collected_papers(questions_rows)
    answers_table = prepare_normalised_answers(questions_rows)

    output_dir = args.output_dir

    theories_path = output_dir / args.theories_output
    _write_csv(theories_path, ("theory_id", "theory_name", "number_of_collected_papers"), theories_table)

    papers_path = output_dir / args.papers_output
    _write_csv(papers_path, ("theory_id", "paper_url", "paper_name", "paper_year"), papers_table)

    answers_path = output_dir / args.answers_output
    _write_csv(
        answers_path,
        ("theory_id", "paper_url", "paper_name", "paper_year", *QUESTION_COLUMNS),
        answers_table,
    )

    print(f"Wrote {len(theories_table)} theory rows to {theories_path}")
    print(f"Wrote {len(papers_table)} theory-paper rows to {papers_path}")
    print(f"Wrote {len(answers_table)} question rows to {answers_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
