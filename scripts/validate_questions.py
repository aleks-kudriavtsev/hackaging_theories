"""Validate question answers against a ground-truth dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ is None:  # pragma: no cover - convenience for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from theories_pipeline import question_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate questions.csv exports against labelled ground-truth answers.",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to the questions.csv file produced by the pipeline.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="CSV or JSON file with expected answers for specific papers and questions.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path to write the validation report as JSON.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return a zero exit code even when mismatches or missing entries are detected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = question_validation.validate_from_paths(args.questions, args.ground_truth)
    print(question_validation.format_report(report))

    if args.report:
        question_validation.write_report(report, args.report)

    if report.has_failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

