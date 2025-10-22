"""Balance theory assignments in a generated ontology payload.

This helper implements the "optimization ontology" stage that follows the
LLM-driven ontology synthesis.  It enforces the competition rule requiring each
theory to reference between one and four supporting publications (three being
the preferred target) and redistributes ambiguous articles to theories with
fewer supporting papers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline.ontology_optimization import optimise_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the ontology JSON file")
    parser.add_argument(
        "--output",
        help=(
            "Optional output path. When omitted the optimisation is applied in-place "
            "and the input file is overwritten."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3,
        help="Preferred number of papers per theory (defaults to 3)",
    )
    parser.add_argument(
        "--minimum",
        type=int,
        default=1,
        help="Minimum allowed number of papers per theory",
    )
    parser.add_argument(
        "--maximum",
        type=int,
        default=4,
        help="Maximum allowed number of papers per theory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = optimise_file(
        args.input,
        args.output,
        target=args.target,
        minimum=args.minimum,
        maximum=args.maximum,
    )

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

