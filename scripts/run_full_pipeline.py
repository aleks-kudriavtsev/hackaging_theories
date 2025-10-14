"""Run the four-step aging theory literature pipeline end-to-end.

The helper script chains together the standalone step scripts that ship with
this repository:

1. ``step1_pubmed_search.py`` – harvest PubMed review metadata.
2. ``step2_filter_reviews.py`` – filter the records via OpenAI relevance check.
3. ``step3_fetch_fulltext.py`` – download PubMed Central full texts when
   available.
4. ``step4_extract_theories.py`` – extract theory mentions from the curated
   corpus using an OpenAI model.

It simply orchestrates these components so users can kick off the full workflow
with a single command while still retaining the ability to run individual
stages in isolation when debugging or extending the pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def run_command(args: list[str]) -> None:
    """Execute *args* in a subprocess, bubbling up any failures."""

    print("\n>>>", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the four-stage PubMed → theory extraction pipeline"
    )
    parser.add_argument(
        "--query",
        default=None,
        help=(
            "Optional override for the PubMed query (defaults to the built-in "
            "aging theory review search)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/pipeline",
        help="Directory where intermediate JSON artefacts will be written.",
    )
    parser.add_argument(
        "--filter-model",
        default="gpt-4o-mini",
        help="OpenAI model identifier for the relevance filter (step 2).",
    )
    parser.add_argument(
        "--extract-model",
        default="gpt-4o-mini",
        help="OpenAI model identifier for theory extraction (step 4).",
    )
    parser.add_argument(
        "--filter-delay",
        type=float,
        default=0.5,
        help="Seconds to pause between OpenAI requests during filtering.",
    )
    parser.add_argument(
        "--extract-delay",
        type=float,
        default=0.5,
        help="Seconds to pause between OpenAI requests during theory extraction.",
    )
    parser.add_argument(
        "--extract-max-chars",
        type=int,
        default=12000,
        help="Maximum characters of review text to forward to the extractor.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_reviews = output_dir / "start_reviews.json"
    filtered_reviews = output_dir / "filtered_reviews.json"
    fulltext_reviews = output_dir / "filtered_reviews_fulltext.json"
    theories_json = output_dir / "aging_theories.json"

    # Step 1 – PubMed harvest
    step1_cmd = [
        sys.executable,
        str(REPO_ROOT / "step1_pubmed_search.py"),
        "--output",
        str(start_reviews),
    ]
    if args.query:
        step1_cmd.extend(["--query", args.query])
    run_command(step1_cmd)

    # Step 2 – OpenAI relevance filter
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "step2_filter_reviews.py"),
            "--input",
            str(start_reviews),
            "--output",
            str(filtered_reviews),
            "--model",
            args.filter_model,
            "--delay",
            str(args.filter_delay),
        ]
    )

    # Step 3 – PMC full-text enrichment
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "step3_fetch_fulltext.py"),
            "--input",
            str(filtered_reviews),
            "--output",
            str(fulltext_reviews),
        ]
    )

    # Step 4 – Theory extraction
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "step4_extract_theories.py"),
            "--input",
            str(fulltext_reviews),
            "--output",
            str(theories_json),
            "--model",
            args.extract_model,
            "--delay",
            str(args.extract_delay),
            "--max-chars",
            str(args.extract_max_chars),
        ]
    )

    print("\nPipeline completed successfully.")
    print(f"Start reviews: {start_reviews}")
    print(f"Filtered reviews: {filtered_reviews}")
    print(f"Full texts: {fulltext_reviews}")
    print(f"Theory catalogue: {theories_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

