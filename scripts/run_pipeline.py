"""Convenience wrapper to execute the aging theory pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class PipelinePaths:
    start_reviews: str
    filtered_reviews: str
    fulltext_reviews: str
    theories: str
    ontology: str


def build_paths(workdir: str) -> PipelinePaths:
    os.makedirs(workdir, exist_ok=True)
    return PipelinePaths(
        start_reviews=os.path.join(workdir, "start_reviews.json"),
        filtered_reviews=os.path.join(workdir, "filtered_reviews.json"),
        fulltext_reviews=os.path.join(workdir, "filtered_reviews_fulltext.json"),
        theories=os.path.join(workdir, "aging_theories.json"),
        ontology=os.path.join(workdir, "aging_ontology.json"),
    )


def run_step(command: List[str], label: str) -> None:
    print(f"\n→ {label}")
    print("  $", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Step '{label}' failed with exit code {exc.returncode}.") from exc


def ensure_env(var: str, step: str) -> None:
    if not os.environ.get(var):
        raise SystemExit(f"{step} requires the environment variable {var} to be set.")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end PubMed → OpenAI aging theory pipeline",
    )
    parser.add_argument(
        "--workdir",
        default="data/pipeline",
        help="Directory to store intermediate and final JSON artefacts.",
    )
    parser.add_argument(
        "--query",
        default='(("aging theory"[TIAB] OR "ageing theory"[TIAB] OR "theories of aging"[TIAB]) AND review[PTYP])',
        help="PubMed query passed to step1_pubmed_search.py.",
    )
    parser.add_argument(
        "--filter-model",
        default="gpt-4o-mini",
        help="OpenAI model used for relevance filtering (step 2).",
    )
    parser.add_argument(
        "--filter-delay",
        type=float,
        default=0.5,
        help="Delay between OpenAI calls during filtering (seconds).",
    )
    parser.add_argument(
        "--theory-model",
        default="gpt-4o-mini",
        help="OpenAI model used for theory extraction (step 4).",
    )
    parser.add_argument(
        "--theory-delay",
        type=float,
        default=0.5,
        help="Delay between OpenAI calls during theory extraction (seconds).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12000,
        help="Maximum characters from each review to send to the LLM (step 4).",
    )
    parser.add_argument(
        "--ontology-model",
        default="gpt-4o-mini",
        help="OpenAI model used for ontology generation (step 5).",
    )
    parser.add_argument(
        "--ontology-top-n",
        type=int,
        default=60,
        help="Maximum number of theories to summarise when prompting step 5.",
    )
    parser.add_argument(
        "--ontology-examples",
        type=int,
        default=3,
        help="Number of representative titles to pass for each theory in step 5.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps even if their output files already exist.",
    )
    return parser.parse_args(argv)


def maybe_skip(path: str, force: bool, label: str) -> bool:
    if not force and os.path.exists(path):
        print(f"Skipping {label}; {path} already exists. Use --force to regenerate.")
        return True
    return False


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    paths = build_paths(args.workdir)

    # Step 1 – PubMed search
    if not maybe_skip(paths.start_reviews, args.force, "step 1"):
        run_step(
            [
                sys.executable,
                "scripts/step1_pubmed_search.py",
                "--query",
                args.query,
                "--output",
                paths.start_reviews,
            ],
            "Collect PubMed review metadata",
        )

    # Step 2 – LLM-based filtering
    ensure_env("OPENAI_API_KEY", "Step 2")
    if not maybe_skip(paths.filtered_reviews, args.force, "step 2"):
        if not os.path.exists(paths.start_reviews):
            raise SystemExit(
                f"Step 2 requires the output from step 1 ({paths.start_reviews}) to exist."
            )
        run_step(
            [
                sys.executable,
                "scripts/step2_filter_reviews.py",
                "--input",
                paths.start_reviews,
                "--output",
                paths.filtered_reviews,
                "--model",
                args.filter_model,
                "--delay",
                str(args.filter_delay),
            ],
            "Filter reviews with OpenAI",
        )

    # Step 3 – Retrieve PMC full texts
    if not maybe_skip(paths.fulltext_reviews, args.force, "step 3"):
        if not os.path.exists(paths.filtered_reviews):
            raise SystemExit(
                f"Step 3 requires the output from step 2 ({paths.filtered_reviews}) to exist."
            )
        run_step(
            [
                sys.executable,
                "scripts/step3_fetch_fulltext.py",
                "--input",
                paths.filtered_reviews,
                "--output",
                paths.fulltext_reviews,
            ],
            "Fetch PMC full texts",
        )

    # Step 4 – Extract theories via LLM
    ensure_env("OPENAI_API_KEY", "Step 4")
    if not maybe_skip(paths.theories, args.force, "step 4"):
        if not os.path.exists(paths.fulltext_reviews):
            raise SystemExit(
                f"Step 4 requires the output from step 3 ({paths.fulltext_reviews}) to exist."
            )
        run_step(
            [
                sys.executable,
                "scripts/step4_extract_theories.py",
                "--input",
                paths.fulltext_reviews,
                "--output",
                paths.theories,
                "--model",
                args.theory_model,
                "--delay",
                str(args.theory_delay),
                "--max-chars",
                str(args.max_chars),
            ],
            "Extract theories from reviews",
        )

    # Step 5 – Ontology generation
    ensure_env("OPENAI_API_KEY", "Step 5")
    if not maybe_skip(paths.ontology, args.force, "step 5"):
        if not os.path.exists(paths.theories):
            raise SystemExit(
                f"Step 5 requires the output from step 4 ({paths.theories}) to exist."
            )
        run_step(
            [
                sys.executable,
                "scripts/step5_generate_ontology.py",
                "--input",
                paths.theories,
                "--output",
                paths.ontology,
                "--model",
                args.ontology_model,
                "--top-n",
                str(args.ontology_top_n),
                "--examples-per-theory",
                str(args.ontology_examples),
            ],
            "Generate ontology from theories",
        )

    print("\nPipeline completed. Results available at:")
    print(f"  Step 1 metadata: {paths.start_reviews}")
    print(f"  Step 2 filtered: {paths.filtered_reviews}")
    print(f"  Step 3 full texts: {paths.fulltext_reviews}")
    print(f"  Step 4 theories: {paths.theories}")
    print(f"  Step 5 ontology: {paths.ontology}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
