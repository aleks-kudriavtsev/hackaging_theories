"""Convenience wrapper to execute the aging theory pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = doi.strip().lower()
    if not cleaned:
        return None
    prefixes = (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "doi:",
    )
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    return cleaned.strip() or None


def _merge_lists(existing: Sequence[str] | None, incoming: Sequence[str] | None) -> List[str]:
    merged: List[str] = []

    def _add_items(values: Sequence[str] | str | None) -> None:
        if values is None:
            return
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            iterable = values
        else:
            iterable = [values]
        for value in iterable:
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned and cleaned not in merged:
                    merged.append(cleaned)

    _add_items(existing)
    _add_items(incoming)
    return merged


def _normalise_sources(record: Mapping[str, object]) -> List[str]:
    raw_sources = record.get("sources")
    if isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes)):
        return _merge_lists(raw_sources, [])
    provenance = record.get("provenance")
    if isinstance(provenance, str):
        return _merge_lists([provenance], [])
    return []


def _merge_fulltext(existing: Mapping[str, str] | None, incoming: Mapping[str, str] | None) -> Dict[str, str] | None:
    merged: Dict[str, str] = {}
    for payload in (existing, incoming):
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                if key not in merged and value.strip():
                    merged[key] = value.strip()
    return merged or None


def _merge_records(base: Dict[str, object], incoming: Mapping[str, object]) -> Dict[str, object]:
    merged = dict(base)
    scalar_fields = [
        "pmid",
        "title",
        "abstract",
        "journal",
        "publication_year",
        "doi",
        "openalex_id",
        "openalex_url",
    ]
    for field in scalar_fields:
        current = merged.get(field)
        new_value = incoming.get(field)
        if (not current or (isinstance(current, str) and not current.strip())) and new_value:
            merged[field] = new_value

    merged["publication_types"] = _merge_lists(
        merged.get("publication_types"), incoming.get("publication_types")
    )
    merged["authors"] = _merge_lists(merged.get("authors"), incoming.get("authors"))
    merged["sources"] = _merge_lists(merged.get("sources"), _normalise_sources(incoming))

    fulltext_existing = merged.get("fulltext_links")
    fulltext_incoming = incoming.get("fulltext_links")
    merged_fulltext = _merge_fulltext(fulltext_existing, fulltext_incoming)
    if merged_fulltext is not None:
        merged["fulltext_links"] = merged_fulltext

    open_access = merged.get("open_access")
    if not open_access and incoming.get("open_access"):
        merged["open_access"] = incoming.get("open_access")

    doi_value = merged.get("doi")
    if isinstance(doi_value, str):
        normalised = _normalise_doi(doi_value)
        if normalised:
            merged["doi"] = normalised

    return merged



@dataclass
class PipelinePaths:
    pubmed_reviews: str
    openalex_reviews: str
    google_scholar_reviews: str
    start_reviews: str
    filtered_reviews: str
    fulltext_reviews: str
    theories: str
    ontology: str


def build_paths(workdir: str) -> PipelinePaths:
    os.makedirs(workdir, exist_ok=True)
    return PipelinePaths(
        pubmed_reviews=os.path.join(workdir, "start_reviews_pubmed.json"),
        openalex_reviews=os.path.join(workdir, "start_reviews_openalex.json"),
        google_scholar_reviews=os.path.join(workdir, "start_reviews_google_scholar.json"),
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
        "--output-dir",
        dest="workdir",
        default="data/pipeline",
        help=(
            "Directory to store intermediate and final JSON artefacts. "
            "(--output-dir is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--query",
        default='(("aging theory"[TIAB] OR "ageing theory"[TIAB] OR "theories of aging"[TIAB]) AND review[PTYP])',
        help="PubMed query passed to step1_pubmed_search.py.",
    )
    parser.add_argument(
        "--filter-model",
        default="gpt-5-nano",
        help=(
            "OpenAI model used for relevance filtering (step 2). Defaults to the "
            "balanced gpt-5-nano tier so we stay within the ~$10 per million "
            "articles budget while still handling batched title/abstract "
            "prompts with solid accuracy."
        ),
    )
    parser.add_argument(
        "--filter-delay",
        type=float,
        default=0.5,
        help="Delay between OpenAI calls during filtering (seconds).",
    )
    parser.add_argument(
        "--theory-model",
        "--extract-model",
        dest="theory_model",
        default="gpt-5-nano",
        help=(
            "OpenAI model used for theory extraction (step 4). Defaults to gpt-5-"
            "nano, which offers enough context for consolidated review prompts "
            "while keeping token costs in line with the $10 per million articles "
            "budget. (--extract-model is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--hypothesis-review-model",
        dest="hypothesis_review_model",
        default="gpt-4.1-nano",
        help=(
            "OpenAI model used for post-extraction hypothesis review. Defaults to "
            "gpt-4.1-nano so the audit pass gains structured reasoning quality "
            "without pushing the per-run budget past ~$10 per million articles."
        ),
    )
    parser.add_argument(
        "--theory-delay",
        "--extract-delay",
        dest="theory_delay",
        type=float,
        default=0.5,
        help=(
            "Delay between OpenAI calls during theory extraction (seconds). "
            "(--extract-delay is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--max-chars",
        "--extract-max-chars",
        dest="max_chars",
        type=int,
        default=12000,
        help=(
            "Maximum characters from each review to send to the LLM (step 4). "
            "(--extract-max-chars is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--ontology-model",
        default="gpt-5-mini",
        help=(
            "OpenAI model used for ontology generation (step 5). Defaults to the "
            "gpt-5-mini tier, which provides broader synthesis context for "
            "ontology assembly while staying on budget."
        ),
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
    if not maybe_skip(paths.pubmed_reviews, args.force, "step 1 (PubMed)"):
        run_step(
            [
                sys.executable,
                "scripts/step1_pubmed_search.py",
                "--query",
                args.query,
                "--output",
                paths.pubmed_reviews,
            ],
            "Collect PubMed review metadata",
        )

    # Step 1b – OpenAlex search
    if not maybe_skip(paths.openalex_reviews, args.force, "step 1b (OpenAlex)"):
        run_step(
            [
                sys.executable,
                "scripts/step1b_openalex_search.py",
                "--output",
                paths.openalex_reviews,
            ],
            "Collect OpenAlex review metadata",
        )

    # Step 1c – optional Google Scholar search handler
    google_scripts = [
        "scripts/step1c_google_scholar.py",
        "scripts/step1c_google_scholar_search.py",
        "scripts/step1_google_scholar_search.py",
    ]
    google_script = next((path for path in google_scripts if os.path.exists(path)), None)
    if google_script:
        if not maybe_skip(paths.google_scholar_reviews, args.force, "step 1c (Google Scholar)"):
            run_step(
                [
                    sys.executable,
                    google_script,
                    "--output",
                    paths.google_scholar_reviews,
                ],
                "Collect Google Scholar metadata",
            )
    else:
        print("Skipping Google Scholar step; no handler script found.")

    # Merge and deduplicate source records
    source_files = [
        paths.pubmed_reviews,
        paths.openalex_reviews,
        paths.google_scholar_reviews,
    ]
    merged_records: List[Dict[str, object]] = []
    index: Dict[str, int] = {}
    for source_path in source_files:
        if not os.path.exists(source_path):
            continue
        with open(source_path, "r", encoding="utf-8") as handle:
            try:
                records = json.load(handle)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Unable to parse JSON from {source_path}: {exc}")
        if not isinstance(records, list):
            raise SystemExit(f"Expected list payload in {source_path}")
        for record in records:
            if not isinstance(record, dict):
                continue
            # Normalise DOI for deduplication purposes
            doi_key = _normalise_doi(record.get("doi"))
            pmid = (record.get("pmid") or "").strip() if isinstance(record.get("pmid"), str) else None
            openalex_id = record.get("openalex_id")
            if isinstance(openalex_id, str):
                openalex_id = openalex_id.strip() or None

            keys = []
            if doi_key:
                keys.append(f"doi:{doi_key}")
            if pmid:
                keys.append(f"pmid:{pmid}")
            if openalex_id:
                keys.append(f"openalex:{openalex_id}")

            existing_idx = None
            for key in keys:
                if key in index:
                    existing_idx = index[key]
                    break

            if existing_idx is None:
                record_copy = dict(record)
                # Ensure consistent sources representation
                record_copy["sources"] = _normalise_sources(record)
                record_copy["publication_types"] = _merge_lists(record.get("publication_types"), [])
                record_copy["authors"] = _merge_lists(record.get("authors"), [])
                doi_value = record_copy.get("doi")
                if isinstance(doi_value, str):
                    normalised = _normalise_doi(doi_value)
                    if normalised:
                        record_copy["doi"] = normalised
                merged_records.append(record_copy)
                existing_idx = len(merged_records) - 1
            else:
                merged_records[existing_idx] = _merge_records(merged_records[existing_idx], record)

            for key in keys:
                index[key] = existing_idx

    if not merged_records:
        raise SystemExit("No records collected from PubMed, OpenAlex, or Google Scholar.")

    with open(paths.start_reviews, "w", encoding="utf-8") as handle:
        json.dump(merged_records, handle, ensure_ascii=False, indent=2)
    print(f"Merged {len(merged_records)} unique records into {paths.start_reviews}")

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
