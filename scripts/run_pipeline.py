"""Convenience wrapper to execute the aging theory pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts import collect_theories


DEFAULT_ONTOLOGY_MAX_THEORIES = 40


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


def _normalise_term(term: str) -> str:
    return " ".join(term.split())


def _query_variants(term: str) -> List[str]:
    cleaned = _normalise_term(term)
    if not cleaned:
        return []
    variants: List[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate:
            key = candidate.lower()
            if key not in seen:
                seen.add(key)
                variants.append(candidate)

    lower = cleaned.lower()
    _add(cleaned)
    if " " in cleaned:
        _add(f'"{cleaned}"')
    if "aging" not in lower and "ageing" not in lower:
        quoted = f'"{cleaned}"' if " " in cleaned else cleaned
        _add(f"{quoted} aging")
    if "theory" not in lower:
        _add(f"{cleaned} theory")
    return variants


def _clean_suggestions(values: Any) -> List[str]:
    suggestions: List[str] = []
    seen: set[str] = set()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for value in values:
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    key = cleaned.lower()
                    if key not in seen:
                        seen.add(key)
                        suggestions.append(cleaned)
    return suggestions


def _theory_suggestions(theory: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    for key in ("preferred_label", "label"):
        value = theory.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                names.append(cleaned)
    aliases = theory.get("aliases")
    if isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)):
        for alias in aliases:
            if isinstance(alias, str):
                cleaned = alias.strip()
                if cleaned:
                    names.append(cleaned)

    suggestions: List[str] = []
    seen: set[str] = set()
    for name in names:
        for variant in _query_variants(name):
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                suggestions.append(variant)

    if not suggestions and names:
        fallback = _normalise_term(names[0])
        if fallback:
            suggestions.append(fallback)
    return suggestions


def _convert_group_node(group: Mapping[str, Any]) -> tuple[str, Dict[str, Any]] | None:
    raw_name = group.get("name") or group.get("label")
    if not isinstance(raw_name, str):
        raw_name = "Unnamed group"
    name = raw_name.strip() or "Unnamed group"

    entry: Dict[str, Any] = {}
    group_suggestions = _clean_suggestions(group.get("suggested_queries"))
    if group_suggestions:
        entry["suggested_queries"] = group_suggestions

    subtargets: Dict[str, Dict[str, Any]] = {}
    subgroups = group.get("subgroups")
    if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
        for child in subgroups:
            if isinstance(child, Mapping):
                converted = _convert_group_node(child)
                if converted:
                    child_name, child_entry = converted
                    subtargets[child_name] = child_entry

    theories = group.get("theories")
    if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
        for theory in theories:
            if not isinstance(theory, Mapping):
                continue
            theory_name = theory.get("preferred_label") or theory.get("label") or theory.get("theory_id")
            if not isinstance(theory_name, str):
                continue
            cleaned_name = theory_name.strip()
            if not cleaned_name:
                continue
            theory_entry: Dict[str, Any] = {}
            suggestions = _theory_suggestions(theory)
            if suggestions:
                theory_entry["suggested_queries"] = suggestions
            else:
                theory_entry["suggested_queries"] = [cleaned_name]
            subtargets.setdefault(cleaned_name, theory_entry)

    if subtargets:
        entry["subtheories"] = subtargets

    if not entry:
        entry["suggested_queries"] = [name]

    return name, entry


def _ontology_targets_from_path(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    ontology = payload.get("ontology") if isinstance(payload, Mapping) else None
    final = ontology.get("final") if isinstance(ontology, Mapping) else None
    groups = final.get("groups") if isinstance(final, Mapping) else None
    if not isinstance(groups, Sequence):
        groups = []

    targets: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        if isinstance(group, Mapping):
            converted = _convert_group_node(group)
            if converted:
                name, entry = converted
                targets[name] = entry
    return targets


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
        "--collector-query",
        "--base-query",
        dest="collector_query",
        default="aging theory",
        help=(
            "Base query string supplied to the ontology-driven collector stage. "
            "(--base-query is accepted as an alias.)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Global paper export limit applied during the collector stage.",
    )
    parser.add_argument(
        "--state-dir",
        dest="state_dir",
        help="Override the collector's retrieval state directory.",
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
        "--filter-processes",
        type=int,
        help=(
            "Number of worker processes to use during step 2 filtering. "
            "Defaults to the auto-scaling behaviour inside the step when omitted."
        ),
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
        "--theory-processes",
        "--extract-processes",
        dest="theory_processes",
        type=int,
        help=(
            "Number of worker processes to use during step 4 theory extraction. "
            "(--extract-processes is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--chunk-chars",
        "--extract-chunk-chars",
        dest="chunk_chars",
        type=int,
        default=12000,
        help=(
            "Maximum characters from each review chunk sent to the LLM during "
            "step 4. (--extract-chunk-chars is accepted for backwards compatibility.)"
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        "--extract-chunk-overlap",
        dest="chunk_overlap",
        type=int,
        default=1000,
        help=(
            "Number of characters to overlap between successive review chunks "
            "in step 4. (--extract-chunk-overlap is accepted for backwards "
            "compatibility.)"
        ),
    )
    parser.add_argument(
        "--max-chars",
        "--extract-max-chars",
        dest="compat_max_chars",
        type=int,
        default=None,
        help="Deprecated alias for --chunk-chars.",
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
        "--ontology-processes",
        type=int,
        default=None,
        help=(
            "Number of worker processes to use during ontology generation. When "
            "omitted, step 5 auto-scales based on the registry size, targeting "
            "roughly 25 theories per worker while respecting the available CPU "
            "cores."
        ),
    )
    parser.add_argument(
        "--ontology-max-theories",
        type=int,
        default=DEFAULT_ONTOLOGY_MAX_THEORIES,
        help=(
            "Maximum number of theories permitted per reconciled ontology group in "
            "step 5 before automatic subgroup splitting is triggered."
        ),
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


def _load_theory_registry(path: str) -> Mapping[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, Mapping):
        return None

    registry = data.get("theory_registry")
    if not isinstance(registry, Mapping):
        return None

    return registry


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    paths = build_paths(args.workdir)
    workdir_path = Path(args.workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    if args.compat_max_chars is not None:
        args.chunk_chars = args.compat_max_chars

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
                *(
                    ["--processes", str(args.filter_processes)]
                    if args.filter_processes is not None
                    else []
                ),
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
    skip_step4 = False
    if not args.force:
        registry = _load_theory_registry(paths.theories)
        if registry is not None:
            print(
                f"Skipping step 4; {paths.theories} already exists. Use --force to regenerate."
            )
            skip_step4 = True
        else:
            status = "missing" if not os.path.exists(paths.theories) else "outdated or corrupt"
            print(
                f"Cached step 4 output at {paths.theories} is {status}; regenerating."
            )

    if not skip_step4:
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
                "--chunk-chars",
                str(args.chunk_chars),
                "--chunk-overlap",
                str(args.chunk_overlap),
                *(
                    ["--processes", str(args.theory_processes)]
                    if args.theory_processes is not None
                    else []
                ),
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
                "--max-theories-per-group",
                str(args.ontology_max_theories),
                *(
                    ["--processes", str(args.ontology_processes)]
                    if args.ontology_processes is not None
                    else []
                ),
            ],
            "Generate ontology from theories",
        )

    # Step 6 – Ontology-driven literature collection and classification
    if not os.path.exists(paths.ontology):
        raise SystemExit(
            "Ontology file missing after step 5; cannot continue to collection stage."
        )

    try:
        ontology_targets = _ontology_targets_from_path(paths.ontology)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Unable to parse ontology file {paths.ontology}: {exc}") from exc

    if not ontology_targets:
        print("No ontology groups with suggested queries found; skipping collector stage.")
    else:
        collector_config_path = Path("config/pipeline.yaml")
        try:
            base_collector_config = collect_theories.load_config(collector_config_path)
        except Exception as exc:  # pragma: no cover - configuration errors surface immediately
            raise SystemExit(
                f"Unable to load collector configuration at {collector_config_path}: {exc}"
            ) from exc

        collector_config = copy.deepcopy(base_collector_config)
        collector_outputs_dir = workdir_path / "collector"
        collector_outputs_dir.mkdir(parents=True, exist_ok=True)

        corpus_cfg = dict(collector_config.get("corpus") or {})
        corpus_cfg["targets"] = ontology_targets
        corpus_cfg["ontology_suggestions_path"] = paths.ontology
        collector_config["corpus"] = corpus_cfg

        outputs_cfg = dict(collector_config.get("outputs") or {})
        outputs_cfg["papers"] = str(collector_outputs_dir / "papers.csv")
        outputs_cfg["theories"] = str(collector_outputs_dir / "theories.csv")
        outputs_cfg["questions"] = str(collector_outputs_dir / "questions.csv")
        outputs_cfg["cache_dir"] = str(collector_outputs_dir / "cache")
        collector_config["outputs"] = outputs_cfg

        os.makedirs(outputs_cfg["cache_dir"], exist_ok=True)

        collector_args: List[str] = [str(args.collector_query)]
        if args.limit is not None:
            collector_args += ["--limit", str(args.limit)]
        if args.state_dir:
            collector_args += ["--state-dir", args.state_dir]

        collector_parser = collect_theories.build_parser()
        collector_namespace = collector_parser.parse_args(collector_args)

        print("\n→ Step 6 (collector)")
        collector_exit = collect_theories.run_pipeline(
            collector_namespace,
            parser=collector_parser,
            config=collector_config,
            config_path=collector_config_path,
        )
        if collector_exit != 0:
            return collector_exit

    print("\nPipeline completed. Results available at:")
    print(f"  Step 1 metadata: {paths.start_reviews}")
    print(f"  Step 2 filtered: {paths.filtered_reviews}")
    print(f"  Step 3 full texts: {paths.fulltext_reviews}")
    print(f"  Step 4 theories: {paths.theories}")
    print(f"  Step 5 ontology: {paths.ontology}")
    outputs = collector_config.get("outputs") if 'collector_config' in locals() else None
    if isinstance(outputs, Mapping):
        papers_path = outputs.get("papers")
        theories_path = outputs.get("theories")
        questions_path = outputs.get("questions")
        if papers_path:
            print(f"  Step 6 papers: {papers_path}")
        if theories_path:
            print(f"  Step 6 assignments: {theories_path}")
        if questions_path:
            print(f"  Step 6 Q&A: {questions_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
