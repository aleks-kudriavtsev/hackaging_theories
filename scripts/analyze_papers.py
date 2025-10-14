"""Analyze previously collected papers and refresh question answers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline import (
    LiteratureRetriever,
    PaperMetadata,
    QuestionExtractor,
    TheoryClassifier,
    TheoryOntology,
    classify_and_extract_parallel,
    export_question_answers,
)
from theories_pipeline.config_utils import (
    MissingSecretError,
    ensure_real_api_keys,
    resolve_api_keys,
)
from theories_pipeline.llm import LLMClient, LLMClientConfig

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configuration files")
        return yaml.safe_load(text)
    return json.loads(text)


def _load_papers_from_csv(path: Path) -> List[PaperMetadata]:
    papers: List[PaperMetadata] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            authors = [author.strip() for author in row["authors"].split(";") if author.strip()]
            year = int(row["year"]) if row.get("year") else None
            doi = row.get("doi") or None
            papers.append(
                PaperMetadata(
                    identifier=row["identifier"],
                    title=row["title"],
                    authors=authors,
                    abstract=row["abstract"],
                    source=row["source"],
                    year=year,
                    doi=doi,
                )
            )
    return papers


def _ensure_cache_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_API_KEY_OVERRIDE_MAP = {
    "openalex_api_key": "openalex",
    "crossref_api_key": "crossref_contact",
    "pubmed_api_key": "pubmed",
    "serpapi_key": "serpapi",
    "semantic_scholar_key": "semantic_scholar",
    "scihub_email": "scihub_email",
    "scihub_rapidapi_key": "scihub_rapidapi",
    "annas_archive_api_key": "annas_archive",
}


def _resolve_workers(cli_value: int | None, config_value: Any, default: int) -> int:
    if cli_value is not None:
        return max(1, int(cli_value))
    if isinstance(config_value, int) and config_value > 0:
        return int(config_value)
    return max(1, int(default))


def _load_api_keys(
    args: argparse.Namespace,
    config_api_keys: Mapping[str, Any],
    *,
    base_path: Path | None,
) -> Dict[str, str | None]:
    resolved = resolve_api_keys(config_api_keys, base_path=base_path)
    resolved = ensure_real_api_keys(resolved)
    overrides = {
        target_key: getattr(args, cli_attr)
        for cli_attr, target_key in _API_KEY_OVERRIDE_MAP.items()
        if getattr(args, cli_attr, None)
    }
    if overrides:
        resolved = {**resolved, **overrides}
        resolved = ensure_real_api_keys(resolved)
    return resolved


def _maybe_build_llm_client(
    llm_cfg: Mapping[str, Any] | None,
    *,
    model_override: str | None,
    temperature_override: float | None,
    batch_override: int | None,
    cache_override: Path | None,
    api_key_override: str | None,
    api_keys: Mapping[str, str | None],
) -> LLMClient | None:
    config_data = llm_cfg if isinstance(llm_cfg, Mapping) else {}

    llm_model = model_override or config_data.get("model")
    if not llm_model:
        return None

    temperature_raw = (
        temperature_override if temperature_override is not None else config_data.get("temperature", 0.0)
    )
    batch_size_raw = batch_override or config_data.get("batch_size", 4)
    cache_dir_value = cache_override or config_data.get("cache_dir") or Path("data/cache/llm")
    max_retries = int(config_data.get("max_retries", 3))
    retry_backoff = float(config_data.get("retry_backoff", 2.0))
    request_timeout = float(config_data.get("request_timeout", 60.0))
    api_key = api_key_override or config_data.get("api_key")
    api_key_key = config_data.get("api_key_key")
    if (not api_key) and api_key_key:
        api_key = api_keys.get(api_key_key)

    config_obj = LLMClientConfig(
        model=llm_model,
        temperature=float(temperature_raw),
        batch_size=int(batch_size_raw),
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        request_timeout=request_timeout,
        cache_dir=Path(cache_dir_value),
    )
    return LLMClient(config_obj, api_key=api_key or None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        type=Path,
        help="Path to the pipeline configuration file",
    )
    parser.add_argument(
        "--papers",
        type=Path,
        help="Optional papers CSV; defaults to config outputs or seed data",
    )
    precedence_note = "Overrides the {name} API key (CLI > config file > environment > defaults)"
    parser.add_argument(
        "--openalex-api-key",
        help=precedence_note.format(name="OpenAlex"),
    )
    parser.add_argument(
        "--crossref-api-key",
        help=precedence_note.format(name="Crossref"),
    )
    parser.add_argument(
        "--pubmed-api-key",
        help=precedence_note.format(name="PubMed"),
    )
    parser.add_argument(
        "--serpapi-key",
        help=precedence_note.format(name="SerpApi"),
    )
    parser.add_argument(
        "--semantic-scholar-key",
        help=precedence_note.format(name="Semantic Scholar"),
    )
    parser.add_argument(
        "--scihub-email",
        help=precedence_note.format(name="Sci-Hub email"),
    )
    parser.add_argument(
        "--scihub-rapidapi-key",
        help=precedence_note.format(name="Sci-Hub RapidAPI"),
    )
    parser.add_argument(
        "--annas-archive-api-key",
        help=precedence_note.format(name="Anna's Archive"),
    )
    parser.add_argument(
        "--llm-model",
        help="Optional OpenAI model name for GPT-assisted classification",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        help="Override sampling temperature for GPT classification",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        help="Number of papers to classify per GPT request batch",
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        help="Directory to cache GPT responses (default: data/cache/llm)",
    )
    parser.add_argument(
        "--llm-api-key",
        help="Explicit API key for GPT classification (overrides config/env)",
    )
    parser.add_argument(
        "--extraction-llm-model",
        help="Optional OpenAI model name for GPT-assisted question extraction",
    )
    parser.add_argument(
        "--extraction-llm-temperature",
        type=float,
        help="Override sampling temperature for GPT extraction",
    )
    parser.add_argument(
        "--extraction-llm-batch-size",
        type=int,
        help="Number of extraction prompts to batch per GPT request",
    )
    parser.add_argument(
        "--extraction-llm-cache-dir",
        type=Path,
        help="Directory to cache GPT extraction responses",
    )
    parser.add_argument(
        "--extraction-llm-api-key",
        help="Explicit API key for GPT extraction (overrides config/env)",
    )
    parser.add_argument(
        "--parallel-fetch",
        type=int,
        help="Number of worker threads to fetch provider pages in parallel",
    )
    parser.add_argument(
        "--classification-workers",
        type=int,
        help="Number of worker threads for GPT classification/extraction",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    try:
        api_keys = _load_api_keys(
            args,
            config.get("api_keys", {}),
            base_path=config_path.parent,
        )
    except MissingSecretError as exc:
        parser.error(str(exc))

    pipeline_cfg: Mapping[str, Any] = config.get("pipeline", {}) if isinstance(config.get("pipeline"), Mapping) else {}
    parallel_fetch = _resolve_workers(args.parallel_fetch, pipeline_cfg.get("parallel_fetch"), 1)
    classification_workers = _resolve_workers(
        args.classification_workers, pipeline_cfg.get("classification_workers"), 1
    )

    papers_path = args.papers or Path(config["outputs"]["papers"])
    if papers_path.exists():
        papers = _load_papers_from_csv(papers_path)
    else:
        retriever = LiteratureRetriever(
            Path(config["data_sources"]["seed_papers"]), parallel_fetch=parallel_fetch
        )
        papers = retriever.search("", limit=None)

    ontology = TheoryOntology.from_targets_config(config.get("corpus", {}).get("targets", {}))
    classification_cfg = config.get("classification", {}) if isinstance(config, Mapping) else {}
    extraction_cfg = config.get("extraction", {}) if isinstance(config, Mapping) else {}

    llm_client = _maybe_build_llm_client(
        classification_cfg.get("llm") if isinstance(classification_cfg, Mapping) else {},
        model_override=args.llm_model,
        temperature_override=args.llm_temperature,
        batch_override=args.llm_batch_size,
        cache_override=args.llm_cache_dir,
        api_key_override=args.llm_api_key,
        api_keys=api_keys,
    )
    extraction_llm_client = _maybe_build_llm_client(
        extraction_cfg.get("llm") if isinstance(extraction_cfg, Mapping) else {},
        model_override=args.extraction_llm_model,
        temperature_override=args.extraction_llm_temperature,
        batch_override=args.extraction_llm_batch_size,
        cache_override=args.extraction_llm_cache_dir,
        api_key_override=args.extraction_llm_api_key,
        api_keys=api_keys,
    )
    classifier = TheoryClassifier.from_config(
        config.get("classification", {}), ontology=ontology, llm_client=llm_client
    )
    extractor = QuestionExtractor(config.get("extraction"), llm_client=extraction_llm_client)

    assignment_groups, answer_groups = classify_and_extract_parallel(
        papers, classifier, extractor, workers=classification_workers
    )
    assignments = [assignment for group in assignment_groups for assignment in group]
    question_answers = [answer for group in answer_groups for answer in group]
    theory_counts: Counter[str] = Counter(assignment.theory for assignment in assignments)

    export_question_answers(question_answers, Path(config["outputs"]["questions"]))

    cache_dir = _ensure_cache_dir(Path(config["outputs"].get("cache_dir", "data/cache")))
    coverage_counts = classifier.summarize(assignments)
    coverage_summary = ontology.coverage(coverage_counts)
    summary = {
        "paper_count": len(papers),
        "theory_counts": dict(theory_counts),
        "question_coverage": _question_coverage(question_answers),
        "ontology_coverage": {
            name: {
                "count": record.count,
                "target": record.target,
                "deficit": record.deficit,
                "met": record.met,
                "depth": record.depth,
            }
            for name, record in coverage_summary.items()
        },
    }
    summary_path = cache_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote refreshed question answers to {config['outputs']['questions']}")
    print(f"Summary saved to {summary_path}")
    print()
    print(ontology.format_coverage_report(coverage_counts))


def _question_coverage(answers: Iterable[Any]) -> Dict[str, Dict[str, int]]:
    coverage: Dict[str, Dict[str, int]] = {}
    for answer in answers:
        question_id = answer.question_id
        coverage.setdefault(question_id, {"answered": 0, "fallback": 0})
        if answer.answer.startswith(answer.question):
            coverage[question_id]["fallback"] += 1
        else:
            coverage[question_id]["answered"] += 1
    return coverage


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
