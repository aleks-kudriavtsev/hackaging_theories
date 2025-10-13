"""Analyze previously collected papers and refresh question answers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    export_question_answers,
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


def _maybe_build_llm_client(config: Mapping[str, Any], args: argparse.Namespace) -> LLMClient | None:
    classification_cfg = config.get("classification", {}) if isinstance(config, Mapping) else {}
    llm_cfg = classification_cfg.get("llm", {}) if isinstance(classification_cfg, Mapping) else {}

    llm_model = args.llm_model or llm_cfg.get("model")
    if not llm_model:
        return None

    temperature = args.llm_temperature if args.llm_temperature is not None else llm_cfg.get("temperature", 0.0)
    batch_size = args.llm_batch_size or llm_cfg.get("batch_size", 4)
    cache_dir = args.llm_cache_dir or llm_cfg.get("cache_dir") or Path("data/cache/llm")
    max_retries = llm_cfg.get("max_retries", 3)
    retry_backoff = llm_cfg.get("retry_backoff", 2.0)
    request_timeout = llm_cfg.get("request_timeout", 60.0)

    config_obj = LLMClientConfig(
        model=llm_model,
        temperature=float(temperature),
        batch_size=int(batch_size),
        max_retries=int(max_retries),
        retry_backoff=float(retry_backoff),
        request_timeout=float(request_timeout),
        cache_dir=Path(cache_dir),
    )
    return LLMClient(config_obj)


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
    args = parser.parse_args()

    config = load_config(args.config)

    papers_path = args.papers or Path(config["outputs"]["papers"])
    if papers_path.exists():
        papers = _load_papers_from_csv(papers_path)
    else:
        retriever = LiteratureRetriever(Path(config["data_sources"]["seed_papers"]))
        papers = retriever.search("", limit=None)

    ontology = TheoryOntology.from_targets_config(config.get("corpus", {}).get("targets", {}))
    llm_client = _maybe_build_llm_client(config, args)
    classifier = TheoryClassifier.from_config(
        config.get("classification", {}), ontology=ontology, llm_client=llm_client
    )
    extractor = QuestionExtractor(config.get("extraction"))

    theory_counts: Counter[str] = Counter()
    question_answers = []
    assignments = []
    for paper_assignments in classifier.classify_batch(papers):
        assignments.extend(paper_assignments)
        theory_counts.update([assignment.theory for assignment in paper_assignments])
        question_answers.extend(extractor.extract(paper))

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
