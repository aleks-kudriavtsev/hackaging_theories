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
    export_question_answers,
)

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
    args = parser.parse_args()

    config = load_config(args.config)

    papers_path = args.papers or Path(config["outputs"]["papers"])
    if papers_path.exists():
        papers = _load_papers_from_csv(papers_path)
    else:
        retriever = LiteratureRetriever(Path(config["data_sources"]["seed_papers"]))
        papers = retriever.search("", limit=None)

    classifier = TheoryClassifier.from_config(config["classification"]["keywords"])
    extractor = QuestionExtractor(config.get("extraction"))

    theory_counts: Counter[str] = Counter()
    question_answers = []
    for paper in papers:
        assignments = classifier.classify(paper)
        theory_counts.update([assignment.theory for assignment in assignments])
        question_answers.extend(extractor.extract(paper))

    export_question_answers(question_answers, Path(config["outputs"]["questions"]))

    cache_dir = _ensure_cache_dir(Path(config["outputs"].get("cache_dir", "data/cache")))
    summary = {
        "paper_count": len(papers),
        "theory_counts": dict(theory_counts),
        "question_coverage": _question_coverage(question_answers),
    }
    summary_path = cache_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote refreshed question answers to {config['outputs']['questions']}")
    print(f"Summary saved to {summary_path}")


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
