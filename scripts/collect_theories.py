"""Collect papers and classify theories for the Hackaging challenge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline import (
    LiteratureRetriever,
    QuestionExtractor,
    TheoryClassifier,
    export_papers,
    export_question_answers,
    export_theories,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search query for literature retrieval")
    parser.add_argument(
        "--config",
        default="config/pipeline.yaml",
        type=Path,
        help="Path to the pipeline configuration file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of papers to retrieve (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    retriever = LiteratureRetriever(Path(config["data_sources"]["seed_papers"]))
    limit = args.limit or config.get("pipeline", {}).get("max_results")
    papers = retriever.search(args.query, limit=limit)

    classifier = TheoryClassifier.from_config(config["classification"]["keywords"])
    assignments = []
    for paper in papers:
        assignments.extend(classifier.classify(paper))

    extractor = QuestionExtractor(config.get("extraction", {}).get("keyword_templates"))
    question_answers = []
    for paper in papers:
        question_answers.extend(extractor.extract(paper))

    outputs = config["outputs"]
    export_papers(papers, Path(outputs["papers"]))
    export_theories(assignments, Path(outputs["theories"]))
    export_question_answers(question_answers, Path(outputs["questions"]))

    print(f"Exported {len(papers)} papers to {outputs['papers']}")
    print(f"Exported {len(assignments)} theory assignments to {outputs['theories']}")
    print(f"Exported {len(question_answers)} question answers to {outputs['questions']}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
