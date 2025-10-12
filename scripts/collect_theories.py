"""Collect papers and classify theories for the Hackaging challenge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline import (
    LiteratureRetriever,
    PaperMetadata,
    ProviderConfig,
    QuestionExtractor,
    TheoryClassifier,
    TheoryOntology,
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


def build_provider_configs(
    config: Mapping[str, Any], limit_to: Iterable[str] | None
) -> List[ProviderConfig]:
    providers_cfg = config.get("providers", [])
    api_keys = config.get("api_keys", {})
    selected = set(limit_to or [])
    configs: List[ProviderConfig] = []
    for item in providers_cfg:
        name = item["name"]
        if selected and name not in selected:
            continue
        api_key = item.get("api_key")
        api_key_key = item.get("api_key_key")
        if not api_key and api_key_key:
            api_key = api_keys.get(api_key_key)
        configs.append(
            ProviderConfig(
                name=name,
                type=item["type"],
                enabled=item.get("enabled", True),
                api_key=api_key,
                base_url=item.get("base_url"),
                query_shards=item.get("query_shards"),
                batch_size=item.get("batch_size", 200),
                rate_limit_per_sec=item.get("rate_limit_per_sec"),
                timeout=item.get("timeout"),
                extra=item.get("extra", {}),
            )
        )
    return configs


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render_query(template: str, context: Mapping[str, Any]) -> str:
    return template.format_map(_SafeDict(context))


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


def _existing_total(retriever: LiteratureRetriever, state_prefix: str) -> int:
    state = retriever.state_store.get(state_prefix)
    if not state:
        return 0
    seen = state.get("seen_identifiers")
    if isinstance(seen, list):
        return len(seen)
    papers = state.get("papers")
    if isinstance(papers, list):
        return len(papers)
    return 0


def collect_for_entry(
    retriever: LiteratureRetriever,
    *,
    name: str,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    providers: Iterable[str] | None,
    resume: bool,
    state_prefix: str,
) -> Tuple[Dict[str, Any], List[PaperMetadata]]:
    query_templates = config.get("queries") or [context.get("base_query", name)]
    queries = [render_query(template, context | {"query": context.get("base_query", name)}) for template in query_templates]
    queries = [q.strip() for q in queries if q.strip()]
    target = config.get("target")
    result = retriever.collect_queries(
        queries,
        target=target,
        providers=list(providers) if providers else None,
        state_key=state_prefix,
        resume=resume,
    )

    entry_map = {paper.identifier: paper for paper in result.papers}
    summary = dict(result.summary)

    subtheory_cfg = config.get("subtheories", {})
    if subtheory_cfg:
        sub_summaries: Dict[str, Any] = {}
        prioritized = []
        for sub_name, sub_config in subtheory_cfg.items():
            sub_state_prefix = f"{state_prefix}::sub::{slugify(sub_name)}"
            existing = _existing_total(retriever, sub_state_prefix) if resume else 0
            sub_target = sub_config.get("target")
            fill_ratio = (existing / sub_target) if sub_target else 0.0
            prioritized.append((fill_ratio, sub_name, sub_config, sub_state_prefix))
        prioritized.sort(key=lambda item: (item[0], item[1]))
        for _ratio, sub_name, sub_config, sub_state_prefix in prioritized:
            sub_summary, sub_papers = collect_for_entry(
                retriever,
                name=sub_name,
                config=sub_config,
                context=context | {"subtheory": sub_name},
                providers=providers,
                resume=resume,
                state_prefix=sub_state_prefix,
            )
            sub_summaries[sub_name] = sub_summary
            for paper in sub_papers:
                entry_map.setdefault(paper.identifier, paper)
        summary["subtheories"] = sub_summaries

    return summary, list(entry_map.values())


def validate_targets(summary: Mapping[str, Any]) -> None:
    for name, stats in summary.items():
        target = stats.get("target")
        total_unique = stats.get("total_unique", 0)
        if target and total_unique < target:
            raise RuntimeError(
                f"Target for '{name}' not met: retrieved {total_unique}, expected >= {target}"
            )
        sub_summary = stats.get("subtheories", {})
        if sub_summary:
            validate_targets(sub_summary)


def format_summary(name: str, summary: Mapping[str, Any], indent: int = 0) -> str:
    padding = " " * indent
    line = (
        f"{padding}- {name}: {summary.get('total_unique', 0)} papers"
        f" (target: {summary.get('target', 'n/a')}, new: {summary.get('newly_retrieved', 0)})"
    )
    sub_lines = [line]
    for provider, count in summary.get("providers", {}).items():
        sub_lines.append(f"{padding}    provider[{provider}]: {count}")
    for sub_name, sub_summary in summary.get("subtheories", {}).items():
        sub_lines.append(format_summary(sub_name, sub_summary, indent=indent + 2))
    return "\n".join(sub_lines)


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
        help="Global maximum number of papers to export (overrides config)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Limit retrieval to the specified providers (by name)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore cached retrieval state and start fresh",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Override the retrieval state directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    corpus_cfg: Mapping[str, Any] = config.get("corpus", {})
    state_dir = Path(args.state_dir or corpus_cfg.get("cache_dir") or config["outputs"].get("cache_dir", "data/cache"))
    provider_configs = build_provider_configs(config, args.providers)
    retriever = LiteratureRetriever(
        Path(config["data_sources"]["seed_papers"]),
        provider_configs=provider_configs,
        state_dir=state_dir,
    )

    global_limit = args.limit or corpus_cfg.get("global_limit")
    resume = not args.no_resume

    context = {"base_query": args.query}
    targets = corpus_cfg.get("targets", {})
    ontology = TheoryOntology.from_targets_config(targets)
    collected_papers: Dict[str, PaperMetadata] = {}
    summary_report: Dict[str, Any] = {}

    prioritized = []
    for theory_name, theory_cfg in targets.items():
        state_prefix = f"theory::{slugify(theory_name)}"
        existing = _existing_total(retriever, state_prefix) if resume else 0
        theory_target = theory_cfg.get("target")
        fill_ratio = (existing / theory_target) if theory_target else 0.0
        prioritized.append((fill_ratio, theory_name, theory_cfg, state_prefix))

    prioritized.sort(key=lambda item: (item[0], item[1]))

    for _ratio, theory_name, theory_cfg, state_prefix in prioritized:
        theory_summary, theory_papers = collect_for_entry(
            retriever,
            name=theory_name,
            config=theory_cfg,
            context=context | {"theory": theory_name},
            providers=args.providers,
            resume=resume,
            state_prefix=state_prefix,
        )
        summary_report[theory_name] = theory_summary
        for paper in theory_papers:
            collected_papers.setdefault(paper.identifier, paper)

    papers = list(collected_papers.values())

    validate_targets(summary_report)
    if global_limit is not None and len(papers) > global_limit:
        papers = papers[: global_limit]

    classifier = TheoryClassifier.from_config(
        config["classification"]["keywords"], ontology=ontology
    )
    assignments = []
    for paper in papers:
        assignments.extend(classifier.classify(paper))

    coverage_counts = classifier.summarize(assignments)
    coverage_summary = ontology.coverage(coverage_counts)
    quota_status = {
        name: {
            "count": record.count,
            "target": record.target,
            "deficit": record.deficit,
            "met": record.met,
            "depth": record.depth,
        }
        for name, record in coverage_summary.items()
    }

    extractor = QuestionExtractor(config.get("extraction"))
    question_answers = []
    for paper in papers:
        question_answers.extend(extractor.extract(paper))

    outputs = config["outputs"]
    export_papers(papers, Path(outputs["papers"]))
    export_theories(assignments, Path(outputs["theories"]))
    export_question_answers(question_answers, Path(outputs["questions"]))

    retriever.state_store.write_summary(
        {"retrieval": summary_report, "quota_status": quota_status}
    )

    print(f"Exported {len(papers)} papers to {outputs['papers']}")
    print(f"Exported {len(assignments)} theory assignments to {outputs['theories']}")
    print(f"Exported {len(question_answers)} question answers to {outputs['questions']}")

    for theory_name, summary in summary_report.items():
        print(format_summary(theory_name, summary))

    print()
    print(ontology.format_coverage_report(coverage_counts))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
