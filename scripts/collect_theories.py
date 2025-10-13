"""Collect papers and classify theories for the Hackaging challenge."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

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
    classify_and_extract_parallel,
    export_papers,
    export_question_answers,
    export_theories,
)
from theories_pipeline.config_utils import MissingSecretError, resolve_api_keys
from theories_pipeline.llm import LLMClient, LLMClientConfig
from theories_pipeline.ontology import OntologyNode
from theories_pipeline.query_expansion import QueryExpander, QueryExpansionSettings
from theories_pipeline.review_bootstrap import (
    ReviewDocument,
    build_bootstrap_ontology,
    extract_theories_from_review,
    merge_bootstrap_into_targets,
    pull_top_cited_reviews,
    write_bootstrap_cache,
)
try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None


logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configuration files")
        return yaml.safe_load(text)
    return json.loads(text)


def _resolve_workers(cli_value: int | None, config_value: Any, default: int) -> int:
    if cli_value is not None:
        return max(1, int(cli_value))
    if isinstance(config_value, int) and config_value > 0:
        return int(config_value)
    return max(1, int(default))


def build_provider_configs(
    config: Mapping[str, Any],
    limit_to: Iterable[str] | None,
    api_keys: Mapping[str, str | None],
) -> List[ProviderConfig]:
    providers_cfg = config.get("providers", [])
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
        extra_cfg = dict(item.get("extra", {})) if isinstance(item.get("extra"), Mapping) else {}
        for key in ("categories", "date_window", "window_days", "server"):
            if key in item and key not in extra_cfg:
                extra_cfg[key] = item[key]
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
                extra=extra_cfg,
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


def _run_bootstrap_phase(
    retriever: LiteratureRetriever,
    llm_client: LLMClient | None,
    corpus_cfg: Mapping[str, Any],
    *,
    context: Mapping[str, Any],
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Sequence[ReviewDocument]]]:
    bootstrap_cfg_raw = corpus_cfg.get("bootstrap")
    if not isinstance(bootstrap_cfg_raw, Mapping):
        return {}, {}, {}
    if not bootstrap_cfg_raw.get("enabled", True):
        return {}, {}, {}
    seed_queries = bootstrap_cfg_raw.get("queries")
    if not seed_queries:
        return {}, {}, {}

    extra_context = bootstrap_cfg_raw.get("context")
    effective_context = context
    if isinstance(extra_context, Mapping):
        effective_context = context | extra_context

    providers = bootstrap_cfg_raw.get("providers")
    min_citations = int(bootstrap_cfg_raw.get("min_citations", 0))
    limit_per_query = bootstrap_cfg_raw.get("limit_per_query")
    max_per_query = bootstrap_cfg_raw.get("max_per_query")
    state_prefix = str(bootstrap_cfg_raw.get("state_prefix", "bootstrap::reviews"))
    resume = bool(bootstrap_cfg_raw.get("resume", True))
    citation_overrides = bootstrap_cfg_raw.get("citation_overrides")
    citation_mapping = citation_overrides if isinstance(citation_overrides, Mapping) else None

    provider_filter: Sequence[str] | None = None
    if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
        provider_filter = tuple(str(provider) for provider in providers)

    review_map = pull_top_cited_reviews(
        retriever,
        seed_queries,
        providers=provider_filter,
        min_citations=min_citations,
        limit_per_query=limit_per_query if isinstance(limit_per_query, int) else None,
        max_per_query=max_per_query if isinstance(max_per_query, int) else None,
        state_prefix=state_prefix,
        resume=resume,
        citation_overrides=citation_mapping,
        context=effective_context,
    )

    review_docs = [doc for documents in review_map.values() for doc in documents]
    if not review_docs:
        return dict(bootstrap_cfg_raw), {}, review_map

    max_theories = bootstrap_cfg_raw.get("max_theories")
    theory_cap = int(max_theories) if isinstance(max_theories, int) else None
    extraction_results = [
        extract_theories_from_review(review, llm_client=llm_client, max_theories=theory_cap)
        for review in review_docs
    ]
    bootstrap_nodes = build_bootstrap_ontology(extraction_results)

    cache_path = Path(bootstrap_cfg_raw.get("cache_path") or "data/cache/bootstrap_ontology.json")
    if bootstrap_nodes:
        write_bootstrap_cache(cache_path, seed_queries=seed_queries, review_map=review_map, bootstrap_nodes=bootstrap_nodes)
    elif cache_path.exists():
        logger.debug("Bootstrap produced no ontology updates; cache at %s left untouched", cache_path)

    enriched_config = dict(bootstrap_cfg_raw)
    enriched_config["bootstrap_nodes"] = bootstrap_nodes
    return enriched_config, bootstrap_nodes, review_map


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


def _has_expansion_config(targets: Mapping[str, Any]) -> bool:
    for data in targets.values():
        if not isinstance(data, Mapping):
            continue
        if isinstance(data.get("expansion"), Mapping):
            return True
        sub = data.get("subtheories")
        if isinstance(sub, Mapping) and _has_expansion_config(sub):
            return True
    return False


def collect_for_entry(
    retriever: LiteratureRetriever,
    *,
    name: str,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    providers: Iterable[str] | None,
    resume: bool,
    state_prefix: str,
    ontology: TheoryOntology | None,
    expander: QueryExpander | None,
    default_expansion: QueryExpansionSettings | None,
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

    node = None
    if ontology is not None:
        try:
            node = ontology.get(name)
        except KeyError:
            node = None
    if node is None:
        node = OntologyNode(name=name, target=config.get("target"))

    expansion_cfg = config.get("expansion") if isinstance(config.get("expansion"), Mapping) else None
    expansion_settings = None
    if expander:
        if expansion_cfg:
            expansion_settings = expander.settings_for(expansion_cfg)
        elif default_expansion:
            expansion_settings = expander.settings_for({})

    expansion_session = None
    if (
        expansion_settings
        and expansion_settings.enabled
        and target
        and summary.get("total_unique", 0) < target
        and expander is not None
    ):
        logger.info("Target unmet for %s; attempting adaptive query expansion", name)
        expansion_session = expander.expand(
            node,
            base_queries=queries,
            papers=result.papers,
            settings=expansion_settings,
            context={"current_total": summary.get("total_unique", 0)},
        )
        if expansion_session is not None:
            adaptive_queries = expansion_session.selected_queries()
            if adaptive_queries:
                merged_queries = queries + adaptive_queries
                rerun = retriever.collect_queries(
                    merged_queries,
                    target=target,
                    providers=list(providers) if providers else None,
                    state_key=state_prefix,
                    resume=True,
                )
                before_total = summary.get("total_unique", 0)
                summary = dict(rerun.summary)
                entry_map = {paper.identifier: paper for paper in rerun.papers}
                new_unique = max(0, summary.get("total_unique", 0) - before_total)
                summary["expansion"] = {
                    "enabled": True,
                    "queries": adaptive_queries,
                    "before_total": before_total,
                    "after_total": summary.get("total_unique", 0),
                    "new_unique": new_unique,
                }
                expander.record_performance(
                    expansion_session,
                    before_total=before_total,
                    after_total=summary.get("total_unique", 0),
                    new_unique=new_unique,
                )
            else:
                summary["expansion"] = {"enabled": True, "queries": []}

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
                ontology=ontology,
                expander=expander,
                default_expansion=default_expansion,
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


def _maybe_build_llm_client(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    api_keys: Mapping[str, str | None],
) -> LLMClient | None:
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
    api_key = args.llm_api_key or llm_cfg.get("api_key")
    api_key_key = llm_cfg.get("api_key_key")
    if (not api_key) and api_key_key:
        api_key = api_keys.get(api_key_key)
    api_key = api_key or None

    config_obj = LLMClientConfig(
        model=llm_model,
        temperature=float(temperature),
        batch_size=int(batch_size),
        max_retries=int(max_retries),
        retry_backoff=float(retry_backoff),
        request_timeout=float(request_timeout),
        cache_dir=Path(cache_dir),
    )
    return LLMClient(config_obj, api_key=api_key)


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
        api_keys = resolve_api_keys(
            config.get("api_keys", {}), base_path=config_path.parent
        )
    except MissingSecretError as exc:
        parser.error(str(exc))

    corpus_cfg: Mapping[str, Any] = config.get("corpus", {})
    pipeline_cfg: Mapping[str, Any] = config.get("pipeline", {}) if isinstance(config.get("pipeline"), Mapping) else {}
    parallel_fetch = _resolve_workers(args.parallel_fetch, pipeline_cfg.get("parallel_fetch"), 1)
    classification_workers = _resolve_workers(
        args.classification_workers, pipeline_cfg.get("classification_workers"), 1
    )
    state_dir = Path(args.state_dir or corpus_cfg.get("cache_dir") or config["outputs"].get("cache_dir", "data/cache"))
    provider_configs = build_provider_configs(config, args.providers, api_keys)
    retriever = LiteratureRetriever(
        Path(config["data_sources"]["seed_papers"]),
        provider_configs=provider_configs,
        state_dir=state_dir,
        parallel_fetch=parallel_fetch,
    )

    llm_client = _maybe_build_llm_client(config, args, api_keys)

    expansion_cfg_raw = corpus_cfg.get("expansion")
    expansion_cfg = expansion_cfg_raw if isinstance(expansion_cfg_raw, Mapping) else None

    base_targets = corpus_cfg.get("targets", {})
    context: Dict[str, Any] = {"base_query": args.query, "query": args.query}
    bootstrap_config, bootstrap_nodes, bootstrap_reviews = _run_bootstrap_phase(
        retriever,
        llm_client,
        corpus_cfg,
        context=context,
    )

    update_runtime = bool(bootstrap_config.get("update_targets", False)) if bootstrap_config else False
    if bootstrap_nodes:
        runtime_targets = merge_bootstrap_into_targets(
            base_targets,
            bootstrap_nodes,
            inject_missing=update_runtime,
        )
        ontology_targets = merge_bootstrap_into_targets(
            base_targets,
            bootstrap_nodes,
            inject_missing=True,
        )
        review_total = sum(len(items) for items in bootstrap_reviews.values())
        logger.info(
            "Bootstrap discovered %d root theories from %d review papers",
            len(bootstrap_nodes),
            review_total,
        )
    else:
        runtime_targets = base_targets
        ontology_targets = base_targets

    wants_expansion = (expansion_cfg is not None) or _has_expansion_config(runtime_targets)
    default_expansion: QueryExpansionSettings | None = None
    expander: QueryExpander | None = None
    if wants_expansion:
        default_data = expansion_cfg or {"enabled": False}
        default_expansion = QueryExpansionSettings.from_mapping(default_data)
        expander = QueryExpander(llm_client=llm_client, default_settings=default_expansion)
    elif expansion_cfg:
        logger.warning("Ignoring non-mapping expansion configuration: %s", type(expansion_cfg))

    global_limit = args.limit or corpus_cfg.get("global_limit")
    resume = not args.no_resume

    ontology = TheoryOntology.from_targets_config(ontology_targets)
    collected_papers: Dict[str, PaperMetadata] = {}
    summary_report: Dict[str, Any] = {}

    prioritized = []
    for theory_name, theory_cfg in runtime_targets.items():
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
            ontology=ontology,
            expander=expander,
            default_expansion=default_expansion,
        )
        summary_report[theory_name] = theory_summary
        for paper in theory_papers:
            collected_papers.setdefault(paper.identifier, paper)

    papers = list(collected_papers.values())

    validate_targets(summary_report)
    if global_limit is not None and len(papers) > global_limit:
        papers = papers[: global_limit]

    classifier = TheoryClassifier.from_config(
        config.get("classification", {}), ontology=ontology, llm_client=llm_client
    )
    extractor = QuestionExtractor(config.get("extraction"))
    assignment_groups, answer_groups = classify_and_extract_parallel(
        papers, classifier, extractor, workers=classification_workers
    )
    assignments = [assignment for group in assignment_groups for assignment in group]
    question_answers = [answer for group in answer_groups for answer in group]

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
