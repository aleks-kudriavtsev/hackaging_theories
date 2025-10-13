"""Collect papers and classify theories for the Hackaging challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline import (
    LiteratureRetriever,
    OntologyManager,
    PaperMetadata,
    ProviderConfig,
    QuestionExtractor,
    TheoryClassifier,
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


_API_KEY_OVERRIDE_MAP = {
    "openalex_api_key": "openalex",
    "crossref_api_key": "crossref_contact",
    "pubmed_api_key": "pubmed",
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
    overrides = {
        target_key: getattr(args, cli_attr)
        for cli_attr, target_key in _API_KEY_OVERRIDE_MAP.items()
        if getattr(args, cli_attr, None)
    }
    if overrides:
        resolved = {**resolved, **overrides}
    return resolved


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


def _build_quickstart_node(query: str, target: int) -> Dict[str, Any]:
    name = query.strip() or "Quickstart Query"
    node = {
        "name": name,
        "target": int(target),
        "queries": [query],
        "subtheories": {},
        "metadata": {
            "source": "quickstart",
            "query": query,
            "generated_at": time.time(),
        },
    }
    return node


def _quickstart_config(node: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    name = str(node.get("name")) if node.get("name") else "Quickstart Query"
    config: Dict[str, Dict[str, Any]] = {
        name: {key: value for key, value in node.items() if key != "name"}
    }
    config[name].setdefault("subtheories", {})
    return config


def _persist_quickstart_node(node: Mapping[str, Any], slug: str) -> Path:
    cache_dir = PROJECT_ROOT / "data" / "cache" / "ontologies"
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": node.get("name"),
        "target": node.get("target"),
        "queries": node.get("queries"),
        "metadata": node.get("metadata", {}),
        "subtheories": node.get("subtheories", {}),
    }
    path = cache_dir / f"{slug}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


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
    ontology_manager: OntologyManager | None,
    expander: QueryExpander | None,
    default_expansion: QueryExpansionSettings | None,
    retrieval_options: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[PaperMetadata]]:
    query_templates = config.get("queries") or [context.get("base_query", name)]
    queries = [render_query(template, context | {"query": context.get("base_query", name)}) for template in query_templates]
    queries = [q.strip() for q in queries if q.strip()]
    base_queries = list(queries)
    target = config.get("target")
    min_citation_override = config.get("min_citation_count")
    if min_citation_override is None and retrieval_options is not None:
        min_citation_override = retrieval_options.get("min_citation_count")
    try:
        min_citation_value = (
            int(min_citation_override)
            if min_citation_override is not None
            else None
        )
    except (TypeError, ValueError):
        min_citation_value = None

    prefer_reviews_override = config.get("prefer_reviews")
    if prefer_reviews_override is None and retrieval_options is not None:
        prefer_reviews_override = retrieval_options.get("prefer_reviews")
    prefer_reviews_flag = (
        _coerce_bool(prefer_reviews_override)
        if prefer_reviews_override is not None
        else False
    )

    sort_by_citations_override = config.get("sort_by_citations")
    if sort_by_citations_override is None and retrieval_options is not None:
        sort_by_citations_override = retrieval_options.get("sort_by_citations")
    sort_by_citations_flag = (
        _coerce_bool(sort_by_citations_override)
        if sort_by_citations_override is not None
        else False
    )

    existing_state: Mapping[str, Any] = {}
    if state_prefix:
        try:
            existing_state = retriever.state_store.get(state_prefix)
        except AttributeError:
            existing_state = {}

    enrichment_sources: List[Mapping[str, Any]] = []
    config_enrichment = config.get("enrichment")
    if isinstance(config_enrichment, Mapping):
        enrichment_sources.append(config_enrichment)
    context_enrichment = context.get("enrichment")
    if isinstance(context_enrichment, Mapping):
        enrichment_sources.append(context_enrichment)

    new_theory_entries: List[Mapping[str, Any]] = []
    query_shard_entries: List[Any] = []
    for source in enrichment_sources:
        raw_theories = source.get("new_theories")
        if isinstance(raw_theories, Mapping):
            new_theory_entries.append(raw_theories)
        elif isinstance(raw_theories, Iterable) and not isinstance(raw_theories, (str, bytes)):
            for item in raw_theories:
                if isinstance(item, Mapping):
                    new_theory_entries.append(item)
        raw_shards = source.get("query_shards")
        if isinstance(raw_shards, Mapping):
            query_shard_entries.append(raw_shards)
        elif isinstance(raw_shards, Iterable) and not isinstance(raw_shards, (str, bytes)):
            for item in raw_shards:
                if isinstance(item, (str, Mapping)):
                    query_shard_entries.append(item)
        elif isinstance(raw_shards, str):
            query_shard_entries.append(raw_shards)

    enrichment_state_raw = (
        existing_state.get("enrichment") if isinstance(existing_state, Mapping) else {}
    )
    enrichment_state: Dict[str, Dict[str, Any]] = {
        "new_theories": {},
        "query_shards": {},
    }
    if isinstance(enrichment_state_raw, Mapping):
        stored_theories = enrichment_state_raw.get("new_theories")
        if isinstance(stored_theories, Mapping):
            enrichment_state["new_theories"] = {
                str(key): dict(value) if isinstance(value, Mapping) else {"status": value}
                for key, value in stored_theories.items()
            }
        stored_shards = enrichment_state_raw.get("query_shards")
        if isinstance(stored_shards, Mapping):
            enrichment_state["query_shards"] = {
                str(key): dict(value) if isinstance(value, Mapping) else {"status": value}
                for key, value in stored_shards.items()
            }

    added_theories: List[Dict[str, Any]] = []
    used_query_shards: List[Dict[str, Any]] = []
    pruned_shards: List[Dict[str, Any]] = []

    if ontology_manager and new_theory_entries:
        for entry in new_theory_entries:
            entry_name = entry.get("name") or entry.get("theory")
            if not isinstance(entry_name, str):
                continue
            entry_name = entry_name.strip()
            if not entry_name:
                continue
            parent_name = entry.get("parent")
            if not isinstance(parent_name, str) or not parent_name.strip():
                parent_name = name
            else:
                parent_name = parent_name.strip()
            config_payload = entry.get("config")
            if isinstance(config_payload, Mapping):
                node_config = dict(config_payload)
            else:
                node_config = {}
            if "target" not in node_config and entry.get("target") is not None:
                try:
                    node_config["target"] = int(entry.get("target"))
                except (TypeError, ValueError):
                    pass
            metadata_payload = (
                dict(entry.get("metadata"))
                if isinstance(entry.get("metadata"), Mapping)
                else {}
            )
            keywords_raw = entry.get("keywords")
            keywords_list: List[str] | None = None
            if isinstance(keywords_raw, Iterable) and not isinstance(keywords_raw, (str, bytes)):
                keywords_list = [str(item).strip() for item in keywords_raw if str(item).strip()]
            is_new = False
            if ontology_manager.has_node(entry_name):
                logger.debug("Ontology already contains node '%s'; skipping append", entry_name)
            else:
                try:
                    ontology_manager.append_node(
                        entry_name,
                        parent=parent_name,
                        config=node_config,
                        keywords=keywords_list,
                        metadata=metadata_payload,
                    )
                    is_new = True
                except ValueError as exc:
                    logger.warning("Failed to append ontology node '%s': %s", entry_name, exc)
                    continue
            record = {
                "name": entry_name,
                "parent": parent_name,
                "config": node_config,
                "metadata": metadata_payload,
                "keywords": keywords_list or [],
            }
            enrichment_state["new_theories"].setdefault(entry_name, {}).update(record)
            if is_new:
                added_theories.append(record)
    elif new_theory_entries:
        logger.debug(
            "Received new ontology theories without a manager; entries will be ignored"
        )

    for entry in query_shard_entries:
        if isinstance(entry, Mapping):
            query_text = entry.get("query") or entry.get("text")
            prune_flag = bool(entry.get("prune")) or str(entry.get("status", "")).lower() == "pruned"
            metadata_payload = {
                key: value
                for key, value in entry.items()
                if key not in {"query", "text", "prune", "status"}
            }
        else:
            query_text = entry
            prune_flag = False
            metadata_payload = {}
        if not isinstance(query_text, str):
            continue
        query_text = query_text.strip()
        if not query_text:
            continue
        shard_id = hashlib.sha1(query_text.encode("utf-8")).hexdigest()
        record = enrichment_state["query_shards"].setdefault(shard_id, {})
        record["query"] = query_text
        if metadata_payload:
            meta_existing = record.setdefault("metadata", {})
            if isinstance(meta_existing, MutableMapping):
                meta_existing.update(metadata_payload)
            else:
                record["metadata"] = dict(metadata_payload)
        if prune_flag:
            record["status"] = "pruned"
            pruned_shards.append({"query": query_text})
            continue
        if record.get("status") == "pruned":
            continue
        if query_text not in queries:
            queries.append(query_text)
        record.setdefault("status", "pending")
        used_query_shards.append({"id": shard_id, "query": query_text})

    ontology = ontology_manager.ontology if ontology_manager else None

    result = retriever.collect_queries(
        queries,
        target=target,
        providers=list(providers) if providers else None,
        state_key=state_prefix,
        resume=resume,
        min_citation_count=min_citation_value,
        prefer_reviews=prefer_reviews_flag,
        sort_by_citations=sort_by_citations_flag,
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
                    min_citation_count=min_citation_value,
                    prefer_reviews=prefer_reviews_flag,
                    sort_by_citations=sort_by_citations_flag,
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
        else:
            summary["expansion"] = {"enabled": True, "queries": []}

    timestamp = time.time()
    if state_prefix:
        state_payload = retriever.state_store.get(state_prefix)
        if not isinstance(state_payload, MutableMapping):
            state_payload = {}
        else:
            state_payload = dict(state_payload)
        for shard in used_query_shards:
            shard_state = enrichment_state["query_shards"].setdefault(
                shard["id"], {"query": shard["query"]}
            )
            shard_state["status"] = "consumed"
            shard_state["last_used"] = timestamp
        for theory_record in added_theories:
            theory_state = enrichment_state["new_theories"].setdefault(
                theory_record["name"], {}
            )
            theory_state.update(theory_record)
            theory_state["status"] = "added"
            theory_state["updated_at"] = timestamp
        if pruned_shards:
            for shard in pruned_shards:
                shard_id = hashlib.sha1(shard["query"].encode("utf-8")).hexdigest()
                shard_state = enrichment_state["query_shards"].setdefault(
                    shard_id, {"query": shard["query"]}
                )
                shard_state["status"] = "pruned"
                shard_state["updated_at"] = timestamp
        state_payload["enrichment"] = enrichment_state
        retriever.state_store.set(state_prefix, state_payload)

    enrichment_report: Dict[str, Any] = {}
    new_query_list = [item["query"] for item in used_query_shards if item["query"] not in base_queries]
    if added_theories:
        enrichment_report["new_theories"] = [item["name"] for item in added_theories]
    if new_query_list:
        enrichment_report["queries_used"] = new_query_list
    if pruned_shards:
        enrichment_report["pruned_queries"] = [item["query"] for item in pruned_shards]
    if enrichment_report:
        summary["enrichment"] = enrichment_report

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
                ontology_manager=ontology_manager,
                expander=expander,
                default_expansion=default_expansion,
                retrieval_options=retrieval_options,
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
        "--quickstart",
        action="store_true",
        help=(
            "Generate a temporary ontology node from the CLI query instead of "
            "requiring corpus.targets"
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Target paper quota for the quickstart ontology node",
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
        api_keys = _load_api_keys(
            args,
            config.get("api_keys", {}),
            base_path=config_path.parent,
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

    base_targets_raw = corpus_cfg.get("targets", {})
    if isinstance(base_targets_raw, Mapping):
        base_targets: Dict[str, Any] = dict(base_targets_raw)
    else:
        base_targets = {}

    quickstart_active = bool(args.quickstart or not base_targets)
    quickstart_node: Dict[str, Any] | None = None
    quickstart_cache_path: Path | None = None
    if quickstart_active:
        if args.target_count is None:
            parser.error(
                "--target-count is required when --quickstart is used or corpus.targets is empty"
            )
        quickstart_node = _build_quickstart_node(args.query, args.target_count)
        quickstart_slug = slugify(args.query)
        quickstart_node.setdefault("metadata", {}).setdefault("slug", quickstart_slug)
        quickstart_cache_path = _persist_quickstart_node(quickstart_node, quickstart_slug)
        logger.info(
            "Quickstart mode active; generated ontology node persisted to %s",
            quickstart_cache_path,
        )
        base_targets = _quickstart_config(quickstart_node)
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

    retrieval_defaults: Dict[str, Any] = {}
    if "min_citation_count" in corpus_cfg:
        try:
            retrieval_defaults["min_citation_count"] = int(corpus_cfg.get("min_citation_count"))
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid corpus.min_citation_count value: %s",
                corpus_cfg.get("min_citation_count"),
            )
    if "prefer_reviews" in corpus_cfg:
        retrieval_defaults["prefer_reviews"] = _coerce_bool(corpus_cfg.get("prefer_reviews"))
    if "sort_by_citations" in corpus_cfg:
        retrieval_defaults["sort_by_citations"] = _coerce_bool(corpus_cfg.get("sort_by_citations"))

    ontology_manager = OntologyManager(
        ontology_targets, storage_path=state_dir / "runtime_ontology.json"
    )
    ontology = ontology_manager.ontology
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
            ontology_manager=ontology_manager,
            expander=expander,
            default_expansion=default_expansion,
            retrieval_options=retrieval_defaults,
        )
        summary_report[theory_name] = theory_summary
        for paper in theory_papers:
            collected_papers.setdefault(paper.identifier, paper)

    papers = list(collected_papers.values())

    if not quickstart_active:
        validate_targets(summary_report)
    if global_limit is not None and len(papers) > global_limit:
        papers = papers[: global_limit]

    ontology = ontology_manager.ontology

    classifier = TheoryClassifier.from_config(
        config.get("classification", {}), ontology=ontology, llm_client=llm_client
    )
    classifier.attach_manager(ontology_manager)
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

    if quickstart_active and quickstart_cache_path is not None:
        print(f"Quickstart ontology node cached at {quickstart_cache_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
