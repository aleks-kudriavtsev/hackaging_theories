"""Bootstrap helpers for discovering theory candidates from review papers.

This module implements a lightweight review bootstrap workflow that can be used
before the main corpus collection stage.  The helpers retrieve highly cited
review papers for a set of seed queries, normalise their metadata, and then
extract provisional theory/subtheory hierarchies either via the configured LLM
client or a deterministic keyword heuristic.  The resulting hierarchy can be
merged into the runtime ontology so the downstream classifier is aware of newly
proposed theories even when they are missing from ``corpus.targets``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .literature import LiteratureRetriever, PaperMetadata
from .llm import LLMClient, LLMClientError, LLMMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewDocument:
    """Wrapper containing metadata for a review paper selected during bootstrap."""

    query: str
    paper: PaperMetadata
    citations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "identifier": self.paper.identifier,
            "title": self.paper.title,
            "authors": list(self.paper.authors),
            "abstract": self.paper.abstract,
            "source": self.paper.source,
            "year": self.paper.year,
            "doi": self.paper.doi,
            "citations": self.citations,
        }

    @property
    def text(self) -> str:
        analysis = self.paper.analysis_text
        return analysis if analysis else "\n".join(filter(None, (self.paper.title, self.paper.abstract)))


@dataclass(frozen=True)
class BootstrapResult:
    """Theory hierarchy extracted from a single review document."""

    review: ReviewDocument
    theories: List[Dict[str, Any]]


@dataclass(frozen=True)
class _SeedQueryConfig:
    name: str
    query: str
    min_citations: int
    providers: Optional[Sequence[str]]
    limit: Optional[int]
    max_reviews: Optional[int]
    state_key: Optional[str]


@dataclass
class _AggregatedNode:
    name: str
    citations: int = 0
    reviews: set[str] = field(default_factory=set)
    queries: set[str] = field(default_factory=set)
    children: Dict[str, "_AggregatedNode"] = field(default_factory=dict)

    def to_config(self) -> Dict[str, Any]:
        child_items = {name: child.to_config() for name, child in sorted(self.children.items())}
        bootstrap_info: Dict[str, Any] = {
            "citations": self.citations,
            "reviews": sorted(self.reviews),
        }
        if self.queries:
            bootstrap_info["queries"] = sorted(self.queries)
        direct_count, child_citations = self.child_metrics()
        branch_size = self.leaf_count()
        if direct_count or branch_size:
            bootstrap_info["child_summary"] = {
                "count": direct_count,
                "citations": child_citations,
                "branch_size": branch_size,
            }
        return {
            "bootstrap": bootstrap_info,
            "subtheories": child_items,
        }

    def child_metrics(self) -> tuple[int, int]:
        count = len(self.children)
        citations = sum(child.citations for child in self.children.values())
        return count, citations

    def leaf_count(self) -> int:
        if not self.children:
            return 1
        return sum(child.leaf_count() for child in self.children.values())


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "bootstrap"


def _render_query(template: str, context: Mapping[str, Any] | None) -> str:
    if not context:
        return template
    try:
        return template.format_map(_SafeDict(context))
    except KeyError:  # pragma: no cover - defensive
        return template


def _looks_like_review(paper: PaperMetadata) -> bool:
    if paper.is_review is True:
        return True
    if paper.is_review is False:
        return False
    text = " ".join(part for part in (paper.title, paper.abstract, paper.full_text) if part)
    if not text:
        return False
    return bool(re.search(r"\b(review|systematic review|meta-?analysis)\b", text, re.IGNORECASE))


def _clean_label(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value.strip())
    cleaned = re.sub(r"[\s\-:,.;]+$", "", cleaned)
    return cleaned


_CITATION_PATTERN = re.compile(r"citations?\s*[:=]\s*(\d+)", re.IGNORECASE)


def _extract_citations(
    paper: PaperMetadata,
    overrides: Mapping[str, int] | None,
) -> int:
    if overrides and paper.identifier in overrides:
        try:
            return int(overrides[paper.identifier])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            logger.debug("Invalid citation override for %s", paper.identifier)
    if paper.citation_count is not None:
        try:
            return int(paper.citation_count)
        except (TypeError, ValueError):
            logger.debug("Invalid citation count on %s: %s", paper.identifier, paper.citation_count)
    sources = [paper.abstract, paper.full_text] + [section.text for section in paper.sections]
    for source in sources:
        if not source:
            continue
        match = _CITATION_PATTERN.search(source)
        if match:
            return int(match.group(1))
    return 0


def _normalize_seed_queries(
    seed_queries: Sequence[Any] | Mapping[str, Any],
    *,
    default_min_citations: int,
    default_limit: Optional[int],
    default_max_reviews: Optional[int],
    default_providers: Optional[Sequence[str]],
    default_state_prefix: str,
    context: Mapping[str, Any] | None,
) -> List[_SeedQueryConfig]:
    configs: List[_SeedQueryConfig] = []
    if isinstance(seed_queries, Mapping):
        items = seed_queries.items()
    else:
        items = [(str(item), item) for item in seed_queries]
    for name, raw in items:
        if isinstance(raw, Mapping):
            template = str(raw.get("query", name))
            providers = raw.get("providers")
            min_citations = int(raw.get("min_citations", default_min_citations))
            limit = raw.get("limit")
            max_reviews = raw.get("max_reviews")
            state_key = raw.get("state_key")
        else:
            template = str(raw)
            providers = None
            min_citations = default_min_citations
            limit = None
            max_reviews = None
            state_key = None
        query = _render_query(template, context)
        config = _SeedQueryConfig(
            name=name,
            query=query,
            min_citations=min_citations,
            providers=tuple(providers) if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)) else None,
            limit=int(limit) if isinstance(limit, int) else default_limit,
            max_reviews=int(max_reviews) if isinstance(max_reviews, int) else default_max_reviews,
            state_key=str(state_key) if state_key else f"{default_state_prefix}::{_slugify(name)}",
        )
        configs.append(config)
    return configs


def pull_top_cited_reviews(
    retriever: LiteratureRetriever,
    seed_queries: Sequence[Any] | Mapping[str, Any],
    *,
    providers: Optional[Sequence[str]] = None,
    min_citations: int = 0,
    limit_per_query: Optional[int] = None,
    max_per_query: Optional[int] = None,
    state_prefix: str = "bootstrap::reviews",
    resume: bool = True,
    citation_overrides: Mapping[str, int] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Dict[str, List[ReviewDocument]]:
    """Retrieve candidate review papers for each seed query.

    Parameters
    ----------
    retriever:
        Active :class:`LiteratureRetriever` instance.
    seed_queries:
        Iterable or mapping describing the seed queries.  Items may either be
        strings or mappings supporting the keys ``query``, ``providers``,
        ``min_citations``, ``limit``, ``max_reviews``, and ``state_key``.
    providers:
        Optional global provider filter that applies when a seed does not define
        its own provider list.
    min_citations:
        Global citation threshold; per-seed overrides take precedence.
    limit_per_query:
        Optional maximum number of papers fetched from providers for each seed.
    max_per_query:
        Optional cap on the number of review documents returned per seed after
        filtering and sorting by citation count.
    state_prefix:
        Prefix used to derive ``state_key`` values when the seed does not
        specify one explicitly.
    resume:
        Whether to resume from cached provider state.
    citation_overrides:
        Optional mapping of ``paper.identifier`` to explicit citation counts.
    context:
        Additional substitution variables used when rendering seed query
        templates.
    """

    seed_configs = _normalize_seed_queries(
        seed_queries,
        default_min_citations=min_citations,
        default_limit=limit_per_query,
        default_max_reviews=max_per_query,
        default_providers=providers,
        default_state_prefix=state_prefix,
        context=context,
    )

    results: Dict[str, List[ReviewDocument]] = {}
    for seed in seed_configs:
        selected_providers = seed.providers if seed.providers is not None else providers
        try:
            retrieval = retriever.collect_queries(
                [seed.query],
                target=seed.limit,
                providers=list(selected_providers) if selected_providers else None,
                state_key=seed.state_key,
                resume=resume,
            )
        except Exception as exc:  # pragma: no cover - defensive, surfaces in logs
            logger.error("Failed to collect review papers for '%s': %s", seed.query, exc)
            continue
        documents: List[ReviewDocument] = []
        seen_ids: set[str] = set()
        for paper in retrieval.papers:
            if paper.identifier in seen_ids:
                continue
            seen_ids.add(paper.identifier)
            if selected_providers and paper.source not in selected_providers:
                continue
            if not _looks_like_review(paper):
                continue
            citations = _extract_citations(paper, citation_overrides)
            if citations < seed.min_citations:
                continue
            documents.append(ReviewDocument(query=seed.query, paper=paper, citations=citations))
        documents.sort(key=lambda doc: (-doc.citations, doc.paper.title.lower()))
        if seed.max_reviews is not None:
            documents = documents[: seed.max_reviews]
        results[seed.name] = documents
    return results


def normalise_review_metadata(reviews: Iterable[ReviewDocument]) -> List[Dict[str, Any]]:
    """Convert review documents into serialisable metadata dictionaries."""

    return [review.to_dict() for review in reviews]


def _parse_llm_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        candidates = payload.get("theories") or payload.get("nodes") or payload.get("results")
        if candidates is None and {"name", "subtheories"} <= payload.keys():
            candidates = [payload]
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        candidates = payload
    else:
        return []
    results: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("theory") or "").strip()
        if not name:
            continue
        raw_children = item.get("subtheories") or item.get("children") or []
        children = _parse_llm_payload(raw_children)
        results.append({"name": name, "subtheories": children})
    return results


_THEORY_LINE = re.compile(r"(?P<name>[A-Z][A-Za-z0-9\-&\s]+?(?:Theory|Model))", re.MULTILINE)
_SUB_LIST = re.compile(r"sub(?:theories|domains|concepts)?\s*[:=-]\s*(?P<subs>[^\n]+)", re.IGNORECASE)
_BULLET_SUB = re.compile(r"^-\s*([A-Za-z0-9][^\n]+)$", re.MULTILINE)


def _deterministic_hierarchy(review: ReviewDocument, max_theories: Optional[int]) -> List[Dict[str, Any]]:
    text = review.text
    if not text:
        return []
    nodes: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for match in _THEORY_LINE.finditer(text):
        name = match.group("name").strip()
        normalized = _clean_label(name)
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        segment_start = max(0, match.start() - 80)
        segment_end = min(len(text), match.end() + 200)
        segment = text[segment_start:segment_end]
        subtheories: List[Dict[str, Any]] = []
        list_match = _SUB_LIST.search(segment)
        if list_match:
            for candidate in re.split(r"[,;]", list_match.group("subs")):
                cleaned = _clean_label(candidate)
                if not cleaned:
                    continue
                subtheories.append({"name": cleaned, "subtheories": []})
        else:
            for bullet in _BULLET_SUB.findall(segment):
                cleaned = _clean_label(bullet)
                if not cleaned:
                    continue
                subtheories.append({"name": cleaned, "subtheories": []})
        nodes.append({"name": normalized, "subtheories": subtheories})
        if max_theories is not None and len(nodes) >= max_theories:
            break
    return nodes


def extract_theories_from_review(
    review: ReviewDocument,
    *,
    llm_client: LLMClient | None = None,
    max_theories: Optional[int] = None,
) -> BootstrapResult:
    """Extract theory candidates from a review document."""

    llm_nodes: List[Dict[str, Any]] = []
    if llm_client is not None:
        system_prompt = (
            "You analyse gerontology review papers and return structured theory "
            "hierarchies. Respond with JSON containing a 'theories' array where "
            "each item has 'name' and optional 'subtheories'."
        )
        user_prompt = (
            f"Title: {review.paper.title}\n"
            f"Citations: {review.citations}\n"
            f"Query: {review.query}\n"
            f"Abstract: {review.paper.abstract}\n"
            f"Full Text (truncated):\n{review.text[:4000]}"
        )
        messages = [LLMMessage("system", system_prompt), LLMMessage("user", user_prompt)]
        try:
            response = llm_client.generate([messages])[0]
            llm_payload = json.loads(response.content)
            llm_nodes = _parse_llm_payload(llm_payload)
        except (LLMClientError, json.JSONDecodeError, IndexError) as exc:
            logger.warning("LLM extraction failed for %s: %s", review.paper.identifier, exc)
            llm_nodes = []
    if not llm_nodes:
        llm_nodes = _deterministic_hierarchy(review, max_theories)
    if max_theories is not None:
        llm_nodes = llm_nodes[:max_theories]
    return BootstrapResult(review=review, theories=llm_nodes)


def _merge_node(
    container: Dict[str, _AggregatedNode],
    node_data: Mapping[str, Any],
    review: ReviewDocument,
) -> None:
    name_raw = node_data.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        return
    name = _clean_label(name_raw)
    aggregated = container.get(name)
    if aggregated is None:
        aggregated = _AggregatedNode(name=name)
        container[name] = aggregated
    aggregated.citations += review.citations
    aggregated.reviews.add(review.paper.identifier)
    aggregated.queries.add(review.query)
    child_container = aggregated.children
    for child in node_data.get("subtheories", []):
        if isinstance(child, Mapping):
            _merge_node(child_container, child, review)


def _aggregate_children(children: Iterable[_AggregatedNode]) -> tuple[int, set[str], set[str]]:
    total_citations = 0
    reviews: set[str] = set()
    queries: set[str] = set()
    for child in children:
        total_citations += child.citations
        reviews.update(child.reviews)
        queries.update(child.queries)
    return total_citations, reviews, queries


def _next_other_name(existing: set[str], used: set[str]) -> str:
    index = 1
    while True:
        name = "Other" if index == 1 else f"Other {index}"
        if name not in existing and name not in used:
            used.add(name)
            return name
        index += 1


def _rebalance_node(node: _AggregatedNode, max_children: int | None) -> None:
    if max_children is None or max_children < 1:
        for child in node.children.values():
            _rebalance_node(child, max_children)
        return

    for child in node.children.values():
        _rebalance_node(child, max_children)

    children = list(node.children.values())
    if len(children) <= max_children:
        return

    sorted_children = sorted(children, key=lambda child: (-child.citations, child.name.lower()))
    group_count = max_children
    groups: list[list[_AggregatedNode]] = [[] for _ in range(group_count)]
    for index, child in enumerate(sorted_children):
        groups[index % group_count].append(child)

    existing_names = set(node.children.keys())
    new_children: Dict[str, _AggregatedNode] = {}
    used_names: set[str] = set()

    for group in groups:
        if not group:
            continue
        if len(group) == 1:
            single = group[0]
            new_children[single.name] = single
            continue
        aggregator_name = _next_other_name(existing_names, used_names)
        citations, reviews, queries = _aggregate_children(group)
        aggregator = _AggregatedNode(name=aggregator_name, citations=citations, reviews=reviews, queries=queries)
        aggregator.children = {child.name: child for child in group}
        _rebalance_node(aggregator, max_children)
        new_children[aggregator.name] = aggregator

    node.children = new_children


def _rebalance_root(children: Dict[str, _AggregatedNode], max_children: int | None) -> Dict[str, _AggregatedNode]:
    if max_children is None or max_children < 1 or len(children) <= max_children:
        for child in children.values():
            _rebalance_node(child, max_children)
        return children

    pseudo_root = _AggregatedNode(name="__root__", children=dict(children))
    _rebalance_node(pseudo_root, max_children)
    return pseudo_root.children


def build_bootstrap_ontology(
    results: Iterable[BootstrapResult],
    *,
    max_children: int | None = None,
) -> Dict[str, Any]:
    """Aggregate bootstrap results into a nested ontology mapping."""

    root: Dict[str, _AggregatedNode] = {}
    for result in results:
        for node in result.theories:
            _merge_node(root, node, result.review)

    balanced = _rebalance_root(root, max_children)
    return {name: aggregated.to_config() for name, aggregated in sorted(balanced.items())}


def merge_bootstrap_into_targets(
    targets: Mapping[str, Any],
    bootstrap_nodes: Mapping[str, Any],
    *,
    inject_missing: bool,
) -> Dict[str, Any]:
    """Merge bootstrap ontology nodes into the corpus targets configuration."""

    if not bootstrap_nodes:
        return json.loads(json.dumps(targets))  # Deep copy for consistency

    def _deep_copy(data: Mapping[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(data))

    merged = _deep_copy(targets)

    def _apply(
        container: MutableMapping[str, Any],
        name: str,
        node_data: Mapping[str, Any],
        allow_create: bool,
    ) -> None:
        entry = container.get(name)
        if entry is None:
            if not allow_create:
                return
            entry = {"target": None, "subtheories": {}}
            container[name] = entry
        elif not isinstance(entry, MutableMapping):
            entry = {"target": entry}
            container[name] = entry
        bootstrap_info = node_data.get("bootstrap")
        if bootstrap_info:
            existing = entry.get("bootstrap", {}) if isinstance(entry.get("bootstrap"), MutableMapping) else {}
            citations = int(existing.get("citations", 0)) + int(bootstrap_info.get("citations", 0))
            reviews = set(existing.get("reviews", [])) | set(bootstrap_info.get("reviews", []))
            queries = set(existing.get("queries", [])) | set(bootstrap_info.get("queries", []))
            updated = {"citations": citations, "reviews": sorted(reviews)}
            if queries:
                updated["queries"] = sorted(queries)
            entry["bootstrap"] = updated
        subtarget = entry.get("subtheories")
        if not isinstance(subtarget, MutableMapping):
            subtarget = {}
            entry["subtheories"] = subtarget
        for child_name, child_data in node_data.get("subtheories", {}).items():
            _apply(subtarget, child_name, child_data, True)

    for theory_name, node_data in bootstrap_nodes.items():
        existing = theory_name in merged
        if not existing and not inject_missing:
            # Skip nodes entirely missing from the base config when injection is disabled.
            continue
        _apply(merged, theory_name, node_data, existing or inject_missing)
    return merged


def write_bootstrap_cache(
    path: Path,
    *,
    seed_queries: Sequence[Any] | Mapping[str, Any],
    review_map: Mapping[str, Sequence[ReviewDocument]],
    bootstrap_nodes: Mapping[str, Any],
) -> None:
    """Persist bootstrap artefacts for reproducibility and auditing."""

    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "queries": seed_queries if not isinstance(seed_queries, Mapping) else {k: v for k, v in seed_queries.items()},
        "reviews": {name: normalise_review_metadata(items) for name, items in review_map.items()},
        "ontology": bootstrap_nodes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ReviewDocument",
    "BootstrapResult",
    "pull_top_cited_reviews",
    "normalise_review_metadata",
    "extract_theories_from_review",
    "build_bootstrap_ontology",
    "merge_bootstrap_into_targets",
    "write_bootstrap_cache",
]
