"""Bootstrap an ontology from review articles for quickstart runs."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from .filtering import RelevanceFilter
from .literature import LiteratureRetriever, PaperMetadata
from .llm import LLMClient, LLMClientError, LLMMessage

logger = logging.getLogger(__name__)

_THEORY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9\- ]+? [Tt]heor(?:y|ies))\b")


@dataclass(frozen=True)
class BootstrapSummary:
    """Summary of the ontology bootstrap step."""

    query: str
    accepted: int
    rejected: int
    theory_count: int
    snapshot_path: Path

    def to_dict(self) -> Mapping[str, object]:
        return {
            "query": self.query,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "theory_count": self.theory_count,
            "snapshot_path": str(self.snapshot_path),
        }


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "ontology"


def _extract_theories_from_text(text: str) -> List[str]:
    matches = _THEORY_PATTERN.findall(text)
    cleaned = []
    seen = set()
    for match in matches:
        label = match.strip().replace("  ", " ")
        normalized = label.title()
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def _extract_candidates(
    papers: Iterable[PaperMetadata],
    *,
    llm_client: LLMClient | None,
    max_labels: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    snippets: List[str] = []
    for paper in papers:
        text = " ".join(filter(None, (paper.title, paper.abstract)))
        snippets.append(text)
        for label in _extract_theories_from_text(text):
            counter[label] += 1
    if not llm_client or not snippets:
        return Counter(dict(counter.most_common(max_labels)))

    payload = "\n".join(f"- {snippet}" for snippet in snippets[:20])
    messages = [
        LLMMessage(
            role="system",
            content=(
                "You are creating a taxonomy of scientific theories of aging. "
                "List distinct theory names mentioned in the bullet list. Respond "
                "with JSON containing a 'theories' array of strings."
            ),
        ),
        LLMMessage(
            role="user",
            content=f"Candidate snippets:\n{payload}\nReturn JSON only.",
        ),
    ]
    try:
        response = llm_client.generate([messages])[0]
        data = json.loads(response.content)
        theories = data.get("theories", [])
        if isinstance(theories, Sequence):
            for item in theories:
                if isinstance(item, str):
                    counter[item.strip().title()] += 1
    except (json.JSONDecodeError, LLMClientError, IndexError) as exc:  # pragma: no cover - defensive
        logger.warning("Failed to parse LLM ontology suggestions: %s", exc)
    return Counter(dict(counter.most_common(max_labels)))


def _build_targets(
    theory_counts: Mapping[str, int],
    *,
    total_target: int,
    min_target: int,
    query: str,
) -> Dict[str, MutableMapping[str, object]]:
    if not theory_counts:
        fallback_name = f"{query.title()} Overview"
        return {
            fallback_name: {
                "target": max(min_target, total_target),
                "queries": [f"\"{query}\" review"],
                "metadata": {"source": "bootstrap", "strategy": "fallback"},
            }
        }

    theories = list(theory_counts.keys())
    per_node = max(min_target, total_target // max(len(theories), 1))
    targets: Dict[str, MutableMapping[str, object]] = {}
    for name, count in theory_counts.items():
        targets[name] = {
            "target": max(min_target, per_node),
            "queries": [
                f"\"{name}\" aging",
                f"\"{name}\" longevity",
                f"aging theory {name}",
            ],
            "metadata": {
                "source": "bootstrap",
                "seed_mentions": int(count),
            },
        }
    return targets


def bootstrap_ontology(
    query: str,
    retriever: LiteratureRetriever,
    *,
    llm_client: LLMClient | None = None,
    providers: Sequence[str] | None = None,
    resume: bool = True,
    target_count: int = 300,
    config: Mapping[str, object] | None = None,
) -> Tuple[Dict[str, MutableMapping[str, object]], BootstrapSummary, List[PaperMetadata]]:
    """Return an ontology targets dict for a quickstart query."""

    bootstrap_cfg = config or {}
    review_limit = int(bootstrap_cfg.get("review_limit", 200))
    relevance_threshold = float(bootstrap_cfg.get("relevance_threshold", 0.35))
    min_target = int(bootstrap_cfg.get("min_target", 25))
    cache_dir = Path(bootstrap_cfg.get("cache_dir", "data/cache/ontologies"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    state_key = f"bootstrap::{_slugify(query)}"
    retrieval = retriever.collect_queries(
        [f"{query} review", query],
        target=review_limit,
        providers=providers,
        state_key=state_key,
        resume=resume,
    )
    papers = retrieval.papers

    filter_client = RelevanceFilter(query, threshold=relevance_threshold, llm_client=llm_client)
    decisions = filter_client.filter(papers)
    accepted_ids = {decision.paper_id for decision in decisions if decision.accepted}
    accepted = [paper for paper in papers if paper.identifier in accepted_ids]
    rejected = [paper for paper in papers if paper.identifier not in accepted_ids]
    if not accepted:
        accepted = papers

    theory_counts = _extract_candidates(accepted, llm_client=llm_client, max_labels=bootstrap_cfg.get("max_theories", 12))
    targets = _build_targets(theory_counts, total_target=target_count, min_target=min_target, query=query)

    snapshot = {
        "query": query,
        "targets": targets,
        "filter": [decision.to_dict() for decision in decisions],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
    }
    snapshot_path = cache_dir / f"{_slugify(query)}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = BootstrapSummary(
        query=query,
        accepted=len(accepted),
        rejected=len(rejected),
        theory_count=len(targets),
        snapshot_path=snapshot_path,
    )
    return targets, summary, accepted


__all__ = ["bootstrap_ontology", "BootstrapSummary"]
