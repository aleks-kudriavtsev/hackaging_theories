"""Utilities for post-processing ontology outputs.

This module implements the "optimization ontology" step requested for the
Hackaging pipeline.  The goal is to balance the number of supporting articles
assigned to each theory after the ontology has been generated:

* Prefer around three papers per theory (with a hard lower bound of one and an
  upper bound of four).
* When a theory collects more than the permitted number of papers, split it
  into additional theories so that every resulting node respects the
  constraints.
* Articles that appear in many theories are re-assigned to those with the
  lowest paper counts in order to even out coverage across the ontology.

The entry point :func:`optimise_ontology_payload` mutates the in-memory
representation of an ``aging_ontology.json`` payload to enforce those rules and
returns a structured summary of the adjustments performed.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .llm import LLMClient, LLMClientError, LLMResponse


logger = logging.getLogger(__name__)

_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_PIPELINE_ID = "ontology-splitter:v2"


def _slugify(value: str) -> str:
    """Return a filesystem-friendly slug for ``value``."""

    normalized = value.strip().lower()
    candidate = _SLUG_TOKEN.sub("-", normalized).strip("-")
    return candidate or "theory"


def _normalise_articles(articles: Iterable[Any]) -> List[str]:
    """Return a deduplicated list of article identifiers preserving order."""

    seen: set[str] = set()
    cleaned: List[str] = []
    for entry in articles:
        text = str(entry).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _ensure_list(mapping: MutableMapping[str, Any], key: str) -> List[Any]:
    value = mapping.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        cleaned = list(value)
    elif value is None:
        cleaned = []
    else:
        cleaned = [value]
    mapping[key] = cleaned
    return cleaned


def _theory_label(theory: Mapping[str, Any]) -> str:
    for key in ("label", "preferred_label", "name", "theory_id"):
        value = theory.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Theory"


def _ensure_theory_id(theory: MutableMapping[str, Any], used: set[str]) -> str:
    """Guarantee that ``theory`` exposes a unique ``theory_id`` key."""

    for key in ("theory_id", "id", "identifier"):
        value = theory.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate in used:
                base = candidate
                counter = 2
                while candidate in used:
                    candidate = f"{base}-{counter}"
                    counter += 1
                theory[key] = candidate
            used.add(candidate)
            if key != "theory_id":
                theory.setdefault("theory_id", candidate)
            return theory["theory_id"]

    slug = _slugify(_theory_label(theory))
    candidate = slug
    counter = 2
    while candidate in used:
        candidate = f"{slug}-{counter}"
        counter += 1
    theory["theory_id"] = candidate
    used.add(candidate)
    return candidate


@dataclass
class TheoryRef:
    """Lightweight wrapper exposing convenience helpers for a theory node."""

    key: str
    label: str
    group: MutableMapping[str, Any]
    node: MutableMapping[str, Any]

    def articles(self) -> List[str]:
        current = self.node.get("supporting_articles")
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            cleaned = _normalise_articles(current)
            if cleaned != list(current):
                self.node["supporting_articles"] = cleaned
            return cleaned
        self.node["supporting_articles"] = []
        return []

    def update_articles(self, values: Iterable[Any]) -> None:
        self.node["supporting_articles"] = _normalise_articles(values)


@dataclass
class TheorySplitCandidate:
    """Container describing a theory that exceeds the configured cap."""

    ref: TheoryRef
    articles: List[str]


@dataclass
class TheorySplitProposal:
    """Single theory proposal returned from the LLM splitter."""

    label: str
    rationale: str
    article_ids: List[str]


@dataclass
class TheorySplitDecision:
    """LLM-backed decision describing how to restructure a theory."""

    candidate: TheorySplitCandidate
    proposals: List[TheorySplitProposal]
    overall_rationale: str
    source: str = "llm"

    @property
    def changed(self) -> bool:
        if not self.proposals:
            return False
        candidate_articles = set(self.candidate.articles)
        first = self.proposals[0]
        if len(self.proposals) > 1:
            return True
        if set(first.article_ids) != candidate_articles:
            return True
        if first.label.strip() != self.candidate.ref.label.strip():
            return True
        return False


@dataclass
class OptimizationSummary:
    """Structured report returned by :func:`optimise_ontology_payload`."""

    changed: bool
    iterations: int = 0
    created_theories: List[Dict[str, Any]] = field(default_factory=list)
    updated_theories: List[Dict[str, Any]] = field(default_factory=list)
    article_reassignments: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "iterations": self.iterations,
            "created_theories": self.created_theories,
            "updated_theories": self.updated_theories,
            "article_reassignments": self.article_reassignments,
            "decisions": self.decisions,
        }


class OntologyOptimizer:
    """Encapsulate the optimization algorithm for ontology payloads."""

    def __init__(
        self,
        groups: Sequence[MutableMapping[str, Any]],
        *,
        target: int = 3,
        minimum: int = 1,
        maximum: int = 4,
        llm_client: Optional[LLMClient] = None,
        llm_model: Optional[str] = "gpt-5-mini",
        llm_temperature: Optional[float] = None,
        batch_size: int = 4,
        max_concurrency: int = 1,
        article_catalog: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        if minimum <= 0:
            raise ValueError("minimum must be greater than zero")
        if maximum < minimum:
            raise ValueError("maximum must be greater than or equal to minimum")
        if target < minimum:
            target = minimum
        if target > maximum:
            target = maximum
        self.groups = groups
        self.target = target
        self.minimum = minimum
        self.maximum = maximum
        self._used_ids: set[str] = set()
        self.summary = OptimizationSummary(changed=False)
        self.llm_client = llm_client
        effective_model = llm_model
        if effective_model is None and llm_client is not None:
            effective_model = getattr(getattr(llm_client, "config", None), "model", None)
        self.llm_model = effective_model
        effective_temperature = llm_temperature
        if effective_temperature is None and llm_client is not None:
            effective_temperature = getattr(getattr(llm_client, "config", None), "temperature", None)
        self.llm_temperature = effective_temperature
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self.article_catalog = article_catalog or {}
        self._batches_executed = 0
        self._fallback_decisions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> OptimizationSummary:
        self._refresh_used_ids()

        candidates = self._collect_split_candidates()
        if not candidates:
            self.summary.iterations = 0
            return self.summary

        decisions: List[TheorySplitDecision] = []
        if self.llm_client is not None:
            decisions.extend(self._evaluate_with_llm(candidates))
        else:
            for candidate in candidates:
                decisions.append(self._fallback_decision(candidate, "LLM client unavailable"))

        for decision in decisions:
            applied = self._apply_decision(decision)
            self.summary.decisions.append(
                {
                    "theory_id": decision.candidate.ref.key,
                    "rationale": decision.overall_rationale,
                    "source": decision.source,
                    "proposals": [
                        {
                            "label": proposal.label,
                            "articles": list(proposal.article_ids),
                        }
                        for proposal in decision.proposals
                    ],
                }
            )
            if applied:
                self.summary.changed = True

        self.summary.iterations = len(decisions)
        return self.summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_used_ids(self) -> None:
        self._used_ids.clear()
        for ref in self._collect_refs():
            self._used_ids.add(ref.key)

    def _collect_refs(self) -> List[TheoryRef]:
        refs: List[TheoryRef] = []

        def visit(group: MutableMapping[str, Any]) -> None:
            theories = group.get("theories")
            if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
                for node in theories:
                    if not isinstance(node, MutableMapping):
                        continue
                    key = _ensure_theory_id(node, self._used_ids)
                    label = _theory_label(node)
                    refs.append(TheoryRef(key=key, label=label, group=group, node=node))

            subgroups = group.get("subgroups")
            if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
                for child in subgroups:
                    if isinstance(child, MutableMapping):
                        visit(child)

        for group in self.groups:
            if isinstance(group, MutableMapping):
                visit(group)

        return refs

    def _unique_theory_id(self, candidate: str) -> str:
        if candidate not in self._used_ids:
            return candidate
        base = candidate
        counter = 2
        while candidate in self._used_ids:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _collect_split_candidates(self) -> List[TheorySplitCandidate]:
        candidates: List[TheorySplitCandidate] = []
        for ref in self._collect_refs():
            articles = ref.articles()
            if len(articles) > self.maximum:
                candidates.append(TheorySplitCandidate(ref=ref, articles=list(articles)))
        return candidates

    def _evaluate_with_llm(self, candidates: Sequence[TheorySplitCandidate]) -> List[TheorySplitDecision]:
        batches: List[Sequence[TheorySplitCandidate]] = []
        for start in range(0, len(candidates), self.batch_size):
            batches.append(candidates[start : start + self.batch_size])

        decisions: List[TheorySplitDecision] = []
        semaphore = None
        try:
            from threading import Semaphore

            semaphore = Semaphore(self.max_concurrency)
        except Exception:  # pragma: no cover - fallback when threading unavailable
            semaphore = None

        def worker(batch_index: int, batch_candidates: Sequence[TheorySplitCandidate]) -> Tuple[int, List[TheorySplitDecision]]:
            if semaphore is not None:
                semaphore.acquire()
            try:
                messages_batch = [self._build_messages(candidate) for candidate in batch_candidates]
                responses = self.llm_client.generate(
                    messages_batch,
                    model=self.llm_model,
                    temperature=self.llm_temperature,
                )
            except LLMClientError as error:
                logger.warning("LLM split request failed: %s", error)
                result: List[TheorySplitDecision] = [
                    self._fallback_decision(candidate, f"LLM error: {error}")
                    for candidate in batch_candidates
                ]
                return batch_index, result
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception("Unexpected LLM failure during ontology optimisation: %s", exc)
                result = [
                    self._fallback_decision(candidate, "Unexpected LLM failure")
                    for candidate in batch_candidates
                ]
                return batch_index, result
            finally:
                if semaphore is not None:
                    semaphore.release()

            batch_decisions: List[TheorySplitDecision] = []
            if len(responses) != len(batch_candidates):
                logger.warning(
                    "LLM returned %d responses for %d candidates; falling back for missing entries",
                    len(responses),
                    len(batch_candidates),
                )
            for candidate, response in zip(batch_candidates, responses):
                decision = self._parse_response(candidate, response)
                batch_decisions.append(decision)
            for candidate in batch_candidates[len(responses) :]:
                batch_decisions.append(
                    self._fallback_decision(candidate, "Missing LLM response")
                )
            return batch_index, batch_decisions

        futures: List[Future[Tuple[int, List[TheorySplitDecision]]]] = []
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            for index, batch_candidates in enumerate(batches):
                futures.append(executor.submit(worker, index, batch_candidates))

            batch_results: Dict[int, List[TheorySplitDecision]] = {}
            for future in as_completed(futures):
                batch_index, batch_decisions = future.result()
                batch_results[batch_index] = batch_decisions

        for index in range(len(batches)):
            decisions.extend(batch_results.get(index, []))

        self._batches_executed = len(batches)
        return decisions

    def _fallback_decision(self, candidate: TheorySplitCandidate, reason: str) -> TheorySplitDecision:
        self._fallback_decisions += 1
        articles = candidate.articles
        chunk_size = min(self.maximum, max(self.target, self.minimum))
        proposals: List[TheorySplitProposal] = []
        start = 0
        variant = 1
        while start < len(articles):
            end = min(start + chunk_size, len(articles))
            chunk = articles[start:end]
            if len(chunk) < self.minimum and proposals:
                proposals[-1].article_ids.extend(chunk)
                break
            label = candidate.ref.label if start == 0 else f"{candidate.ref.label} Variant {variant}"
            proposals.append(
                TheorySplitProposal(
                    label=label,
                    rationale=f"{reason} (chunk {variant})" if start != 0 else reason,
                    article_ids=list(chunk),
                )
            )
            start = end
            variant += 1

        if not proposals:
            proposals.append(
                TheorySplitProposal(
                    label=candidate.ref.label,
                    rationale=reason,
                    article_ids=list(articles),
                )
            )

        return TheorySplitDecision(
            candidate=candidate,
            proposals=proposals,
            overall_rationale=reason,
            source="fallback",
        )

    def _build_messages(self, candidate: TheorySplitCandidate) -> List[Mapping[str, str]]:
        system_prompt = (
            "You are an ontology editor who restructures aging theories so that each node "
            "contains between {min_count} and {max_count} supporting articles."
        ).format(min_count=self.minimum, max_count=self.maximum)

        article_lines = [self._format_article_context(article_id) for article_id in candidate.articles]
        article_text = "\n".join(article_lines)
        user_prompt = (
            "Review the theory '{label}' which currently has {count} supporting articles.\n"
            "Each article is listed with any available metadata.\n"
            "Determine whether multiple distinct theories are present.\n"
            "Return JSON with keys: overall_rationale (string) and theories (array).\n"
            "Each element of theories must contain label (string), rationale (string), and article_ids (array of IDs).\n"
            "Assign every article to exactly one theory and ensure no theory exceeds {max_count} articles.\n"
            "If you recommend keeping a single theory, still return one entry describing it."
        ).format(label=candidate.ref.label, count=len(candidate.articles), max_count=self.maximum)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nArticles:\n{article_text}"},
        ]

    def _format_article_context(self, article_id: str) -> str:
        details = self.article_catalog.get(article_id)
        if not isinstance(details, Mapping):
            return f"- {article_id}"
        parts = [f"- {article_id}"]
        title = str(details.get("title") or "").strip()
        if title:
            parts.append(f"Title: {title}")
        journal = str(details.get("journal") or "").strip()
        year = details.get("publication_year")
        if journal or year:
            meta = journal
            if year:
                meta = f"{meta} ({year})" if meta else f"({year})"
            if meta:
                parts.append(meta)
        summary = str(details.get("summary") or details.get("abstract") or "").strip()
        if summary:
            parts.append(f"Summary: {summary}")
        return " | ".join(parts)

    def _parse_response(
        self, candidate: TheorySplitCandidate, response: LLMResponse
    ) -> TheorySplitDecision:
        text = response.content.strip()
        match = _JSON_BLOCK.search(text)
        payload_text = text if match is None else match.group(0)
        try:
            data = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse LLM response for %s: %s", candidate.ref.key, exc)
            return self._fallback_decision(candidate, "Unparseable LLM response")

        theories_data = data.get("theories")
        overall_rationale = str(data.get("overall_rationale") or "LLM rationale unavailable").strip()
        proposals: List[TheorySplitProposal] = []
        if isinstance(theories_data, Sequence):
            for entry in theories_data:
                if not isinstance(entry, Mapping):
                    continue
                label = str(entry.get("label") or candidate.ref.label).strip() or candidate.ref.label
                rationale = str(entry.get("rationale") or overall_rationale or "LLM decision").strip()
                article_ids_raw = entry.get("article_ids")
                if isinstance(article_ids_raw, Sequence) and not isinstance(article_ids_raw, (str, bytes)):
                    article_ids = [str(article).strip() for article in article_ids_raw if str(article).strip()]
                else:
                    article_ids = list(candidate.articles)
                proposals.append(
                    TheorySplitProposal(
                        label=label,
                        rationale=rationale or "LLM decision",
                        article_ids=article_ids,
                    )
                )

        if not proposals:
            return self._fallback_decision(candidate, "LLM returned no proposals")

        return TheorySplitDecision(
            candidate=candidate,
            proposals=proposals,
            overall_rationale=overall_rationale or proposals[0].rationale,
            source="llm",
        )

    def _normalise_proposals(
        self, candidate: TheorySplitCandidate, proposals: Sequence[TheorySplitProposal]
    ) -> List[TheorySplitProposal]:
        base_articles = list(candidate.articles)
        assigned: set[str] = set()
        normalised: List[TheorySplitProposal] = []

        for proposal in proposals:
            unique_articles: List[str] = []
            for article_id in proposal.article_ids:
                if article_id in base_articles and article_id not in assigned:
                    assigned.add(article_id)
                    unique_articles.append(article_id)
            if not unique_articles:
                continue
            normalised.append(
                TheorySplitProposal(
                    label=proposal.label,
                    rationale=proposal.rationale,
                    article_ids=unique_articles,
                )
            )

        # Assign any leftover articles to the first available bucket respecting limits.
        remaining = [article for article in base_articles if article not in assigned]
        for article in remaining:
            placed = False
            for proposal in normalised:
                if len(proposal.article_ids) < self.maximum:
                    proposal.article_ids.append(article)
                    placed = True
                    break
            if not placed:
                normalised.append(
                    TheorySplitProposal(
                        label=f"{proposals[0].label} (extra)",
                        rationale=proposals[0].rationale,
                        article_ids=[article],
                    )
                )

        if not normalised:
            normalised.append(
                TheorySplitProposal(
                    label=proposals[0].label,
                    rationale=proposals[0].rationale,
                    article_ids=list(base_articles),
                )
            )

        final: List[TheorySplitProposal] = []
        for proposal in normalised:
            articles = list(proposal.article_ids)
            while len(articles) > self.maximum:
                chunk = articles[: self.maximum]
                final.append(
                    TheorySplitProposal(
                        label=proposal.label,
                        rationale=proposal.rationale,
                        article_ids=chunk,
                    )
                )
                articles = articles[self.maximum :]
            if not articles:
                continue
            final.append(
                TheorySplitProposal(
                    label=proposal.label,
                    rationale=proposal.rationale,
                    article_ids=articles,
                )
            )

        # Merge proposals that fall below the minimum threshold into neighbours.
        index = 0
        while index < len(final):
            proposal = final[index]
            if len(proposal.article_ids) >= self.minimum or len(final) == 1:
                index += 1
                continue
            if index + 1 < len(final):
                final[index + 1].article_ids = proposal.article_ids + final[index + 1].article_ids
                final.pop(index)
                continue
            if index > 0:
                final[index - 1].article_ids.extend(proposal.article_ids)
                final.pop(index)
                continue
            index += 1

        return final

    def _apply_decision(self, decision: TheorySplitDecision) -> bool:
        candidate = decision.candidate
        proposals = self._normalise_proposals(candidate, decision.proposals)
        if not proposals:
            return False

        base_articles = candidate.articles
        template = deepcopy(candidate.ref.node)

        primary = proposals[0]
        candidate.ref.node["label"] = primary.label
        candidate.ref.update_articles(primary.article_ids)
        metadata = candidate.ref.node.setdefault("metadata", {})
        metadata["optimization_pipeline"] = _PIPELINE_ID
        metadata["optimization_rationale"] = primary.rationale or decision.overall_rationale
        metadata["optimization_source"] = decision.source
        metadata["optimization_batch"] = self._batches_executed
        self.summary.updated_theories.append(
            {
                "theory_id": candidate.ref.key,
                "label": primary.label,
                "articles": list(primary.article_ids),
                "rationale": metadata["optimization_rationale"],
                "source": decision.source,
            }
        )

        changed = decision.changed or set(primary.article_ids) != set(base_articles)

        theories = _ensure_list(candidate.ref.group, "theories")
        for index, proposal in enumerate(proposals[1:], start=1):
            new_node = deepcopy(template)
            new_id = self._unique_theory_id(f"{candidate.ref.key}_opt{index}")
            new_node["theory_id"] = new_id
            new_node["label"] = proposal.label
            new_node["supporting_articles"] = list(proposal.article_ids)
            new_metadata = new_node.setdefault("metadata", {})
            new_metadata["optimization_origin"] = candidate.ref.key
            new_metadata["optimization_variant"] = index
            new_metadata["optimization_pipeline"] = _PIPELINE_ID
            new_metadata["optimization_rationale"] = proposal.rationale or decision.overall_rationale
            new_metadata["optimization_source"] = decision.source
            theories.append(new_node)
            self._used_ids.add(new_id)
            self.summary.created_theories.append(
                {
                    "theory_id": new_id,
                    "origin": candidate.ref.key,
                    "articles": list(proposal.article_ids),
                    "label": proposal.label,
                    "rationale": new_metadata["optimization_rationale"],
                    "source": decision.source,
                }
            )
            changed = True

        return changed


def optimise_ontology_payload(
    payload: MutableMapping[str, Any],
    *,
    target: int = 3,
    minimum: int = 1,
    maximum: int = 4,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = "gpt-5-mini",
    llm_temperature: Optional[float] = None,
    batch_size: int = 4,
    max_concurrency: int = 1,
) -> OptimizationSummary:
    """Apply optimisation rules to an ``aging_ontology.json`` style payload."""

    ontology = payload.get("ontology") if isinstance(payload, MutableMapping) else None
    final = ontology.get("final") if isinstance(ontology, MutableMapping) else None
    groups = final.get("groups") if isinstance(final, MutableMapping) else None

    if not isinstance(groups, list):
        logger.debug("Ontology payload missing 'final.groups'; skipping optimisation")
        return OptimizationSummary(changed=False)

    article_catalog = _build_article_catalog(payload)
    optimizer = OntologyOptimizer(
        groups,
        target=target,
        minimum=minimum,
        maximum=maximum,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        article_catalog=article_catalog,
    )
    summary = optimizer.run()

    if summary.changed:
        optimisation_meta = payload.setdefault("optimization_ontology", {})
        optimisation_meta.update(summary.to_dict())
        optimisation_meta.update(
            {
                "llm_model": optimizer.llm_model,
                "llm_temperature": optimizer.llm_temperature,
                "llm_batches": optimizer._batches_executed,
                "fallback_decisions": optimizer._fallback_decisions,
            }
        )

    return summary


def optimise_file(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    target: int = 3,
    minimum: int = 1,
    maximum: int = 4,
    llm_client: Optional[LLMClient] = None,
    llm_model: Optional[str] = "gpt-5-mini",
    llm_temperature: Optional[float] = None,
    batch_size: int = 4,
    max_concurrency: int = 1,
) -> OptimizationSummary:
    """Load ``input_path``, optimise it and persist the result to ``output_path``."""

    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
        if not isinstance(payload, MutableMapping):
            raise RuntimeError(f"Expected mapping at {input_path}; found {type(payload)!r}")

    summary = optimise_ontology_payload(
        payload,
        target=target,
        minimum=minimum,
        maximum=maximum,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )

    output = input_path if output_path is None else output_path
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary


def _build_article_catalog(payload: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    catalog: Dict[str, Mapping[str, Any]] = {}

    prompt_summary = payload.get("prompt_summary") if isinstance(payload, Mapping) else None
    if isinstance(prompt_summary, Sequence):
        for entry in prompt_summary:
            if not isinstance(entry, Mapping):
                continue
            representatives = entry.get("representative_articles")
            if not isinstance(representatives, Sequence):
                continue
            for article in representatives:
                if not isinstance(article, Mapping):
                    continue
                article_id = str(
                    article.get("id")
                    or article.get("identifier")
                    or article.get("doi")
                    or article.get("title")
                    or ""
                ).strip()
                if not article_id:
                    continue
                catalog.setdefault(article_id, article)  # prefer first occurrence

    return catalog

