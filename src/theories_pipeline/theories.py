"""Classify papers into gerontological theories for the Hackaging challenge."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

from .literature import PaperMetadata
from .ontology import TheoryOntology
from .llm import LLMClient, LLMClientError, LLMMessage, LLMRateLimitError, LLMResponse
from .ontology_summaries import (
    clean_summary,
    extract_quote_snippets,
    fallback_summary,
    format_bootstrap_highlights,
    format_keywords_line,
)

if TYPE_CHECKING:
    from .ontology_manager import OntologyManager, OntologyUpdate


@dataclass(frozen=True)
class TheoryAssignment:
    """Represents the mapping of a paper to a theory label."""

    paper_id: str
    theory: str
    score: float
    depth: int = 0


@dataclass(frozen=True)
class AggregatedTheory:
    """Summary of the papers collected for a single theory."""

    theory_id: str
    theory_name: str
    paper_ids: Tuple[str, ...]
    number_of_collected_papers: int


@dataclass(frozen=True)
class TheoryAggregationResult:
    """Container for aggregated theory statistics."""

    theories: Tuple[AggregatedTheory, ...]
    theory_ids_by_name: Mapping[str, str]
    theory_index: Mapping[str, AggregatedTheory]
    paper_to_theory_ids: Mapping[str, Tuple[str, ...]]


_SLUG_TOKEN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    normalized = value.strip().lower()
    slug = _SLUG_TOKEN.sub("-", normalized)
    return slug.strip("-") or "theory"


def aggregate_theory_assignments(
    assignments: Iterable[TheoryAssignment],
    ontology: TheoryOntology,
) -> TheoryAggregationResult:
    """Aggregate theory assignments and assign stable identifiers.

    Parameters
    ----------
    assignments:
        All raw theory assignments produced by the classifier.
    ontology:
        The ontology providing canonical theory names and metadata.

    Returns
    -------
    TheoryAggregationResult
        Aggregated counts, stable theory identifiers, and lookups that map
        papers to the theories they support.
    """

    papers_by_theory: Dict[str, set[str]] = {name: set() for name in ontology.names()}
    for assignment in assignments:
        if assignment.score <= 0:
            continue
        papers_by_theory.setdefault(assignment.theory, set()).add(assignment.paper_id)

    seen_ids: set[str] = set()
    ids_by_name: Dict[str, str] = {}
    for name in ontology.names():
        node = ontology.get(name)
        metadata = getattr(node, "metadata", {}) or {}
        preferred: str | None = None
        for key in ("theory_id", "id", "slug", "identifier"):
            candidate = metadata.get(key)
            if isinstance(candidate, str) and candidate.strip():
                preferred = candidate.strip()
                break
        base = _slugify(preferred if preferred is not None else name)
        candidate_id = base
        counter = 2
        while candidate_id in seen_ids:
            candidate_id = f"{base}-{counter}"
            counter += 1
        seen_ids.add(candidate_id)
        ids_by_name[name] = candidate_id

    aggregated: List[AggregatedTheory] = []
    paper_to_ids: Dict[str, List[str]] = {}
    for name in ontology.names():
        papers = papers_by_theory.get(name)
        if not papers:
            continue
        sorted_papers = tuple(sorted(papers))
        theory_id = ids_by_name[name]
        aggregated.append(
            AggregatedTheory(
                theory_id=theory_id,
                theory_name=name,
                paper_ids=sorted_papers,
                number_of_collected_papers=len(sorted_papers),
            )
        )
        for paper_id in sorted_papers:
            paper_to_ids.setdefault(paper_id, []).append(theory_id)

    aggregated.sort(key=lambda item: item.theory_name.lower())
    for ids in paper_to_ids.values():
        ids.sort()

    theory_index = {entry.theory_id: entry for entry in aggregated}
    paper_to_theory_ids = {paper_id: tuple(ids) for paper_id, ids in paper_to_ids.items()}

    return TheoryAggregationResult(
        theories=tuple(aggregated),
        theory_ids_by_name=ids_by_name,
        theory_index=theory_index,
        paper_to_theory_ids=paper_to_theory_ids,
    )


class TheoryClassifier:
    """Hybrid classifier using keywords with optional GPT assistance."""

    def __init__(
        self,
        keyword_map: Mapping[str, Iterable[str]],
        ontology: TheoryOntology,
        *,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.keyword_map = {
            theory: [kw.lower() for kw in keywords]
            for theory, keywords in keyword_map.items()
        }
        self.ontology = ontology
        self._ontology_names = set(ontology.names())
        self.llm_client = llm_client
        self._name_lookup = {name.lower(): name for name in ontology.names()}
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------
    def _score_keywords(self, paper: PaperMetadata) -> Dict[str, float]:
        analysis_text = paper.analysis_text if paper.analysis_text else paper.abstract
        text = " ".join(part for part in (paper.title, analysis_text) if part).lower()
        scores: Dict[str, float] = {}
        for theory, keywords in self.keyword_map.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches:
                scores[theory] = matches / len(keywords)
        return scores

    def aggregate_scores(self, scores: Mapping[str, float]) -> Dict[str, float]:
        aggregated: Dict[str, float] = {name: float(scores.get(name, 0.0)) for name in self.ontology.names()}
        for name, score in scores.items():
            if name not in aggregated:
                aggregated[name] = float(score)
        for name in reversed(self.ontology.post_order()):
            node = self.ontology.get(name)
            child_scores = [aggregated[child] for child in node.children]
            best_child = max(child_scores, default=0.0)
            aggregated[name] = max(aggregated.get(name, 0.0), best_child)
        return aggregated

    def classify(self, paper: PaperMetadata) -> List[TheoryAssignment]:
        return self.classify_batch([paper])[0]

    def classify_batch(self, papers: Sequence[PaperMetadata]) -> List[List[TheoryAssignment]]:
        if not papers:
            return []
        if not self.llm_client:
            return [self._keyword_assignments(paper) for paper in papers]

        messages_batch = [self._build_llm_messages(paper) for paper in papers]
        try:
            responses = self.llm_client.generate(messages_batch)
        except LLMRateLimitError:
            self._logger.warning(
                "Rate limit encountered during LLM classification; falling back to keywords"
            )
            return [self._keyword_assignments(paper) for paper in papers]
        except LLMClientError as exc:
            self._logger.error("LLM classification failed: %s", exc)
            return [self._keyword_assignments(paper) for paper in papers]

        assignments_batch: List[List[TheoryAssignment]] = []
        for paper, response in zip(papers, responses):
            llm_scores = self._scores_from_llm_response(response)
            if llm_scores is None:
                assignments_batch.append(self._keyword_assignments(paper))
                continue
            keyword_scores = self._score_keywords(paper)
            merged_scores: Dict[str, float] = dict(keyword_scores)
            for name, score in llm_scores.items():
                merged_scores[name] = max(merged_scores.get(name, 0.0), score)
            aggregated = self.aggregate_scores(merged_scores)
            assignments_batch.append(self._assignments_from_scores(paper.identifier, aggregated))

        return assignments_batch

    def summarize(self, assignments: Iterable[TheoryAssignment]) -> Dict[str, int]:
        counts: Dict[str, set[str]] = {name: set() for name in self.ontology.names()}
        for assignment in assignments:
            if assignment.score <= 0:
                continue
            if assignment.theory in counts:
                counts[assignment.theory].add(assignment.paper_id)
        return {name: len(ids) for name, ids in counts.items()}

    def update_ontology(
        self,
        ontology: TheoryOntology,
        *,
        keyword_updates: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        """Refresh the classifier state after an ontology update."""

        self.ontology = ontology
        self._ontology_names = set(ontology.names())
        self._name_lookup = {name.lower(): name for name in ontology.names()}
        if keyword_updates:
            for name, keywords in keyword_updates.items():
                normalized = [kw.lower() for kw in keywords]
                if normalized:
                    self.keyword_map[name] = normalized

    def attach_manager(self, manager: "OntologyManager") -> None:
        """Subscribe to ontology updates emitted by an :class:`OntologyManager`."""

        def _listener(ontology: TheoryOntology, update: "OntologyUpdate") -> None:
            keyword_updates: Dict[str, Iterable[str]] = {}
            for name, data in update.added.items():
                keywords = data.get("keywords")
                if keywords:
                    keyword_updates[name] = keywords
            for name, keywords in update.keyword_updates.items():
                if keywords:
                    keyword_updates[name] = keywords
            self.update_ontology(ontology, keyword_updates=keyword_updates)

        manager.register_listener(_listener)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        ontology: TheoryOntology,
        llm_client: LLMClient | None = None,
    ) -> "TheoryClassifier":
        keyword_cfg: Mapping[str, Iterable[str] | Mapping[str, object]]
        if "keywords" in config and isinstance(config["keywords"], Mapping):
            keyword_cfg = config["keywords"]  # type: ignore[assignment]
        else:
            keyword_cfg = config  # type: ignore[assignment]
        keyword_map = cls._normalize_keyword_config(keyword_cfg)
        return cls(keyword_map, ontology, llm_client=llm_client)

    @staticmethod
    def _normalize_keyword_config(
        config: Mapping[str, Iterable[str] | Mapping[str, object]]
    ) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}

        def visit(name: str, value: Iterable[str] | Mapping[str, object]) -> None:
            if isinstance(value, Mapping):
                keywords = value.get("keywords")
                if isinstance(keywords, Iterable) and not isinstance(keywords, (str, bytes)):
                    normalized[name] = [kw.lower() for kw in keywords]
                for child_name, child_value in value.get("subtheories", {}).items():
                    visit(child_name, child_value)
            else:
                normalized[name] = [kw.lower() for kw in value]

        for theory, value in config.items():
            visit(theory, value)

        return normalized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _keyword_assignments(self, paper: PaperMetadata) -> List[TheoryAssignment]:
        aggregated = self.aggregate_scores(self._score_keywords(paper))
        return self._assignments_from_scores(paper.identifier, aggregated)

    def _assignments_from_scores(
        self, paper_id: str, aggregated: Mapping[str, float]
    ) -> List[TheoryAssignment]:
        assignments = [
            TheoryAssignment(
                paper_id=paper_id,
                theory=theory,
                score=score,
                depth=self.ontology.depth(theory) if theory in self._ontology_names else 0,
            )
            for theory, score in aggregated.items()
            if score > 0.0
        ]
        assignments.sort(key=lambda item: (item.depth, -item.score, item.theory))
        return assignments

    def _build_llm_messages(self, paper: PaperMetadata) -> List[LLMMessage]:
        ontology_description = self._format_ontology_prompt()
        paper_text = self._format_paper_prompt(paper)
        system_prompt = (
            "You are an expert gerontology analyst. Assign the paper to the "
            "most appropriate theory and optional subtheory from the provided "
            "ontology. Respond with JSON containing a 'predictions' array."
        )
        user_prompt = (
            f"Ontology:\n{ontology_description}\n\nPaper:\n{paper_text}\n\n"
            "Return a JSON object like {\"predictions\":[{\"theory\":...,\"subtheory\":...,\"confidence\":0.0}]}."
            " Use a confidence between 0 and 1."
        )
        return [LLMMessage("system", system_prompt), LLMMessage("user", user_prompt)]

    def _format_ontology_prompt(self) -> str:
        levels: Dict[int, List[str]] = {}
        for name in self.ontology.names():
            depth = self.ontology.depth(name)
            levels.setdefault(depth, []).append(name)
        lines: List[str] = []
        for depth in sorted(levels):
            indent = "  " * depth
            for name in sorted(levels[depth]):
                node = self.ontology.get(name)
                parent = node.parent
                keywords = self.keyword_map.get(name)
                summary = clean_summary(node.metadata.get("summary"))
                if not summary:
                    bootstrap_meta = node.metadata.get("bootstrap") if isinstance(node.metadata, Mapping) else None
                    summary = fallback_summary(
                        name,
                        keywords=keywords,
                        bootstrap=bootstrap_meta if isinstance(bootstrap_meta, Mapping) else None,
                    )
                header = f"{indent}- {name}"
                if parent:
                    header += f" (child of {parent})"
                if summary:
                    header += f": {summary}"
                lines.append(header)
                detail_indent = f"{indent}    "
                keyword_line = format_keywords_line(keywords)
                if keyword_line:
                    lines.append(f"{detail_indent}Key terms: {keyword_line}")
                bootstrap_line = format_bootstrap_highlights(node.metadata)
                if bootstrap_line:
                    lines.append(f"{detail_indent}Bootstrap: {bootstrap_line}")
                for quote in extract_quote_snippets(node.metadata):
                    lines.append(f"{detail_indent}Quote: {quote}")
        return "\n".join(lines)

    def _format_paper_prompt(self, paper: PaperMetadata) -> str:
        primary_text = paper.analysis_text.strip()
        if primary_text:
            if paper.full_text and paper.full_text.strip():
                text_label = "Full text"
            elif paper.sections:
                text_label = "Section text"
            else:
                text_label = "Abstract"
        else:
            text_label = "Abstract"
            primary_text = paper.abstract.strip()
        excerpt = primary_text[:4000] if primary_text else ""
        if primary_text and len(primary_text) > 4000:
            excerpt = f"{excerpt}..."
        if not excerpt:
            excerpt = "<no abstract provided>"
        authors = ", ".join(paper.authors) if paper.authors else "Unknown"
        return (
            f"Title: {paper.title}\n"
            f"Authors: {authors}\n"
            f"Source: {paper.source}\n"
            f"{text_label}: {excerpt}"
        )

    def _scores_from_llm_response(self, response: LLMResponse) -> Dict[str, float] | None:
        content = response.content.strip()
        if not content:
            return None
        data = self._extract_json(content)
        if data is None:
            self._logger.warning("LLM response not valid JSON: %s", content[:200])
            return None
        predictions = data.get("predictions") or data.get("theories") or data.get("assignments")
        if isinstance(predictions, Mapping):
            predictions = [predictions]
        if not isinstance(predictions, list):
            self._logger.warning("LLM response missing predictions: %s", content[:200])
            return None
        scores: Dict[str, float] = {}
        for item in predictions:
            if not isinstance(item, Mapping):
                continue
            theory_name = self._match_name(item.get("theory"))
            if not theory_name:
                continue
            confidence = self._coerce_confidence(item.get("confidence"))
            if confidence <= 0:
                continue
            scores[theory_name] = max(scores.get(theory_name, 0.0), confidence)
            sub_candidates = self._collect_subtheories(item)
            for sub_name in sub_candidates:
                matched = self._match_name(sub_name)
                if matched:
                    scores[matched] = max(scores.get(matched, 0.0), confidence)
        return scores if scores else None

    def _collect_subtheories(self, item: Mapping[str, object]) -> List[str]:
        sub_names: List[str] = []
        direct = item.get("subtheory") or item.get("sub_theory") or item.get("sub")
        if isinstance(direct, str) and direct.strip():
            sub_names.append(direct.strip())
        sub_list = item.get("subtheories") or item.get("sub_theories")
        if isinstance(sub_list, Sequence):
            for entry in sub_list:
                if isinstance(entry, str) and entry.strip():
                    sub_names.append(entry.strip())
                elif isinstance(entry, Mapping):
                    name = entry.get("name") or entry.get("theory")
                    if isinstance(name, str) and name.strip():
                        sub_names.append(name.strip())
        return sub_names

    def _extract_json(self, text: str) -> MutableMapping[str, object] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _coerce_confidence(self, value: object) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            confidence = float(value)
        elif isinstance(value, str):
            cleaned = value.strip().rstrip("%")
            if not cleaned:
                return 0.0
            try:
                confidence = float(cleaned)
            except ValueError:
                return 0.0
        else:
            return 0.0
        if confidence > 1.5:
            confidence /= 100.0
        return max(0.0, min(1.0, confidence))

    def _match_name(self, name: object) -> str | None:
        if not isinstance(name, str):
            return None
        return self._name_lookup.get(name.strip().lower())
