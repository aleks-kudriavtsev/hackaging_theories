"""Classify papers into gerontological theories for the Hackaging challenge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .literature import PaperMetadata
from .ontology import TheoryOntology


@dataclass(frozen=True)
class TheoryAssignment:
    """Represents the mapping of a paper to a theory label."""

    paper_id: str
    theory: str
    score: float
    depth: int = 0


class TheoryClassifier:
    """Keyword-based hierarchical classifier with ontology rollups."""

    def __init__(self, keyword_map: Mapping[str, Iterable[str]], ontology: TheoryOntology) -> None:
        self.keyword_map = {
            theory: [kw.lower() for kw in keywords]
            for theory, keywords in keyword_map.items()
        }
        self.ontology = ontology
        self._ontology_names = set(ontology.names())

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------
    def _score_keywords(self, paper: PaperMetadata) -> Dict[str, float]:
        text = " ".join([paper.title, paper.abstract]).lower()
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
        aggregated = self.aggregate_scores(self._score_keywords(paper))
        assignments = [
            TheoryAssignment(
                paper_id=paper.identifier,
                theory=theory,
                score=score,
                depth=self.ontology.depth(theory) if theory in self._ontology_names else 0,
            )
            for theory, score in aggregated.items()
            if score > 0.0
        ]
        assignments.sort(key=lambda item: (item.depth, -item.score, item.theory))
        return assignments

    def summarize(self, assignments: Iterable[TheoryAssignment]) -> Dict[str, int]:
        counts: Dict[str, set[str]] = {name: set() for name in self.ontology.names()}
        for assignment in assignments:
            if assignment.score <= 0:
                continue
            if assignment.theory in counts:
                counts[assignment.theory].add(assignment.paper_id)
        return {name: len(ids) for name, ids in counts.items()}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Iterable[str] | Mapping[str, object]],
        *,
        ontology: TheoryOntology,
    ) -> "TheoryClassifier":
        keyword_map = cls._normalize_keyword_config(config)
        return cls(keyword_map, ontology)

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
