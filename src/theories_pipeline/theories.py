"""Classify papers into gerontological theories for the Hackaging challenge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .literature import PaperMetadata


@dataclass(frozen=True)
class TheoryAssignment:
    """Represents the mapping of a paper to a theory label."""

    paper_id: str
    theory: str
    score: float


class TheoryClassifier:
    """Simple keyword-based classifier.

    The implementation is intentionally transparent and deterministic so that
    teams can extend or swap it for ML models while retaining the same API.
    """

    def __init__(self, keyword_map: Mapping[str, Iterable[str]]) -> None:
        self.keyword_map = {
            theory: [kw.lower() for kw in keywords]
            for theory, keywords in keyword_map.items()
        }

    def classify(self, paper: PaperMetadata) -> List[TheoryAssignment]:
        text = " ".join([paper.title, paper.abstract]).lower()
        assignments: List[TheoryAssignment] = []
        for theory, keywords in self.keyword_map.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches:
                score = matches / len(keywords)
                assignments.append(TheoryAssignment(paper.identifier, theory, score))
        assignments.sort(key=lambda item: item.score, reverse=True)
        return assignments

    @classmethod
    def from_config(cls, config: Dict[str, Iterable[str]]) -> "TheoryClassifier":
        return cls(config)
