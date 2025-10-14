"""Relevance filtering utilities for literature retrieval."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

from .literature import PaperMetadata
from .llm import LLMClient, LLMClientError, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterDecision:
    """Decision returned by :class:`RelevanceFilter`."""

    paper_id: str
    accepted: bool
    score: float
    reason: str | None = None

    def to_dict(self) -> Mapping[str, object]:
        return {
            "paper_id": self.paper_id,
            "accepted": self.accepted,
            "score": self.score,
            "reason": self.reason,
        }


class RelevanceFilter:
    """Simple heuristic relevance filter with optional LLM assistance."""

    def __init__(
        self,
        query: str,
        *,
        threshold: float = 0.35,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.query = query
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.llm_client = llm_client
        self._query_terms = self._tokenize(query)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]

    def _heuristic_score(self, paper: PaperMetadata) -> float:
        content = " ".join(part for part in (paper.title, paper.abstract) if part).lower()
        if not content:
            return 0.0
        matches = sum(1 for term in self._query_terms if term in content)
        if not matches:
            return 0.0
        return matches / math.sqrt(len(self._query_terms) * max(len(content.split()), 1))

    def _llm_score(self, paper: PaperMetadata) -> float | None:
        if not self.llm_client:
            return None
        summary = json.dumps(
            {
                "query": self.query,
                "title": paper.title,
                "abstract": paper.abstract,
                "source": paper.source,
            },
            ensure_ascii=False,
        )
        messages: Sequence[LLMMessage] = [
            LLMMessage(
                role="system",
                content=(
                    "You are assisting with a literature review on theories of aging. "
                    "Return JSON with a numeric 'score' between 0 and 1 indicating how "
                    "relevant the paper is to the query."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    "Given the following JSON payload decide if the paper is relevant to the query.\n"
                    f"{summary}\n"
                    "Respond with {\"score\": <number between 0 and 1>, \"reason\": <short text>}."
                ),
            ),
        ]
        try:
            response: LLMResponse = self.llm_client.generate([messages])[0]
        except (LLMClientError, IndexError) as exc:  # pragma: no cover - defensive
            self._logger.warning("LLM relevance scoring failed: %s", exc)
            return None
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            self._logger.debug("Failed to parse LLM response for %s", paper.identifier)
            return None
        score = data.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
        return None

    def evaluate(self, paper: PaperMetadata) -> FilterDecision:
        llm_score = self._llm_score(paper)
        heuristic_score = self._heuristic_score(paper)
        score = llm_score if llm_score is not None else heuristic_score
        accepted = score >= self.threshold
        reason = None
        if llm_score is not None:
            reason = "llm" if accepted else "llm_below_threshold"
        elif heuristic_score > 0.0:
            reason = "heuristic"
        else:
            reason = "no_match"
        return FilterDecision(
            paper_id=paper.identifier,
            accepted=accepted,
            score=score,
            reason=reason,
        )

    def filter(self, papers: Iterable[PaperMetadata]) -> List[FilterDecision]:
        return [self.evaluate(paper) for paper in papers]


__all__ = ["FilterDecision", "RelevanceFilter"]
