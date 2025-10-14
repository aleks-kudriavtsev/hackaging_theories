"""Relevance filtering for retrieved literature records.

This module implements a small, configurable filtering layer that scores
retrieved :class:`~theories_pipeline.literature.PaperMetadata` entries before
they are passed to downstream ontology bootstrap or classification steps.
Scoring combines lightweight keyword heuristics with an optional
:class:`~theories_pipeline.llm.LLMClient` backed evaluation in order to keep the
behaviour deterministic when the LLM path is disabled.  Filter decisions are
encapsulated in :class:`FilterDecision` records which can be persisted in the
retrieval state for reproducibility.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .literature import PaperMetadata

try:  # pragma: no cover - optional import for typing / runtime usage
    from .llm import LLMClient, LLMMessage, LLMResponse
except ImportError:  # pragma: no cover - during type checking the client may be absent
    LLMClient = None  # type: ignore[assignment]
    LLMMessage = None  # type: ignore[assignment]
    LLMResponse = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+", flags=re.IGNORECASE)


def _normalise(text: str) -> str:
    return text.casefold()


def _coerce_keywords(keywords: Iterable[str] | None) -> list[str]:
    if not keywords:
        return []
    coerced: list[str] = []
    for entry in keywords:
        if entry is None:
            continue
        value = str(entry).strip()
        if value:
            coerced.append(value)
    return coerced


@dataclass(frozen=True)
class FilterDecision:
    """Deterministic record describing how a paper was scored."""

    identifier: str
    score: float
    accepted: bool
    rationale: str | None
    details: Mapping[str, Any] | None = None

    def to_record(self, *, threshold: float) -> Mapping[str, Any]:
        payload = {
            "identifier": self.identifier,
            "score": float(self.score),
            "accepted": bool(self.accepted),
            "threshold": float(threshold),
        }
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        if self.details is not None:
            payload["details"] = self.details
        return payload


class RelevanceFilter:
    """Hybrid keyword/LLM filter for retrieval results."""

    DEFAULT_LLM_PROMPT = (
        "Rate how relevant the paper is to the topic '{topic}'. "
        "Respond with JSON containing numeric 'score' between 0 and 1 and "
        "a short 'rationale'."
    )

    def __init__(
        self,
        *,
        keywords: Sequence[str] | None = None,
        threshold: float = 0.0,
        llm_client: LLMClient | None = None,
        llm_weight: float = 0.5,
        llm_prompt: str | None = None,
        max_text_chars: int = 2000,
    ) -> None:
        self.keywords = _coerce_keywords(keywords)
        self.threshold = float(threshold)
        self.llm_client = llm_client
        self.llm_weight = self._clamp_weight(llm_weight)
        self.llm_prompt = llm_prompt or self.DEFAULT_LLM_PROMPT
        self.max_text_chars = max(0, int(max_text_chars))

    @staticmethod
    def _clamp_weight(weight: float) -> float:
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            return 0.5
        if numeric <= 0:
            return 0.0
        if numeric >= 1:
            return 1.0
        return numeric

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        papers: Sequence[PaperMetadata],
        *,
        context: Mapping[str, Any] | None = None,
        existing_decisions: MutableMapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[list[PaperMetadata], list[FilterDecision]]:
        """Return accepted papers and associated decisions."""

        accepted: list[PaperMetadata] = []
        decisions: list[FilterDecision] = []
        context = context or {}

        for paper in papers:
            identifier = paper.identifier
            if existing_decisions and identifier in existing_decisions:
                stored = existing_decisions[identifier]
                decision = self._decision_from_record(identifier, stored)
                decisions.append(decision)
                if decision.accepted:
                    accepted.append(paper)
                continue

            decision = self._score_paper(paper, context=context)
            decisions.append(decision)
            if existing_decisions is not None:
                existing_decisions[identifier] = dict(decision.to_record(threshold=self.threshold))
            if decision.accepted:
                accepted.append(paper)

        return accepted, decisions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _decision_from_record(self, identifier: str, record: Mapping[str, Any]) -> FilterDecision:
        score = float(record.get("score", 0.0))
        stored_threshold = float(record.get("threshold", self.threshold))
        rationale = record.get("rationale")
        accepted = bool(record.get("accepted", score >= self.threshold))
        if stored_threshold != self.threshold:
            accepted = score >= self.threshold
        details = record.get("details") if isinstance(record.get("details"), Mapping) else None
        return FilterDecision(
            identifier=identifier,
            score=score,
            accepted=accepted,
            rationale=rationale if isinstance(rationale, str) else None,
            details=details,
        )

    def _score_paper(
        self,
        paper: PaperMetadata,
        *,
        context: Mapping[str, Any],
    ) -> FilterDecision:
        heuristic_score, heuristic_details, heuristic_reason = self._heuristic_score(paper)

        llm_score: float | None = None
        llm_reason: str | None = None
        llm_details: dict[str, Any] = {}

        if self.llm_client is not None:
            llm_score, llm_reason, llm_details = self._llm_score(paper, context=context)

        combined_score = self._combine_scores(heuristic_score, llm_score)
        accepted = combined_score >= self.threshold

        details: dict[str, Any] = {
            "heuristic": heuristic_details,
            "threshold": self.threshold,
            "method": "heuristic" if self.llm_client is None else "hybrid",
        }
        if llm_score is not None:
            details["llm"] = {
                "score": llm_score,
                "cached": llm_details.get("cached"),
                "metadata": llm_details.get("metadata"),
            }
        rationale = llm_reason or heuristic_reason
        return FilterDecision(
            identifier=paper.identifier,
            score=combined_score,
            accepted=accepted,
            rationale=rationale,
            details=details,
        )

    def _heuristic_score(self, paper: PaperMetadata) -> tuple[float, Mapping[str, Any], str]:
        if not self.keywords:
            details = {
                "keyword_count": 0,
                "matched_keywords": [],
                "section_matches": {},
            }
            return 1.0, details, "No keywords configured"

        keywords = {kw.casefold(): kw for kw in self.keywords}
        title_text = _normalise(paper.title or "")
        abstract_text = _normalise(paper.abstract or "")
        body_text = _normalise(paper.analysis_text or "")

        title_matches = {raw for key, raw in keywords.items() if key in title_text}
        abstract_matches = {raw for key, raw in keywords.items() if key in abstract_text}
        body_matches = {raw for key, raw in keywords.items() if key in body_text}
        body_only = body_matches - title_matches - abstract_matches

        total_keywords = len(keywords)
        coverage = len(title_matches | abstract_matches | body_matches) / total_keywords
        base_score = (
            0.5 * len(title_matches)
            + 0.3 * len(abstract_matches)
            + 0.2 * len(body_only)
        ) / total_keywords
        score = min(1.0, base_score + 0.1 * coverage)

        details = {
            "keyword_count": total_keywords,
            "matched_keywords": sorted(title_matches | abstract_matches | body_matches),
            "section_matches": {
                "title": sorted(title_matches),
                "abstract": sorted(abstract_matches),
                "body": sorted(body_only),
            },
            "coverage": coverage,
            "base_score": base_score,
        }
        if not details["matched_keywords"]:
            reason = "No keyword matches"
        else:
            reason = f"Matched {len(details['matched_keywords'])}/{total_keywords} keywords"
        return score, details, reason

    def _combine_scores(self, heuristic: float, llm: float | None) -> float:
        heuristic_clamped = max(0.0, min(1.0, heuristic))
        if llm is None:
            return heuristic_clamped
        llm_clamped = max(0.0, min(1.0, float(llm)))
        weight = self.llm_weight
        if weight <= 0:
            return heuristic_clamped
        if weight >= 1:
            return llm_clamped
        return heuristic_clamped * (1.0 - weight) + llm_clamped * weight

    def _llm_score(
        self,
        paper: PaperMetadata,
        *,
        context: Mapping[str, Any],
    ) -> tuple[float | None, str | None, Mapping[str, Any]]:
        if self.llm_client is None:
            return None, None, {}

        topic = (
            context.get("topic")
            or context.get("theory")
            or context.get("name")
            or ""
        )
        prompt_context = {
            "topic": topic,
            "theory": context.get("theory", topic),
            "query": context.get("query", ""),
            "base_query": context.get("base_query", ""),
        }
        prompt = self._render_prompt(prompt_context)
        content = self._prepare_paper_payload(paper)

        if LLMMessage is None:
            raise RuntimeError("LLM support is unavailable: the LLMMessage class is missing")

        messages = [
            LLMMessage(role="system", content=prompt),
            LLMMessage(role="user", content=content),
        ]

        try:
            responses = self.llm_client.generate([messages])  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive runtime fallback
            logger.warning("LLM scoring failed for %s: %s", paper.identifier, exc)
            return None, None, {"error": str(exc)}

        if not responses:
            return None, None, {"error": "empty_response"}

        response: LLMResponse = responses[0]
        parsed_score, rationale = self._parse_llm_response(response.content)
        details = {
            "cached": response.cached,
            "metadata": response.metadata,
        }
        return parsed_score, rationale, details

    def _render_prompt(self, context: Mapping[str, Any]) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:  # pragma: no cover - defensive fallback
                return ""

        try:
            return self.llm_prompt.format_map(_SafeDict(context))
        except Exception:  # pragma: no cover - fallback to raw prompt
            return self.llm_prompt

    def _prepare_paper_payload(self, paper: PaperMetadata) -> str:
        abstract = paper.abstract.strip() if paper.abstract else ""
        body = paper.analysis_text.strip() if paper.analysis_text else ""
        if self.max_text_chars:
            body = body[: self.max_text_chars]
        payload_parts = [
            f"Title: {paper.title.strip() if paper.title else 'N/A'}",
            f"Abstract: {abstract or 'N/A'}",
        ]
        if body:
            payload_parts.append("Content:\n" + body)
        return "\n\n".join(payload_parts)

    def _parse_llm_response(self, text: str) -> tuple[float | None, str | None]:
        cleaned = text.strip()
        if not cleaned:
            return None, None
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, Mapping):
                raw_score = payload.get("score")
                if raw_score is not None:
                    score = float(raw_score)
                    rationale = payload.get("rationale") or payload.get("reason")
                    return float(max(0.0, min(1.0, score))), (
                        str(rationale) if rationale is not None else None
                    )
        except json.JSONDecodeError:
            pass
        match = re.search(r"([01]?\.\d+)", cleaned)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score)), None
            except ValueError:  # pragma: no cover - defensive
                return None, None
        return None, cleaned


__all__ = ["RelevanceFilter", "FilterDecision"]
