import json
from typing import Any, List, Mapping, Sequence

import pytest

from theories_pipeline.filtering import RelevanceFilter
from theories_pipeline.literature import PaperMetadata
from theories_pipeline.llm import LLMResponse


def _paper(identifier: str, *, title: str, abstract: str = "", full_text: str = "") -> PaperMetadata:
    return PaperMetadata(
        identifier=identifier,
        title=title,
        authors=("Author",),
        abstract=abstract,
        source="test",
        full_text=full_text,
    )


def test_relevance_filter_heuristic_only() -> None:
    papers = [
        _paper(
            "p-positive",
            title="Digital aging technology",
            abstract="Explores gerontology and technology",
        ),
        _paper("p-negative", title="Completely unrelated topic"),
    ]

    state: dict[str, Mapping[str, Any]] = {}
    relevance_filter = RelevanceFilter(keywords=["aging", "technology"], threshold=0.25)

    accepted, decisions = relevance_filter.apply(
        papers,
        context={"theory": "Digital Aging"},
        existing_decisions=state,
    )

    assert [paper.identifier for paper in accepted] == ["p-positive"]
    positive = next(decision for decision in decisions if decision.identifier == "p-positive")
    negative = next(decision for decision in decisions if decision.identifier == "p-negative")
    assert positive.accepted is True
    assert negative.accepted is False
    assert state["p-positive"]["accepted"] is True
    assert state["p-negative"]["accepted"] is False
    assert "technology" in positive.details["heuristic"]["matched_keywords"]


class DummyLLMClient:
    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages_batch: Sequence[Sequence[Any]]) -> List[LLMResponse]:
        del messages_batch
        if self.calls >= len(self._responses):
            raise RuntimeError("Exhausted fake responses")
        response = self._responses[self.calls]
        self.calls += 1
        return [response]


def test_relevance_filter_llm_path_uses_cached_state() -> None:
    papers = [
        _paper("p-heuristic", title="Gerontology and aging"),
        _paper("p-llm", title="Completely different subject"),
    ]

    responses = [
        LLMResponse(content=json.dumps({"score": 0.4, "rationale": "Relevant terms present"}), cached=False),
        LLMResponse(content=json.dumps({"score": 0.9, "rationale": "LLM judged relevant"}), cached=False),
    ]
    client = DummyLLMClient(responses)

    state: dict[str, Mapping[str, Any]] = {}
    relevance_filter = RelevanceFilter(
        keywords=["gerontology"],
        threshold=0.45,
        llm_client=client,  # type: ignore[arg-type]
        llm_weight=0.7,
    )

    accepted_first, decisions_first = relevance_filter.apply(
        papers,
        context={"theory": "Gerontology"},
        existing_decisions=state,
    )

    assert client.calls == 2
    assert {paper.identifier for paper in accepted_first} == {"p-heuristic", "p-llm"}
    llm_decision = next(decision for decision in decisions_first if decision.identifier == "p-llm")
    assert llm_decision.score >= 0.45
    assert state["p-llm"]["details"]["llm"]["score"] == pytest.approx(0.9)

    accepted_second, _ = relevance_filter.apply(
        papers,
        context={"theory": "Gerontology"},
        existing_decisions=state,
    )

    assert client.calls == 2, "LLM should not be invoked when decisions are cached"
    assert {paper.identifier for paper in accepted_second} == {"p-heuristic", "p-llm"}
