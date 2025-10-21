from __future__ import annotations

import json

import pytest

from theories_pipeline.extraction import QUESTION_CHOICES, QuestionExtractor
from theories_pipeline.literature import PaperMetadata
from theories_pipeline.llm import LLMResponse


class DummyLLMClient:
    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.calls: int = 0
        self.messages = []

    def generate(self, messages_batch, *, model=None, temperature=None):  # type: ignore[override]
        self.messages.append(messages_batch)
        payload = (
            self.payloads[self.calls]
            if self.calls < len(self.payloads)
            else self.payloads[-1]
        )
        self.calls += 1
        return [LLMResponse(content=payload, cached=False)]


def test_extractor_returns_categorical_answers() -> None:
    paper = PaperMetadata(
        identifier="p1",
        title="Comparative longevity interventions",
        authors=["Researcher"],
        abstract=(
            "We quantified plasma IL-6 as a biomarker of aging using ELISA measurements. "
            "Mechanistic experiments demonstrate that inhibiting mTOR signaling extends longevity. "
            "The dietary intervention extends lifespan in mice and is discussed as a validated strategy. "
            "These changes were reversible after treatment cessation, indicating reversibility. "
            "A comparative regression across species predicted lifespan from mitochondrial GC content. "
            "Our analysis of naked mole rat longevity includes detailed survival modeling. "
            "We analyzed avian longevity datasets to assess extended lifespan in birds. "
            "Body size correlated with lifespan (r = 0.72) across sampled mammals. "
            "A randomized trial of calorie restriction extended lifespan in rodents."
        ),
        source="Test",
    )
    extractor = QuestionExtractor()
    answers = extractor.extract(paper)
    assert len(answers) == len(QUESTION_CHOICES)
    mapped = {answer.question_id: answer for answer in answers}

    assert mapped["Q1"].answer == "Yes, quantitatively shown"
    assert mapped["Q2"].answer == "Mechanism supported by experiments"
    assert mapped["Q3"].answer == "Validated longevity intervention"
    assert mapped["Q4"].answer == "Changes appear reversible"
    assert mapped["Q5"].answer == "Yes, quantitatively shown"
    assert mapped["Q6"].answer == "Primary focus of the paper"
    assert mapped["Q7"].answer == "Primary focus of the paper"
    assert mapped["Q8"].answer == "Link supported by data"
    assert mapped["Q9"].answer == "Experimental evidence presented"

    for answer in answers:
        assert answer.confidence > 0
        assert answer.answer in QUESTION_CHOICES[answer.question_id]


def test_extractor_uses_llm_when_available() -> None:
    llm_payload = json.dumps(
        {
            "answer": "Yes, quantitatively shown",
            "confidence": 0.92,
            "rationale": "Evidence sentences describe quantified biomarker levels.",
        }
    )
    llm_client = DummyLLMClient(llm_payload)
    extractor = QuestionExtractor({"llm": {"enabled": True}}, llm_client=llm_client)
    paper = PaperMetadata(
        identifier="p2",
        title="Study of aging",
        authors=["Author"],
        abstract="This manuscript discusses experiments about senescence and cellular processes.",
        source="Test",
    )

    answers = extractor.extract(paper)
    mapped = {answer.question_id: answer for answer in answers}
    q1 = mapped["Q1"]

    assert q1.answer == "Yes, quantitatively shown"
    assert q1.confidence > 0.9
    assert q1.gpt_confidence is not None
    assert q1.gpt_confidence == pytest.approx(0.92, rel=1e-3)

    evidence = json.loads(q1.evidence or "{}")
    assert evidence["heuristic"]["confidence"] == pytest.approx(0.1)
    assert evidence["heuristic"]["answer"] == "No evidence found"
    assert evidence["gpt"]["rationale"].startswith("Evidence sentences")
    assert llm_client.calls == len(QUESTION_CHOICES)


def test_extractor_llm_decline_falls_back_to_heuristics() -> None:
    llm_payload = json.dumps(
        {
            "answer": "unknown",
            "confidence": 0.2,
            "rationale": "The evidence does not clearly support an answer.",
        }
    )
    llm_client = DummyLLMClient(llm_payload)
    extractor = QuestionExtractor({"llm": {"enabled": True}}, llm_client=llm_client)
    paper = PaperMetadata(
        identifier="p3",
        title="Biomarker quantification",
        authors=["Author"],
        abstract="We measured plasma IL-6 levels as an aging biomarker in mice.",
        source="Test",
    )

    answers = extractor.extract(paper)
    mapped = {answer.question_id: answer for answer in answers}
    q1 = mapped["Q1"]

    assert q1.answer == "Yes, quantitatively shown"
    assert q1.confidence >= 0.85
    assert q1.heuristic_confidence == pytest.approx(0.9)
    assert q1.gpt_confidence == pytest.approx(0.2)

    evidence = json.loads(q1.evidence or "{}")
    assert evidence["gpt"]["answer"] == "unknown"
    assert evidence["heuristic"]["answer"] == "Yes, quantitatively shown"
    assert llm_client.calls == len(QUESTION_CHOICES)
