from __future__ import annotations

import json

from theories_pipeline.literature import PaperMetadata
from theories_pipeline.ontology import TheoryOntology
from theories_pipeline.theories import TheoryAssignment, TheoryClassifier
from theories_pipeline.llm import LLMResponse, LLMRateLimitError


def test_classifier_returns_sorted_assignments() -> None:
    paper = PaperMetadata(
        identifier="p1",
        title="Activity engagement for seniors",
        authors=["Author"],
        abstract="The study applies activity theory and socioemotional selectivity theory.",
        source="Test",
    )
    ontology = TheoryOntology.from_targets_config(
        {
            "Activity Theory": {"target": 1},
            "Socioemotional Selectivity Theory": {"target": 1},
        }
    )
    classifier = TheoryClassifier(
        {
            "Activity Theory": ["activity"],
            "Socioemotional Selectivity Theory": ["socioemotional"],
        },
        ontology,
    )
    assignments = classifier.classify(paper)
    assert [assignment.theory for assignment in assignments] == [
        "Activity Theory",
        "Socioemotional Selectivity Theory",
    ]
    assert all(isinstance(item, TheoryAssignment) for item in assignments)


def test_classifier_rolls_up_scores_to_parents() -> None:
    paper = PaperMetadata(
        identifier="p2",
        title="A deep dive into engagement",
        authors=["Author"],
        abstract="Engagement as a subtheory of activity theory",
        source="Test",
    )
    ontology = TheoryOntology.from_targets_config(
        {
            "Activity Theory": {
                "target": 2,
                "subtheories": {
                    "Engagement": {"target": 1},
                },
            }
        }
    )
    classifier = TheoryClassifier({"Engagement": ["engagement"]}, ontology)
    assignments = classifier.classify(paper)
    theories = {assignment.theory for assignment in assignments}
    assert theories == {"Engagement", "Activity Theory"}
    counts = classifier.summarize(assignments)
    assert counts == {"Activity Theory": 1, "Engagement": 1}


class _DummyLLM:
    def __init__(self, responses=None, error=None):
        self._responses = responses or []
        self._error = error
        self.calls = 0

    def generate(self, messages_batch, *, model=None, temperature=None):  # type: ignore[override]
        self.calls += 1
        if self._error:
            raise self._error
        return list(self._responses)


def test_classifier_uses_llm_predictions_when_available() -> None:
    ontology = TheoryOntology.from_targets_config(
        {
            "Activity Theory": {
                "target": 1,
                "subtheories": {"Engagement": {"target": 1}},
            }
        }
    )
    paper = PaperMetadata(
        identifier="p3",
        title="Paper about engagement",
        authors=("Author",),
        abstract="Discusses engagement as part of activity theory.",
        source="Test",
    )
    response = LLMResponse(
        content=json.dumps(
            {
                "predictions": [
                    {
                        "theory": "Activity Theory",
                        "subtheory": "Engagement",
                        "confidence": 0.9,
                    }
                ]
            }
        ),
        cached=False,
    )
    llm_client = _DummyLLM(responses=[response])
    classifier = TheoryClassifier(
        {"Activity Theory": ["activity"], "Engagement": ["engagement"]},
        ontology,
        llm_client=llm_client,
    )
    assignments = classifier.classify(paper)
    theories = {assignment.theory for assignment in assignments}
    assert "Activity Theory" in theories
    assert "Engagement" in theories
    assert llm_client.calls == 1


def test_classifier_falls_back_to_keywords_on_rate_limit() -> None:
    ontology = TheoryOntology.from_targets_config(
        {"Activity Theory": {"target": 1}}
    )
    paper = PaperMetadata(
        identifier="p4",
        title="Activity focus",
        authors=("Author",),
        abstract="Activity and engagement are key.",
        source="Test",
    )
    llm_client = _DummyLLM(error=LLMRateLimitError("rate limit"))
    classifier = TheoryClassifier(
        {"Activity Theory": ["activity"]},
        ontology,
        llm_client=llm_client,
    )
    assignments = classifier.classify(paper)
    assert any(assignment.theory == "Activity Theory" for assignment in assignments)
