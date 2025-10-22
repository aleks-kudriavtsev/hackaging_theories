from __future__ import annotations

import json

from theories_pipeline.literature import PaperMetadata
from theories_pipeline.ontology import TheoryOntology
from theories_pipeline.ontology_manager import OntologyManager
from theories_pipeline.theories import TheoryAssignment, TheoryClassifier
from theories_pipeline.llm import LLMResponse, LLMRateLimitError


def test_classifier_returns_sorted_assignments() -> None:
    paper = PaperMetadata(
        identifier="p1",
        title="Activity engagement for seniors",
        authors=["Author"],
        abstract="",
        full_text="The study applies activity theory and socioemotional selectivity theory.",
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


def test_ontology_preserves_dynamic_metadata() -> None:
    config = {
        "Quickstart Query": {
            "target": None,
            "metadata": {"source": "quickstart"},
            "queries": ["aging review"],
            "subtheories": {
                "Activity Theory": {
                    "bootstrap": {"citations": 150, "reviews": ["rev-1"]},
                    "subtheories": {},
                }
            },
        }
    }
    ontology = TheoryOntology.from_targets_config(config)
    root_node = ontology.get("Quickstart Query")
    assert root_node.metadata["source"] == "quickstart"
    assert root_node.metadata["queries"] == ["aging review"]
    activity = ontology.get("Activity Theory")
    assert activity.parent == "Quickstart Query"
    assert activity.metadata["bootstrap"]["citations"] == 150


def test_classifier_handles_runtime_appended_nodes(tmp_path) -> None:
    manager = OntologyManager({"Quickstart Query": {"metadata": {"source": "quickstart"}}}, storage_path=tmp_path / "ontology.json")
    classifier = TheoryClassifier({"Quickstart Query": ["quickstart"]}, manager.ontology)
    classifier.attach_manager(manager)

    manager.append_node(
        "Activity Theory",
        parent="Quickstart Query",
        config={"bootstrap": {"citations": 120}},
        keywords=["activity"],
        metadata={"source": "review_bootstrap"},
    )

    paper = PaperMetadata(
        identifier="dyn-1",
        title="Activity theory insights",
        authors=["Author"],
        abstract="Explores activity participation among older adults.",
        source="Test",
    )
    assignments = classifier.classify(paper)
    theories = {assignment.theory for assignment in assignments}
    assert "Activity Theory" in theories
    assert "Quickstart Query" in theories
    assert classifier.keyword_map["Activity Theory"] == ["activity"]
    runtime_node = classifier.ontology.get("Activity Theory")
    assert runtime_node.metadata["bootstrap"]["citations"] == 120
    assert runtime_node.metadata["source"] == "review_bootstrap"


def test_format_ontology_prompt_includes_metadata() -> None:
    ontology = TheoryOntology.from_targets_config(
        {
            "Activity Theory": {
                "metadata": {
                    "summary": "Active engagement fosters healthy aging.",
                    "bootstrap": {
                        "citations": 120,
                        "reviews": ["rev-1", "rev-2"],
                        "queries": ["activity aging"],
                    },
                    "quotes": ["Active participation supports well-being."],
                },
                "subtheories": {
                    "Participation": {
                        "metadata": {},
                    }
                },
            }
        }
    )
    classifier = TheoryClassifier(
        {
            "Activity Theory": ["activity", "engagement"],
            "Participation": ["participation"],
        },
        ontology,
    )
    prompt = classifier._format_ontology_prompt()
    lines = prompt.splitlines()
    assert any("Activity Theory" in line and "Active engagement fosters healthy aging." in line for line in lines)
    assert any("Key terms: activity, engagement" in line for line in lines)
    assert any("Bootstrap: citations=120; reviews=rev-1, rev-2; queries=activity aging" in line for line in lines)
    assert any("Quote: Active participation supports well-being." in line for line in lines)
    assert any(
        "Participation" in line
        and "focuses on participation in aging research" in line.lower()
        for line in lines
    )


def test_depth_deficits_groups_nodes_and_updates_report() -> None:
    config = {
        "Root": {
            "target": 3,
            "subtheories": {
                "Child A": {
                    "target": 2,
                    "subtheories": {"Grandchild": {"target": 1}},
                },
                "Child B": {"target": 1},
            },
        }
    }
    ontology = TheoryOntology.from_targets_config(config)
    counts = {
        "Root": 2,
        "Child A": 1,
        "Child B": 1,
        "Grandchild": 0,
    }

    depth_deficits = ontology.depth_deficits(counts)
    assert sorted(depth_deficits.keys()) == [0, 1, 2]
    assert [record.name for record in depth_deficits[0]] == ["Root"]
    assert [record.name for record in depth_deficits[1]] == ["Child A"]
    assert [record.name for record in depth_deficits[2]] == ["Grandchild"]

    report = ontology.format_coverage_report(counts)
    assert "Deficit summary by depth:" in report
    assert "Depth 2: total deficit" in report
    assert "Grandchild (-1)" in report
