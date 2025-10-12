from __future__ import annotations

from theories_pipeline.literature import PaperMetadata
from theories_pipeline.ontology import TheoryOntology
from theories_pipeline.theories import TheoryAssignment, TheoryClassifier


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
