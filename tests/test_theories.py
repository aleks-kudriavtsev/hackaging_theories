from __future__ import annotations

from theories_pipeline.literature import PaperMetadata
from theories_pipeline.theories import TheoryAssignment, TheoryClassifier


def test_classifier_returns_sorted_assignments() -> None:
    paper = PaperMetadata(
        identifier="p1",
        title="Activity engagement for seniors",
        authors=["Author"],
        abstract="The study applies activity theory and socioemotional selectivity theory.",
        source="Test",
    )
    classifier = TheoryClassifier(
        {
            "Activity Theory": ["activity"],
            "Socioemotional Selectivity Theory": ["socioemotional"],
        }
    )
    assignments = classifier.classify(paper)
    assert [assignment.theory for assignment in assignments] == [
        "Activity Theory",
        "Socioemotional Selectivity Theory",
    ]
    assert all(isinstance(item, TheoryAssignment) for item in assignments)
