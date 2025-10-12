from __future__ import annotations

from theories_pipeline.extraction import QUESTION_CHOICES, QuestionExtractor
from theories_pipeline.literature import PaperMetadata


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

    assert mapped["Q1"].answer == "yes_quantitative"
    assert mapped["Q2"].answer == "mechanistic_evidence"
    assert mapped["Q3"].answer == "validated_intervention"
    assert mapped["Q4"].answer == "reversible"
    assert mapped["Q5"].answer == "yes_quantitative"
    assert mapped["Q6"].answer == "primary_focus"
    assert mapped["Q7"].answer == "primary_focus"
    assert mapped["Q8"].answer == "supported"
    assert mapped["Q9"].answer == "experimental"

    for answer in answers:
        assert answer.confidence > 0
        assert answer.answer in QUESTION_CHOICES[answer.question_id]
