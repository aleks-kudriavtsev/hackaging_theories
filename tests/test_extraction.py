from __future__ import annotations

from theories_pipeline.extraction import QuestionExtractor
from theories_pipeline.literature import PaperMetadata


def test_extractor_returns_sentence_matches() -> None:
    paper = PaperMetadata(
        identifier="p1",
        title="Goal-oriented engagement",
        authors=["Researcher"],
        abstract=(
            "The research question investigates how goals change with age. "
            "Our mixed-method design combines surveys and interviews. "
            "Participants included 45 older adults from community centers. "
            "Key outcomes measured were engagement levels and wellbeing. "
            "Interventions focused on community-led activity programs. "
            "Findings highlight increased participation when goals are socioemotional. "
            "Limitations include a small sample size. "
            "Future work will examine digital interventions."
        ),
        source="Test",
    )
    extractor = QuestionExtractor(
        {
            "Q1": ["goal"],
            "Q2": ["research question"],
            "Q3": ["design"],
            "Q4": ["participants"],
            "Q5": ["outcomes"],
            "Q6": ["interventions"],
            "Q7": ["findings"],
            "Q8": ["limitations"],
            "Q9": ["future"],
        }
    )
    answers = extractor.extract(paper)
    mapped = {answer.question_id: answer.answer for answer in answers}
    assert mapped["Q3"].startswith("Our mixed-method design")
    assert mapped["Q4"].startswith("Participants included")
    assert mapped["Q8"].startswith("Limitations include")
