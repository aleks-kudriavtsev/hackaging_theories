from theories_pipeline.filtering import RelevanceFilter
from theories_pipeline.literature import PaperMetadata


def make_paper(identifier: str, title: str, abstract: str) -> PaperMetadata:
    return PaperMetadata(
        identifier=identifier,
        title=title,
        authors=("Doe",),
        abstract=abstract,
        source="test",
    )


def test_relevance_filter_accepts_matching_paper():
    paper = make_paper("1", "Aging theory review", "Comprehensive overview of modern aging theories")
    flt = RelevanceFilter("aging theory")
    decision = flt.evaluate(paper)
    assert decision.accepted
    assert decision.score > 0


def test_relevance_filter_rejects_unrelated_paper():
    paper = make_paper("2", "Quantum field methods", "Study of particle physics with no biology")
    flt = RelevanceFilter("aging theory", threshold=0.2)
    decision = flt.evaluate(paper)
    assert not decision.accepted
    assert decision.score == 0
