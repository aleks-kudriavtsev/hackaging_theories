from theories_pipeline.extraction import QuestionExtractor
from theories_pipeline.literature import PaperMetadata
from theories_pipeline.ontology import TheoryOntology
from theories_pipeline.pipeline_utils import classify_and_extract_parallel
from theories_pipeline.theories import TheoryClassifier


def _make_paper(identifier: str, title: str, abstract: str) -> PaperMetadata:
    return PaperMetadata(
        identifier=identifier,
        title=title,
        authors=("Author",),
        abstract=abstract,
        source="test",
    )


def test_classify_and_extract_parallel_matches_sequential() -> None:
    ontology = TheoryOntology.from_targets_config({"Activity Theory": {"target": 1}})
    classifier = TheoryClassifier.from_config(
        {"Activity Theory": ["activity", "engagement"]},
        ontology=ontology,
        llm_client=None,
    )
    extractor = QuestionExtractor()
    papers = [
        _make_paper("p1", "Activity engagement", "This paper studies activity theory."),
        _make_paper("p2", "Neutral study", "This work does not mention key terms."),
    ]

    parallel_assignments, parallel_answers = classify_and_extract_parallel(
        papers, classifier, extractor, workers=2, batch_size=1
    )

    sequential_assignments = classifier.classify_batch(papers)
    sequential_answers = [extractor.extract(paper) for paper in papers]

    assert parallel_assignments == sequential_assignments
    assert parallel_answers == sequential_answers
