from pathlib import Path

from theories_pipeline.literature import PaperMetadata, RetrievalResult
from theories_pipeline.review_bootstrap import bootstrap_ontology


class DummyRetriever:
    def __init__(self, papers):
        self.papers = papers
        self.state_store = type(
            "State",
            (),
            {
                "get": staticmethod(lambda _key: {}),
            },
        )()

    def collect_queries(self, queries, *, target, providers, state_key, resume):  # noqa: D401 - test stub
        return RetrievalResult(papers=self.papers, newly_added=len(self.papers), summary={"target": target})


def make_paper(identifier: str, title: str, abstract: str) -> PaperMetadata:
    return PaperMetadata(
        identifier=identifier,
        title=title,
        authors=("Doe",),
        abstract=abstract,
        source="test",
    )


def test_bootstrap_generates_targets(tmp_path: Path):
    papers = [
        make_paper("1", "Disposable Soma Theory", "The disposable soma theory of aging is summarised."),
        make_paper("2", "Oxidative Stress Theory", "Oxidative stress theory overview."),
    ]
    retriever = DummyRetriever(papers)
    targets, summary, accepted = bootstrap_ontology(
        "aging theory",
        retriever,
        llm_client=None,
        providers=None,
        resume=False,
        target_count=100,
        config={"cache_dir": tmp_path, "min_target": 10, "review_limit": 5},
    )
    assert "Disposable Soma Theory" in targets
    assert summary.theory_count >= 1
    assert summary.snapshot_path.exists()
    assert accepted
