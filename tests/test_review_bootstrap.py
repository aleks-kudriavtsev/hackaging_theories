from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from theories_pipeline.literature import PaperMetadata, RetrievalResult
from theories_pipeline.review_bootstrap import (
    BootstrapResult,
    ReviewDocument,
    build_bootstrap_ontology,
    extract_theories_from_review,
    merge_bootstrap_into_targets,
    normalise_review_metadata,
    pull_top_cited_reviews,
)


class _FakeRetriever:
    def __init__(self, mapping: Mapping[str, Sequence[PaperMetadata]]) -> None:
        self.mapping = mapping
        self.calls: List[Dict[str, object]] = []

    def collect_queries(
        self,
        queries: Sequence[str],
        *,
        target: int | None,
        providers: Sequence[str] | None,
        state_key: str | None,
        resume: bool,
    ) -> RetrievalResult:
        query = queries[0]
        self.calls.append(
            {
                "query": query,
                "providers": tuple(providers) if providers else None,
                "state_key": state_key,
                "resume": resume,
                "target": target,
            }
        )
        papers = list(self.mapping.get(query, ()))
        return RetrievalResult(papers=papers, newly_added=len(papers), summary={})


def _paper(identifier: str, *, title: str, abstract: str, source: str, citations: str) -> PaperMetadata:
    return PaperMetadata(
        identifier=identifier,
        title=title,
        authors=("Author",),
        abstract=f"{abstract} Citations: {citations}",
        source=source,
        year=2020,
        doi=None,
        full_text="",
        citation_count=int(citations),
        is_review=True,
    )


def test_pull_top_cited_reviews_filters_and_orders() -> None:
    query = "aging review"
    papers = [
        _paper(
            "p1",
            title="Comprehensive review of Activity Theory",
            abstract="This review summarises",
            source="openalex",
            citations="50",
        ),
        PaperMetadata(
            identifier="p2",
            title="Primary research on biomarkers",
            authors=("B",),
            abstract="Experimental study",
            source="openalex",
            year=2021,
            doi=None,
            full_text="",
            citation_count=5,
            is_review=False,
        ),
        _paper(
            "p3",
            title="Meta-analysis of socioemotional selectivity",
            abstract="A systematic review",
            source="openalex",
            citations="10",
        ),
    ]
    retriever = _FakeRetriever({query: papers})
    overrides = {"p3": 80}

    results = pull_top_cited_reviews(
        retriever,
        [query],
        providers=["openalex"],
        min_citations=20,
        max_per_query=2,
        citation_overrides=overrides,
        context={"base_query": query},
    )

    assert query in results
    docs = results[query]
    assert [doc.paper.identifier for doc in docs] == ["p3", "p1"]
    metadata = normalise_review_metadata(docs)
    assert metadata[0]["citations"] == 80
    assert metadata[0]["query"] == query


def test_extract_theories_and_build_bootstrap_tree() -> None:
    paper = PaperMetadata(
        identifier="rev-1",
        title="Activity theory review",
        authors=("Author",),
        abstract="This review surveys classic theories of aging.",
        source="openalex",
        year=2019,
        doi=None,
        full_text=(
            "Activity Theory discusses engagement. Subtheories: Engagement; Participation.\n"
            "Socioemotional Selectivity Theory emphasises emotional goals.\n"
        ),
        citation_count=120,
        is_review=True,
    )
    review = ReviewDocument(query="aging review", paper=paper, citations=120)
    secondary = PaperMetadata(
        identifier="rev-2",
        title="Follow-up review on activity theory",
        authors=("Author",),
        abstract="Another review discussing continuity.",
        source="openalex",
        year=2020,
        doi=None,
        full_text="Activity Theory advocates participation.",
        citation_count=30,
        is_review=True,
    )
    review_two = ReviewDocument(query="aging review", paper=secondary, citations=30)
    result = extract_theories_from_review(review, llm_client=None)
    result_two = extract_theories_from_review(review_two, llm_client=None)
    nodes = build_bootstrap_ontology([result, result_two])

    assert "Activity Theory" in nodes
    activity = nodes["Activity Theory"]
    assert activity["bootstrap"]["citations"] == 150
    assert sorted(activity["bootstrap"]["reviews"]) == ["rev-1", "rev-2"]
    sub_names = set(activity["subtheories"].keys())
    assert {name.lower() for name in sub_names} == {"engagement", "participation"}


def _leaf_count_from_config(node: Mapping[str, Any]) -> int:
    sub = node.get("subtheories")
    if not isinstance(sub, Mapping) or not sub:
        return 1
    return sum(_leaf_count_from_config(child) for child in sub.values())


def test_build_bootstrap_ontology_balances_children_evenly() -> None:
    paper = PaperMetadata(
        identifier="rev-balanced",
        title="Balancing review",
        authors=("Author",),
        abstract="",
        source="openalex",
        year=2021,
        doi=None,
        full_text="",
        citation_count=90,
        is_review=True,
    )
    review = ReviewDocument(query="aging review", paper=paper, citations=90)
    parent = {
        "name": "Composite Theory",
        "subtheories": [
            {"name": f"Theory {index}", "subtheories": []}
            for index in range(1, 11)
        ],
    }
    result = BootstrapResult(review=review, theories=[parent])

    ontology = build_bootstrap_ontology([result], max_children=3)

    composite = ontology["Composite Theory"]
    subtheories = composite["subtheories"]
    assert len(subtheories) == 3

    leaf_counts = [_leaf_count_from_config(node) for node in subtheories.values()]
    assert max(leaf_counts) - min(leaf_counts) <= 1

    branch_sizes = [
        node.get("bootstrap", {}).get("child_summary", {}).get("branch_size", 0)
        for node in subtheories.values()
    ]
    assert max(branch_sizes) - min(branch_sizes) <= 1

def test_merge_bootstrap_into_targets_handles_missing_nodes() -> None:
    base = {
        "Activity Theory": {
            "target": 100,
            "subtheories": {"Participation": {"target": 50}},
        }
    }
    bootstrap_nodes = {
        "Activity Theory": {
            "bootstrap": {
                "citations": 120,
                "reviews": ["rev-1"],
                "queries": ["aging review"],
            },
            "subtheories": {
                "Engagement": {
                    "bootstrap": {"citations": 120, "reviews": ["rev-1"]},
                    "subtheories": {},
                },
                "Participation": {
                    "bootstrap": {"citations": 40, "reviews": ["rev-2"]},
                    "subtheories": {},
                },
            },
        },
        "Continuity Theory": {
            "bootstrap": {"citations": 60, "reviews": ["rev-3"]},
            "subtheories": {},
        },
    }

    merged_existing = merge_bootstrap_into_targets(base, bootstrap_nodes, inject_missing=False)
    assert "Continuity Theory" not in merged_existing
    activity = merged_existing["Activity Theory"]
    assert activity["bootstrap"]["citations"] == 120
    assert sorted(activity["bootstrap"]["reviews"]) == ["rev-1"]
    assert "Engagement" in activity["subtheories"]
    assert activity["subtheories"]["Participation"]["bootstrap"]["citations"] == 40

    merged_with_new = merge_bootstrap_into_targets(base, bootstrap_nodes, inject_missing=True)
    assert "Continuity Theory" in merged_with_new
    assert merged_with_new["Continuity Theory"]["bootstrap"]["citations"] == 60
