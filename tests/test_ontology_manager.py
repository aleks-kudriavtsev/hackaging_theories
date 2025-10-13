from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from theories_pipeline.literature import PaperMetadata, RetrievalResult
from theories_pipeline.ontology_manager import OntologyManager
from theories_pipeline.theories import TheoryClassifier

from scripts.collect_theories import collect_for_entry


def test_ontology_manager_updates_classifier(tmp_path: Path) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage)
    classifier = TheoryClassifier({"Activity Theory": ["activity"]}, manager.ontology)
    classifier.attach_manager(manager)

    manager.append_node(
        "Digital Aging",
        parent="Activity Theory",
        config={"target": 2},
        keywords=["digital aging", "technology"],
    )

    assert "Digital Aging" in manager.ontology.names()
    assert "Digital Aging" in classifier.ontology.names()
    assert classifier.keyword_map["Digital Aging"] == ["digital aging", "technology"]
    payload = json.loads(storage.read_text(encoding="utf-8"))
    stored_names = [entry["name"] for entry in payload.get("nodes", [])]
    assert "Digital Aging" in stored_names


class _DummyStateStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self._data.get(key, {})))

    def set(self, key: str, value: Mapping[str, Any]) -> None:
        self._data[key] = json.loads(json.dumps(value))

    def clear(self, key: str) -> None:  # pragma: no cover - compatibility
        self._data.pop(key, None)

    def write_summary(self, summary: Mapping[str, Any]) -> None:  # pragma: no cover - compatibility
        self._data["summary"] = json.loads(json.dumps(summary))


class _DummyRetriever:
    def __init__(self) -> None:
        self.state_store = _DummyStateStore()
        self.last_queries: List[str] = []

    def collect_queries(
        self,
        queries: Iterable[str],
        *,
        target: int | None = None,
        providers: Iterable[str] | None = None,
        state_key: str | None,
        resume: bool,
        min_citation_count: int | None = None,
        prefer_reviews: bool = False,
        sort_by_citations: bool = False,
    ) -> RetrievalResult:
        del providers, resume, min_citation_count, prefer_reviews, sort_by_citations
        self.last_queries = list(queries)
        papers = [
            PaperMetadata(
                identifier="p1",
                title="Digital aging study",
                authors=("Author",),
                abstract="Explores technology and aging",
                source="Test",
            )
        ]
        summary = {
            "target": target,
            "total_unique": len(papers),
            "newly_retrieved": len(papers),
            "providers": {},
            "queries": list(self.last_queries),
            "met_target": True,
            "prior_total": 0,
        }
        state_payload = {
            "seen_identifiers": [paper.dedupe_key for paper in papers],
            "papers": [paper.to_dict() for paper in papers],
            "provider_totals": {},
            "queries": {},
        }
        if state_key:
            self.state_store.set(state_key, state_payload)
        return RetrievalResult(papers=papers, newly_added=len(papers), summary=summary)


def test_collect_for_entry_enrichment(tmp_path: Path) -> None:
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=tmp_path / "runtime.json")
    retriever = _DummyRetriever()
    summary, papers = collect_for_entry(
        retriever,
        name="Activity Theory",
        config={
            "queries": ["{query} gerontology"],
            "target": 1,
            "enrichment": {
                "new_theories": [
                    {
                        "name": "Digital Aging",
                        "keywords": ["digital aging"],
                        "parent": "Activity Theory",
                        "target": 2,
                    }
                ],
                "query_shards": [
                    {"query": "digital aging seniors", "source": "llm"},
                    {"query": "outdated technology", "prune": True},
                ],
            },
        },
        context={"base_query": "activity theory", "query": "activity theory"},
        providers=None,
        resume=True,
        state_prefix="theory::activity-theory",
        ontology_manager=manager,
        expander=None,
        default_expansion=None,
        retrieval_options={},
    )

    assert papers and papers[0].title == "Digital aging study"
    assert "enrichment" in summary
    enrichment_summary = summary["enrichment"]
    assert enrichment_summary["new_theories"] == ["Digital Aging"]
    assert "digital aging seniors" in enrichment_summary["queries_used"]
    assert "outdated technology" in enrichment_summary["pruned_queries"]
    assert "Digital Aging" in manager.ontology.names()

    state = retriever.state_store.get("theory::activity-theory")
    enrichment_state = state["enrichment"]
    theory_state = enrichment_state["new_theories"]["Digital Aging"]
    assert theory_state["status"] == "added"
    shard_states = list(enrichment_state["query_shards"].values())
    statuses = {item["query"]: item["status"] for item in shard_states}
    assert statuses["digital aging seniors"] == "consumed"
    assert statuses["outdated technology"] == "pruned"
    assert "digital aging seniors" in retriever.last_queries

