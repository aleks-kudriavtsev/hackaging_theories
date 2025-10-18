from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from theories_pipeline.literature import PaperMetadata, RetrievalResult
from theories_pipeline.ontology_manager import OntologyManager, RuntimeNodeSpec
from theories_pipeline.theories import TheoryClassifier

from scripts.collect_theories import collect_for_entry


def test_ontology_manager_updates_classifier(tmp_path: Path) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage)
    classifier = TheoryClassifier({"Activity Theory": ["activity"]}, manager.ontology)
    classifier.attach_manager(manager)

    spec = RuntimeNodeSpec(
        name="Digital Aging",
        parent=None,
        config={"target": 2},
        keywords=["digital aging", "technology"],
        metadata={"summary": "LLM proposed"},
        provenance={"mode": "child", "source": "test"},
    )
    manager.append_child("Activity Theory", spec)

    assert "Digital Aging" in manager.ontology.names()
    assert "Digital Aging" in classifier.ontology.names()
    assert classifier.keyword_map["Digital Aging"] == ["digital aging", "technology"]
    payload = json.loads(storage.read_text(encoding="utf-8"))
    stored_names = [entry["name"] for entry in payload.get("nodes", [])]
    assert "Digital Aging" in stored_names
    runtime_node = classifier.ontology.get("Digital Aging")
    provenance = runtime_node.metadata.get("runtime_provenance", {})
    assert provenance.get("mode") == "child"
    assert provenance.get("source") == "test"


def test_update_keywords_refreshes_classifier(tmp_path: Path) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage)
    classifier = TheoryClassifier({"Activity Theory": ["activity"]}, manager.ontology)
    classifier.attach_manager(manager)

    paper = PaperMetadata(
        identifier="p-keyword",
        title="Digital aging interventions",
        authors=("Author",),
        abstract="Study of digital aging technologies in older adults.",
        source="Test",
    )

    initial_assignments = classifier.classify(paper)
    assert not any(assignment.theory == "Activity Theory" for assignment in initial_assignments)

    changed = manager.update_keywords("Activity Theory", ["digital aging"], merge=True)
    assert changed is True

    payload = json.loads(storage.read_text(encoding="utf-8"))
    stored_entry = next(entry for entry in payload["nodes"] if entry["name"] == "Activity Theory")
    assert "digital aging" in stored_entry.get("keywords", [])

    updated_assignments = classifier.classify(paper)
    assert any(assignment.theory == "Activity Theory" for assignment in updated_assignments)


def test_append_sibling_infers_parent(tmp_path: Path) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager(
        {
            "Activity Theory": {
                "target": 1,
                "subtheories": {"Engagement": {"target": 1}},
            }
        },
        storage_path=storage,
    )
    sibling_spec = RuntimeNodeSpec(
        name="Participation",
        parent=None,
        config={"target": 2},
        keywords=["participation"],
        provenance={"mode": "sibling", "source": "test"},
    )
    manager.append_sibling("Engagement", sibling_spec)
    assert manager.ontology.parent("Participation") == "Activity Theory"
    payload = json.loads(storage.read_text(encoding="utf-8"))
    saved = next(entry for entry in payload["nodes"] if entry["name"] == "Participation")
    assert saved["parent"] == "Activity Theory"
    assert saved["provenance"]["mode"] == "sibling"


def test_append_node_generates_summary_when_missing(tmp_path: Path) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage)
    added = manager.append_node(
        "Digital Engagement",
        parent="Activity Theory",
        keywords=["digital", "engagement"],
        metadata={},
    )
    assert added is True
    node = manager.ontology.get("Digital Engagement")
    summary = node.metadata.get("summary")
    assert isinstance(summary, str) and summary
    assert "digital" in summary.lower()


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


class _StubBootstrapper:
    def __init__(self, proposals: Sequence[RuntimeNodeSpec]) -> None:
        self.proposals = list(proposals)
        self.requests: List[Any] = []

    def propose_labels(self, request: Any) -> Any:
        self.requests.append(request)
        return SimpleNamespace(proposals=list(self.proposals))


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
    filtering_state = state.get("filtering")
    assert filtering_state, "Filter decisions should be stored in state"
    paper_state = filtering_state["p1"]
    assert paper_state["accepted"] is True
    assert paper_state["score"] >= 1.0


def test_runtime_bootstrap_persists_between_runs(tmp_path: Path) -> None:
    storage_path = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage_path)
    retriever = _DummyRetriever()
    proposals = [
        RuntimeNodeSpec(
            name="Digital Practices",
            parent=None,
            config={"target": 5},
            keywords=["digital"],
            metadata={"summary": "LLM"},
            provenance={"mode": "child", "source": "stub"},
        )
    ]
    bootstrapper = _StubBootstrapper(proposals)

    summary, _ = collect_for_entry(
        retriever,
        name="Activity Theory",
        config={
            "queries": ["activity theory aging"],
            "target": 1,
            "runtime_labels": {"threshold": 1, "mode": "child", "max_new_labels": 1},
        },
        context={"base_query": "activity theory", "query": "activity theory"},
        providers=None,
        resume=True,
        state_prefix="theory::activity-theory",
        ontology_manager=manager,
        expander=None,
        default_expansion=None,
        retrieval_options={},
        filter_llm_client=None,
        label_bootstrapper=bootstrapper,
    )

    assert bootstrapper.requests, "bootstrapper should have been invoked"
    assert "Digital Practices" in manager.ontology.names()
    enrichment = summary.get("enrichment", {})
    assert "bootstrap_labels" in enrichment
    assert "Digital Practices" in enrichment["bootstrap_labels"].get("child", [])

    # Subsequent run should reuse cached runtime ontology and skip bootstrap call
    second_manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage_path)
    second_bootstrapper = _StubBootstrapper(proposals)

    summary_two, _ = collect_for_entry(
        retriever,
        name="Activity Theory",
        config={
            "queries": ["activity theory aging"],
            "target": 1,
            "runtime_labels": {"threshold": 1, "mode": "child", "max_new_labels": 1},
        },
        context={"base_query": "activity theory", "query": "activity theory"},
        providers=None,
        resume=True,
        state_prefix="theory::activity-theory",
        ontology_manager=second_manager,
        expander=None,
        default_expansion=None,
        retrieval_options={},
        filter_llm_client=None,
        label_bootstrapper=second_bootstrapper,
    )

    assert not second_bootstrapper.requests, "bootstrapper should be skipped on rerun"
    assert "Digital Practices" in second_manager.ontology.names()
    state_payload = retriever.state_store.get("theory::activity-theory")
    bootstrap_state = state_payload["enrichment"]["bootstrap_labels"]["child"]
    assert bootstrap_state["status"] == "complete"
    assert summary_two.get("enrichment", {}).get("bootstrap_labels") is None


def test_runtime_autofragment_overflow(tmp_path: Path) -> None:
    storage_path = tmp_path / "runtime.json"
    manager = OntologyManager(
        {
            "Activity Theory": {
                "target": 2,
                "subtheories": {
                    "Engagement": {"target": 1},
                    "Participation": {"target": 3},
                },
            }
        },
        storage_path=storage_path,
    )
    retriever = _DummyRetriever()
    proposals = [
        RuntimeNodeSpec(
            name="Engagement - Digital", 
            parent=None,
            config={"target": 1},
            keywords=["digital"],
            metadata={"notes": "auto"},
            provenance={"mode": "child", "source": "stub"},
        )
    ]
    bootstrapper = _StubBootstrapper(proposals)

    summary, _ = collect_for_entry(
        retriever,
        name="Engagement",
        config={
            "queries": ["engagement aging"],
            "target": 1,
            "runtime_labels": {"threshold": 1, "mode": "sibling", "max_new_labels": 2},
        },
        context={"base_query": "activity theory", "theory": "Engagement"},
        providers=None,
        resume=True,
        state_prefix="theory::activity-theory::sub::engagement",
        ontology_manager=manager,
        expander=None,
        default_expansion=None,
        retrieval_options={},
        filter_llm_client=None,
        label_bootstrapper=bootstrapper,
    )

    assert bootstrapper.requests, "autofragment should invoke bootstrapper"
    request = bootstrapper.requests[0]
    assert request.mode == "child"
    assert request.parent == "Engagement"

    enrichment = summary.get("enrichment", {})
    reasons = enrichment.get("bootstrap_reasons", {})
    assert "child" in reasons
    reason_payload = reasons["child"]
    assert reason_payload["strategy"] == "sibling_imbalance"
    assert reason_payload["target_parent"] == "Engagement"

    state_payload = retriever.state_store.get("theory::activity-theory::sub::engagement")
    stored_reason = state_payload["enrichment"]["bootstrap_reasons"]["child"]
    assert stored_reason["strategy"] == "sibling_imbalance"
    assert "Engagement - Digital" in manager.ontology.names()
