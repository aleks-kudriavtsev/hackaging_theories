"""Integration tests for ontology suggestion plumbing in the retrieval loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_theories import collect_for_entry
from theories_pipeline.literature import RetrievalResult


class _DummyStateStore:
    def __init__(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Dict[str, Any]:
        stored = self.data.get(key, {})
        return json.loads(json.dumps(stored))

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value


class _DummyRetriever:
    def __init__(self) -> None:
        self.state_store = _DummyStateStore()
        self.queries: List[List[str]] = []

    def collect_queries(self, queries: List[str], **_: Any) -> RetrievalResult:
        self.queries.append(list(queries))
        return RetrievalResult(papers=[], newly_added=0, summary={"total_unique": 0})


def test_collect_for_entry_tracks_ontology_suggestions() -> None:
    retriever = _DummyRetriever()
    summary, papers = collect_for_entry(
        retriever,
        name="Mitochondrial Theory",
        config={"suggested_queries": ["mitochondrial aging"], "target": 0},
        context={"base_query": "cellular senescence"},
        providers=None,
        resume=False,
        state_prefix="theory::mitochondrial-theory",
        ontology_manager=None,
        expander=None,
        default_expansion=None,
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=None,
    )

    assert papers == []
    assert summary["query_source"] == "suggested"
    assert summary["suggested_queries"] == ["mitochondrial aging"]
    assert summary["suggested_queries_used"] is True
    assert retriever.queries == [["mitochondrial aging"]]

    state_entry = retriever.state_store.data["theory::mitochondrial-theory"]
    query_state = state_entry["queries"]
    assert query_state["templates"] == ["mitochondrial aging"]
    assert query_state["source"] == "suggested"
    assert query_state["suggested"] == ["mitochondrial aging"]
    assert query_state["suggested_used"] is True
