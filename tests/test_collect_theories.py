from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import pytest
from theories_pipeline.literature import PaperMetadata
from theories_pipeline.ontology_manager import OntologyManager, OntologyUpdate, RuntimeNodeSpec
from theories_pipeline.query_expansion import (
    QueryCandidate,
    QueryExpander,
    QueryExpansionSession,
    QueryExpansionSettings,
)
from theories_pipeline.review_bootstrap import ReviewDocument


def _load_collect_theories_module():
    module_name = "collect_theories_test_module"
    if module_name in globals():  # pragma: no cover - defensive cache
        return globals()[module_name]

    spec = importlib.util.spec_from_file_location(module_name, Path("scripts/collect_theories.py"))
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Unable to load collect_theories script module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals()[module_name] = module
    return module


collect_theories = _load_collect_theories_module()


def test_audit_runtime_ontology_nodes_detects_missing_children(caplog):
    runtime_targets = {
        "Root": {
            "target": 2,
            "subtheories": {
                "Present": {"target": 1},
            },
        }
    }
    ontology_targets = {
        "Root": {
            "target": 2,
            "subtheories": {
                "Present": {"target": 1},
                "Missing": {"target": 1},
            },
        }
    }

    with caplog.at_level(logging.WARNING):
        collect_theories.audit_runtime_ontology_nodes(
            runtime_targets,
            ontology_targets,
            strict=False,
        )

    audit_messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Missing" in audit_messages
    assert "runtime nodes" in audit_messages

    with pytest.raises(SystemExit):
        collect_theories.audit_runtime_ontology_nodes(
            runtime_targets,
            ontology_targets,
            strict=True,
        )


def test_audit_runtime_node_counts_detects_discrepancies(caplog):
    runtime_targets = {
        "Root": {
            "target": 1,
            "subtheories": {
                "OnlyRuntime": {"target": 1},
            },
        }
    }
    ontology_targets = {
        "Root": {
            "target": 1,
            "subtheories": {
                "OnlyRuntime": {"target": 1},
                "MissingInRuntime": {"target": 1},
            },
        }
    }

    with caplog.at_level(logging.WARNING):
        collect_theories.audit_runtime_node_counts(
            runtime_targets,
            ontology_targets,
            strict=False,
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "node-count mismatch" in messages

    with pytest.raises(SystemExit):
        collect_theories.audit_runtime_node_counts(
            runtime_targets,
            ontology_targets,
            strict=True,
        )


class DummyStateStore:
    def __init__(self):
        self.written = None

    def get(self, _key):  # pragma: no cover - deterministic noop
        return {}

    def write_summary(self, payload):
        self.written = payload


class DummyRetriever:
    def __init__(self, *_args, **_kwargs):
        self.state_store = DummyStateStore()


class DummyOntology:
    def __init__(self) -> None:
        self._node = SimpleNamespace(metadata={}, parent=None, children=())

    def names(self):
        return ()

    def get(self, _name):  # pragma: no cover - simple stub
        return self._node

    def target(self, _name):  # pragma: no cover - simple stub
        return None

    def coverage(self, _counts):
        return {}

    def format_coverage_report(self, _counts):
        return "dummy coverage"

    def depth_deficits(self, _counts):
        return {}

    def deficit_summary_by_depth(self, _counts):
        return {}


class DummyOntologyManager:
    def __init__(self, *_args, **_kwargs):
        self.ontology = DummyOntology()

    def append_node(self, *args, **kwargs):  # pragma: no cover - unused in tests
        return None

    def get_node_config(self, _name):  # pragma: no cover - defensive stub
        return {}


class RecordingOntologyManager:
    def __init__(self, base_config: Mapping[str, Any], *, storage_path=None):
        self.base_config = base_config
        self.ontology = DummyOntology()
        self.appended: List[Dict[str, Any]] = []
        self._known_nodes: set[str] = set()

        def _register(mapping: Mapping[str, Any]) -> None:
            for name, value in mapping.items():
                self._known_nodes.add(str(name))
                if isinstance(value, Mapping):
                    sub_map = value.get("subtheories")
                    if isinstance(sub_map, Mapping):
                        _register(sub_map)

        _register(base_config)

    def has_node(self, name: str) -> bool:
        return name in self._known_nodes

    def append_node(
        self,
        name: str,
        *,
        parent: str | None = None,
        config: Mapping[str, Any] | None = None,
        keywords: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.appended.append(
            {
                "name": name,
                "parent": parent,
                "config": dict(config or {}),
                "keywords": list(keywords) if keywords else [],
                "metadata": dict(metadata or {}),
                "provenance": dict(provenance or {}),
            }
        )
        self._known_nodes.add(name)

    def get_node_config(self, name: str):
        def _search(mapping: Mapping[str, Any]):
            for node_name, data in mapping.items():
                if node_name == name and isinstance(data, Mapping):
                    return data
                if isinstance(data, Mapping):
                    sub_map = data.get("subtheories")
                    if isinstance(sub_map, Mapping):
                        found = _search(sub_map)
                        if found is not None:
                            return found
            return None

        base_entry = _search(self.base_config)
        if base_entry is not None:
            return base_entry
        for entry in self.appended:
            if entry["name"] == name:
                payload = dict(entry.get("config") or {})
                payload.setdefault("subtheories", {})
                return payload
        return None

class DummyClassifier:
    @classmethod
    def from_config(cls, _config, *, ontology, llm_client):
        return cls()

    def attach_manager(self, _manager):
        return None

    def summarize(self, _assignments, *, include_ids: bool = False):
        if include_ids:
            return {}
        return {}


def _prepare_args(tmp_path: Path, config_path: Path, *, quickstart: bool, target_count: int | None):
    return argparse.Namespace(
        query="Test Query",
        config=config_path,
        quickstart=quickstart,
        target_count=target_count,
        limit=None,
        providers=None,
        openalex_api_key=None,
        crossref_api_key=None,
        pubmed_api_key=None,
        no_resume=False,
        state_dir=None,
        llm_model=None,
        llm_temperature=None,
        llm_batch_size=None,
        llm_cache_dir=None,
        llm_api_key=None,
        parallel_fetch=None,
        classification_workers=None,
        verify_bootstrap=False,
        bootstrap_verify_dir=None,
    )


def _write_config(tmp_path: Path, targets: dict[str, object]) -> Path:
    config_path = tmp_path / "config.json"
    config = {
        "api_keys": {},
        "data_sources": {"seed_papers": str(tmp_path / "seed.json")},
        "providers": [],
        "outputs": {
            "papers": str(tmp_path / "out" / "papers.csv"),
            "theories": str(tmp_path / "out" / "theories.csv"),
            "questions": str(tmp_path / "out" / "questions.csv"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "corpus": {
            "cache_dir": str(tmp_path / "literature"),
            "targets": targets,
        },
        "extraction": {},
        "classification": {},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _patch_runtime(
    monkeypatch,
    tmp_path: Path,
    *,
    args,
    validate_calls: List[dict],
    bootstrap_hook: Callable[..., Sequence[Any]] | None = None,
    collect_hook: Callable[..., tuple[Mapping[str, Any], Sequence[Any]]] | None = None,
    ontology_factory: Callable[..., Any] = DummyOntologyManager,
) -> None:
    monkeypatch.setattr(collect_theories.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(collect_theories, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collect_theories, "resolve_api_keys", lambda *_a, **_k: {})
    monkeypatch.setattr(collect_theories, "build_provider_configs", lambda *_a, **_k: [])
    monkeypatch.setattr(collect_theories, "LiteratureRetriever", DummyRetriever)
    if bootstrap_hook is None:
        bootstrap_hook = lambda *_a, **_k: ({}, {}, {})
    monkeypatch.setattr(collect_theories, "_run_bootstrap_phase", bootstrap_hook)
    monkeypatch.setattr(collect_theories, "_existing_total", lambda *_a, **_k: 0)

    if collect_hook is None:
        def collect_hook(*_a, **_k):  # type: ignore[misc]
            return {"providers": {"stub": 1}}, [SimpleNamespace(identifier="paper-1")]

    monkeypatch.setattr(collect_theories, "collect_for_entry", collect_hook)
    monkeypatch.setattr(collect_theories, "OntologyManager", ontology_factory)
    monkeypatch.setattr(collect_theories, "TheoryClassifier", DummyClassifier)
    monkeypatch.setattr(collect_theories, "QuestionExtractor", lambda _cfg: object())
    monkeypatch.setattr(
        collect_theories,
        "classify_and_extract_parallel",
        lambda papers, classifier, extractor, workers: ([["assignment"]], [["answer"]]),
    )
    monkeypatch.setattr(collect_theories, "export_papers", lambda *a, **k: None)
    monkeypatch.setattr(collect_theories, "export_theories", lambda *a, **k: None)
    monkeypatch.setattr(collect_theories, "export_question_answers", lambda *a, **k: None)

    def fake_validate(summary):
        validate_calls.append(summary)

    monkeypatch.setattr(collect_theories, "validate_targets", fake_validate)


def test_main_uses_config_targets_when_available(monkeypatch, tmp_path):
    targets = {
        "Managed Node": {
            "target": 5,
            "queries": ["foo"],
        }
    }
    config_path = _write_config(tmp_path, targets)
    args = _prepare_args(tmp_path, config_path, quickstart=False, target_count=None)
    validate_calls: list[dict] = []
    _patch_runtime(monkeypatch, tmp_path, args=args, validate_calls=validate_calls)

    collect_theories.main()

    assert validate_calls, "validate_targets should run for managed ontologies"
    assert tmp_path.joinpath("data", "cache", "ontologies").exists() is False


def test_verify_bootstrap_flag_enables_override(monkeypatch, tmp_path):
    targets = {"Managed Node": {"target": 5, "queries": ["foo"]}}
    config_path = _write_config(tmp_path, targets)
    args = _prepare_args(tmp_path, config_path, quickstart=False, target_count=None)
    args.verify_bootstrap = True
    args.bootstrap_verify_dir = tmp_path / "verify_out"

    captured_override: Dict[str, Any] = {}

    def bootstrap_hook(*_a, **kwargs):
        override = kwargs.get("verification_override")
        if isinstance(override, Mapping):
            captured_override.update(override)
        return {}, {}, {}

    validate_calls: list[dict] = []
    _patch_runtime(
        monkeypatch,
        tmp_path,
        args=args,
        validate_calls=validate_calls,
        bootstrap_hook=bootstrap_hook,
    )

    collect_theories.main()

    assert captured_override["enabled"] is True
    assert captured_override["output_dir"] == args.bootstrap_verify_dir


def test_run_bootstrap_phase_emits_verification_files(tmp_path):
    retriever = SimpleNamespace(
        collect_queries=lambda *_a, **_k: SimpleNamespace(
            papers=[
                PaperMetadata(
                    identifier="rev-verify",
                    title="Verification review",
                    authors=("Author",),
                    abstract="Discusses Activity Theory. Subtheories: Engagement; Participation.",
                    source="openalex",
                    year=2020,
                    doi=None,
                    full_text="",
                    citation_count=50,
                    is_review=True,
                )
            ],
            newly_added=1,
            summary={},
        )
    )

    corpus_cfg: Mapping[str, Any] = {
        "bootstrap": {
            "enabled": True,
            "queries": ["aging theory"],
            "min_citations": 0,
            "max_theories": 2,
        }
    }

    verification_dir = tmp_path / "verify"
    config, nodes, review_map = collect_theories._run_bootstrap_phase(  # type: ignore[attr-defined]
        retriever,
        llm_client=None,
        corpus_cfg=corpus_cfg,
        context={},
        verification_override={"output_dir": verification_dir},
    )

    verification_cfg = config.get("verification", {})
    assert verification_cfg.get("enabled") is True
    assert Path(verification_cfg["output_dir"]) == verification_dir
    assert Path(verification_cfg["results_path"]).exists()
    assert Path(verification_cfg["report_path"]).exists()
    summary = verification_cfg.get("summary", {})
    assert summary.get("total_reviews") == 1
    assert summary.get("total_nodes", 0) >= 1

    assert nodes, "Bootstrap nodes should not be empty"
    assert any(review_map.values()), "Review map should contain retrieved papers"


def test_quickstart_generates_cache_and_skips_validation(monkeypatch, tmp_path, capsys):
    config_path = _write_config(tmp_path, {})
    args = _prepare_args(tmp_path, config_path, quickstart=True, target_count=12)
    validate_calls: list[dict] = []
    _patch_runtime(monkeypatch, tmp_path, args=args, validate_calls=validate_calls)

    collect_theories.main()

    assert not validate_calls, "validate_targets should be skipped in quickstart mode"

    slug_path = tmp_path / "data" / "cache" / "ontologies" / "test-query.json"
    assert slug_path.exists(), "Quickstart cache was not created"
    payload = json.loads(slug_path.read_text(encoding="utf-8"))
    assert payload["target"] == 12
    assert payload["name"] == "Test Query"

    captured = capsys.readouterr().out
    assert "Quickstart ontology node cached" in captured


def test_quickstart_bootstrap_enrichment_updates_cache(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, {})
    args = _prepare_args(tmp_path, config_path, quickstart=True, target_count=20)

    bootstrap_nodes: Dict[str, Any] = {
        "Activity Theory": {
            "bootstrap": {
                "citations": 150,
                "reviews": ["rev-1"],
                "queries": ["activity theory aging"],
            },
            "subtheories": {
                "Engagement": {
                    "bootstrap": {
                        "citations": 40,
                        "reviews": ["rev-1"],
                        "queries": ["engagement aging"],
                    },
                    "subtheories": {},
                }
            },
        }
    }
    review_doc = ReviewDocument(
        query="Test Query",
        paper=PaperMetadata(
            identifier="rev-1",
            title="Activity theory review",
            authors=("Author",),
            abstract="A review of activity theory",
            source="openalex",
            year=2021,
            doi=None,
            full_text="",
            citation_count=150,
            is_review=True,
        ),
        citations=150,
    )
    bootstrap_reviews = {"Test Query": [review_doc]}

    collect_calls: List[Dict[str, Any]] = []

    def capture_collect(*_a, **kwargs):
        collect_calls.append(
            {
                "name": kwargs.get("name"),
                "context": kwargs.get("context"),
                "config": kwargs.get("config"),
            }
        )
        return {"providers": {}}, []

    ontology_instances: List[RecordingOntologyManager] = []

    def make_ontology(base_config, **kwargs):
        manager = RecordingOntologyManager(base_config, **kwargs)
        ontology_instances.append(manager)
        return manager

    validate_calls: list[dict] = []
    _patch_runtime(
        monkeypatch,
        tmp_path,
        args=args,
        validate_calls=validate_calls,
        bootstrap_hook=lambda *_a, **_k: ({}, bootstrap_nodes, bootstrap_reviews),
        collect_hook=capture_collect,
        ontology_factory=make_ontology,
    )

    collect_theories.main()

    assert not validate_calls, "Quickstart should continue skipping validation"
    assert collect_calls, "collect_for_entry should be invoked"
    assert len(collect_calls) == 1
    call = collect_calls[0]
    assert call["name"] == "Test Query"
    enrichment = call["context"].get("enrichment")
    assert enrichment, "Quickstart context should include enrichment payload"
    theories = enrichment.get("new_theories", [])
    assert any(entry["name"] == "Activity Theory" and entry["parent"] == "Test Query" for entry in theories)
    assert any(entry["name"] == "Engagement" and entry["parent"] == "Activity Theory" for entry in theories)
    shards = enrichment.get("query_shards", [])
    shard_queries = {item["query"] for item in shards}
    assert {"activity theory aging", "engagement aging"} <= shard_queries

    config_subs = call["config"].get("subtheories", {})
    assert "Activity Theory" in config_subs
    assert "Engagement" in config_subs["Activity Theory"].get("subtheories", {})

    assert ontology_instances, "OntologyManager should be constructed"
    base_config = ontology_instances[0].base_config
    assert "Activity Theory" in base_config

    slug_path = tmp_path / "data" / "cache" / "ontologies" / "test-query.json"
    payload = json.loads(slug_path.read_text(encoding="utf-8"))
    assert "Activity Theory" in payload["subtheories"]
    assert payload["subtheories"]["Activity Theory"]["bootstrap"]["citations"] == 150


def test_quickstart_reuses_cached_snapshot(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, {})
    args = _prepare_args(tmp_path, config_path, quickstart=True, target_count=15)

    cache_dir = tmp_path / "data" / "cache" / "ontologies"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_payload = {
        "name": "Test Query",
        "target": 9,
        "queries": ["Test Query"],
        "metadata": {"source": "quickstart"},
        "subtheories": {
            "Activity Theory": {
                "target": 5,
                "subtheories": {
                    "Engagement": {"target": 2, "subtheories": {}},
                },
            }
        },
    }
    cache_path = cache_dir / "test-query.json"
    cache_path.write_text(json.dumps(cached_payload), encoding="utf-8")

    collect_calls: List[Dict[str, Any]] = []

    def capture_collect(*_a, **kwargs):
        collect_calls.append(
            {
                "name": kwargs.get("name"),
                "config": kwargs.get("config"),
            }
        )
        return {"providers": {}}, []

    validate_calls: list[dict] = []
    _patch_runtime(
        monkeypatch,
        tmp_path,
        args=args,
        validate_calls=validate_calls,
        collect_hook=capture_collect,
    )

    collect_theories.main()

    assert collect_calls, "collect_for_entry should run"
    config_payload = collect_calls[0]["config"]
    subs = config_payload.get("subtheories", {})
    assert "Activity Theory" in subs, "Cached subtheories should be reused"
    engagement = subs["Activity Theory"].get("subtheories", {})
    assert "Engagement" in engagement, "Nested cached nodes should persist"
    slug_path = tmp_path / "data" / "cache" / "ontologies" / "test-query.json"
    persisted = json.loads(slug_path.read_text(encoding="utf-8"))
    assert persisted["target"] == 15, "Target should be updated from CLI flag"


def test_subtheory_quota_marks_waiting(monkeypatch):
    original_collect_for_entry = collect_theories.collect_for_entry

    class DummyDecision:
        def __init__(self, identifier: str):
            self.identifier = identifier
            self.accepted = True

        def to_record(self, *, threshold: float) -> Dict[str, Any]:
            return {"accepted": True, "score": 1.0, "threshold": threshold}

    class DummyRelevanceFilter:
        def __init__(self, *args, threshold: float = 0.0, **kwargs):
            self.threshold = float(threshold)

        def apply(self, papers, *, context, existing_decisions):
            decisions = [DummyDecision(p.identifier) for p in papers]
            return list(papers), decisions

    class StubStateStore:
        def __init__(self):
            self.storage: Dict[str, Any] = {}

        def get(self, key):
            return self.storage.get(key, {})

        def set(self, key, value):
            self.storage[key] = value

    class StubRetriever:
        def __init__(self):
            self.state_store = StubStateStore()

        def collect_queries(
            self,
            *_args,
            **_kwargs,
        ):
            paper = SimpleNamespace(identifier="root-paper")
            summary = {"total_unique": 1, "providers": {"stub": 1}}
            return SimpleNamespace(papers=[paper], summary=summary)

    monkeypatch.setattr(collect_theories, "RelevanceFilter", DummyRelevanceFilter)

    existing_counts = {
        f"root::sub::{collect_theories.slugify(name)}": count
        for name, count in {"Alpha": 3, "Beta": 8, "Gamma": 1}.items()
    }
    monkeypatch.setattr(
        collect_theories,
        "_existing_total",
        lambda _retriever, prefix: existing_counts.get(prefix, 0),
    )

    called_subs: List[str] = []

    def stub_collect_for_entry(*args, **kwargs):
        if kwargs.get("context", {}).get("subtheory"):
            called_subs.append(kwargs["name"])
            return {"total_unique": 0}, []
        return original_collect_for_entry(*args, **kwargs)

    monkeypatch.setattr(collect_theories, "collect_for_entry", stub_collect_for_entry)

    config = {
        "queries": ["root"],
        "target": 90,
        "max_subtheories": 2,
        "subtheories": {
            "Alpha": {"queries": ["alpha"], "target": 10},
            "Beta": {"queries": ["beta"], "target": 10},
            "Gamma": {"queries": ["gamma"], "target": 10},
        },
    }

    retriever = StubRetriever()

    summary, papers = original_collect_for_entry(
        retriever,
        name="Root",
        config=config,
        context={"base_query": "root"},
        providers=None,
        resume=True,
        state_prefix="root",
        ontology_manager=None,
        expander=None,
        default_expansion=None,
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=None,
    )

    assert papers, "Top-level retrieval should return accepted papers"
    assert len(called_subs) == 2
    assert set(called_subs) == {"Alpha", "Gamma"}

    sub_summaries = summary.get("subtheories", {})
    assert summary.get("subtheory_quota") == 2
    assert summary.get("subtheory_queue") == ["Beta"]

    assert sub_summaries["Beta"]["status"] == "waiting"
    assert sub_summaries["Beta"]["queue_position"] == 1
    assert sub_summaries["Alpha"].get("status") == "active"
    assert sub_summaries["Gamma"].get("status") == "active"


def test_query_expansion_keywords_and_bootstrap(tmp_path: Path, monkeypatch) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=storage)

    class _StateStore:
        def __init__(self) -> None:
            self.data: Dict[str, Any] = {}

        def get(self, key: str) -> Dict[str, Any]:
            return json.loads(json.dumps(self.data.get(key, {})))

        def set(self, key: str, value: Mapping[str, Any]) -> None:
            self.data[key] = json.loads(json.dumps(value))

        def write_summary(self, value: Mapping[str, Any]) -> None:
            self.data["summary"] = json.loads(json.dumps(value))

    class _Decision:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier
            self.accepted = True

        def to_record(self, *, threshold: float) -> Dict[str, Any]:
            return {"accepted": True, "score": 1.0, "threshold": threshold}

    class _AcceptAllFilter:
        def __init__(self, *args, **kwargs) -> None:
            self.threshold = 0.0

        def apply(self, papers, *, context, existing_decisions):
            del context, existing_decisions
            decisions = [_Decision(paper.identifier) for paper in papers]
            return list(papers), decisions

    class _ExpansionRetriever:
        def __init__(self) -> None:
            self.state_store = _StateStore()
            self.calls: List[List[str]] = []

        def collect_queries(
            self,
            queries: Sequence[str],
            *,
            target: int | None = None,
            providers: Sequence[str] | None = None,
            state_key: str | None,
            resume: bool,
            min_citation_count: int | None = None,
            prefer_reviews: bool = False,
            sort_by_citations: bool = False,
        ) -> Any:
            del target, providers, state_key, resume, min_citation_count, prefer_reviews, sort_by_citations
            query_list = [str(item) for item in queries]
            self.calls.append(query_list)
            papers = [
                PaperMetadata(
                    identifier="p-exp",
                    title="Frailty and digital aging biomarkers",
                    authors=("Author",),
                    abstract="Explores digital aging biomarkers and resilience.",
                    source="Test",
                )
            ]
            summary = {
                "total_unique": len(papers),
                "providers": {},
                "queries": query_list,
            }
            return SimpleNamespace(papers=papers, summary=summary)

    class _StubExpander:
        def __init__(self) -> None:
            self.settings = QueryExpansionSettings(
                enabled=True,
                max_new_queries=2,
                bootstrap_new_theories=True,
                bootstrap_candidate_limit=2,
                bootstrap_mode="child",
                bootstrap_max_labels=1,
            )
            self.performance: List[Any] = []

        def settings_for(self, override: Mapping[str, Any]) -> QueryExpansionSettings:
            del override
            return self.settings

        def expand(self, node, *, base_queries, papers, settings, context):
            del node, base_queries, papers, settings, context
            return QueryExpansionSession(
                session_id="expansion",
                node_name="Activity Theory",
                base_queries=["activity theory"],
                candidates=[
                    QueryCandidate(query='"Digital aging" biomarkers', source="gpt"),
                    QueryCandidate(query="Frailty OR Resilience", source="embedding"),
                ],
            )

        def record_performance(self, session, **kwargs):
            self.performance.append((session, kwargs))

    class _RecordingBootstrapper:
        def __init__(self) -> None:
            self.requests: List[Any] = []

        def propose_labels(self, request) -> Any:
            self.requests.append(request)
            spec = RuntimeNodeSpec(
                name="Adaptive Candidate",
                parent=None,
                config={},
                keywords=("resilience",),
                metadata={},
                provenance={},
            )
            return SimpleNamespace(proposals=[spec])

    monkeypatch.setattr(collect_theories, "RelevanceFilter", _AcceptAllFilter)

    retriever = _ExpansionRetriever()
    expander = _StubExpander()
    bootstrapper = _RecordingBootstrapper()

    summary, papers = collect_theories.collect_for_entry(
        retriever,
        name="Activity Theory",
        config={
            "queries": ["activity theory"],
            "target": 2,
            "expansion": {
                "enabled": True,
                "bootstrap_new_theories": True,
                "bootstrap_candidate_limit": 2,
                "bootstrap_max_labels": 1,
            },
        },
        context={"base_query": "activity theory"},
        providers=None,
        resume=False,
        state_prefix="activity::expansion",
        ontology_manager=manager,
        expander=expander,
        default_expansion=None,
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=bootstrapper,
    )

    assert papers and papers[0].identifier == "p-exp"
    expansion_summary = summary.get("expansion")
    assert expansion_summary, "Expansion summary should be recorded"
    assert expansion_summary["queries"] == ['"Digital aging" biomarkers', "Frailty OR Resilience"]
    assert expansion_summary["keywords"] == [
        "digital aging",
        "biomarkers",
        "frailty",
        "resilience",
    ]
    assert expansion_summary["bootstrap_candidates"] == ["digital aging", "biomarkers"]
    assert expansion_summary["bootstrap_generated"] == ["Adaptive Candidate"]

    assert bootstrapper.requests, "Bootstrapper should be invoked"
    request = bootstrapper.requests[0]
    assert tuple(request.keywords) == ("digital aging", "biomarkers")

    payload = json.loads(storage.read_text(encoding="utf-8"))
    stored_entry = next(entry for entry in payload["nodes"] if entry["name"] == "Activity Theory")
    assert set(stored_entry.get("keywords", [])) >= {"digital aging", "biomarkers"}
    assert manager.has_node("Adaptive Candidate")

    enrichment = summary.get("enrichment", {})
    assert enrichment.get("expansion_keywords") == [
        "digital aging",
        "biomarkers",
        "frailty",
        "resilience",
    ]
    assert enrichment.get("expansion_candidates") == ["digital aging", "biomarkers"]
    assert enrichment.get("bootstrap_labels", {}).get("expansion::child") == ["Adaptive Candidate"]

    state_payload = retriever.state_store.data.get("activity::expansion")
    assert state_payload and "enrichment" in state_payload
    expansion_state = state_payload["enrichment"].get("expansion")
    assert expansion_state
    assert expansion_state["keywords"] == [
        "digital aging",
        "biomarkers",
        "frailty",
        "resilience",
    ]
    assert expansion_state["bootstrap_candidates"] == ["digital aging", "biomarkers"]
    assert expansion_state["bootstrap_generated"] == ["Adaptive Candidate"]


def test_collect_for_entry_processes_runtime_queue(tmp_path: Path, monkeypatch) -> None:
    storage = tmp_path / "runtime.json"
    manager = OntologyManager(
        {"Root Theory": {"queries": ["root aging"], "target": 3}},
        storage_path=storage,
    )

    class _StateStore:
        def __init__(self) -> None:
            self.data: Dict[str, Any] = {}

        def get(self, _key: str) -> Dict[str, Any]:
            return {}

        def set(self, key: str, value: Mapping[str, Any]) -> None:
            self.data[key] = dict(value)

        def write_summary(self, value: Mapping[str, Any]) -> None:
            self.data["summary"] = dict(value)

    class _Decision:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier
            self.accepted = True

        def to_record(self, *, threshold: float) -> Dict[str, Any]:
            return {"accepted": True, "score": 1.0, "threshold": threshold}

    class _AcceptAllFilter:
        def __init__(self, *args, **kwargs) -> None:
            self.threshold = 0.0

        def apply(self, papers, *, context, existing_decisions):
            del context, existing_decisions
            decisions = [_Decision(paper.identifier) for paper in papers]
            return list(papers), decisions

    class _StubRetriever:
        def __init__(self) -> None:
            self.state_store = _StateStore()
            self.calls: List[Tuple[str | None, Sequence[str]]] = []

        def collect_queries(
            self,
            queries: Sequence[str],
            *,
            target: int | None = None,
            providers: Sequence[str] | None = None,
            state_key: str | None,
            resume: bool,
            min_citation_count: int | None = None,
            prefer_reviews: bool = False,
            sort_by_citations: bool = False,
        ) -> Any:
            del target, providers, resume, min_citation_count, prefer_reviews, sort_by_citations
            query_list = [str(item) for item in queries]
            self.calls.append((state_key, query_list))
            if state_key and "::runtime::" in state_key:
                papers = [
                    PaperMetadata(
                        identifier="child-2",
                        title="Child runtime paper",
                        authors=("Author",),
                        abstract="",
                        source="Test",
                    )
                ]
            elif "child followup query" in query_list:
                papers = [
                    PaperMetadata(
                        identifier="child-1",
                        title="Child theory paper",
                        authors=("Author",),
                        abstract="",
                        source="Test",
                    )
                ]
            else:
                papers = [
                    PaperMetadata(
                        identifier="root-1",
                        title="Root theory paper",
                        authors=("Author",),
                        abstract="",
                        source="Test",
                    )
                ]
            summary = {
                "total_unique": len(papers),
                "providers": {"stub": len(papers)},
                "queries": query_list,
            }
            return SimpleNamespace(papers=papers, summary=summary)

    class _StubSession:
        def selected_queries(self) -> Sequence[str]:
            return ["child followup query"]

    class _StubExpander:
        def __init__(self) -> None:
            self.settings = QueryExpansionSettings(enabled=True, use_gpt=False, use_embeddings=False)

        def settings_for(self, _override: Mapping[str, Any] | None) -> QueryExpansionSettings:
            return self.settings

        def expand(self, *args, **kwargs) -> _StubSession:
            return _StubSession()

        def record_performance(self, *args, **kwargs) -> None:
            return None

    def _stub_promote(**kwargs) -> List[str]:
        ontology_manager = kwargs["ontology_manager"]
        parent_name = kwargs["node_name"]
        spec = RuntimeNodeSpec(
            name="Child Theory",
            parent=parent_name,
            config={"queries": ["child followup query"], "target": 1},
            keywords=("child",),
            metadata={},
            provenance={"source": "test"},
        )
        ontology_manager.append_child(parent_name, spec)
        return [spec.name]

    monkeypatch.setattr(collect_theories, "RelevanceFilter", _AcceptAllFilter)
    monkeypatch.setattr(collect_theories, "_promote_successful_queries", _stub_promote)

    retriever = _StubRetriever()
    expander = _StubExpander()

    summary, papers = collect_theories.collect_for_entry(
        retriever,
        name="Root Theory",
        config={"queries": ["root aging"], "target": 3},
        context={"base_query": "root aging"},
        providers=None,
        resume=False,
        state_prefix="root",
        ontology_manager=manager,
        expander=expander,
        default_expansion=QueryExpansionSettings(enabled=True, use_gpt=False, use_embeddings=False),
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=None,
    )

    sub_summaries = summary.get("subtheories", {})
    assert "Child Theory" in sub_summaries, "Runtime child should be processed"
    child_summary = sub_summaries["Child Theory"]
    assert child_summary.get("target") == 1
    assert any(state_key and "::runtime::" in state_key for state_key, _ in retriever.calls)
    identifiers = {paper.identifier for paper in papers}
    assert "child-1" in identifiers
    assert "child-2" in identifiers


def test_promote_successful_queries_emits_update(tmp_path: Path) -> None:
    cache_dir = tmp_path / "query-cache"
    settings = QueryExpansionSettings(cache_dir=cache_dir)
    expander = QueryExpander(llm_client=None, default_settings=settings)
    manager = OntologyManager({"Activity Theory": {"target": 1}}, storage_path=tmp_path / "runtime.json")
    updates: List[OntologyUpdate] = []

    def _listener(_ontology, update: OntologyUpdate) -> None:
        updates.append(update)

    manager.register_listener(_listener)

    def _paper(identifier: str, title: str) -> PaperMetadata:
        return PaperMetadata(
            identifier=identifier,
            title=title,
            authors=("Author",),
            abstract="Explores digital biomarkers and resilience in ageing.",
            source="Test",
        )

    session_one = QueryExpansionSession(
        session_id="s1",
        node_name="Activity Theory",
        base_queries=["activity theory"],
        candidates=[QueryCandidate(query="digital biomarkers", source="gpt")],
    )
    papers_one = [_paper("p1", "Digital biomarkers in social engagement")]
    expander.record_performance(
        session_one,
        before_total=0,
        after_total=2,
        new_unique=len(papers_one),
        new_papers=papers_one,
    )

    session_two = QueryExpansionSession(
        session_id="s2",
        node_name="Activity Theory",
        base_queries=["activity theory"],
        candidates=[QueryCandidate(query="digital biomarkers", source="gpt")],
    )
    papers_two = [
        _paper("p2", "Wearables and digital biomarkers for ageing"),
        _paper("p3", "Community resilience and technology"),
    ]
    expander.record_performance(
        session_two,
        before_total=2,
        after_total=5,
        new_unique=len(papers_two),
        new_papers=papers_two,
    )

    records = expander.consistent_queries(
        "Activity Theory",
        selected_queries=["digital biomarkers"],
        min_successes=2,
        min_new_unique=2,
    )
    assert records and records[0].new_unique_total >= 3

    promoted = collect_theories._promote_successful_queries(
        expander=expander,
        ontology_manager=manager,
        node_name="Activity Theory",
        selected_queries=["digital biomarkers"],
        new_papers=papers_two,
        min_successes=2,
        min_new_unique=2,
    )

    assert promoted, "Adaptive query should produce a runtime node"
    assert updates, "Ontology manager should emit an update"
    latest_update = updates[-1]
    added_names = set(latest_update.added)
    assert set(promoted) <= added_names

    promoted_name = promoted[0]
    node = manager.ontology.get(promoted_name)
    provenance = node.metadata.get("runtime_provenance", {})
    assert provenance.get("query") == "digital biomarkers"
    assert provenance.get("supporting_papers"), "Provenance should list supporting papers"

def test_collect_for_entry_records_rejection_registry(monkeypatch):
    recorded: List[tuple[tuple[str, ...], Dict[str, Any]]] = []

    class RegistryStateStore:
        def __init__(self) -> None:
            self.storage: Dict[str, Any] = {}

        def get(self, key: str) -> Dict[str, Any]:
            return copy.deepcopy(self.storage.get(key, {}))

        def set(self, key: str, value: Mapping[str, Any]) -> None:
            self.storage[key] = copy.deepcopy(dict(value))

        def latest_rejection(self, _key: str):  # pragma: no cover - unused in this test
            return None

        def record_rejection(self, keys, **payload) -> None:
            recorded.append((tuple(keys), dict(payload)))

    class RegistryRetriever:
        def __init__(self) -> None:
            self.state_store = RegistryStateStore()

        def collect_queries(self, *_args, **_kwargs):
            paper = PaperMetadata(
                identifier="https://openalex.org/W1234567890",
                title="Aging Theory",
                authors=(),
                abstract="",
                source="openalex",
                doi="10.1000/test",
            )
            summary = {"total_unique": 1, "providers": {"stub": 1}}
            return SimpleNamespace(papers=[paper], summary=summary)

    class RejectingFilter:
        def __init__(self, *args, threshold: float = 0.0, **kwargs) -> None:
            self.threshold = float(threshold)

        def apply(self, papers, *, context, existing_decisions):
            decisions = [
                collect_theories.FilterDecision(
                    identifier=paper.identifier,
                    score=0.0,
                    accepted=False,
                    rationale="irrelevant",
                    details={"reason": "test"},
                )
                for paper in papers
            ]
            return [], decisions

    monkeypatch.setattr(collect_theories, "RelevanceFilter", RejectingFilter)

    retriever = RegistryRetriever()

    summary, papers = collect_theories.collect_for_entry(
        retriever,
        name="Aging",
        config={"queries": ["aging"]},
        context={"base_query": "aging"},
        providers=None,
        resume=False,
        state_prefix="aging",
        ontology_manager=None,
        expander=None,
        default_expansion=None,
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=None,
    )

    assert not papers
    assert summary["filtering"]["rejected"] == 1
    assert recorded, "Rejected paper should be written to registry"
    keys, payload = recorded[0]
    assert "doi:10.1000/test" in keys
    assert payload["node"] == "Aging"


def test_collect_for_entry_respects_rejection_registry(monkeypatch):
    class RegistryStateStore:
        def __init__(self) -> None:
            self.storage: Dict[str, Any] = {}
            self.registry: Dict[str, Any] = {
                "doi:10.2000/skip": {
                    "decisions": [
                        {
                            "identifier": "https://openalex.org/W999",
                            "node": "SeedNode",
                            "rationale": "cached rejection",
                            "score": 0.05,
                            "threshold": 0.6,
                            "accepted": False,
                            "details": {"note": "existing"},
                        }
                    ]
                }
            }

        def get(self, key: str) -> Dict[str, Any]:
            return copy.deepcopy(self.storage.get(key, {}))

        def set(self, key: str, value: Mapping[str, Any]) -> None:
            self.storage[key] = copy.deepcopy(dict(value))

        def latest_rejection(self, key: str):
            entry = self.registry.get(key)
            if not entry:
                return None
            decisions = entry.get("decisions")
            if not decisions:
                return None
            return copy.deepcopy(decisions[-1])

        def record_rejection(self, *_args, **_kwargs) -> None:
            raise AssertionError("Registry should not record duplicate rejections")

    class RegistryRetriever:
        def __init__(self) -> None:
            self.state_store = RegistryStateStore()

        def collect_queries(self, *_args, **_kwargs):
            paper = PaperMetadata(
                identifier="https://openalex.org/W999",
                title="Existing Paper",
                authors=(),
                abstract="",
                source="openalex",
                doi="10.2000/skip",
            )
            summary = {"total_unique": 1, "providers": {"stub": 1}}
            return SimpleNamespace(papers=[paper], summary=summary)

    def fail_score(self, paper, *, context):  # pragma: no cover - defensive
        raise AssertionError("_score_paper should not be called for registered rejection")

    monkeypatch.setattr(collect_theories.RelevanceFilter, "_score_paper", fail_score)

    retriever = RegistryRetriever()

    summary, papers = collect_theories.collect_for_entry(
        retriever,
        name="SkipNode",
        config={"queries": ["skip"], "filtering": {"threshold": 0.6}},
        context={"base_query": "skip"},
        providers=None,
        resume=False,
        state_prefix="skip",
        ontology_manager=None,
        expander=None,
        default_expansion=None,
        retrieval_options=None,
        filter_llm_client=None,
        label_bootstrapper=None,
    )

    assert not papers
    assert summary["filtering"]["rejected"] == 1
    stored = retriever.state_store.storage["skip"]["filtering"]
    record = stored["https://openalex.org/W999"]
    assert record["details"]["source"] == "global_registry"
