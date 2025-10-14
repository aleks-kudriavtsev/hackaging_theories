from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Sequence

from theories_pipeline.literature import PaperMetadata
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
    def coverage(self, _counts):
        return {}

    def format_coverage_report(self, _counts):
        return "dummy coverage"


class DummyOntologyManager:
    def __init__(self, *_args, **_kwargs):
        self.ontology = DummyOntology()

    def append_node(self, *args, **kwargs):  # pragma: no cover - unused in tests
        return None


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

class DummyClassifier:
    @classmethod
    def from_config(cls, _config, *, ontology, llm_client):
        return cls()

    def attach_manager(self, _manager):
        return None

    def summarize(self, _assignments):
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
