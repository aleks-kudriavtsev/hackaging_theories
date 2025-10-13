from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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


def _patch_runtime(monkeypatch, tmp_path: Path, *, args, validate_calls: list[dict]):
    monkeypatch.setattr(collect_theories.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(collect_theories, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collect_theories, "resolve_api_keys", lambda *_a, **_k: {})
    monkeypatch.setattr(collect_theories, "build_provider_configs", lambda *_a, **_k: [])
    monkeypatch.setattr(collect_theories, "LiteratureRetriever", DummyRetriever)
    monkeypatch.setattr(collect_theories, "_run_bootstrap_phase", lambda *_a, **_k: ({}, {}, {}))
    monkeypatch.setattr(collect_theories, "_existing_total", lambda *_a, **_k: 0)

    def fake_collect_for_entry(*_a, **_k):
        return {"providers": {"stub": 1}}, [SimpleNamespace(identifier="paper-1")]

    monkeypatch.setattr(collect_theories, "collect_for_entry", fake_collect_for_entry)
    monkeypatch.setattr(collect_theories, "OntologyManager", DummyOntologyManager)
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
