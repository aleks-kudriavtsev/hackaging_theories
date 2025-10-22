import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "step5_generate_ontology.py"
SPEC = importlib.util.spec_from_file_location("step5_generate_ontology", MODULE_PATH)
assert SPEC and SPEC.loader
_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)

_coerce_articles_from_documents = _MODULE._coerce_articles_from_documents
_load_json_payload = _MODULE._load_json_payload
_load_registry_builder = _MODULE._load_registry_builder
_limit_group_theories = _MODULE._limit_group_theories


def test_load_json_payload_returns_documents_when_registry_missing(tmp_path: Path) -> None:
    payloads = [
        {"index": 0, "record": {"id": "a1", "theory_extraction": {"theories": ["A"]}}},
        {"index": 1, "record": {"id": "a2", "theory_extraction": {"theories": ["B"]}}},
    ]
    target = tmp_path / "checkpoint.json"
    with target.open("w", encoding="utf-8") as fh:
        for entry in payloads:
            fh.write(json.dumps(entry))
            fh.write("\n")

    loaded = _load_json_payload(str(target))
    assert isinstance(loaded, list)
    assert loaded == payloads


def test_coerce_articles_from_documents_prefers_index_sorting() -> None:
    documents = [
        {"index": 3, "record": {"id": "c", "theory_extraction": {"theories": ["C"]}}},
        {"index": 1, "record": {"id": "a", "theory_extraction": {"theories": ["A"]}}},
        {"index": 2, "record": {"id": "b", "theory_extraction": {"theories": ["B"]}}},
    ]

    articles = _coerce_articles_from_documents(documents)
    assert [article["id"] for article in articles] == ["a", "b", "c"]


def test_coerce_articles_handles_embedded_article_arrays() -> None:
    documents = [
        {
            "articles": [
                {"id": "alpha", "theory_extraction": {"theories": ["Alpha"]}},
                {"id": "beta", "theory_extraction": {"theories": ["Beta"]}},
            ]
        }
    ]

    articles = _coerce_articles_from_documents(documents)
    assert {article["id"] for article in articles} == {"alpha", "beta"}


def test_coerce_articles_accepts_standalone_records() -> None:
    documents = [
        {"id": "solo", "theory_extraction": {"theories": ["Solo"]}},
        {"id": "extra", "other": "value"},
    ]

    articles = _coerce_articles_from_documents(documents)
    assert [article["id"] for article in articles] == ["solo"]


def test_load_registry_builder_handles_direct_execution(monkeypatch) -> None:
    module_name = "scripts.step4_extract_theories"

    def explode(name: str):
        raise ImportError("forced failure")

    monkeypatch.delitem(_MODULE.sys.modules, module_name, raising=False)
    monkeypatch.setattr(_MODULE.importlib, "import_module", explode)

    builder = _load_registry_builder()
    try:
        assert callable(builder)
        assert builder.__module__ == module_name
    finally:
        _MODULE.sys.modules.pop(module_name, None)


def test_limit_group_theories_splits_large_groups() -> None:
    theories = [
        {"theory_id": f"T{index}", "preferred_label": f"Theory {index}"}
        for index in range(45)
    ]
    groups = [{"name": "Cellular", "theories": theories}]
    reconciliation: dict = {}

    processed = _limit_group_theories(groups, 40, reconciliation=reconciliation)

    assert len(processed) == 1
    main_group = processed[0]
    assert len(main_group["theories"]) == 40
    assert "subgroups" in main_group
    overflow_group = main_group["subgroups"][0]
    assert overflow_group["name"].startswith("Cellular (auto-split")
    assert [theory["theory_id"] for theory in overflow_group["theories"]] == [
        f"T{index}" for index in range(40, 45)
    ]

    adjustments = reconciliation.get("group_splits")
    assert adjustments and adjustments[0]["limit"] == 40
    assert adjustments[0]["overflow_groups"][0] == [f"T{index}" for index in range(40, 45)]
