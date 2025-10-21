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
