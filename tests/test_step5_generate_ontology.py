import importlib.util
import json
from pathlib import Path
from typing import Set


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "step5_generate_ontology.py"
SPEC = importlib.util.spec_from_file_location("step5_generate_ontology", MODULE_PATH)
assert SPEC and SPEC.loader
_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_MODULE)

_coerce_articles_from_documents = _MODULE._coerce_articles_from_documents
_load_json_payload = _MODULE._load_json_payload
_load_registry_builder = _MODULE._load_registry_builder
_limit_group_theories = _MODULE._limit_group_theories
_reconcile_groups = _MODULE._reconcile_groups
_build_llm_pass_audit = _MODULE._build_llm_pass_audit
consolidate_group_summaries = _MODULE.consolidate_group_summaries
refine_group_hierarchy = _MODULE.refine_group_hierarchy
refine_groups_with_llm = _MODULE.refine_groups_with_llm


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


def test_build_llm_pass_audit_clones_metadata() -> None:
    consolidation = {"status": "completed", "model": "gpt-5-mini"}
    refinement = {"status": "pending", "model": "gpt-4.1"}

    audit = _build_llm_pass_audit(consolidation, refinement, [])

    assert audit["consolidation"]["status"] == "completed"
    assert audit["refinement"]["model"] == "gpt-4.1"

    consolidation["status"] = "mutated"
    assert audit["consolidation"]["status"] == "completed"


def test_consolidate_group_summaries_merges_parent_groups() -> None:
    input_groups = [
        {
            "name": "Cellular Senescence",
            "description": "Loss of proliferation capacity",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
        {
            "name": "Cellular Ageing",
            "description": "Synonym of senescence",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
    ]

    def fake_call(messages, api_key, *, model, temperature):
        assert api_key == "test-key"
        assert model == _MODULE.GROUP_CONSOLIDATION_MODEL
        assert messages and messages[0]["role"] == "system"
        payload = {
            "parent_merges": [
                {
                    "parent": {"group_id": "G1"},
                    "children": [
                        {"group_id": "G2"},
                    ],
                }
            ]
        }
        metadata = {
            "id": "cmpl-merge",
            "model": model,
            "usage": {"prompt_tokens": 20, "completion_tokens": 12},
        }
        return payload, metadata

    consolidated, metadata = consolidate_group_summaries(
        input_groups,
        "test-key",
        call_model=fake_call,
    )

    assert metadata["status"] == "completed"
    assert metadata["applied_merge_count"] == 1
    assert metadata["suggested_merge_count"] == 1
    assert len(consolidated) == 1
    parent_group = consolidated[0]
    assert parent_group["name"] == "Cellular Senescence"
    assert "subgroups" in parent_group
    child_group = parent_group["subgroups"][0]
    assert child_group["name"] == "Cellular Ageing"
    assert child_group["theories"][0]["supporting_articles"] == ["A1"]


def test_refine_groups_with_llm_runs_two_passes() -> None:
    input_groups = [
        {
            "name": "Cellular Senescence",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
        {
            "name": "Cellular Ageing",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
        {
            "name": "Oxidative Stress",
            "theories": [
                {
                    "theory_id": "T2",
                    "preferred_label": "Oxidative Stress",
                    "supporting_articles": ["A2"],
                }
            ],
        },
    ]

    call_counter = {"count": 0}

    def fake_call(messages, api_key, *, model, temperature):
        call_counter["count"] += 1
        assert api_key == "test-key"
        assert model == _MODULE.GROUP_CONSOLIDATION_MODEL
        assert messages and "Focus for this pass" in messages[1]["content"]
        if call_counter["count"] == 1:
            payload = {
                "parent_merges": [
                    {
                        "parent": {"group_id": "G1"},
                        "children": [
                            {"group_id": "G2"},
                        ],
                    }
                ]
            }
        else:
            payload = {"parent_merges": []}
        metadata = {"id": f"cmpl-{call_counter['count']}", "model": model}
        return payload, metadata

    cache: Set[str] = set()
    enriched, metadata = refine_groups_with_llm(
        input_groups,
        "test-key",
        cache=cache,
        call_model=fake_call,
    )

    assert call_counter["count"] == 2
    assert metadata["status"] == "completed"
    assert len(metadata["passes"]) == 2
    assert metadata["passes"][0]["suggested_merge_count"] == 1
    assert metadata["passes"][1]["suggested_merge_count"] == 0
    assert len(enriched) == 2
    parent_group = enriched[0]
    assert parent_group["name"] == "Cellular Senescence"
    assert parent_group["subgroups"][0]["name"] == "Cellular Ageing"
    assert parent_group["subgroups"][0]["theories"][0]["supporting_articles"] == ["A1"]


def test_refine_groups_with_llm_uses_cache_to_skip_calls() -> None:
    groups = [
        {
            "name": "Cellular Senescence",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        }
    ]

    def fake_call(messages, api_key, *, model, temperature):
        raise AssertionError("call_model should not be invoked when cached")

    cache: Set[str] = set()
    # Seed the cache by running once with a call that records the key.
    def priming_call(messages, api_key, *, model, temperature):
        return {"parent_merges": []}, {"id": "seed", "model": model}

    enriched, metadata = refine_groups_with_llm(
        groups,
        "test-key",
        cache=cache,
        call_model=priming_call,
    )

    assert metadata["status"] == "completed"
    assert enriched == groups

    enriched_again, metadata_again = refine_groups_with_llm(
        groups,
        "test-key",
        cache=cache,
        call_model=fake_call,
    )

    assert metadata_again["status"] == "skipped"
    assert metadata_again.get("reason") == "cached"
    assert enriched_again == groups


def test_refinement_and_reconciliation_merge_synonymous_groups() -> None:
    input_groups = [
        {
            "name": "Cellular Senescence",
            "description": "Loss of proliferation capacity",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
        {
            "name": "Cellular Ageing",
            "description": "Synonym of senescence",
            "theories": [
                {
                    "theory_id": "T1",
                    "preferred_label": "Cellular Senescence",
                    "supporting_articles": ["A1"],
                }
            ],
        },
    ]

    def fake_call(messages, api_key, *, model, temperature):
        assert api_key == "test-key"
        assert model == _MODULE.REFINEMENT_MODEL
        assert temperature == 0.4
        assert messages and messages[0]["role"] == "system"
        payload = {
            "groups": [
                {
                    "name": "Cellular Senescence",
                    "description": "Unified senescence bucket",
                },
                {
                    "name": "Cellular Ageing",
                    "parent": "Cellular Senescence",
                    "theories": [
                        {
                            "theory_id": "T1",
                            "preferred_label": "Cellular Senescence",
                            "supporting_articles": ["A1"],
                        }
                    ],
                },
            ]
        }
        metadata = {
            "id": "cmpl-test",
            "model": model,
            "usage": {"prompt_tokens": 42, "completion_tokens": 25},
        }
        return payload, metadata

    refined_groups, metadata = refine_group_hierarchy(
        input_groups,
        "test-key",
        call_model=fake_call,
    )

    assert metadata["status"] == "completed"
    assert metadata["materialised_group_count"] == 1
    assert metadata["response_metadata"]["usage"]["prompt_tokens"] == 42

    assert len(refined_groups) == 1
    parent_group = refined_groups[0]
    assert parent_group["name"] == "Cellular Senescence"
    assert "subgroups" in parent_group
    nested = parent_group["subgroups"][0]
    assert nested["name"] == "Cellular Ageing"
    assert nested["theories"][0]["theory_id"] == "T1"

    registry = {
        "T1": {
            "label": "Cellular Senescence",
            "aliases": [],
            "supporting_articles": ["A1"],
        }
    }
    article_index = {"A1": ["T1"]}

    reconciled, reconciliation = _reconcile_groups(
        refined_groups,
        registry,
        article_index=article_index,
    )

    assert len(reconciled) == 1
    reconciled_child = reconciled[0]["subgroups"][0]
    assert reconciled_child["theories"][0]["supporting_articles"] == ["A1"]
    assert "duplicate_theories" not in reconciliation


def test_consolidation_and_refinement_pipeline() -> None:
    input_groups = [
        {
            "name": "Reactive Oxygen Species",
            "description": "ROS driven damage",
            "theories": [
                {
                    "theory_id": "T_ROS",
                    "preferred_label": "Oxidative Stress Theory",
                    "supporting_articles": ["A10", "A11"],
                }
            ],
        },
        {
            "name": "Mitochondrial Damage",
            "description": "Mitochondrial dysfunction",
            "theories": [
                {
                    "theory_id": "T_MITO",
                    "preferred_label": "Mitochondrial Free Radical Theory",
                    "supporting_articles": ["A20"],
                }
            ],
        },
    ]

    def fake_merge(messages, api_key, *, model, temperature):
        assert api_key == "test-key"
        payload = {
            "parent_merges": [
                {
                    "parent": {
                        "name": "Damage Accumulation",
                        "description": "Parent bucket for damage theories",
                    },
                    "children": [
                        {"group_id": "G1"},
                        {"group_id": "G2"},
                    ],
                }
            ]
        }
        metadata = {"id": "cmpl-merge", "model": model}
        return payload, metadata

    consolidated, merge_metadata = consolidate_group_summaries(
        input_groups,
        "test-key",
        call_model=fake_merge,
    )

    assert merge_metadata["created_parent_groups"] == ["Damage Accumulation"]
    assert len(consolidated) == 1
    consolidated_parent = consolidated[0]
    assert consolidated_parent["name"] == "Damage Accumulation"
    assert {child["name"] for child in consolidated_parent["subgroups"]} == {
        "Reactive Oxygen Species",
        "Mitochondrial Damage",
    }

    captured_prompt: dict = {}

    def fake_refine(messages, api_key, *, model, temperature):
        assert api_key == "test-key"
        captured_prompt["content"] = messages[1]["content"]
        payload = {"groups": json.loads(json.dumps(consolidated))}
        metadata = {"id": "cmpl-refine", "model": model}
        return payload, metadata

    refined, refinement_metadata = refine_group_hierarchy(
        consolidated,
        "test-key",
        call_model=fake_refine,
    )

    assert refinement_metadata["status"] == "completed"
    assert "Damage Accumulation" in captured_prompt["content"]
    assert refined == consolidated

    audit = _build_llm_pass_audit(merge_metadata, refinement_metadata, [])
    assert audit["consolidation"]["applied_merge_count"] == 1
    assert audit["refinement"]["status"] == "completed"
