from __future__ import annotations

import json
from pathlib import Path

from theories_pipeline.ontology_suggestions import (
    load_ontology_query_suggestions,
    merge_query_suggestions,
)


def test_load_ontology_query_suggestions(tmp_path: Path) -> None:
    payload = {
        "ontology": {
            "final": {
                "groups": [
                    {
                        "name": "Group Alpha",
                        "suggested_queries": ["Group Alpha aging"],
                        "subgroups": [
                            {
                                "label": "Nested Cluster",
                                "suggested_queries": ["Nested Cluster theory"],
                            }
                        ],
                        "theories": [
                            {
                                "preferred_label": "First Theory",
                                "suggested_queries": ["First Theory query"],
                            },
                            {
                                "preferred_label": "Fallback Theory",
                            },
                        ],
                    },
                    {
                        "label": "Group Beta",
                        "theories": [],
                    },
                ]
            }
        }
    }
    ontology_path = tmp_path / "aging_ontology.json"
    ontology_path.write_text(json.dumps(payload), encoding="utf-8")

    suggestions = load_ontology_query_suggestions(ontology_path)

    assert "Group Alpha" in suggestions
    alpha = suggestions["Group Alpha"]
    assert alpha["suggested_queries"] == ["Group Alpha aging"]

    nested = alpha["subtheories"]["Nested Cluster"]
    assert nested["suggested_queries"] == ["Nested Cluster theory"]

    first_theory = alpha["subtheories"]["First Theory"]
    assert first_theory["suggested_queries"] == ["First Theory query"]

    fallback_theory = alpha["subtheories"]["Fallback Theory"]
    assert "Fallback Theory" in fallback_theory["suggested_queries"]

    beta = suggestions["Group Beta"]
    assert beta["suggested_queries"] == ["Group Beta"]


def test_merge_query_suggestions_updates_targets() -> None:
    targets = {
        "Group Alpha": {
            "queries": ["alpha base"],
            "suggested_queries": ["existing alpha"],
            "subtheories": {
                "First Theory": {"target": 5},
            },
        },
        "Group Beta": 3,
    }

    suggestions = {
        "Group Alpha": {
            "suggested_queries": ["group alpha aging"],
            "subtheories": {
                "First Theory": {"suggested_queries": ["first theory aging"]},
            },
        },
        "Group Beta": {"suggested_queries": ["group beta aging"]},
    }

    applied = merge_query_suggestions(targets, suggestions)

    assert targets["Group Alpha"]["queries"] == ["alpha base"], "Explicit queries should remain"
    assert targets["Group Alpha"]["suggested_queries"] == [
        "existing alpha",
        "group alpha aging",
    ]
    assert targets["Group Alpha"]["subtheories"]["First Theory"]["suggested_queries"] == [
        "first theory aging"
    ]

    beta_entry = targets["Group Beta"]
    assert beta_entry["target"] == 3
    assert beta_entry["suggested_queries"] == ["group beta aging"]
    assert beta_entry["queries"] == ["group beta aging"], "Suggestions should seed missing queries"

    assert applied["Group Alpha"]["suggested_queries"] == [
        "existing alpha",
        "group alpha aging",
    ]
    assert applied["Group Alpha"]["subtheories"]["First Theory"]["suggested_queries"] == [
        "first theory aging"
    ]
    assert applied["Group Beta"]["suggested_queries"] == ["group beta aging"]
    assert applied["Group Beta"]["queries"] == ["group beta aging"]
