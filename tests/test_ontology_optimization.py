from __future__ import annotations

import json
from pathlib import Path

from theories_pipeline.ontology_optimization import optimise_ontology_payload, optimise_file


def _base_payload() -> dict:
    return {
        "ontology": {
            "final": {
                "groups": [
                    {
                        "name": "Root",
                        "theories": [],
                        "subgroups": [],
                    }
                ]
            }
        }
    }


def test_split_large_theory_creates_additional_nodes() -> None:
    payload = _base_payload()
    group = payload["ontology"]["final"]["groups"][0]
    group["theories"].append(
        {
            "label": "Cellular Damage",
            "supporting_articles": [f"A{index}" for index in range(1, 9)],
        }
    )

    summary = optimise_ontology_payload(payload)

    assert summary.changed is True
    assert summary.iterations >= 1
    theories = group["theories"]
    counts = [len(entry.get("supporting_articles", [])) for entry in theories]
    assert all(1 <= count <= 4 for count in counts)
    assert sum(counts) == 8
    assert summary.created_theories


def test_rebalance_articles_moves_shared_entries() -> None:
    payload = _base_payload()
    group = payload["ontology"]["final"]["groups"][0]
    group["theories"] = [
        {
            "label": "Metabolic",
            "supporting_articles": ["shared", "M1", "M2", "M3"],
        },
        {
            "label": "Genetic",
            "supporting_articles": ["shared"],
        },
    ]

    summary = optimise_ontology_payload(payload)

    assert summary.changed is True
    metabolic, genetic = group["theories"]
    assert "shared" not in metabolic["supporting_articles"]
    assert "shared" in genetic["supporting_articles"]
    assert summary.article_reassignments[0]["article"] == "shared"
    assert summary.article_reassignments[0]["to"] == genetic.get("theory_id")


def test_optimise_file_writes_payload(tmp_path: Path) -> None:
    payload = _base_payload()
    group = payload["ontology"]["final"]["groups"][0]
    group["theories"].append(
        {
            "label": "Inflammatory",
            "supporting_articles": [f"I{index}" for index in range(5)],
        }
    )
    source = tmp_path / "input.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    summary = optimise_file(str(source))

    assert summary.changed is True
    updated = json.loads(source.read_text(encoding="utf-8"))
    theories = updated["ontology"]["final"]["groups"][0]["theories"]
    assert any(entry.get("label", "").startswith("Inflammatory") for entry in theories)
    assert all(1 <= len(entry.get("supporting_articles", [])) <= 4 for entry in theories)

