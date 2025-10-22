from __future__ import annotations

import json
from pathlib import Path

from theories_pipeline.llm import LLMResponse
from theories_pipeline.ontology_optimization import optimise_ontology_payload, optimise_file


class DummyLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, messages_batch, *, model=None, temperature=None):  # noqa: D401
        self.calls += 1
        batch = []
        for _ in messages_batch:
            if not self.responses:
                batch.append(LLMResponse(content="{}", cached=False))
                continue
            batch.append(LLMResponse(content=self.responses.pop(0), cached=False))
        return batch


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

    llm = DummyLLM(
        [
            json.dumps(
                {
                    "overall_rationale": "Evidence clusters by mechanism",
                    "theories": [
                        {
                            "label": "Cellular Damage Core",
                            "rationale": "Articles A1-A4 describe oxidative damage",
                            "article_ids": ["A1", "A2", "A3", "A4"],
                        },
                        {
                            "label": "Cellular Damage Variant",
                            "rationale": "Articles A5-A8 focus on mitochondrial damage",
                            "article_ids": ["A5", "A6", "A7", "A8"],
                        },
                    ],
                }
            )
        ]
    )

    summary = optimise_ontology_payload(payload, llm_client=llm, batch_size=2)

    assert summary.changed is True
    assert summary.iterations == 1
    theories = group["theories"]
    assert len(theories) == 2
    labels = {entry["label"] for entry in theories}
    assert "Cellular Damage Core" in labels
    assert "Cellular Damage Variant" in labels
    for entry in theories:
        assert 1 <= len(entry.get("supporting_articles", [])) <= 4
        metadata = entry.get("metadata", {})
        assert metadata.get("optimization_rationale")
        assert metadata.get("optimization_pipeline")
    assert summary.created_theories
    assert summary.updated_theories


def test_split_without_llm_uses_fallback() -> None:
    payload = _base_payload()
    group = payload["ontology"]["final"]["groups"][0]
    group["theories"].append(
        {
            "label": "Inflammatory",
            "supporting_articles": [f"I{index}" for index in range(5)],
        }
    )

    summary = optimise_ontology_payload(payload, llm_client=None, maximum=3)

    assert summary.changed is True
    theories = group["theories"]
    counts = [len(entry.get("supporting_articles", [])) for entry in theories]
    assert all(count <= 3 for count in counts)
    assert sum(counts) == 5


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

