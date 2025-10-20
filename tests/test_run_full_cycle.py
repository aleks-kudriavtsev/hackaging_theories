import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_full_cycle


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "pipeline.json"
    payload = {
        "data_sources": {
            "seed_papers": str((PROJECT_ROOT / "data/examples/seed_papers.json").resolve()),
        },
        "providers": [
            {
                "name": "pubmed",
                "type": "pubmed",
                "enabled": False,
            }
        ],
        "outputs": {},
        "corpus": {},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _write_ontology(workdir: Path) -> None:
    payload = {
        "ontology": {
            "final": {
                "groups": [
                    {
                        "name": "Cellular Mechanisms",
                        "suggested_queries": ["cellular aging mechanisms"],
                        "theories": [
                            {
                                "preferred_label": "Senescence Cascade",
                                "suggested_queries": ["senescence cascade aging"],
                                "representative_titles": ["Title A"],
                            },
                            {
                                "preferred_label": "Telomere Attrition",
                            },
                        ],
                    }
                ]
            }
        }
    }
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "aging_ontology.json").write_text(json.dumps(payload), encoding="utf-8")


def test_run_full_cycle_invokes_pipeline_and_collector(tmp_path: Path, tmp_config: Path, monkeypatch) -> None:
    workdir = tmp_path / "cycle"

    pipeline_calls: List[List[str]] = []

    def fake_run_pipeline_main(argv: List[str] | None) -> int:
        pipeline_calls.append(list(argv or []))
        target_dir = Path(argv[argv.index("--workdir") + 1]) if argv else workdir
        _write_ontology(Path(target_dir))
        return 0

    monkeypatch.setattr(run_full_cycle.run_pipeline, "main", fake_run_pipeline_main)

    collector_calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []

    def fake_collect_for_entry(*args, **kwargs):
        collector_calls.append((args, kwargs))
        return {"total_unique": 0}, []

    monkeypatch.setattr(run_full_cycle.collect_theories, "collect_for_entry", fake_collect_for_entry)
    monkeypatch.setattr(run_full_cycle.collect_theories, "_load_api_keys", lambda *a, **k: {})
    monkeypatch.setattr(run_full_cycle.collect_theories, "_maybe_build_llm_client", lambda *a, **k: None)

    class DummyClassifier:
        def attach_manager(self, manager: Any) -> None:  # pragma: no cover - trivial
            self.manager = manager

        def summarize(self, assignments: List[Any]) -> Dict[str, Any]:
            return {}

        @classmethod
        def from_config(cls, *a, **k):  # pragma: no cover - simple factory
            return cls()

    monkeypatch.setattr(run_full_cycle.collect_theories, "TheoryClassifier", DummyClassifier)
    monkeypatch.setattr(run_full_cycle.collect_theories, "QuestionExtractor", lambda *a, **k: object())
    monkeypatch.setattr(
        run_full_cycle.collect_theories,
        "classify_and_extract_parallel",
        lambda papers, classifier, extractor, workers=1: ([], []),
    )
    monkeypatch.setattr(run_full_cycle.collect_theories, "export_papers", lambda *a, **k: None)
    monkeypatch.setattr(run_full_cycle.collect_theories, "export_theories", lambda *a, **k: None)
    monkeypatch.setattr(run_full_cycle.collect_theories, "export_question_answers", lambda *a, **k: None)

    args = [
        "--workdir",
        str(workdir),
        "--config",
        str(tmp_config),
        "--collector-query",
        "aging theory",
        "--limit",
        "5",
        "--no-resume",
    ]

    result = run_full_cycle.main(args)
    assert result == 0

    assert pipeline_calls, "run_pipeline.main should be invoked"
    assert any("--workdir" in call for call in pipeline_calls)
    assert collector_calls, "collect_for_entry should be executed"

    state_dir = workdir / "collector_state"
    assert state_dir.exists(), "Collector state directory should default under the workdir"
