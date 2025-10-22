import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_full_cycle
from theories_pipeline.outputs import (
    COMPETITION_PAPER_COLUMNS,
    COMPETITION_QUESTION_COLUMNS,
    COMPETITION_THEORY_COLUMNS,
    COMPETITION_THEORY_PAPER_COLUMNS,
    QUESTION_COLUMNS,
    QUESTION_CONFIDENCE_COLUMNS,
)


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

        def summarize(self, assignments: List[Any], *, include_ids: bool = False) -> Dict[str, Any]:
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

    papers_path = workdir / "papers.csv"
    theories_path = workdir / "theories.csv"
    theory_papers_path = workdir / "theory_papers.csv"
    questions_path = workdir / "questions.csv"
    for path in [papers_path, theories_path, theory_papers_path, questions_path]:
        assert path.exists()

    with questions_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "theory_id",
            "paper_url",
            "paper_name",
            "paper_year",
            *QUESTION_COLUMNS,
            *QUESTION_CONFIDENCE_COLUMNS,
        ]

    competition_dir = workdir / "competition"
    assert competition_dir.exists()
    competition_paths = {
        "papers": competition_dir / "papers.csv",
        "theories": competition_dir / "theories.csv",
        "theory_papers": competition_dir / "theory_papers.csv",
        "questions": competition_dir / "questions.csv",
    }
    for path in competition_paths.values():
        assert path.exists()

    with competition_paths["papers"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_PAPER_COLUMNS)

    with competition_paths["theories"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_COLUMNS)

    with competition_paths["theory_papers"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_THEORY_PAPER_COLUMNS)

    with competition_paths["questions"].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COMPETITION_QUESTION_COLUMNS)

    accuracy_ground_truth = [
        {
            "theory_id": "placeholder",
            "paper_id": "placeholder",
            "question_id": question,
            "expected_answer": "",
        }
        for question in QUESTION_COLUMNS
    ]
    assert len(accuracy_ground_truth) == len(QUESTION_COLUMNS)
