import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pipeline


DEFAULT_ARGS = run_pipeline.parse_args([])
DEFAULT_QUERY = DEFAULT_ARGS.query
DEFAULT_ONTOLOGY_EXAMPLES = DEFAULT_ARGS.ontology_examples


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _prepare_cached_stage_files(paths: run_pipeline.PipelinePaths) -> None:
    pubmed_record = {
        "pmid": "999999",
        "title": "Cached Theory Review",
        "doi": "10.1000/cached",
        "publication_year": 2022,
        "publication_types": ["Review"],
        "authors": ["Casey Cache"],
        "sources": ["PubMed"],
    }
    openalex_record = {
        "openalex_id": "W00000",
        "title": "Cached Theory Review",
        "doi": "10.1000/cached",
        "publication_year": 2022,
        "publication_types": ["Review"],
        "authors": ["Casey Cache"],
        "provenance": "OpenAlex",
    }
    google_record = {
        "title": "Cached Theory Review",
        "publication_year": 2022,
        "authors": ["Casey Cache"],
        "sources": ["Google Scholar"],
    }

    _write_json(Path(paths.pubmed_reviews), [pubmed_record])
    _write_json(Path(paths.openalex_reviews), [openalex_record])
    _write_json(Path(paths.google_scholar_reviews), [google_record])
    _write_json(Path(paths.filtered_reviews), [])
    _write_json(Path(paths.fulltext_reviews), [])
    _write_json(
        Path(paths.ontology),
        {"ontology": {"final": {"groups": []}}},
    )


@pytest.fixture
def sample_source_payloads(tmp_path):
    paths = run_pipeline.build_paths(str(tmp_path))

    pubmed_records = [
        {
            "pmid": "12345",
            "title": "PubMed Aging Theory",
            "doi": "10.1000/foo",
            "journal": "Journal of Longevity",
            "publication_year": 2020,
            "publication_types": ["Review"],
            "authors": ["Alice Smith", "Bob Jones"],
            "sources": ["PubMed"],
            "fulltext_links": {"pmc": "PMC12345"},
            "open_access": True,
        }
    ]
    openalex_records = [
        {
            "openalex_id": "W98765",
            "title": "PubMed Aging Theory",
            "doi": "https://doi.org/10.1000/foo",
            "journal": "Journal of Longevity",
            "publication_year": 2020,
            "publication_types": ["Review", "Article"],
            "authors": ["Alice Smith", "Carol White"],
            "provenance": "OpenAlex",
            "fulltext_links": {"openalex": "https://example.org/fulltext"},
        },
        {
            "openalex_id": "W11111",
            "title": "Distinct Aging Theory",
            "doi": "10.2000/bar",
            "publication_year": 2021,
            "publication_types": ["Review"],
            "authors": ["Dave Brown"],
            "provenance": "OpenAlex",
        },
    ]

    _write_json(Path(paths.pubmed_reviews), pubmed_records)
    _write_json(Path(paths.openalex_reviews), openalex_records)

    return paths


def test_run_pipeline_executes_all_steps_and_merges(tmp_path, monkeypatch, sample_source_payloads):
    paths = sample_source_payloads
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    commands = []

    def fake_run(command, check=True):
        commands.append(command)
        if "--output" in command:
            output_path = Path(command[command.index("--output") + 1])
            if not output_path.exists():
                _write_json(output_path, [])
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    result = run_pipeline.main(["--workdir", str(tmp_path), "--force"])
    assert result == 0

    expected_scripts = [
        "scripts/step1_pubmed_search.py",
        "scripts/step1b_openalex_search.py",
        "scripts/step2_filter_reviews.py",
        "scripts/step3_fetch_fulltext.py",
        "scripts/step4_extract_theories.py",
        "scripts/step5_generate_ontology.py",
    ]
    assert [cmd[1] for cmd in commands] == expected_scripts

    step1_command = commands[0]
    assert step1_command[0] == sys.executable
    assert step1_command[step1_command.index("--query") + 1] == DEFAULT_QUERY
    assert Path(step1_command[step1_command.index("--output") + 1]) == Path(paths.pubmed_reviews)

    step1b_command = commands[1]
    assert Path(step1b_command[step1b_command.index("--output") + 1]) == Path(paths.openalex_reviews)

    step2_command = commands[2]
    assert Path(step2_command[step2_command.index("--input") + 1]) == Path(paths.start_reviews)
    assert Path(step2_command[step2_command.index("--output") + 1]) == Path(paths.filtered_reviews)

    step3_command = commands[3]
    assert Path(step3_command[step3_command.index("--input") + 1]) == Path(paths.filtered_reviews)
    assert Path(step3_command[step3_command.index("--output") + 1]) == Path(paths.fulltext_reviews)

    step4_command = commands[4]
    assert Path(step4_command[step4_command.index("--input") + 1]) == Path(paths.fulltext_reviews)
    assert Path(step4_command[step4_command.index("--output") + 1]) == Path(paths.theories)

    step5_command = commands[5]
    assert Path(step5_command[step5_command.index("--input") + 1]) == Path(paths.theories)
    assert Path(step5_command[step5_command.index("--output") + 1]) == Path(paths.ontology)
    assert (
        step5_command[step5_command.index("--max-theories-per-group") + 1]
        == str(run_pipeline.DEFAULT_ONTOLOGY_MAX_THEORIES)
    )
    assert (
        step5_command[step5_command.index("--examples-per-theory") + 1]
        == str(DEFAULT_ONTOLOGY_EXAMPLES)
    )

    start_reviews_path = Path(paths.start_reviews)
    with start_reviews_path.open("r", encoding="utf-8") as handle:
        merged_records = json.load(handle)

    assert merged_records == [
        {
            "pmid": "12345",
            "title": "PubMed Aging Theory",
            "doi": "10.1000/foo",
            "journal": "Journal of Longevity",
            "publication_year": 2020,
            "publication_types": ["Review", "Article"],
            "authors": ["Alice Smith", "Bob Jones", "Carol White"],
            "sources": ["PubMed", "OpenAlex"],
            "fulltext_links": {
                "pmc": "PMC12345",
                "openalex": "https://example.org/fulltext",
            },
            "open_access": True,
            "openalex_id": "W98765",
        },
        {
            "openalex_id": "W11111",
            "title": "Distinct Aging Theory",
            "doi": "10.2000/bar",
            "publication_year": 2021,
            "publication_types": ["Review"],
            "authors": ["Dave Brown"],
            "sources": ["OpenAlex"],
            "provenance": "OpenAlex",
        },
    ]


def test_run_pipeline_surfaces_step_failures(tmp_path, monkeypatch, sample_source_payloads):
    paths = sample_source_payloads
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    commands = []

    def fake_run(command, check=True):
        commands.append(command)
        if "--output" in command:
            output_path = Path(command[command.index("--output") + 1])
            if not output_path.exists():
                _write_json(output_path, [])
        if "scripts/step3_fetch_fulltext.py" in command:
            raise subprocess.CalledProcessError(returncode=1, cmd=command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as excinfo:
        run_pipeline.main(["--workdir", str(tmp_path), "--force"])

    assert str(excinfo.value) == "Step 'Fetch PMC full texts' failed with exit code 1."
    assert any("scripts/step3_fetch_fulltext.py" in command for command in commands)
    assert Path(paths.start_reviews).exists()


def test_step4_cache_without_registry_triggers_regeneration(tmp_path, monkeypatch):
    paths = run_pipeline.build_paths(str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    _prepare_cached_stage_files(paths)
    _write_json(Path(paths.theories), {"unexpected": "payload"})

    commands = []

    def fake_run(command, check=True):
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    result = run_pipeline.main(["--workdir", str(tmp_path)])
    assert result == 0

    executed_scripts = [cmd[1] for cmd in commands]
    assert "scripts/step4_extract_theories.py" in executed_scripts


def test_step4_cache_with_registry_is_skipped(tmp_path, monkeypatch):
    paths = run_pipeline.build_paths(str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    _prepare_cached_stage_files(paths)
    _write_json(Path(paths.theories), {"theory_registry": {}})

    commands = []

    def fake_run(command, check=True):
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(run_pipeline.subprocess, "run", fake_run)

    result = run_pipeline.main(["--workdir", str(tmp_path)])
    assert result == 0

    executed_scripts = [cmd[1] for cmd in commands]
    assert "scripts/step4_extract_theories.py" not in executed_scripts
