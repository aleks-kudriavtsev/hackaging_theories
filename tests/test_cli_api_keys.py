import argparse
import importlib.util
from pathlib import Path


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collect_theories = _load_script_module("collect_theories", Path("scripts/collect_theories.py"))
analyze_papers = _load_script_module("analyze_papers", Path("scripts/analyze_papers.py"))


def test_collect_theories_cli_override_applies_to_provider_config(monkeypatch):
    args = argparse.Namespace(
        openalex_api_key="cli-openalex",
        crossref_api_key=None,
        pubmed_api_key=None,
        serpapi_key=None,
        semantic_scholar_key=None,
        scihub_email=None,
        scihub_rapidapi_key=None,
        annas_archive_api_key=None,
    )

    def fake_resolve(config, base_path=None):  # pragma: no cover - patched in test
        assert config == {"openalex": "config"}
        assert base_path == Path("/tmp/config")
        return {"openalex": "resolved-openalex"}

    monkeypatch.setattr(collect_theories, "resolve_api_keys", fake_resolve)

    api_keys = collect_theories._load_api_keys(
        args,
        {"openalex": "config"},
        base_path=Path("/tmp/config"),
    )

    provider_cfg = {
        "providers": [
            {
                "name": "openalex",
                "type": "openalex",
                "api_key_key": "openalex",
            }
        ]
    }

    configs = collect_theories.build_provider_configs(provider_cfg, None, api_keys)
    assert configs[0].api_key == "cli-openalex"


def test_analyze_papers_cli_override_updates_api_keys(monkeypatch):
    args = argparse.Namespace(
        openalex_api_key=None,
        crossref_api_key="cli-crossref",
        pubmed_api_key="cli-pubmed",
        serpapi_key=None,
        semantic_scholar_key=None,
        scihub_email=None,
        scihub_rapidapi_key=None,
        annas_archive_api_key=None,
    )

    def fake_resolve(config, base_path=None):  # pragma: no cover - patched in test
        assert config == {"crossref_contact": "cfg", "pubmed": "cfg"}
        assert base_path == Path("/tmp/analysis")
        return {"crossref_contact": "resolved-crossref", "pubmed": "resolved-pubmed"}

    monkeypatch.setattr(analyze_papers, "resolve_api_keys", fake_resolve)

    api_keys = analyze_papers._load_api_keys(
        args,
        {"crossref_contact": "cfg", "pubmed": "cfg"},
        base_path=Path("/tmp/analysis"),
    )

    assert api_keys["crossref_contact"] == "cli-crossref"
    assert api_keys["pubmed"] == "cli-pubmed"
    # Ensure non-overridden keys remain available when present.
    assert api_keys.get("openalex") is None
