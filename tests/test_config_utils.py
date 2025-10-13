from __future__ import annotations

from pathlib import Path

import pytest

from theories_pipeline.config_utils import MissingSecretError, resolve_api_keys


def test_resolve_api_keys_prefers_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPENALEX", "secret-openalex")
    config = {"openalex": {"env": "TEST_OPENALEX"}}
    resolved = resolve_api_keys(config)
    assert resolved["openalex"] == "secret-openalex"


def test_resolve_api_keys_uses_default_when_env_missing() -> None:
    config = {"crossref": {"env": "DOES_NOT_EXIST", "default": "user@example.com"}}
    resolved = resolve_api_keys(config, env={})
    assert resolved["crossref"] == "user@example.com"


def test_resolve_api_keys_reads_from_file(tmp_path: Path) -> None:
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("cached-key\n", encoding="utf-8")
    config = {"pubmed": {"file": secret_file.name}}
    resolved = resolve_api_keys(config, base_path=tmp_path)
    assert resolved["pubmed"] == "cached-key"


def test_resolve_api_keys_raises_when_required_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    config = {"openai": {"env": "MISSING_SECRET", "required": True}}
    with pytest.raises(MissingSecretError):
        resolve_api_keys(config)


def test_resolve_api_keys_passes_through_plain_string() -> None:
    config = {"custom": "literal-value"}
    resolved = resolve_api_keys(config, env={})
    assert resolved["custom"] == "literal-value"

