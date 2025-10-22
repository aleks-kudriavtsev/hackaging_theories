"""Unit tests for the step4 theory extraction helpers."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "step4_extract_theories.py"
SPEC = importlib.util.spec_from_file_location("step4_extract_theories", MODULE_PATH)
step4 = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # pragma: no cover - defensive
SPEC.loader.exec_module(step4)  # type: ignore[assignment]


def test_chat_completion_json_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """socket.timeout errors are converted to RuntimeError with guidance."""

    def fake_urlopen(*_: Any, **__: Any) -> Any:
        raise socket.timeout("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError) as excinfo:
        step4.chat_completion_json("sys", "user", "key", "model", 0.1, timeout=1)

    message = str(excinfo.value).lower()
    assert "timeout" in message
    assert "retry" in message


def test_build_registry_tracks_synonyms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synonym matches are cached and exposed via the synonym registry."""

    def fake_find_match(*_: Any, **__: Any) -> Any:
        return None

    disambiguation_calls: list[tuple[str, float, str]] = []

    def fake_disambiguate(
        name: str,
        candidates: dict[str, dict[str, Any]],
        record: dict[str, Any],
        api_key: str,
        model: str,
        temperature: float,
        request_timeout: float | None,
    ) -> str | None:
        disambiguation_calls.append((name, temperature, model))
        return next(iter(candidates.keys()), None)

    monkeypatch.setattr(step4, "find_lexical_match", fake_find_match)
    monkeypatch.setattr(step4, "disambiguate_with_llm", fake_disambiguate)

    annotated = [
        {"id": "a", "theory_extraction": {"theories": ["Disposable soma theory"]}},
        {"id": "b", "theory_extraction": {"theories": ["Disposable soma hypothesis"]}},
        {"id": "c", "theory_extraction": {"theories": ["Disposable soma hypothesis"]}},
    ]

    registry, synonyms = step4.build_theory_registry(
        annotated,
        api_key="dummy",
        model="gpt-5-mini",
        temperature=0.3,
        request_timeout=None,
    )

    assert len(disambiguation_calls) == 1
    assert disambiguation_calls[0] == (
        "Disposable soma hypothesis",
        0.3,
        "gpt-5-mini",
    )

    assert len(registry) == 1
    canonical_id, entry = next(iter(registry.items()))
    assert entry["label"] == "Disposable soma theory"
    assert entry["supporting_articles"] == ["a", "b", "c"]
    assert "Disposable soma hypothesis" in entry["aliases"]

    assert synonyms == {
        "Disposable soma hypothesis": {
            "canonical_id": canonical_id,
            "slug": "disposable-soma-hypothesis",
            "source_articles": ["b", "c"],
        }
    }
