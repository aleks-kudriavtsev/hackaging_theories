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
        step4.chat_completion_json("sys", "user", "key", "model", timeout=1)

    message = str(excinfo.value).lower()
    assert "timeout" in message
    assert "retry" in message
