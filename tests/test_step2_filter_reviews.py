import asyncio
import sys
import types
from pathlib import Path
from typing import Dict, List

import importlib.util
import pytest


class _DummyAsyncOpenAI:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - trivial wrapper
        pass

    async def __aenter__(self) -> "_DummyAsyncOpenAI":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial wrapper
        return False


sys.modules.setdefault("openai", types.SimpleNamespace(AsyncOpenAI=_DummyAsyncOpenAI))

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "step2_filter_reviews.py"
SPEC = importlib.util.spec_from_file_location("step2_filter_reviews", MODULE_PATH)
assert SPEC and SPEC.loader
step2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(step2)  # type: ignore[arg-type]


def _record(identifier: str, title: str, abstract: str = "") -> Dict:
    return {
        "id": identifier,
        "title": title,
        "abstract": abstract,
    }


def test_build_prompt_enumerates_items() -> None:
    prompt = step2.build_prompt([
        _record("1", "First", "About aging"),
        _record("2", "Second", "Maybe"),
    ])

    assert "Item 1:" in prompt
    assert "Item 2:" in prompt
    assert "Return a JSON array" in prompt

def test_filter_records_processes_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _record("a", "Aging review", "Discusses aging theory"),
        _record("b", "Other topic", "Not about aging"),
    ]

    prompts: List[str] = []

    async def fake_call_openai(
        client, prompt: str, model: str, semaphore, delay: float
    ) -> Dict:
        prompts.append(prompt)
        return [
            {"relevant": True, "explanation": "Focuses on aging theory"},
            {"relevant": False, "explanation": "Different domain"},
        ]

    monkeypatch.setattr(step2, "_call_openai", fake_call_openai)
    monkeypatch.setattr(step2, "AsyncOpenAI", _DummyAsyncOpenAI)

    kept = asyncio.run(
        step2.async_filter_records(
            records,
            "key",
            "model",
            delay=0,
            batch_size=2,
            concurrency=2,
        )
    )

    assert len(kept) == 1
    assert kept[0]["id"] == "a"
    assert kept[0]["llm_filter"]["relevant"] is True
    assert "Item 1:" in prompts[0]


def test_filter_records_retries_failed_items(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _record("a", "Aging review", "Discusses aging"),
        _record("b", "Edge case", "Ambiguous"),
    ]

    responses: List[object] = [
        RuntimeError("OpenAI returned invalid JSON payload: not-json"),
        [
            {"relevant": True, "explanation": "Retry success"},
            {"relevant": False, "explanation": "Retry fail"},
        ],
        [
            {"relevant": False, "explanation": "Retry fail"},
        ],
    ]

    async def fake_call_openai(
        client, prompt: str, model: str, semaphore, delay: float
    ) -> Dict:
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    monkeypatch.setattr(step2, "_call_openai", fake_call_openai)
    monkeypatch.setattr(step2, "AsyncOpenAI", _DummyAsyncOpenAI)

    kept = asyncio.run(
        step2.async_filter_records(
            records,
            "key",
            "model",
            delay=0,
            batch_size=2,
            concurrency=1,
        )
    )

    assert len(kept) == 1
    assert kept[0]["id"] == "a"
    assert responses == []
    assert records[0]["llm_filter"]["explanation"] == "Retry success"
    assert records[1]["llm_filter"]["relevant"] is False
