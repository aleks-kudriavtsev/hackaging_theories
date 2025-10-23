import asyncio
import sys
import types
from pathlib import Path
from typing import Dict, List

import importlib.util
import json
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


def test_call_openai_raises_when_content_missing() -> None:
    class _FakeChatCompletions:
        def __init__(self, response: object) -> None:
            self._response = response

        async def create(self, *args, **kwargs):
            return self._response

    class _FakeChat:
        def __init__(self, response: object) -> None:
            self.completions = _FakeChatCompletions(response)

    class _FakeClient:
        def __init__(self, response: object) -> None:
            self.chat = _FakeChat(response)

    message = types.SimpleNamespace(content=None, role="assistant")
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    response = types.SimpleNamespace(
        id="resp-123",
        model="gpt-test",
        choices=[choice],
    )

    client = _FakeClient(response)
    semaphore = asyncio.Semaphore(1)

    async def _invoke() -> None:
        await step2._call_openai(client, "prompt", "model", semaphore, delay=0)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(_invoke())

    message_text = str(excinfo.value)
    assert "missing message content" in message_text
    assert "resp-123" in message_text


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


def test_async_filter_records_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    records = [_record("cached", "Aging focus", "Focuses on aging theory")]

    cache_path = tmp_path / "cache.json"
    cache_payload = {
        "id:cached": {"relevant": True, "explanation": "Cached decision"},
    }
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
    cache_store = step2.load_decision_cache(cache_path)

    called = False

    async def fake_call_openai(*args, **kwargs):  # pragma: no cover - should not run
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(step2, "_call_openai", fake_call_openai)
    monkeypatch.setattr(step2, "AsyncOpenAI", _DummyAsyncOpenAI)

    kept = asyncio.run(
        step2.async_filter_records(
            records,
            "key",
            "model",
            delay=0,
            batch_size=1,
            concurrency=1,
            cache=cache_store,
            cache_path=str(cache_path),
        )
    )

    assert not called
    assert len(kept) == 1
    assert kept[0]["id"] == "cached"
    assert kept[0]["llm_filter"]["explanation"] == "Cached decision"


def test_async_filter_records_persists_new_decisions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [_record("new", "Fresh aging review", "Discusses aging theory extensively")]

    responses = [
        [
            {"relevant": True, "explanation": "Looks good"},
        ]
    ]

    async def fake_call_openai(*args, **kwargs):
        return responses.pop(0)

    cache_path = tmp_path / "cache.json"
    cache_store = step2.load_decision_cache(cache_path)

    monkeypatch.setattr(step2, "_call_openai", fake_call_openai)
    monkeypatch.setattr(step2, "AsyncOpenAI", _DummyAsyncOpenAI)

    kept = asyncio.run(
        step2.async_filter_records(
            records,
            "key",
            "model",
            delay=0,
            batch_size=1,
            concurrency=1,
            cache=cache_store,
            cache_path=str(cache_path),
        )
    )

    assert kept and kept[0]["llm_filter"]["relevant"] is True
    assert responses == []
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_keys = step2._record_cache_keys(records[0])
    assert expected_keys
    for key in expected_keys:
        assert key in saved
        assert saved[key]["relevant"] is True


def test_single_item_retry_accepts_legacy_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _record("a", "General", ""),
        _record("b", "Focused", ""),
    ]

    responses: List[object] = [
        {"1": {"relevant": False, "explanation": "Irrelevant"}},
        {"relevant": True, "explanation": "Uses legacy schema"},
    ]

    def fake_call_openai(prompt: str, api_key: str, model: str) -> Dict:
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[return-value]

    monkeypatch.setattr(step2, "call_openai", fake_call_openai)

    kept = step2.filter_records(records, "key", "model", delay=0, batch_size=2)

    assert len(kept) == 1
    assert kept[0]["id"] == "b"
    assert records[1]["llm_filter"]["explanation"] == "Uses legacy schema"
    assert responses == []
