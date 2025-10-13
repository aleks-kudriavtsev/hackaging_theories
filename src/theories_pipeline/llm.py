"""LLM client utilities for GPT-powered theory classification.

This module wraps the OpenAI API with conveniences for request batching,
retries, and transparent caching of responses on disk.  The implementation is
designed so the rest of the pipeline can depend on a small, well-defined
interface without importing ``openai`` directly.  All interaction with the
network client happens through :class:`LLMClient` which can be configured via a
:class:`LLMClientConfig` dataclass instance.

The client stores cached responses in JSON files under ``data/cache/llm`` by
default which allows repeated runs of the classification scripts to reuse prior
LLM outputs without consuming additional API quota.  Cache keys are derived
from the full request payload (model, temperature, and message content) so
changing any of these parameters automatically results in new cache entries.

The implementation is intentionally defensive: the OpenAI dependency is
optional and only required when an instance of :class:`LLMClient` is created.
If the library is missing a clear runtime error is raised.  The retry logic
performs exponential backoff for both rate limit and transient API failures so
callers can simply catch :class:`LLMRateLimitError` to fall back to keyword
classification.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, List, Mapping, MutableMapping, Sequence

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


def _default_cache_dir() -> Path:
    return Path("data/cache/llm")


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for the :class:`LLMClient`."""

    model: str
    temperature: float = 0.0
    batch_size: int = 4
    max_retries: int = 3
    retry_backoff: float = 2.0
    request_timeout: float = 60.0
    cache_dir: Path = field(default_factory=_default_cache_dir)

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if self.batch_size <= 0:
            raise ValueError("batch_size must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


@dataclass(frozen=True)
class LLMMessage:
    """Chat completion style message payload."""

    role: str
    content: str

    def to_dict(self) -> Mapping[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResponse:
    """Simplified representation of an LLM response."""

    content: str
    cached: bool
    metadata: Mapping[str, Any] | None = None


class LLMClientError(RuntimeError):
    """Base error raised by :class:`LLMClient`."""


class LLMRateLimitError(LLMClientError):
    """Error raised when the API reports a rate limit condition."""


class LLMClient:
    """Wrapper around the OpenAI API with caching and retry support."""

    def __init__(
        self,
        config: LLMClientConfig,
        *,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if openai is None and OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required to use GPT-backed classification"
            )

        self.config = config
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client, self._client_mode = self._init_client()
        self._cache_lock = Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def generate(
        self,
        messages_batch: Sequence[Sequence[LLMMessage | Mapping[str, str]]],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> List[LLMResponse]:
        """Generate completions for a batch of prompts.

        Parameters
        ----------
        messages_batch:
            Iterable of chat message sequences. Each sequence corresponds to a
            single completion request.
        model:
            Optional override for the configured model.
        temperature:
            Optional override for the configured sampling temperature.
        """

        effective_model = model or self.config.model
        effective_temperature = (
            self.config.temperature if temperature is None else temperature
        )

        payloads = [
            self._prepare_payload(messages, effective_model, effective_temperature)
            for messages in messages_batch
        ]

        results: List[Optional[LLMResponse]] = [None] * len(payloads)
        pending_indices: List[int] = []
        pending_payloads: List[MutableMapping[str, Any]] = []

        for idx, payload in enumerate(payloads):
            cached = self._read_cache(payload)
            if cached is not None:
                results[idx] = LLMResponse(content=cached, cached=True)
            else:
                pending_indices.append(idx)
                pending_payloads.append(payload)

        if not pending_payloads:
            return [response for response in results if response is not None]

        batch_size = self.config.batch_size
        for start in range(0, len(pending_payloads), batch_size):
            chunk_payloads = pending_payloads[start : start + batch_size]
            chunk_indices = pending_indices[start : start + batch_size]
            for idx, payload in zip(chunk_indices, chunk_payloads):
                response_text = self._execute_with_retries(payload)
                response = LLMResponse(content=response_text, cached=False)
                results[idx] = response
                self._write_cache(payload, response_text)

        return [response for response in results if response is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_client(self) -> tuple[Any, str]:
        if OpenAI is not None:  # openai >= 1.0
            client = OpenAI(api_key=self.api_key)
            return client, "sdk"
        if openai is None:
            raise RuntimeError("OpenAI client library is unavailable")
        if self.api_key:
            openai.api_key = self.api_key
        return openai, "legacy"

    def _prepare_payload(
        self,
        messages: Sequence[LLMMessage | Mapping[str, str]],
        model: str,
        temperature: float,
    ) -> MutableMapping[str, Any]:
        normalized: List[Mapping[str, str]] = []
        for message in messages:
            if isinstance(message, LLMMessage):
                normalized.append(message.to_dict())
            elif isinstance(message, Mapping):
                normalized.append(
                    {"role": str(message["role"]), "content": str(message["content"])}
                )
            else:  # pragma: no cover - defensive programming
                raise TypeError(
                    "messages must be LLMMessage instances or dicts with role/content"
                )
        return {
            "model": model,
            "temperature": temperature,
            "messages": normalized,
        }

    def _cache_key(self, payload: Mapping[str, Any]) -> str:
        digest_source = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(digest_source).hexdigest()

    def _cache_path(self, payload: Mapping[str, Any]) -> Path:
        return self.cache_dir / f"{self._cache_key(payload)}.json"

    def _read_cache(self, payload: Mapping[str, Any]) -> str | None:
        path = self._cache_path(payload)
        with self._cache_lock:
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:  # pragma: no cover - corrupt cache
                self.logger.warning("Invalid LLM cache entry at %s; ignoring", path)
                return None
        return str(data.get("content", ""))

    def _write_cache(self, payload: Mapping[str, Any], content: str) -> None:
        path = self._cache_path(payload)
        payload = dict(payload)
        payload["content"] = content
        with self._cache_lock:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _execute_with_retries(self, payload: MutableMapping[str, Any]) -> str:
        attempt = 0
        max_attempts = self.config.max_retries + 1
        while attempt < max_attempts:
            try:
                return self._invoke_api(payload)
            except LLMRateLimitError:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                delay = self.config.retry_backoff**attempt
                self.logger.warning(
                    "Rate limit encountered, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt,
                    max_attempts,
                )
                time.sleep(delay)
            except LLMClientError:
                attempt += 1
                if attempt >= max_attempts:
                    raise
                delay = self.config.retry_backoff**attempt
                self.logger.warning(
                    "LLM request failed, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt,
                    max_attempts,
                )
                time.sleep(delay)
        raise LLMClientError("Exceeded maximum retry attempts")

    def _invoke_api(self, payload: MutableMapping[str, Any]) -> str:
        try:
            if self._client_mode == "sdk":
                timeout = self.config.request_timeout
                response = self._client.chat.completions.create(  # type: ignore[call-arg]
                    model=payload["model"],
                    messages=payload["messages"],
                    temperature=payload["temperature"],
                    timeout=timeout,
                )
                choice = response.choices[0].message
                content = getattr(choice, "content", "")
            else:
                request_timeout = self.config.request_timeout
                response = self._client.ChatCompletion.create(  # type: ignore[attr-defined]
                    model=payload["model"],
                    messages=payload["messages"],
                    temperature=payload["temperature"],
                    request_timeout=request_timeout,
                )
                content = response["choices"][0]["message"]["content"]
            return content or ""
        except Exception as exc:  # pragma: no cover - network error handling
            if self._is_rate_limit(exc):
                raise LLMRateLimitError(str(exc)) from exc
            raise LLMClientError(str(exc)) from exc

    def _is_rate_limit(self, exc: Exception) -> bool:
        if getattr(exc, "status_code", None) == 429:
            return True
        if getattr(exc, "status", None) == 429:
            return True
        code = getattr(exc, "code", None)
        if isinstance(code, str) and "rate" in code.lower():
            return True
        message = str(exc).lower()
        return "rate limit" in message or "too many requests" in message


__all__ = [
    "LLMClient",
    "LLMClientConfig",
    "LLMClientError",
    "LLMMessage",
    "LLMRateLimitError",
    "LLMResponse",
]

