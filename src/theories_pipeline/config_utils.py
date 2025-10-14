"""Helper utilities for loading configuration values safely.

This module focuses on resolving secret material such as API keys from a
configuration mapping without forcing contributors to commit the secrets to the
repository.  The helpers support reading values directly from the config, from
environment variables, or from external files while providing descriptive
errors when required secrets are missing.  The goal is to keep the CLI entry
points thin and reusable while centralising the security-sensitive logic in a
single location that can be unit tested.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping


class MissingSecretError(RuntimeError):
    """Raised when a required secret cannot be resolved."""


_PLACEHOLDER_API_KEYS: Mapping[str, Iterable[str]] = {
    "openalex": ("sk-your-openalex-key",),
    "pubmed": ("your-pubmed-key",),
    "openai": ("sk-your-openai-key",),
    "scihub_rapidapi": ("your-rapidapi-token",),
    "annas_archive": ("your-rapidapi-token",),
}


def resolve_api_keys(
    config: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    base_path: Path | None = None,
) -> Dict[str, str | None]:
    """Resolve an ``api_keys`` configuration mapping.

    Parameters
    ----------
    config:
        Mapping of key names to configuration descriptors.  Each descriptor can
        be either a literal string value or a mapping describing where to load
        the secret (environment variable, external file, or inline default).
    env:
        Optional environment mapping.  Defaults to :data:`os.environ` which is
        appropriate for production code paths.  Tests can inject a custom
        mapping to avoid mutating the real process environment.
    base_path:
        If provided, relative file references are resolved against this path.
    """

    environment = dict(os.environ if env is None else env)
    root = Path(base_path) if base_path is not None else None

    resolved: Dict[str, str | None] = {}
    for name, descriptor in config.items():
        resolved[name] = _resolve_single_secret(name, descriptor, environment, root)
    return resolved


def ensure_real_api_keys(values: Mapping[str, str | None]) -> Dict[str, str | None]:
    """Ensure configured API keys are not left at documented placeholder values.

    The README uses human-friendly placeholder strings (for example,
    ``"your-pubmed-key"``) when demonstrating how to export credentials.  In
    practice users must replace those values with their real secrets.  When the
    placeholders leak into the runtime configuration the external services
    reject our requests, which can be difficult to diagnose from the network
    trace alone.  This helper validates the final configuration and raises a
    :class:`MissingSecretError` with a descriptive message if any placeholder is
    detected.
    """

    cleaned: Dict[str, str | None] = dict(values)
    offenders: list[tuple[str, str]] = []
    for name, raw_value in cleaned.items():
        if not raw_value:
            continue
        normalized = raw_value.strip().lower()
        placeholders = {value.lower() for value in _PLACEHOLDER_API_KEYS.get(name, ())}
        if normalized in placeholders:
            offenders.append((name, raw_value))
    if offenders:
        examples = ", ".join(f"{name}='{value}'" for name, value in offenders)
        raise MissingSecretError(
            "Placeholder API key detected. Replace the example value with your real "
            f"credentials for: {examples}"
        )
    return cleaned


def _resolve_single_secret(
    name: str,
    descriptor: Any,
    environment: Mapping[str, str],
    base_path: Path | None,
) -> str | None:
    if descriptor is None:
        return None

    if isinstance(descriptor, str):
        expanded = os.path.expandvars(descriptor)
        if expanded and expanded != descriptor:
            return expanded
        if descriptor.lower().startswith("env:"):
            env_name = descriptor.split(":", 1)[1].strip()
            if not env_name:
                return None
            return environment.get(env_name)
        return descriptor or None

    if isinstance(descriptor, MutableMapping):
        if "env" in descriptor:
            env_name = str(descriptor["env"]).strip()
            if env_name:
                value = environment.get(env_name)
                if value:
                    return value
                if descriptor.get("required"):
                    raise MissingSecretError(
                        f"Environment variable '{env_name}' required for API key '{name}'"
                    )
            default = descriptor.get("default")
            if default is not None:
                return str(default)
        if "value" in descriptor:
            raw_value = descriptor.get("value")
            return str(raw_value) if raw_value is not None else None
        if "file" in descriptor:
            file_path = Path(str(descriptor["file"]))
            if not file_path.is_absolute() and base_path is not None:
                file_path = base_path / file_path
            if file_path.exists():
                return file_path.read_text(encoding="utf-8").strip()
            if descriptor.get("required"):
                raise MissingSecretError(
                    f"Secret file '{file_path}' required for API key '{name}' not found"
                )
            default = descriptor.get("default")
            if default is not None:
                return str(default)
        return None

    return None


__all__ = ["MissingSecretError", "ensure_real_api_keys", "resolve_api_keys"]

