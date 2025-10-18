"""Helper utilities for generating ontology node summaries.

This module centralises lightweight heuristics used when formatting
ontology prompts or when an :class:`~theories_pipeline.ontology_manager.OntologyManager`
needs to synthesise a summary for newly appended nodes.  The helpers are
intentionally conservative: they operate on simple metadata structures and
avoid heavy dependencies so they can be used during tests without relying
on live LLM access.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def clean_summary(text: str | None) -> str:
    """Return a normalised summary string."""

    if not isinstance(text, str):
        return ""
    summary = text.strip().strip("\"\'")
    return summary.strip()


def _normalized_keywords(keywords: Iterable[str] | None) -> list[str]:
    if not keywords:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in keywords:
        item = str(raw).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
    return ordered


def fallback_summary(
    name: str,
    *,
    keywords: Iterable[str] | None = None,
    bootstrap: Mapping[str, Any] | None = None,
) -> str:
    """Synthesize a short summary when none has been provided."""

    key_terms = _normalized_keywords(keywords)
    fragments: list[str] = []
    if key_terms:
        focus_terms = ", ".join(key_terms[:3])
        fragments.append(f"focuses on {focus_terms} in aging research")
    if isinstance(bootstrap, Mapping):
        citations = bootstrap.get("citations")
        try:
            citation_int = int(citations) if citations is not None else None
        except (TypeError, ValueError):
            citation_int = None
        if citation_int:
            fragments.append(f"is supported by approximately {citation_int} cited references")
    if not fragments:
        summary = f"{name} is a theory in the aging ontology."
    elif len(fragments) == 1:
        summary = f"{name} {fragments[0]}."
    else:
        joined = ", and ".join([fragments[0], fragments[1]])
        if len(fragments) > 2:
            tail = ", ".join(fragments[2:])
            joined = f"{joined}, {tail}"
        summary = f"{name} {joined}."
    return summary


def format_bootstrap_highlights(metadata: Mapping[str, Any] | None) -> str | None:
    """Format bootstrap metadata into a human readable summary."""

    if not isinstance(metadata, Mapping):
        return None
    bootstrap = metadata.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or not bootstrap:
        return None
    parts: list[str] = []
    citations = bootstrap.get("citations")
    try:
        citation_int = int(citations) if citations is not None else None
    except (TypeError, ValueError):
        citation_int = None
    if citation_int is not None:
        parts.append(f"citations={citation_int}")
    reviews = bootstrap.get("reviews")
    if isinstance(reviews, Sequence) and not isinstance(reviews, (str, bytes)):
        review_list = [str(item).strip() for item in reviews if str(item).strip()]
        if review_list:
            parts.append(f"reviews={', '.join(review_list[:3])}")
    queries = bootstrap.get("queries")
    if isinstance(queries, Sequence) and not isinstance(queries, (str, bytes)):
        query_list = [str(item).strip() for item in queries if str(item).strip()]
        if query_list:
            parts.append(f"queries={', '.join(query_list[:3])}")
    if not parts:
        return None
    return "; ".join(parts)


def extract_quote_snippets(metadata: Mapping[str, Any] | None) -> list[str]:
    """Return notable quote snippets embedded in metadata if available."""

    if not isinstance(metadata, Mapping):
        return []
    for key in ("quotes", "notable_quotes", "key_quotes", "highlights"):
        value = metadata.get(key)
        if isinstance(value, str):
            snippet = value.strip()
            return [snippet] if snippet else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            snippets = [str(item).strip() for item in value if str(item).strip()]
            if snippets:
                return snippets[:3]
    return []


def format_keywords_line(keywords: Iterable[str] | None) -> str | None:
    key_terms = _normalized_keywords(keywords)
    if not key_terms:
        return None
    return ", ".join(key_terms[:5])


__all__ = [
    "clean_summary",
    "fallback_summary",
    "format_bootstrap_highlights",
    "extract_quote_snippets",
    "format_keywords_line",
]
