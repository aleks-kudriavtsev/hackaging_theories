"""Runtime ontology bootstrap helpers for overfull nodes."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Sequence

from .literature import PaperMetadata
from .llm import LLMClient, LLMClientError, LLMMessage
from .ontology_manager import RuntimeNodeSpec

logger = logging.getLogger(__name__)


_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class RuntimeLabelRequest:
    """Context for requesting runtime ontology labels."""

    node: str
    mode: str
    parent: str | None
    papers: Sequence[PaperMetadata]
    max_labels: int


@dataclass(frozen=True)
class RuntimeLabelResponse:
    """Structured response for bootstrapper proposals."""

    proposals: Sequence[RuntimeNodeSpec] = field(default_factory=tuple)


class RuntimeOntologyBootstrapper:
    """LLM-backed helper that proposes runtime ontology labels."""

    def __init__(
        self,
        llm_client: LLMClient | None,
        *,
        max_papers: int = 6,
        max_chars: int = 2000,
    ) -> None:
        self.llm_client = llm_client
        self.max_papers = max(1, int(max_papers))
        self.max_chars = max(200, int(max_chars))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def propose_labels(self, request: RuntimeLabelRequest) -> RuntimeLabelResponse:
        """Return candidate labels or an empty response when unavailable."""

        if not self.llm_client:
            return RuntimeLabelResponse()
        if not request.papers:
            return RuntimeLabelResponse()

        snippets = self._paper_snippets(request.papers)
        prompt = self._build_prompt(request, snippets)
        try:
            response = self.llm_client.generate([prompt])[0]
        except LLMClientError as exc:  # pragma: no cover - network/runtime guard
            logger.warning("LLM bootstrap failed for %s: %s", request.node, exc)
            return RuntimeLabelResponse()

        proposals = self._parse_response(
            response.content,
            request,
            [paper.identifier for paper in request.papers[: self.max_papers]],
        )
        if request.max_labels and len(proposals) > request.max_labels:
            proposals = proposals[: request.max_labels]
        return RuntimeLabelResponse(proposals=proposals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _paper_snippets(self, papers: Sequence[PaperMetadata]) -> List[str]:
        snippets: List[str] = []
        for paper in papers[: self.max_papers]:
            excerpt = paper.analysis_text or paper.abstract or ""
            excerpt = (excerpt or "").strip()[: self.max_chars]
            if excerpt and len(excerpt) == self.max_chars:
                excerpt = f"{excerpt}..."
            author_line = ", ".join(paper.authors) if paper.authors else "Unknown"
            snippets.append(
                "\n".join(
                    [
                        f"Title: {paper.title}",
                        f"Authors: {author_line}",
                        f"Source: {paper.source}",
                        f"Abstract: {excerpt or '<missing>'}",
                    ]
                )
            )
        return snippets

    def _build_prompt(
        self,
        request: RuntimeLabelRequest,
        snippets: Sequence[str],
    ) -> List[LLMMessage]:
        system_prompt = (
            "You expand a hierarchical ontology of gerontology theories. Given context "
            "about the papers collected for a node, propose concise names for "
            "either child subtheories or sibling theories. Respond with JSON containing "
            "a 'labels' array; each entry should include 'name', optional 'keywords', "
            "and optional 'notes' describing the rationale."
        )
        parent_line = request.parent or "<no parent>"
        user_prompt = (
            f"Current node: {request.node}\n"
            f"Relationship requested: {request.mode}\n"
            f"Parent node: {parent_line}\n"
            f"Provide up to {request.max_labels} labels."
        )
        paper_block = "\n\n".join(snippets)
        if paper_block:
            user_prompt = f"{user_prompt}\n\nRecent papers (title/abstract excerpts):\n{paper_block}"
        return [LLMMessage("system", system_prompt), LLMMessage("user", user_prompt)]

    def _parse_response(
        self,
        text: str,
        request: RuntimeLabelRequest,
        paper_ids: Sequence[str],
    ) -> List[RuntimeNodeSpec]:
        cleaned = text.strip()
        if not cleaned:
            return []
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = _JSON_PATTERN.search(cleaned)
            if not match:
                logger.debug("LLM bootstrap response missing JSON: %s", cleaned[:120])
                return []
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.debug("Unable to parse JSON from bootstrap response: %s", cleaned[:120])
                return []
        labels = payload.get("labels") or payload.get("theories") or payload.get("nodes")
        if isinstance(labels, Mapping):
            labels = [labels]
        if not isinstance(labels, Iterable):
            return []
        proposals: List[RuntimeNodeSpec] = []
        for item in labels:
            if not isinstance(item, Mapping):
                continue
            raw_name = item.get("name") or item.get("label") or item.get("theory")
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip()
            if not name:
                continue
            keywords: List[str] = []
            raw_keywords = item.get("keywords")
            if isinstance(raw_keywords, (list, tuple, set)):
                for value in raw_keywords:
                    if isinstance(value, str) and value.strip():
                        keywords.append(value.strip())
            elif isinstance(raw_keywords, str) and raw_keywords.strip():
                keywords.append(raw_keywords.strip())
            config_payload = (
                dict(item.get("config"))
                if isinstance(item.get("config"), Mapping)
                else {}
            )
            metadata_payload = (
                dict(item.get("metadata"))
                if isinstance(item.get("metadata"), Mapping)
                else {}
            )
            notes = item.get("notes") or item.get("description")
            if isinstance(notes, str) and notes.strip():
                metadata_payload.setdefault("notes", notes.strip())
            provenance_payload = {
                "mode": request.mode,
                "papers": list(paper_ids),
                "source": "runtime_llm_bootstrap",
            }
            raw_provenance = item.get("provenance")
            if isinstance(raw_provenance, Mapping):
                for key, value in raw_provenance.items():
                    provenance_payload[str(key)] = value
            proposals.append(
                RuntimeNodeSpec(
                    name=name,
                    parent=None,
                    config=config_payload,
                    keywords=keywords,
                    metadata=metadata_payload,
                    provenance=provenance_payload,
                )
            )
        return proposals


__all__ = [
    "RuntimeLabelRequest",
    "RuntimeLabelResponse",
    "RuntimeOntologyBootstrapper",
]

