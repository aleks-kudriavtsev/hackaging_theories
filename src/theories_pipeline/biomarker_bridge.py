"""Export theory candidates to the aging-biomarker evidence system.

This adapter transfers hypotheses and source identifiers, never evidence grades.
It is deliberately independent of the LLM extraction and ontology optimizers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"
THEORY_CLASSES = {
    "evolutionary_theory", "mechanistic_hypothesis", "conceptual_framework",
    "intervention", "unknown",
}
_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
_FIXTURE_ID = re.compile(r"^(A\d+|paper-(?:ds|fr)-\d+)$", re.I)


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = [values]
    if not isinstance(values, list):
        raise ValueError("Expected a string or a list of strings")
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _doi(value: Any) -> str | None:
    text = str(value or "").strip()
    text = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:\s*)", "", text, flags=re.I)
    return text.lower() if _DOI.fullmatch(text) else None


def _pmid(value: Any) -> str | None:
    text = str(value or "").strip()
    text = re.sub(r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", text, flags=re.I)
    text = re.sub(r"^pmid:\s*", "", text, flags=re.I).strip("/")
    return text if re.fullmatch(r"[1-9]\d*", text) else None


def _fixture_metadata(payload: Mapping[str, Any], input_file: str) -> bool:
    parts = {part.lower() for part in Path(input_file).parts}
    if parts & {"examples", "fixtures", "testdata"}:
        return True
    for field in ("is_fixture", "is_synthetic", "synthetic", "demo"):
        if payload.get(field) is True:
            return True
    kind = str(payload.get("data_kind") or payload.get("dataset_type") or "").lower()
    return kind in {"example", "fixture", "synthetic", "demo"}


def _records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    registry = payload.get("theory_registry")
    if isinstance(registry, dict):
        records = []
        for theory_id, entry in registry.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid registry entry: {theory_id}")
            if entry.get("theory_id", theory_id) != theory_id:
                raise ValueError(f"Conflicting theory ID: {theory_id}")
            records.append(dict(entry, theory_id=theory_id))
        return records
    ontology = payload.get("ontology", payload)
    if not isinstance(ontology, dict):
        raise ValueError("Missing theory_registry or ontology object")
    final = ontology.get("final", ontology)
    groups = final.get("groups") if isinstance(final, dict) else None
    if not isinstance(groups, list):
        raise ValueError("Missing ontology.final.groups")
    records = []

    def visit(items: list[Any]) -> None:
        for group in items:
            if not isinstance(group, dict):
                raise ValueError("Invalid ontology group")
            for entry in group.get("theories", []):
                if not isinstance(entry, dict):
                    raise ValueError("Invalid ontology theory")
                records.append(dict(entry))
            visit(group.get("subgroups", []))

    visit(groups)
    return records


def _article_catalog(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    articles = list(payload.get("articles") or [])
    for summary in payload.get("prompt_summary", []):
        if isinstance(summary, dict):
            articles.extend(summary.get("representative_articles", []))
    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("Invalid article record")
        article_id = str(article.get("id") or article.get("identifier") or "").strip()
        if not article_id:
            continue
        prior = catalog.get(article_id)
        if prior is not None:
            for key in ("doi", "pmid", "title"):
                if prior.get(key) and article.get(key) and prior[key] != article[key]:
                    raise ValueError(f"Conflicting article ID: {article_id}")
            catalog[article_id] = {**prior, **article}
        else:
            catalog[article_id] = article
    return catalog


def export_theory_pack(
    payload: Mapping[str, Any], *, source_repository: str,
    source_commit: str, registry_sha256: str, input_file: str,
) -> dict[str, Any]:
    """Build a portable metadata pack without inferring empirical links.

    Real-looking DOI/PMID values are retained as *unverified identifiers*. A
    receiving system must resolve bibliographic records and assess claim support
    separately. Input confidence, validation flags, and LLM scores are ignored.
    """
    if not source_repository.strip():
        raise ValueError("source_repository is required")
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", source_commit):
        raise ValueError("source_commit must be a complete Git commit hash")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", registry_sha256):
        raise ValueError("registry_sha256 must be a SHA-256 digest")
    catalog = _article_catalog(payload)
    fixture_input = _fixture_metadata(payload, input_file)
    nodes: dict[str, dict[str, Any]] = {}
    for entry in _records(payload):
        theory_id = str(entry.get("theory_id") or entry.get("id") or "").strip()
        label = str(entry.get("label") or entry.get("preferred_label") or "").strip()
        if not theory_id or not label:
            raise ValueError("Every theory must have an explicit stable ID and label")
        reasons = ["fixture_input"] if fixture_input or _fixture_metadata(entry, "") else []
        citations = []
        references = _strings(entry.get("supporting_articles"))
        references.extend("PMID:" + value for value in _strings(entry.get("pmids")))
        references.extend(_strings(entry.get("dois")))
        if entry.get("doi"):
            references.append(str(entry["doi"]))
        for article_id in dict.fromkeys(references):
            article = catalog.get(article_id, {})
            doi_raw = article.get("doi")
            doi = _doi(doi_raw or article_id)
            pmid_raw = article.get("pmid")
            # A numeric local article ID is not automatically a PubMed ID.
            pmid = _pmid(pmid_raw) if pmid_raw else (
                _pmid(article_id) if article_id.lower().startswith(("pmid:", "https://pubmed.ncbi.nlm.nih.gov/")) else None
            )
            citation_reasons = []
            if doi and doi.startswith("10.1000/"):
                citation_reasons.append("example_doi_prefix")
            if _FIXTURE_ID.fullmatch(article_id) and not (doi or pmid):
                citation_reasons.append("fixture_style_id_without_resolvable_identifier")
            if _fixture_metadata(article, ""):
                citation_reasons.append("fixture_article")
            if doi_raw and not doi:
                citation_reasons.append("invalid_doi")
            if pmid_raw and not pmid:
                citation_reasons.append("invalid_pmid")
            reasons.extend(citation_reasons)
            citations.append({
                "article_id": article_id, "doi": doi, "pmid": pmid,
                "title": article.get("title"),
                "verification_status": "quarantined_fixture" if citation_reasons else "identifier_unverified",
                "quarantine_reasons": citation_reasons,
            })
        raw_class = entry.get("theory_class", "unknown")
        theory_class = raw_class if raw_class in THEORY_CLASSES else "unknown"
        node = {
            "theory_id": theory_id, "label": label,
            "aliases": _strings(entry.get("aliases")),
            "theory_class": theory_class,
            "status": "quarantined_fixture" if reasons else "hypothesis_candidate",
            "citations": citations,
            "quarantine_reasons": list(dict.fromkeys(reasons)),
            "warnings": [] if citations else ["no_source_identifiers"],
            "scientific_validation": "not_assessed",
        }
        if theory_id in nodes and nodes[theory_id] != node:
            raise ValueError(f"Conflicting duplicate theory ID: {theory_id}")
        nodes[theory_id] = node
    theories = sorted(nodes.values(), key=lambda node: node["theory_id"])
    return {
        "schema_version": SCHEMA_VERSION, "layer": "theory_metadata",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_repository": source_repository, "source_commit": source_commit,
            "registry_sha256": registry_sha256, "input_file": input_file,
        },
        "theories": theories, "empirical_edges": [], "theory_links": [],
        "summary": {
            "total_theories": len(theories),
            "hypothesis_candidates": sum(n["status"] == "hypothesis_candidate" for n in theories),
            "quarantined_theories": sum(n["status"] == "quarantined_fixture" for n in theories),
        },
        "policy": {
            "llm_confidence_is_evidence": False,
            "theory_membership_implies_causality": False,
            "quarantined_nodes_eligible_for_panel": False,
            "correction_requires_new_source_snapshot": True,
            "empirical_links_require_separate_claim_review": True,
        },
    }


def export_from_path(path: Path, *, source_repository: str, source_commit: str) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    return export_theory_pack(
        payload, source_repository=source_repository, source_commit=source_commit,
        registry_sha256=hashlib.sha256(raw).hexdigest(), input_file=path.as_posix(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    try:
        pack = export_from_path(args.input, source_repository=args.source_repository, source_commit=args.source_commit)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"Theory bridge error: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pack["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
