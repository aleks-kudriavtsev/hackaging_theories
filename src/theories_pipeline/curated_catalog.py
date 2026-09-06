"""Validate and export a curated *research-question* catalog, never panel evidence.

Bibliographic curation is an auditable assertion, not automatic theory validation.
The legacy bridge keeps its conservative identifier_unverified status; this
optional interface carries separate, explicit source-review records downstream.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

KINDS = {"theory", "framework", "intervention_family"}
CLASS_KIND = {"evolutionary_theory": "theory", "mechanistic_hypothesis": "theory",
              "conceptual_framework": "framework", "intervention": "intervention_family"}
CHECK_SCOPE = "bibliographic_identity_and_concept_only"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _objects(value: Any, field: str) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(field + " must be a list of objects")
    return value


def _texts(value: Any, field: str, *, nonempty=False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(field + " must be a list of nonempty strings")
    if nonempty and not value:
        raise ValueError(field + " must not be empty")
    return list(dict.fromkeys(value))


def _required(value: Mapping, fields: list[str], entity: str) -> None:
    for field in fields:
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(entity + " requires " + field)


def _reference(source: Mapping) -> dict:
    _required(source, ["id", "title", "url", "source_type", "population_scope"], "Source")
    doi = str(source.get("doi") or "").lower()
    pmid = str(source.get("pmid") or "")
    if not doi and not pmid:
        raise ValueError("Source requires DOI or PMID")
    if doi and (not re.fullmatch(r"10\.\d{4,9}/\S+", doi) or doi.startswith("10.1000/")):
        raise ValueError("Source has invalid or placeholder DOI")
    if pmid and not re.fullmatch(r"[1-9]\d*", pmid):
        raise ValueError("Source has invalid PMID")
    url = urlparse(source["url"])
    if url.scheme != "https" or not url.hostname or url.hostname in {"example.com", "example.org", "localhost"}:
        raise ValueError("Source URL must identify a real HTTPS bibliographic record")
    if url.hostname == "pubmed.ncbi.nlm.nih.gov" and pmid and url.path.strip("/") != pmid:
        raise ValueError("Source URL/PMID mismatch")
    if url.hostname in {"doi.org", "dx.doi.org"} and doi and unquote(url.path.strip("/")).lower() != doi:
        raise ValueError("Source URL/DOI mismatch")
    review = source.get("bibliographic_review", {})
    _required(review, ["checked_date", "reviewed_by", "verification_scope", "evidence_location"], "Source review")
    if review["verification_scope"] != CHECK_SCOPE or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review["checked_date"]):
        raise ValueError("Source review scope/date is invalid")
    if review.get("identity_matches") is not True:
        raise ValueError("Source identity has not been checked")
    if source.get("fixture") or source.get("is_fixture") or source.get("is_synthetic"):
        raise ValueError("Fixture sources cannot enter a curated catalog")
    result = {key: deepcopy(source[key]) for key in ("id", "title", "doi", "pmid", "url", "source_type", "population_scope") if key in source}
    result.update({key: review[key] for key in ("checked_date", "verification_scope", "reviewed_by", "evidence_location")})
    result["retraction_status"] = "not_systematically_assessed"
    return result


def build_consumer_config(payload: Mapping, pack: Mapping) -> dict | None:
    """Return the aging_biomarkers contract for explicitly curated inputs only.

    Reject malformed graph references and any attempt to import empirical links,
    accepted observations, evidence scores or confirmed predictions via theory.
    """
    if payload.get("catalog_schema") != "curated_theory_discovery/1.0":
        return None
    if pack["summary"]["quarantined_theories"]:
        raise ValueError("Quarantined theory cannot enter a curated catalog")
    forbidden = ("empirical_edges", "observations", "selection_scores", "evidence_scores")
    if any(payload.get(key) for key in forbidden):
        raise ValueError("Theory catalog cannot supply empirical evidence or scores")
    refs = {}
    for source in _objects(payload.get("articles"), "articles"):
        ref = _reference(source)
        if ref["id"] in refs:
            raise ValueError("Duplicate curated source ID")
        refs[ref["id"]] = ref
    theories = []
    registry = payload.get("theory_registry")
    if not isinstance(registry, dict):
        raise ValueError("Curated catalog requires theory_registry")
    for tid, entry in registry.items():
        _required(entry, ["label", "label_ru", "definition", "scope", "theory_class"], tid)
        source_ids = _texts(entry.get("supporting_articles"), tid + ".supporting_articles", nonempty=True)
        if any(sid not in refs for sid in source_ids):
            raise ValueError("Unknown source reference in " + tid)
        kind = CLASS_KIND.get(entry["theory_class"])
        if kind not in KINDS:
            raise ValueError("Unsupported curated theory class")
        theories.append({"theory_id": tid, "label": entry["label"], "label_ru": entry["label_ru"],
                         "kind": kind, "theory_class": entry["theory_class"], "layer": "T",
                         "description": entry["definition"], "scope": entry["scope"],
                         "query_terms": _texts(entry.get("query_terms"), tid + ".query_terms", nonempty=True),
                         "competing_explanations": _texts(entry.get("competing_explanations"), tid + ".competing_explanations", nonempty=True),
                         "source_status": "bibliographic_source_checked", "scientific_validation": "not_assessed",
                         "source_refs": [refs[sid] for sid in source_ids], "causal_claim": False})
    tids = set(registry)
    nodes, aids = {}, set()
    for original in _objects(payload.get("nodes"), "nodes"):
        node = deepcopy(original)
        _required(node, ["id", "level", "label_ru"], "Node")
        nid, level = node["id"], node["level"]
        if nid in nodes or nid in tids or nid.startswith("T:") or level not in {"L1", "L2", "L3"}:
            raise ValueError("Duplicate or invalid clinical node: " + nid)
        if level == "L3":
            _required(node, ["analyte_id", "measurement_type"], nid)
            aids.add(node["analyte_id"])
            if node.get("outside_panel"):
                source_ids = _texts(node.pop("source_ids", None), nid + ".source_ids", nonempty=True)
                if any(sid not in refs for sid in source_ids):
                    raise ValueError("Unknown outside-panel source")
                node["source_refs"] = [refs[sid] for sid in source_ids]
        elif node.get("analyte_id"):
            raise ValueError("Analyte must be L3")
        node.update(layer=level, evidence_status="hypothesis_context", causal_claim=False)
        nodes[nid] = node
    links, seen = [], set()
    for original in _objects(payload.get("links"), "links"):
        link = deepcopy(original)
        _required(link, ["theory_id", "node_id", "relation", "rationale"], "Link")
        tid, nid, rel = link["theory_id"], link["node_id"], link["relation"]
        if tid not in tids or nid not in nodes:
            raise ValueError("Link references an unknown theory or node")
        if rel not in {"organizes", "proposes_marker"}:
            raise ValueError("Theory catalog cannot assert an empirical relation")
        if rel == "proposes_marker" and nodes[nid]["level"] != "L3":
            raise ValueError("proposes_marker requires an L3 node")
        if link.get("observation_ids") or link.get("causal_claim") or link.get("empirical_edge"):
            raise ValueError("Theory catalog cannot import observation/causal assertions")
        identity = (tid, nid, rel)
        if identity in seen:
            raise ValueError("Duplicate theory link")
        seen.add(identity)
        link.update(observation_ids=[], causal_claim=False, empirical_edge=False, status="hypothesis")
        links.append(link)
    predictions, seen = [], set()
    for original in _objects(payload.get("predictions"), "predictions"):
        prediction = deepcopy(original)
        _required(prediction, ["id", "endpoint", "question", "alternative_explanation", "required_test", "falsification_criterion"], "Prediction")
        pid = prediction["id"]
        if pid in seen:
            raise ValueError("Duplicate prediction ID")
        seen.add(pid)
        ptids = _texts(prediction.get("theory_ids"), pid + ".theory_ids", nonempty=True)
        paids = _texts(prediction.get("analyte_ids"), pid + ".analyte_ids", nonempty=True)
        if any(tid not in tids for tid in ptids) or any(aid not in aids for aid in paids):
            raise ValueError("Prediction references an unknown theory or analyte")
        if prediction.get("observation_ids") or prediction.get("confirmed") or prediction.get("causal_claim"):
            raise ValueError("Predictions cannot be confirmed by a theory catalog")
        prediction["query_terms"] = _texts(prediction.get("query_terms", []), pid + ".query_terms")
        prediction.update(observation_ids=[], confirmed=False, causal_claim=False, status="untested_hypothesis")
        predictions.append(prediction)
    questions = deepcopy(_objects(payload.get("search_questions"), "search_questions"))
    for question in questions:
        _required(question, ["id", "theory_id", "question", "supportive_query", "contradictory_query", "interpretation"], "Search question")
        if question["theory_id"] not in tids:
            raise ValueError("Search question references unknown theory")
    result = {"schema_version": "1.0", "source": "hackaging_theories_curated_export", "layer": "T",
            "theories": theories, "nodes": list(nodes.values()), "links": links,
            "predictions": predictions, "search_questions": questions,
            "policy": {"theory_membership_changes_panel_score": False,
                       "bibliographic_review_validates_theory": False,
                       "clinical_evidence_must_be_reviewed_downstream": True,
                       "search_questions_are_not_preregistered_studies": True}}
    if payload.get("consumer_config") is not None and payload["consumer_config"] != result:
        raise ValueError("Pinned consumer_config does not match the validated catalog")
    return result
