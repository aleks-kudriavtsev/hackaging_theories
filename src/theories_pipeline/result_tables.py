"""Helpers to build Hackaging CSV result tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .outputs import QUESTION_COLUMNS

# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

_SEPARATOR_TOKENS: Sequence[str] = (" - ", " — ", " – ")


@dataclass(frozen=True)
class _NormalisationRule:
    question_id: str
    mapping: Mapping[str, str]


_QUESTION_RULES: Mapping[str, _NormalisationRule] = {
    "Q1": _NormalisationRule(
        question_id="Q1",
        mapping={
            "yes, quantitatively shown": "Yes, quantitatively shown",
            "yes, mentioned without data": "Yes, but not shown",
            "yes, but not shown": "Yes, but not shown",
            "yes": "Yes, but not shown",
            "no evidence found": "No",
            "no": "No",
        },
    ),
    "Q2": _NormalisationRule(
        question_id="Q2",
        mapping={
            "mechanism supported by experiments": "Yes",
            "mechanism hypothesized": "Yes",
            "mechanism hypothesised": "Yes",
            "yes": "Yes",
            "no mechanism discussed": "No",
            "no": "No",
        },
    ),
    "Q3": _NormalisationRule(
        question_id="Q3",
        mapping={
            "validated longevity intervention": "Yes",
            "proposed longevity intervention": "Yes",
            "yes": "Yes",
            "no intervention discussed": "No",
            "no": "No",
        },
    ),
    "Q4": _NormalisationRule(
        question_id="Q4",
        mapping={
            "changes appear irreversible": "Yes",
            "changes appear reversible": "No",
            "not discussed": "No",
            "yes": "Yes",
            "no": "No",
        },
    ),
    "Q5": _NormalisationRule(
        question_id="Q5",
        mapping={
            "yes, quantitatively shown": "Yes, quantitatively shown",
            "yes, mentioned without data": "Yes, but not shown",
            "yes, but not shown": "Yes, but not shown",
            "yes": "Yes, but not shown",
            "no evidence found": "No",
            "no": "No",
        },
    ),
    "Q6": _NormalisationRule(
        question_id="Q6",
        mapping={
            "primary focus of the paper": "Yes",
            "mentioned in passing": "No",
            "not mentioned": "No",
            "yes": "Yes",
            "no": "No",
        },
    ),
    "Q7": _NormalisationRule(
        question_id="Q7",
        mapping={
            "primary focus of the paper": "Yes",
            "mentioned in passing": "No",
            "not mentioned": "No",
            "yes": "Yes",
            "no": "No",
        },
    ),
    "Q8": _NormalisationRule(
        question_id="Q8",
        mapping={
            "link supported by data": "Yes",
            "link is speculative": "No",
            "no link reported": "No",
            "yes": "Yes",
            "no": "No",
        },
    ),
    "Q9": _NormalisationRule(
        question_id="Q9",
        mapping={
            "experimental evidence presented": "Yes",
            "observational evidence presented": "Yes",
            "no evidence presented": "No",
            "yes": "Yes",
            "no": "No",
        },
    ),
}


def _clean_answer(answer: str) -> str:
    text = answer.strip()
    if not text:
        return ""
    for token in _SEPARATOR_TOKENS:
        if token in text:
            text = text.split(token, 1)[0].strip()
            break
    return text


def normalize_answer(question_id: str, answer: str | None) -> str:
    """Map detailed extraction answers to the public scoreboard categories."""

    if not answer:
        return ""
    cleaned = _clean_answer(str(answer))
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    rule = _QUESTION_RULES.get(question_id)
    if rule:
        mapped = rule.mapping.get(lowered)
        if mapped is not None:
            return mapped
    # Fall back to simple Yes/No capitalisation when possible.
    if lowered == "yes":
        return "Yes"
    if lowered == "no":
        return "No"
    return cleaned


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

_CORE_COLUMNS: Sequence[str] = ("theory_id", "paper_url", "paper_name", "paper_year")


def prepare_normalised_row(row: Mapping[str, str]) -> dict[str, str]:
    """Return a question row with normalised answers and trimmed metadata."""

    result = {column: (row.get(column, "") or "").strip() for column in _CORE_COLUMNS}
    for question in QUESTION_COLUMNS:
        result[question] = normalize_answer(question, row.get(question))
    return result


def prepare_theory_table(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return sorted theory rows with consistent formatting."""

    entries: list[tuple[int, str, str, str, str]] = []
    for row in rows:
        theory_id = (row.get("theory_id", "") or "").strip()
        if not theory_id:
            continue
        theory_name = (row.get("theory_name", "") or "").strip()
        count_raw = (row.get("number_of_collected_papers", "") or "").strip()
        try:
            count_value = int(count_raw)
        except ValueError:
            count_value = 0
        display_count = count_raw if count_raw else (str(count_value) if count_value else "")
        entries.append((count_value, theory_name.lower(), theory_name, theory_id, display_count))

    entries.sort(key=lambda item: (-item[0], item[1], item[3]))

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for count_value, _, theory_name, theory_id, display_count in entries:
        if theory_id in seen:
            continue
        seen.add(theory_id)
        result.append(
            {
                "theory_id": theory_id,
                "theory_name": theory_name,
                "number_of_collected_papers": display_count if display_count else str(count_value),
            }
        )
    return result


def _parse_year(value: str) -> int:
    text = value.strip()
    if not text:
        return 9999
    try:
        return int(text)
    except ValueError:
        return 9999


def prepare_collected_papers(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Return a per-theory paper table derived from question rows."""

    seen: set[tuple[str, str]] = set()
    collected: list[dict[str, str]] = []
    for row in rows:
        theory_id = (row.get("theory_id", "") or "").strip()
        paper_url = (row.get("paper_url", "") or "").strip()
        if not theory_id or not paper_url:
            continue
        key = (theory_id, paper_url)
        if key in seen:
            continue
        seen.add(key)
        paper_name = (row.get("paper_name", "") or "").strip()
        paper_year = (row.get("paper_year", "") or "").strip()
        collected.append(
            {
                "theory_id": theory_id,
                "paper_url": paper_url,
                "paper_name": paper_name,
                "paper_year": paper_year,
            }
        )

    collected.sort(key=lambda item: (item["theory_id"], _parse_year(item["paper_year"]), item["paper_name"].lower()))
    return collected


def prepare_normalised_answers(rows: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Normalise all question answers and provide stable ordering."""

    normalised = [prepare_normalised_row(row) for row in rows]
    normalised.sort(key=lambda item: (item["theory_id"], item["paper_url"]))
    return normalised


__all__ = [
    "normalize_answer",
    "prepare_collected_papers",
    "prepare_normalised_answers",
    "prepare_normalised_row",
    "prepare_theory_table",
]
