"""Validation helpers for question-answer exports."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

QUESTION_IDS: tuple[str, ...] = tuple(f"Q{index}" for index in range(1, 10))
UNSPECIFIED_THEORY = "<unspecified>"


def _normalise_question_id(question_id: str) -> str:
    text = (question_id or "").strip()
    if not text:
        raise ValueError("Ground-truth row is missing a question identifier")
    upper = text.upper()
    if upper not in QUESTION_IDS:
        raise ValueError(f"Unsupported question identifier: {question_id!r}")
    return upper


def _normalise_answer_label(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    return text


@dataclass
class QuestionRow:
    """Single entry from questions.csv."""

    theory_id: str
    paper_id: str
    paper_name: str | None
    answers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GroundTruthEntry:
    """Expected answer for a paper/question pair."""

    theory_id: str | None
    paper_id: str
    question_id: str
    expected_answer: str


@dataclass
class Mismatch:
    """Details of an answer mismatch."""

    theory_id: str
    paper_id: str
    question_id: str
    expected_label: str
    actual_label: str
    actual_text: str


@dataclass
class TheoryMetrics:
    """Aggregated validation metrics for a single theory."""

    theory_id: str
    expected: int = 0
    found: int = 0
    correct: int = 0

    @property
    def missing(self) -> int:
        return max(self.expected - self.found, 0)

    @property
    def incorrect(self) -> int:
        return max(self.found - self.correct, 0)

    @property
    def accuracy(self) -> float:
        if self.found == 0:
            return 0.0
        return self.correct / self.found

    @property
    def recall(self) -> float:
        if self.expected == 0:
            return 0.0
        return self.found / self.expected

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "expected": self.expected,
            "found": self.found,
            "correct": self.correct,
            "missing": self.missing,
            "incorrect": self.incorrect,
            "accuracy": self.accuracy,
            "recall": self.recall,
        }


@dataclass
class ValidationReport:
    """Full validation results for a question export."""

    overall_expected: int
    overall_found: int
    overall_correct: int
    per_theory: Dict[str, TheoryMetrics]
    mismatches: List[Mismatch]
    missing_entries: List[GroundTruthEntry]

    @property
    def overall_recall(self) -> float:
        if self.overall_expected == 0:
            return 0.0
        return self.overall_found / self.overall_expected

    @property
    def overall_accuracy(self) -> float:
        if self.overall_found == 0:
            return 0.0
        return self.overall_correct / self.overall_found

    @property
    def has_failures(self) -> bool:
        return bool(self.mismatches or self.missing_entries)

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall": {
                "expected": self.overall_expected,
                "found": self.overall_found,
                "correct": self.overall_correct,
                "recall": self.overall_recall,
                "accuracy": self.overall_accuracy,
            },
            "per_theory": {
                theory: metrics.to_dict() for theory, metrics in sorted(self.per_theory.items())
            },
            "missing": [entry.__dict__ for entry in self.missing_entries],
            "mismatches": [
                {
                    "theory_id": mismatch.theory_id,
                    "paper_id": mismatch.paper_id,
                    "question_id": mismatch.question_id,
                    "expected": mismatch.expected_label,
                    "actual_label": mismatch.actual_label,
                    "actual_text": mismatch.actual_text,
                }
                for mismatch in self.mismatches
            ],
        }


def load_questions(path: Path) -> List[QuestionRow]:
    """Load the pipeline question export from ``questions.csv``."""

    rows: List[QuestionRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            theory_id = (raw.get("theory_id") or "").strip()
            paper_id = (raw.get("paper_url") or raw.get("paper_id") or "").strip()
            if not paper_id:
                continue
            answers = {qid: (raw.get(qid) or "").strip() for qid in QUESTION_IDS}
            row = QuestionRow(
                theory_id=theory_id or UNSPECIFIED_THEORY,
                paper_id=paper_id,
                paper_name=(raw.get("paper_name") or "").strip() or None,
                answers=answers,
            )
            rows.append(row)
    return rows


def _load_ground_truth_from_csv(path: Path) -> List[GroundTruthEntry]:
    entries: List[GroundTruthEntry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            paper_id = (raw.get("paper_url") or raw.get("paper_id") or raw.get("identifier") or "").strip()
            if not paper_id:
                continue
            theory_id = (raw.get("theory_id") or raw.get("theory") or "").strip() or None
            question_id = _normalise_question_id(raw.get("question_id") or raw.get("question") or "")
            expected = (raw.get("expected_answer") or raw.get("answer") or "").strip()
            entries.append(
                GroundTruthEntry(
                    theory_id=theory_id,
                    paper_id=paper_id,
                    question_id=question_id,
                    expected_answer=expected,
                )
            )
    return entries


def _load_ground_truth_from_json(path: Path) -> List[GroundTruthEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        if "ground_truth" in payload and isinstance(payload["ground_truth"], Sequence):
            records = payload["ground_truth"]
        elif "entries" in payload and isinstance(payload["entries"], Sequence):
            records = payload["entries"]
        elif "data" in payload and isinstance(payload["data"], Sequence):
            records = payload["data"]
        else:
            raise ValueError("Unsupported JSON structure for ground-truth file")
    elif isinstance(payload, Sequence):
        records = payload
    else:
        raise ValueError("Ground-truth JSON must be a sequence or contain a sequence under known keys")

    entries: List[GroundTruthEntry] = []
    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        paper_id = (
            raw.get("paper_url")
            or raw.get("paper_id")
            or raw.get("identifier")
            or raw.get("id")
            or ""
        )
        paper_id = str(paper_id).strip()
        if not paper_id:
            continue
        theory_id_value = raw.get("theory_id") or raw.get("theory") or None
        theory_id = str(theory_id_value).strip() if theory_id_value else None
        question_id = _normalise_question_id(str(raw.get("question_id") or raw.get("question") or ""))
        expected_value = raw.get("expected_answer") or raw.get("answer") or ""
        expected = str(expected_value).strip()
        entries.append(
            GroundTruthEntry(
                theory_id=theory_id,
                paper_id=paper_id,
                question_id=question_id,
                expected_answer=expected,
            )
        )
    return entries


def load_ground_truth(path: Path) -> List[GroundTruthEntry]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_ground_truth_from_csv(path)
    if suffix == ".json":
        return _load_ground_truth_from_json(path)
    raise ValueError(f"Unsupported ground-truth format: {path.suffix}")


def _index_questions(rows: Iterable[QuestionRow]) -> tuple[
    Dict[tuple[str, str], QuestionRow],
    MutableMapping[str, List[QuestionRow]],
]:
    direct: Dict[tuple[str, str], QuestionRow] = {}
    by_paper: Dict[str, List[QuestionRow]] = {}
    for row in rows:
        key = (row.theory_id, row.paper_id)
        direct[key] = row
        by_paper.setdefault(row.paper_id, []).append(row)
    return direct, by_paper


def validate(questions: Sequence[QuestionRow], ground_truth: Sequence[GroundTruthEntry]) -> ValidationReport:
    direct_index, paper_index = _index_questions(questions)

    per_theory: Dict[str, TheoryMetrics] = {}
    mismatches: List[Mismatch] = []
    missing: List[GroundTruthEntry] = []
    overall_expected = 0
    overall_found = 0
    overall_correct = 0

    for entry in ground_truth:
        overall_expected += 1
        theory_key = entry.theory_id or UNSPECIFIED_THEORY
        metrics = per_theory.setdefault(theory_key, TheoryMetrics(theory_key))
        metrics.expected += 1

        row = None
        if entry.theory_id:
            row = direct_index.get((entry.theory_id, entry.paper_id))
        if row is None:
            candidates = paper_index.get(entry.paper_id, [])
            if entry.theory_id:
                for candidate in candidates:
                    if candidate.theory_id == entry.theory_id:
                        row = candidate
                        break
            if row is None and candidates:
                row = candidates[0]
                if theory_key == UNSPECIFIED_THEORY:
                    theory_key = row.theory_id
                    metrics = per_theory.setdefault(theory_key, TheoryMetrics(theory_key))
                    metrics.expected += 1
                    per_theory[UNSPECIFIED_THEORY].expected -= 1
        if row is None:
            missing.append(entry)
            continue

        metrics.found += 1
        overall_found += 1

        actual_text = row.answers.get(entry.question_id, "")
        actual_label = _normalise_answer_label(actual_text)
        expected_label = _normalise_answer_label(entry.expected_answer)
        if actual_label == expected_label:
            metrics.correct += 1
            overall_correct += 1
        else:
            mismatches.append(
                Mismatch(
                    theory_id=row.theory_id,
                    paper_id=row.paper_id,
                    question_id=entry.question_id,
                    expected_label=expected_label,
                    actual_label=actual_label,
                    actual_text=actual_text,
                )
            )

    if UNSPECIFIED_THEORY in per_theory and per_theory[UNSPECIFIED_THEORY].expected == 0:
        per_theory.pop(UNSPECIFIED_THEORY)

    return ValidationReport(
        overall_expected=overall_expected,
        overall_found=overall_found,
        overall_correct=overall_correct,
        per_theory=per_theory,
        mismatches=mismatches,
        missing_entries=missing,
    )


def validate_from_paths(questions_path: Path, ground_truth_path: Path) -> ValidationReport:
    questions = load_questions(questions_path)
    ground_truth = load_ground_truth(ground_truth_path)
    return validate(questions, ground_truth)


def format_report(report: ValidationReport) -> str:
    overall_found_denom = report.overall_found or 1
    lines = [
        "Question validation summary:",
        f"  Overall recall: {report.overall_recall:.3f} ({report.overall_found}/{report.overall_expected})",
        f"  Overall accuracy: {report.overall_accuracy:.3f} ({report.overall_correct}/{overall_found_denom})",
    ]
    if report.per_theory:
        lines.append("  Per-theory metrics:")
        for theory, metrics in sorted(report.per_theory.items()):
            found_denom = metrics.found or 1
            lines.append(
                f"    - {theory}: recall={metrics.recall:.3f} ({metrics.found}/{metrics.expected}), "
                f"accuracy={metrics.accuracy:.3f} ({metrics.correct}/{found_denom})"
            )
    if report.missing_entries:
        lines.append("  Missing entries:")
        for entry in report.missing_entries:
            lines.append(
                f"    - theory={entry.theory_id or UNSPECIFIED_THEORY} paper={entry.paper_id} question={entry.question_id}"
            )
    if report.mismatches:
        lines.append("  Answer mismatches:")
        for mismatch in report.mismatches:
            lines.append(
                "    - theory={theory} paper={paper} question={question}: expected={expected!r}, actual={actual!r}".format(
                    theory=mismatch.theory_id,
                    paper=mismatch.paper_id,
                    question=mismatch.question_id,
                    expected=mismatch.expected_label,
                    actual=mismatch.actual_label or mismatch.actual_text,
                )
            )
    return "\n".join(lines)


def write_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

