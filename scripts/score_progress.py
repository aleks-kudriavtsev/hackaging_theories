"""Summarise pipeline exports into progress and quality reports."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:  # pragma: no cover - convenience for scripts
    sys.path.insert(0, str(SRC_PATH))

from itertools import zip_longest

from theories_pipeline import outputs as outputs_module  # noqa: E402

QUESTION_COLUMNS = tuple(outputs_module.QUESTION_COLUMNS)
_CONFIDENCE_COLUMNS = tuple(getattr(outputs_module, "QUESTION_CONFIDENCE_COLUMNS", ()))
QUESTION_CONFIDENCE_LOOKUP = {
    question: confidence
    for question, confidence in zip_longest(
        QUESTION_COLUMNS, _CONFIDENCE_COLUMNS, fillvalue=None
    )
    if question is not None and confidence
}

logger = logging.getLogger(__name__)


@dataclass
class QuestionMetrics:
    """Aggregated statistics for a single question column."""

    question_id: str
    answered: int
    yes_count: int
    blank_count: int
    confidences: Sequence[float]

    @property
    def yes_ratio(self) -> float:
        if not self.answered:
            return 0.0
        return self.yes_count / self.answered

    @property
    def average_confidence(self) -> float | None:
        if not self.confidences:
            return None
        return sum(self.confidences) / len(self.confidences)

    @property
    def min_confidence(self) -> float | None:
        if not self.confidences:
            return None
        return min(self.confidences)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV export at {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        logger.debug("Unable to parse integer from %s", text)
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        logger.debug("Unable to parse float from %s", text)
        return None


def _compute_log_score(
    rows: Iterable[Mapping[str, str]]
) -> tuple[float, list[Dict[str, Any]]]:
    total = 0.0
    contributions: list[Dict[str, Any]] = []
    for row in rows:
        count = _parse_int(row.get("number_of_collected_papers"))
        if not count or count <= 0:
            continue
        log_value = math.log10(count)
        contributions.append(
            {
                "theory_id": row.get("theory_id"),
                "theory_name": row.get("theory_name"),
                "count": count,
                "log10_contribution": log_value,
            }
        )
        total += log_value

    contributions.sort(key=lambda item: item["log10_contribution"], reverse=True)

    for entry in contributions:
        if total > 0:
            entry["log10_share"] = entry["log10_contribution"] / total
        else:
            entry["log10_share"] = 0.0

    return total, contributions


def _collect_deficits(
    rows: Sequence[Mapping[str, str]]
) -> tuple[List[Dict[str, Any]], bool]:
    has_target_column = any("target" in row for row in rows)
    has_deficit_column = any("deficit" in row for row in rows)
    if not has_target_column or not has_deficit_column:
        return [], False

    deficits: List[Dict[str, Any]] = []
    for row in rows:
        target = _parse_int(row.get("target"))
        deficit = _parse_int(row.get("deficit"))
        if target is None or deficit is None or deficit <= 0:
            continue
        deficits.append(
            {
                "theory_id": row.get("theory_id"),
                "theory_name": row.get("theory_name"),
                "target": target,
                "count": _parse_int(row.get("number_of_collected_papers")) or 0,
                "deficit": deficit,
            }
        )
    deficits.sort(key=lambda item: (-item["deficit"], item["theory_name"] or ""))
    return deficits, True


def _resolve_confidence_column(
    question_id: str, available_fields: Collection[str]
) -> str | None:
    direct = QUESTION_CONFIDENCE_LOOKUP.get(question_id)
    if direct and direct in available_fields:
        return direct

    fallback = f"{question_id}_confidence"
    if fallback in available_fields:
        return fallback
    return None


def _summarise_questions(rows: Sequence[Mapping[str, str]]) -> Dict[str, QuestionMetrics]:
    metrics: Dict[str, QuestionMetrics] = {}
    available_fields: set[str] = set()
    for row in rows:
        available_fields.update(row.keys())

    for question_id in QUESTION_COLUMNS:
        confidence_column = _resolve_confidence_column(question_id, available_fields)
        answered = 0
        yes_count = 0
        blank_count = 0
        confidences: List[float] = []
        for row in rows:
            answer = (row.get(question_id) or "").strip()
            if not answer:
                blank_count += 1
            else:
                answered += 1
                if answer.lower().startswith("yes"):
                    yes_count += 1
            if confidence_column:
                confidence_value = _parse_float(row.get(confidence_column))
                if confidence_value is not None:
                    confidences.append(confidence_value)
        metrics[question_id] = QuestionMetrics(
            question_id=question_id,
            answered=answered,
            yes_count=yes_count,
            blank_count=blank_count,
            confidences=tuple(confidences),
        )
    return metrics


def _build_markdown(
    log_score: float,
    log_contributions: Sequence[Mapping[str, Any]],
    theory_rows: Sequence[Mapping[str, str]],
    question_metrics: Mapping[str, QuestionMetrics],
    deficits: Sequence[Mapping[str, Any]],
    deficit_data_available: bool,
    question_row_count: int,
    timestamp: str,
) -> str:
    lines = ["# Progress Report", ""]
    lines.append(f"_Generated: {timestamp}_")
    lines.append("")
    lines.append(f"- Σ log₁₀(papers per theory): **{log_score:.3f}**")
    lines.append(f"- Theories evaluated: **{len(theory_rows)}**")
    lines.append(f"- Recorded question rows: **{question_row_count}**")
    lines.append("")

    if question_metrics:
        lines.append("## Question Coverage")
        lines.append("")
        lines.append("| Question | Answered | Yes % | Avg confidence | Blank |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for question_id in QUESTION_COLUMNS:
            metric = question_metrics[question_id]
            yes_pct = metric.yes_ratio * 100
            avg_conf = metric.average_confidence
            avg_display = f"{avg_conf:.3f}" if avg_conf is not None else "n/a"
            lines.append(
                f"| {question_id} | {metric.answered} | {yes_pct:.1f}% | {avg_display} | {metric.blank_count} |"
            )
        lines.append("")

    if log_contributions:
        lines.append("## Σ log₁₀ Contributions")
        lines.append("")
        lines.append("| Theory | Papers | log₁₀ | Share |")
        lines.append("| --- | ---: | ---: | ---: |")
        for entry in log_contributions:
            name = entry.get("theory_name") or entry.get("theory_id") or "Unknown"
            share = float(entry.get("log10_share") or 0.0) * 100
            lines.append(
                f"| {name} | {entry.get('count', 0)} | {entry.get('log10_contribution', 0.0):.3f} | {share:.1f}% |"
            )
        lines.append("")

    lines.append("## Target Deficits")
    lines.append("")
    if not deficit_data_available:
        lines.append("Target/deficit columns not present in export.")
    elif deficits:
        lines.append("| Theory | Papers | Target | Deficit |")
        lines.append("| --- | ---: | ---: | ---: |")
        for entry in deficits:
            name = entry.get("theory_name") or entry.get("theory_id") or "Unknown"
            lines.append(
                f"| {name} | {entry['count']} | {entry['target']} | {entry['deficit']} |"
            )
    else:
        lines.append("All theory targets are currently met.")
    lines.append("")
    return "\n".join(lines)


def generate_progress_report(
    theories_path: Path,
    questions_path: Path,
    report_dir: Path,
    *,
    confidence_threshold: float = 0.6,
) -> Dict[str, Any]:
    """Compute aggregate metrics from exported CSV artefacts."""

    theory_rows = _read_csv(theories_path)
    question_rows = _read_csv(questions_path)

    log_score, log_contributions = _compute_log_score(theory_rows)
    deficits, deficit_data_available = _collect_deficits(theory_rows)
    question_metrics = _summarise_questions(question_rows)

    alerts: List[str] = []
    for question_id, metric in question_metrics.items():
        avg_conf = metric.average_confidence
        min_conf = metric.min_confidence
        if avg_conf is not None and avg_conf < confidence_threshold:
            message = (
                f"Average confidence for {question_id} ({avg_conf:.3f}) below threshold "
                f"{confidence_threshold:.3f}."
            )
            logger.warning(message)
            alerts.append(message)
        elif min_conf is not None and min_conf < confidence_threshold:
            message = (
                f"Minimum confidence for {question_id} ({min_conf:.3f}) below threshold "
                f"{confidence_threshold:.3f}."
            )
            logger.warning(message)
            alerts.append(message)
        if metric.blank_count > 0:
            message = f"Detected {metric.blank_count} blank answers for {question_id}."
            logger.warning(message)
            alerts.append(message)
        if metric.answered == 0 and metric.blank_count == 0:
            message = f"No responses recorded for {question_id}."
            logger.warning(message)
            alerts.append(message)

    timestamp = datetime.now(timezone.utc).isoformat()
    report_payload: Dict[str, Any] = {
        "generated_at": timestamp,
        "log_score": log_score,
        "theory_count": len(theory_rows),
        "question_row_count": len(question_rows),
        "question_metrics": {
            question_id: {
                "answered": metric.answered,
                "yes_ratio": metric.yes_ratio,
                "blank_count": metric.blank_count,
                "average_confidence": metric.average_confidence,
                "min_confidence": metric.min_confidence,
            }
            for question_id, metric in question_metrics.items()
        },
        "log_score_breakdown": log_contributions,
        "deficits": deficits,
        "deficit_data_available": deficit_data_available,
        "alerts": alerts,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "progress_report.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    markdown = _build_markdown(
        log_score,
        log_contributions,
        theory_rows,
        question_metrics,
        deficits,
        deficit_data_available,
        len(question_rows),
        timestamp,
    )
    markdown_path = report_dir / "progress_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    logger.info("Progress report saved to %s and %s", json_path, markdown_path)
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("data/pipeline"),
        help="Directory containing exported CSV artefacts (defaults to data/pipeline).",
    )
    parser.add_argument(
        "--theories",
        type=Path,
        help="Path to theories.csv (defaults to <workdir>/theories.csv).",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        help="Path to questions.csv (defaults to <workdir>/questions.csv).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory for generated reports (defaults to <workdir>/reports).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Alert threshold for average question confidence.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    workdir = args.workdir if isinstance(args.workdir, Path) else Path(args.workdir)
    theories_path = args.theories or (workdir / "theories.csv")
    questions_path = args.questions or (workdir / "questions.csv")
    report_dir = args.report_dir or (workdir / "reports")

    try:
        generate_progress_report(
            theories_path,
            questions_path,
            report_dir,
            confidence_threshold=float(args.confidence_threshold),
        )
    except Exception as exc:  # pragma: no cover - CLI robustness
        logger.error("Failed to generate progress report: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
