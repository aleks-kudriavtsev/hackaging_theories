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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:  # pragma: no cover - convenience for scripts
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline.outputs import QUESTION_COLUMNS, QUESTION_CONFIDENCE_COLUMNS  # noqa: E402

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


def _compute_log_score(rows: Iterable[Mapping[str, str]]) -> float:
    total = 0.0
    for row in rows:
        count = _parse_int(row.get("number_of_collected_papers"))
        if count and count > 0:
            total += math.log10(count)
    return total


def _collect_deficits(rows: Iterable[Mapping[str, str]]) -> List[Dict[str, Any]]:
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
    return deficits


def _summarise_questions(rows: Iterable[Mapping[str, str]]) -> Dict[str, QuestionMetrics]:
    metrics: Dict[str, QuestionMetrics] = {}
    for question_id, confidence_column in zip(QUESTION_COLUMNS, QUESTION_CONFIDENCE_COLUMNS):
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
    theory_rows: Sequence[Mapping[str, str]],
    question_metrics: Mapping[str, QuestionMetrics],
    deficits: Sequence[Mapping[str, Any]],
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

    lines.append("## Target Deficits")
    lines.append("")
    if deficits:
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

    log_score = _compute_log_score(theory_rows)
    deficits = _collect_deficits(theory_rows)
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
        "deficits": deficits,
        "alerts": alerts,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "progress_report.json"
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    markdown = _build_markdown(
        log_score,
        theory_rows,
        question_metrics,
        deficits,
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
