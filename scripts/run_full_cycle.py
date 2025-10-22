"""Regenerate the review artefacts and enrich them with the ontology collector."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

if __package__ is None:  # pragma: no cover - convenience for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts import collect_theories, run_pipeline, score_progress

logger = logging.getLogger(__name__)


def _normalise_strings(values: Any) -> list[str]:
    if isinstance(values, Mapping):
        items = values.values()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        items = values
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        items = []

    seen: set[str] = set()
    results: list[str] = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(text)
    return results


def _extract_metadata(payload: Mapping[str, Any], ignore: Iterable[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    ignore_set = {key.lower() for key in ignore}
    for key, value in payload.items():
        if key.lower() in ignore_set:
            continue
        metadata[key] = deepcopy(value)
    return metadata


def _resolve_name(node: Mapping[str, Any], *, fallback: str = "") -> str | None:
    for key in ("name", "label", "preferred_label", "theory_id"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return None


def _build_theory_entry(
    theory: Mapping[str, Any],
    *,
    suggestions: Mapping[str, Any] | None,
    default_target: int | None,
) -> tuple[str, Dict[str, Any]] | None:
    name = _resolve_name(theory)
    if not name:
        return None

    entry: Dict[str, Any] = {}
    if default_target is not None:
        entry["target"] = int(default_target)

    direct_suggestions = _normalise_strings(theory.get("suggested_queries"))
    if direct_suggestions:
        entry["suggested_queries"] = direct_suggestions
    else:
        suggestion_payload = suggestions.get("suggested_queries") if isinstance(suggestions, Mapping) else None
        hinted = _normalise_strings(suggestion_payload)
        entry["suggested_queries"] = hinted or [name]

    metadata = _extract_metadata(
        theory,
        ignore={
            "preferred_label",
            "label",
            "theory_id",
            "name",
            "aliases",
            "suggested_queries",
            "queries",
        },
    )
    aliases = theory.get("aliases")
    alias_list = _normalise_strings(aliases)
    if alias_list:
        metadata.setdefault("aliases", alias_list)
    if metadata:
        entry["metadata"] = metadata

    return name, entry


def _build_group_entry(
    group: Mapping[str, Any],
    *,
    suggestion_map: Mapping[str, Any],
    default_target: int | None,
) -> tuple[str, Dict[str, Any]] | None:
    name = _resolve_name(group, fallback="Unnamed group")
    if not name:
        return None

    entry: Dict[str, Any] = {}
    if default_target is not None:
        entry["target"] = int(default_target)

    group_suggestions = suggestion_map.get(name)
    suggestion_queries = []
    if isinstance(group_suggestions, Mapping):
        suggestion_queries = _normalise_strings(group_suggestions.get("suggested_queries"))
    if not suggestion_queries:
        suggestion_queries = _normalise_strings(group.get("suggested_queries")) or [name]
    entry["suggested_queries"] = suggestion_queries

    subtargets: Dict[str, Dict[str, Any]] = {}
    suggestion_children = (
        group_suggestions.get("subtheories")
        if isinstance(group_suggestions, Mapping) and isinstance(group_suggestions.get("subtheories"), Mapping)
        else {}
    )

    subgroups = group.get("subgroups")
    if isinstance(subgroups, Sequence) and not isinstance(subgroups, (str, bytes)):
        for child in subgroups:
            if not isinstance(child, Mapping):
                continue
            child_name = _resolve_name(child)
            converted = _build_group_entry(
                child,
                suggestion_map=suggestion_children,
                default_target=default_target,
            )
            if converted:
                sub_name, sub_entry = converted
                subtargets[sub_name] = sub_entry

    theories = group.get("theories")
    if isinstance(theories, Sequence) and not isinstance(theories, (str, bytes)):
        for theory in theories:
            if not isinstance(theory, Mapping):
                continue
            theory_name = _resolve_name(theory)
            theory_suggestions = suggestion_children.get(theory_name, {}) if theory_name else {}
            converted = _build_theory_entry(
                theory,
                suggestions=theory_suggestions if isinstance(theory_suggestions, Mapping) else {},
                default_target=default_target,
            )
            if converted:
                sub_name, sub_entry = converted
                subtargets[sub_name] = sub_entry

    metadata = _extract_metadata(group, {"name", "label", "suggested_queries", "subgroups", "theories"})
    if metadata:
        entry["metadata"] = metadata

    if subtargets:
        entry["subtheories"] = subtargets

    return name, entry


def _targets_from_ontology(
    ontology_path: Path,
    *,
    default_target: int | None,
) -> Dict[str, Dict[str, Any]]:
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology payload missing at {ontology_path}")

    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    ontology = payload.get("ontology") if isinstance(payload, Mapping) else None
    final = ontology.get("final") if isinstance(ontology, Mapping) else None
    groups = final.get("groups") if isinstance(final, Mapping) else None
    if not isinstance(groups, Sequence):
        return {}

    suggestions = collect_theories.load_ontology_query_suggestions(ontology_path)

    targets: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        converted = _build_group_entry(group, suggestion_map=suggestions, default_target=default_target)
        if converted:
            name, entry = converted
            targets[name] = entry
    return targets


def build_parser() -> argparse.ArgumentParser:
    default_args = run_pipeline.parse_args([])
    parser = argparse.ArgumentParser(
        description=(
            "Run the review harvesting pipeline followed by ontology-guided "
            "literature collection in a single command."
        )
    )
    parser.add_argument(
        "--workdir",
        default="data/pipeline",
        help="Directory to store intermediate pipeline artefacts and collector outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Collector configuration file to hydrate with ontology-derived targets.",
    )
    parser.add_argument(
        "--collector-query",
        default=default_args.collector_query,
        help="Base query string forwarded to the collector stage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional global paper export limit applied during collection.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        help="Restrict retrieval to the specified provider names.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Override the retrieval state directory (defaults to <workdir>/collector_state).",
    )
    parser.add_argument(
        "--default-target",
        type=int,
        default=6,
        help=(
            "Optional per-node retrieval quota applied to ontology-derived targets "
            "(default: 6)."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Discard cached retrieval state before the collector run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of pipeline artefacts even when outputs already exist.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Alert threshold for average question confidence in the progress report.",
    )
    return parser


def _prepare_collector_config(
    config: MutableMapping[str, Any],
    *,
    targets: Mapping[str, Any],
    ontology_path: Path,
    workdir: Path,
) -> None:
    corpus_cfg = config.setdefault("corpus", {})
    corpus_cfg["targets"] = targets
    corpus_cfg["ontology_suggestions_path"] = str(ontology_path)
    corpus_cfg.setdefault("base_query", "")
    corpus_cfg.setdefault("cache_dir", str(workdir / "collector_state"))

    outputs_cfg = config.setdefault("outputs", {})
    outputs_cfg.setdefault("papers", str(workdir / "papers.csv"))
    outputs_cfg.setdefault("theories", str(workdir / "theories.csv"))
    outputs_cfg.setdefault("theory_papers", str(workdir / "theory_papers.csv"))
    outputs_cfg.setdefault("questions", str(workdir / "questions.csv"))
    outputs_cfg.setdefault("cache_dir", str(workdir / "cache"))
    outputs_cfg.setdefault("reports", str(workdir / "reports"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ontology_path = workdir / "aging_ontology.json"

    pipeline_args = ["--workdir", str(workdir), "--collector-query", args.collector_query]
    if args.limit is not None:
        pipeline_args.extend(["--limit", str(args.limit)])
    if args.force:
        pipeline_args.append("--force")

    logger.info("Running review pipeline with args: %s", pipeline_args)
    pipeline_result = run_pipeline.main(pipeline_args)
    if pipeline_result != 0:
        return pipeline_result

    targets = _targets_from_ontology(ontology_path, default_target=args.default_target)
    if not targets:
        logger.warning("No ontology groups discovered in %s; skipping collector run.", ontology_path)
        return 0

    config_path = args.config if isinstance(args.config, Path) else Path(args.config)
    config = collect_theories.load_config(config_path)
    _prepare_collector_config(config, targets=targets, ontology_path=ontology_path, workdir=workdir)

    collector_args = [
        args.collector_query,
        "--config",
        str(config_path),
        "--state-dir",
        str(args.state_dir or (workdir / "collector_state")),
    ]
    if args.limit is not None:
        collector_args.extend(["--limit", str(args.limit)])
    if args.providers:
        collector_args.extend(["--providers", *args.providers])
    if args.no_resume:
        collector_args.append("--no-resume")

    collector_parser = collect_theories.build_parser()
    collector_namespace = collector_parser.parse_args(collector_args)

    result = collect_theories.run_pipeline(
        collector_namespace,
        parser=collector_parser,
        config=config,
        config_path=config_path,
    )

    if result == 0:
        outputs_cfg = config.get("outputs", {}) if isinstance(config, Mapping) else {}
        theories_path = Path(outputs_cfg.get("theories", workdir / "theories.csv"))
        questions_path = Path(outputs_cfg.get("questions", workdir / "questions.csv"))
        reports_dir = Path(outputs_cfg.get("reports", workdir / "reports"))
        try:
            score_progress.generate_progress_report(
                theories_path,
                questions_path,
                reports_dir,
                confidence_threshold=float(args.confidence_threshold),
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to generate progress report")

    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
