"""Balance theory assignments in a generated ontology payload.

This helper implements the "optimization ontology" stage that follows the
LLM-driven ontology synthesis.  It enforces the competition rule requiring each
theory to reference between one and four supporting publications (three being
the preferred target) and redistributes ambiguous articles to theories with
fewer supporting papers.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from theories_pipeline.llm import LLMClient, LLMClientConfig
from theories_pipeline.ontology_optimization import optimise_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the ontology JSON file")
    parser.add_argument(
        "--output",
        help=(
            "Optional output path. When omitted the optimisation is applied in-place "
            "and the input file is overwritten."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3,
        help="Preferred number of papers per theory (defaults to 3)",
    )
    parser.add_argument(
        "--minimum",
        type=int,
        default=1,
        help="Minimum allowed number of papers per theory",
    )
    parser.add_argument(
        "--maximum",
        type=int,
        default=4,
        help="Maximum allowed number of papers per theory",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-5-mini",
        help="LLM model identifier used for ontology splitting (set to 'none' to disable)",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature supplied to the LLM",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=4,
        help="Number of theories to evaluate per LLM request",
    )
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent LLM requests",
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=Path("data/cache/llm"),
        help="Directory used to cache LLM responses",
    )
    parser.add_argument(
        "--llm-api-key",
        help="API key for the OpenAI client (defaults to OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for failed LLM calls",
    )
    parser.add_argument(
        "--llm-retry-backoff",
        type=float,
        default=2.0,
        help="Backoff multiplier used between LLM retries",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Request timeout for the LLM client in seconds",
    )
    return parser


def _maybe_build_llm_client(args: argparse.Namespace) -> LLMClient | None:
    model = args.llm_model
    if not model or model.lower() == "none":
        return None

    cache_dir = Path(args.llm_cache_dir)
    config = LLMClientConfig(
        model=model,
        temperature=float(args.llm_temperature),
        batch_size=max(1, int(args.llm_batch_size)),
        max_retries=max(0, int(args.llm_max_retries)),
        retry_backoff=float(args.llm_retry_backoff),
        request_timeout=float(args.llm_timeout),
        cache_dir=cache_dir,
    )

    api_key = args.llm_api_key or os.environ.get("OPENAI_API_KEY")
    return LLMClient(config, api_key=api_key)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    llm_client = _maybe_build_llm_client(args)

    summary = optimise_file(
        args.input,
        args.output,
        target=args.target,
        minimum=args.minimum,
        maximum=args.maximum,
        llm_client=llm_client,
        llm_model=None if llm_client is None else args.llm_model,
        llm_temperature=args.llm_temperature if llm_client is not None else None,
        batch_size=args.llm_batch_size,
        max_concurrency=args.llm_max_concurrency,
    )

    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

