"""Compatibility wrapper around :mod:`scripts.run_pipeline`.

The project historically provided ``scripts/run_full_pipeline.py`` as the main
entrypoint for orchestrating the literature workflow. The richer
``scripts/run_pipeline.py`` supersedes it, but we keep this module as a thin
shim so existing docs and automation continue to work without modification.
"""

from __future__ import annotations

from run_pipeline import main as _run_pipeline_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to :func:`scripts.run_pipeline.main` for backwards compatibility."""

    return _run_pipeline_main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
