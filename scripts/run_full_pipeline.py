"""Compatibility wrapper around :mod:`scripts.run_pipeline`.

The project historically provided ``scripts/run_full_pipeline.py`` as the main
entrypoint for orchestrating the literature workflow. The richer
``scripts/run_pipeline.py`` supersedes it, but we keep this module as a thin
shim so existing docs and automation continue to work without modification.
"""

from __future__ import annotations

import os
from pathlib import Path

from run_pipeline import main as _run_pipeline_main


def main(argv: list[str] | None = None) -> int:
    """Delegate to :func:`scripts.run_pipeline.main` for backwards compatibility."""

    project_root = Path(__file__).resolve().parent.parent
    original_cwd = Path.cwd()
    try:
        os.chdir(project_root)
        return _run_pipeline_main(argv)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
