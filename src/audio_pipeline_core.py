"""Compatibility shim for legacy imports.

This module re-exports the public API from ``diaremot.pipeline.audio_pipeline_core``
so that older entrypoints importing ``audio_pipeline_core`` from the repository
root continue to function. The real implementation lives under ``src/``.
"""
from __future__ import annotations

import os
import sys

import diaremot.pipeline.audio_pipeline_core as _core
from diaremot.pipeline import cli_entry as _cli_entry
from diaremot.pipeline.audio_pipeline_core import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", [])


if __name__ == "__main__":  # pragma: no cover - exercised via explicit tests
    argv = sys.argv[1:]
    if os.environ.get("PYTEST_CURRENT_TEST"):
        argv = []
    sys.exit(_cli_entry.main(argv))
