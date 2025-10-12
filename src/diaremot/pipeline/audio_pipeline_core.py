"""Compatibility shim that re-exports the pipeline implementation pieces."""

from __future__ import annotations

import os
import sys

from . import cli_entry
from .cli_entry import _args_to_config, _build_arg_parser, main
from .config import (
    CORE_DEPENDENCY_REQUIREMENTS,
    DEFAULT_PIPELINE_CONFIG,
    build_pipeline_config,
    dependency_health_summary,
)
from .logging_utils import (
    CoreLogger,
    JSONLWriter,
    RunStats,
    StageGuard,
    _fmt_hms,
    _fmt_hms_ms,
)
from .orchestrator import (
    AudioAnalysisPipelineV2,
    clear_pipeline_cache,
    diagnostics,
    resume,
    run_pipeline,
    verify_dependencies,
)
from .outputs import (
    SEGMENT_COLUMNS,
    default_affect,
    ensure_segment_keys,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_speakers_summary,
    write_timeline_csv,
)
from .speaker_diarization import SpeakerDiarizer

__all__ = [
    "AudioAnalysisPipelineV2",
    "CORE_DEPENDENCY_REQUIREMENTS",
    "DEFAULT_PIPELINE_CONFIG",
    "JSONLWriter",
    "CoreLogger",
    "RunStats",
    "StageGuard",
    "SpeakerDiarizer",
    "SEGMENT_COLUMNS",
    "_args_to_config",
    "_build_arg_parser",
    "_fmt_hms",
    "_fmt_hms_ms",
    "build_pipeline_config",
    "clear_pipeline_cache",
    "default_affect",
    "dependency_health_summary",
    "diagnostics",
    "ensure_segment_keys",
    "main",
    "resume",
    "run_pipeline",
    "verify_dependencies",
    "write_qc_report",
    "write_segments_csv",
    "write_segments_jsonl",
    "write_speakers_summary",
    "write_timeline_csv",
]


if __name__ == "__main__":  # pragma: no cover - exercised via explicit tests
    argv = sys.argv[1:]
    if os.environ.get("PYTEST_CURRENT_TEST") and argv and all(arg.startswith("-") for arg in argv):
        argv = []
    sys.exit(cli_entry.main(argv))
