#!/usr/bin/env python3
"""Lightweight diagnostics for ensuring the DiaRemot pipeline is runnable.

The script assumes **faster-whisper** as the default ASR backend and will
fall back to OpenAI's ``whisper`` package when available.  It performs three
stages:

1. **Dependency check** – import core Python packages and report versions.
2. **Model availability** – ensure critical ONNX models are present.
3. **Smoke test** – optionally run a 1‑second synthetic audio sample through
   the full pipeline.

Usage::

    python pipeline_healthcheck.py [--skip-models] [--skip-smoke] [--local-only]

By default all checks are executed.  Use the flags to skip individual steps or
require models to be available offline.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .audio_pipeline_core import (
    CORE_DEPENDENCY_REQUIREMENTS,
)
from .audio_pipeline_core import (
    diagnostics as core_diagnostics,
)
from .diagnostics_smoke import run_pipeline_smoke_test

# Optional fallbacks that are not required by the core dependency set
OPTIONAL_DEPENDENCIES = ("whisper",)

# Minimal set of ONNX models used by the pipeline
MODELS = {
    "speech_emotion": "hf://Dpngtm/wav2vec2-emotion-recognition/model.onnx",
    "vad": "hf://snakers4/silero-vad/silero_vad.onnx",
    "speaker_embedding": "hf://speechbrain/spkrec-ecapa-voxceleb/embedding_model.onnx",
}


def check_dependencies() -> dict[str, str]:
    """Return the version/health of each core dependency."""

    diag = core_diagnostics(require_versions=True)
    summary = diag.get("summary", {})

    report: dict[str, str] = {}

    for pkg, min_version in CORE_DEPENDENCY_REQUIREMENTS.items():
        entry = summary.get(pkg, {})
        status = entry.get("status", "ok")
        version = entry.get("version")
        issue = entry.get("issue")

        if status == "ok":
            if version:
                report[pkg] = version
            elif min_version:
                report[pkg] = f"unknown (requires >= {min_version})"
            else:
                report[pkg] = "unknown"
            continue

        detail = issue or status
        if version:
            detail = f"{detail}; installed {version}"
        if min_version:
            detail = f"{detail}; requires >= {min_version}"
        report[pkg] = f"{status}: {detail}"

    # Optional dependencies remain best-effort imports
    for pkg in OPTIONAL_DEPENDENCIES:
        try:
            module = importlib.import_module(pkg)
            report[pkg] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostics only
            report[pkg] = f"missing ({exc})"

    return report


def check_models(local_only: bool = False) -> dict[str, str]:
    """Ensure required ONNX models are available."""
    from ..io.onnx_utils import ensure_onnx_model

    report: dict[str, str] = {}
    for name, src in MODELS.items():
        try:
            path = ensure_onnx_model(src, local_files_only=local_only)
            report[name] = str(path)
        except Exception as exc:  # pragma: no cover - network issues etc.
            report[name] = f"missing ({exc})"
    return report


def run_smoke_test() -> dict[str, str]:
    """Run a very small end‑to‑end pipeline test."""

    result = run_pipeline_smoke_test(tmp_dir=Path("healthcheck_tmp"))
    if result.success:
        output = str(result.output_dir) if result.output_dir else ""
        return {"success": True, "output": output}

    return {"success": False, "error": result.error or "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaRemo pipeline health check")
    parser.add_argument("--skip-models", action="store_true", help="skip model availability checks")
    parser.add_argument(
        "--skip-smoke", action="store_true", help="skip running the pipeline smoke test"
    )
    parser.add_argument("--local-only", action="store_true", help="do not download missing models")
    args = parser.parse_args()

    report = {"dependencies": check_dependencies()}
    if not args.skip_models:
        report["models"] = check_models(local_only=args.local_only)
    if not args.skip_smoke:
        report["smoke_test"] = run_smoke_test()

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
