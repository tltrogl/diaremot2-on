"""Core orchestration logic for the DiaRemot audio analysis pipeline."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from . import speaker_diarization as _speaker_diarization
from .config import (
    DEFAULT_PIPELINE_CONFIG,
    build_pipeline_config,
)
from .config import (
    diagnostics as config_diagnostics,
)
from .config import (
    verify_dependencies as config_verify_dependencies,
)
from .core.affect_mixin import AffectMixin
from .core.component_factory import ComponentFactoryMixin
from .core.output_mixin import OutputMixin
from .core.paralinguistics_mixin import ParalinguisticsMixin
from .errors import StageExecutionError, coerce_stage_error
from .logging_utils import CoreLogger, RunStats, StageGuard, _fmt_hms_ms
from .pipeline_checkpoint_system import PipelineCheckpointManager, ProcessingStage
from .runtime_env import configure_local_cache_env
from .stages import PIPELINE_STAGES, PipelineState

# Backwards-compatible aliases for test hooks and StageGuard shims
DiarizationConfig = _speaker_diarization.DiarizationConfig
SpeakerDiarizer = _speaker_diarization.SpeakerDiarizer

configure_local_cache_env()
CACHE_VERSION = "v3"  # Incremented to handle new checkpoint logic


__all__ = [
    "AudioAnalysisPipelineV2",
    "build_pipeline_config",
    "run_pipeline",
    "resume",
    "diagnostics",
    "verify_dependencies",
    "clear_pipeline_cache",
]


def clear_pipeline_cache(cache_root: Path | None = None) -> None:
    """Remove cached diarization/transcription artefacts."""

    cache_dir = Path(cache_root) if cache_root else Path(".cache")
    if cache_dir.exists():
        import shutil

        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except PermissionError:
            raise RuntimeError("Could not clear cache directory due to insufficient permissions")
    cache_dir.mkdir(parents=True, exist_ok=True)


def verify_dependencies(strict: bool = False) -> tuple[bool, list[str]]:
    """Expose lightweight dependency verification for external callers."""

    return config_verify_dependencies(strict)


def run_pipeline(
    input_path: str,
    outdir: str,
    *,
    config: dict[str, Any] | None = None,
    clear_cache: bool = False,
) -> dict[str, Any]:
    """Execute the pipeline for ``input_path`` writing artefacts to ``outdir``."""

    if clear_cache:
        try:
            clear_pipeline_cache(Path(config.get("cache_root", ".cache")) if config else None)
        except RuntimeError:
            if config is None:
                config = dict(DEFAULT_PIPELINE_CONFIG)
            config["ignore_tx_cache"] = True

    merged_config = build_pipeline_config(config)
    pipe = AudioAnalysisPipelineV2(merged_config)
    return pipe.process_audio_file(input_path, outdir)


def resume(
    input_path: str,
    outdir: str,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resume a previous run using available checkpoints/caches."""

    merged_config = build_pipeline_config(config)
    merged_config["ignore_tx_cache"] = False
    pipe = AudioAnalysisPipelineV2(merged_config)
    stage, _data, metadata = pipe.checkpoints.get_resume_point(input_path)
    if metadata is not None:
        pipe.corelog.info(
            "Resuming from %s checkpoint created at %s",
            metadata.stage.value if hasattr(metadata.stage, "value") else metadata.stage,
            metadata.timestamp,
        )
    return pipe.process_audio_file(input_path, outdir)


def diagnostics(require_versions: bool = False) -> dict[str, Any]:
    """Return diagnostic information about optional runtime dependencies."""

    return config_diagnostics(require_versions=require_versions)


# Main Pipeline Class
class AudioAnalysisPipelineV2(
    ComponentFactoryMixin,
    AffectMixin,
    ParalinguisticsMixin,
    OutputMixin,
):
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}

        self.run_id = cfg.get("run_id") or (
            time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        self.schema_version = "2.0.0"

        # Paths
        self.log_dir = Path(cfg.get("log_dir", "logs"))
        self.cache_root = Path(cfg.get("cache_root", ".cache"))
        # Support multiple cache roots for reading (first is primary for writes)
        extra_roots = cfg.get("cache_roots", [])
        if isinstance(extra_roots, str | Path):
            extra_roots = [extra_roots]
        self.cache_roots: list[Path] = [self.cache_root] + [Path(p) for p in extra_roots]
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # Persist config for later checks
        self.cfg = dict(cfg)
        self.cache_version = CACHE_VERSION
        self.paralinguistics_module = None

        # Quiet mode env + logging
        self.quiet = bool(cfg.get("quiet", False))
        if self.quiet:
            import os as _os

            _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            _os.environ.setdefault("CT2_VERBOSE", "0")
            try:
                from transformers.utils.logging import set_verbosity_error as _setv

                _setv()
            except Exception:
                pass

        # Logger & Stats
        self.corelog = CoreLogger(
            self.run_id,
            self.log_dir / "run.jsonl",
            console_level=(logging.WARNING if self.quiet else logging.INFO),
        )
        self.stats = RunStats(run_id=self.run_id, file_id="", schema_version=self.schema_version)

        # Checkpoint manager
        self.checkpoints = PipelineCheckpointManager(cfg.get("checkpoint_dir", "checkpoints"))

        # Optional early dependency verification
        if bool(cfg.get("validate_dependencies", False)):
            ok, problems = config_verify_dependencies(
                strict=bool(cfg.get("strict_dependency_versions", False))
            )
            if not ok:
                raise RuntimeError(
                    "Dependency verification failed:\n  - " + "\n  - ".join(problems)
                )

        # Initialize components with error handling
        self._init_components(cfg)


    def process_audio_file(self, input_audio_path: str, out_dir: str) -> dict[str, Any]:
        """Main processing entry point coordinating modular stages."""

        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        self.stats.file_id = Path(input_audio_path).name

        state = PipelineState(input_audio_path=input_audio_path, out_dir=outp)

        try:
            for stage in PIPELINE_STAGES:
                with StageGuard(self.corelog, self.stats, stage.name) as guard:
                    try:
                        stage.runner(self, state, guard)
                    except StageExecutionError:
                        raise
                    except Exception as exc:
                        context = {
                            "stage": stage.name,
                            "file_id": self.stats.file_id,
                            "has_audio": state.y is not None,
                            "has_turns": bool(getattr(state, "turns", None)),
                            "has_transcript": bool(getattr(state, "norm_tx", None)),
                            "audio_sha16": getattr(state, "audio_sha16", None),
                        }
                        raise coerce_stage_error(
                            stage.name,
                            f"Stage '{stage.name}' execution failed",
                            context=context,
                            cause=exc,
                        ) from exc
        except StageExecutionError as exc:
            self.corelog.error(f"Pipeline failed: {exc}")
            raise
        except Exception as exc:
            self.corelog.error(f"Pipeline failed with unhandled error: {exc}")
            if not state.segments_final and state.norm_tx:
                state.segments_final = [
                    self.ensure_segment(seg, self.stats.file_id)
                    for seg in state.norm_tx
                ]
            try:
                self._write_outputs(
                    input_audio_path,
                    outp,
                    state.segments_final,
                    state.speakers_summary,
                    state.health,
                    state.turns,
                    state.overlap_stats,
                    state.per_speaker_interrupts,
                    state.conv_metrics,
                    state.duration_s,
                    state.sed_info,
                )
            except Exception as write_error:
                self.corelog.error(f"Failed to write outputs: {write_error}")

        outputs = {
            "csv": str((outp / "diarized_transcript_with_emotion.csv").resolve()),
            "jsonl": str((outp / "segments.jsonl").resolve()),
            "timeline": str((outp / "timeline.csv").resolve()),
            "summary_html": str((outp / "summary.html").resolve()),
            "summary_pdf": str((outp / "summary.pdf").resolve()),
            "qc_report": str((outp / "qc_report.json").resolve()),
            "speaker_registry": getattr(
                self.diar_conf,
                "registry_path",
                str(Path("registry") / "speaker_registry.json"),
            ),
        }

        if state.sed_info:
            timeline_csv = state.sed_info.get("timeline_csv")
            if timeline_csv:
                outputs["events_timeline"] = str(Path(timeline_csv).resolve())
            timeline_jsonl = state.sed_info.get("timeline_jsonl")
            if timeline_jsonl:
                outputs["events_jsonl"] = str(Path(timeline_jsonl).resolve())

        timeline_csv_fallback = outp / "events_timeline.csv"
        if "events_timeline" not in outputs and timeline_csv_fallback.exists():
            outputs["events_timeline"] = str(timeline_csv_fallback.resolve())
        timeline_jsonl_fallback = outp / "events.jsonl"
        if "events_jsonl" not in outputs and timeline_jsonl_fallback.exists():
            outputs["events_jsonl"] = str(timeline_jsonl_fallback.resolve())

        spk_path = outp / "speakers_summary.csv"
        if spk_path.exists():
            outputs["speakers_summary"] = str(spk_path.resolve())

        manifest = {
            "run_id": self.run_id,
            "file_id": self.stats.file_id,
            "out_dir": str(outp.resolve()),
            "outputs": outputs,
        }

        if getattr(self.stats, "issues", None):
            dedup_issues = sorted({str(issue) for issue in self.stats.issues})
            manifest["issues"] = dedup_issues

        try:
            dep_ok = bool(self.stats.config_snapshot.get("dependency_ok", True))
            dep_summary = self.stats.config_snapshot.get("dependency_summary", {}) or {}
            unhealthy = [k for k, v in dep_summary.items() if v.get("status") != "ok"]
            if dep_ok and not unhealthy:
                self.corelog.info("[deps] All core dependencies loaded successfully.")
            else:
                self.corelog.warn("[deps] Issues detected: " + ", ".join(unhealthy))
            manifest["dependency_ok"] = dep_ok and not unhealthy
            manifest["dependency_unhealthy"] = unhealthy
        except Exception:
            pass

        try:
            if hasattr(self, "tx") and hasattr(self.tx, "get_model_info"):
                tx_info = self.tx.get_model_info()
                manifest["transcriber"] = tx_info
                if tx_info.get("fallback_triggered"):
                    self.corelog.warn(
                        "[tx] Fallback engaged: " + str(tx_info.get("fallback_reason", "unknown"))
                    )
            if "background_sed" in getattr(self.stats, "config_snapshot", {}):
                manifest["background_sed"] = self.stats.config_snapshot.get("background_sed")
        except Exception:
            pass

        self.corelog.event("done", "stop", **manifest)

        try:
            stage_names = [stage.name for stage in PIPELINE_STAGES]
            failures = {f.get("stage"): f for f in getattr(self.stats, "failures", [])}
            self.corelog.info("[ALERT] Stage summary:")
            for st in stage_names:
                if st in failures:
                    failure = failures[st]
                    elapsed_ms = float(failure.get("elapsed_ms", 0.0))
                    self.corelog.warn(
                        f"  - {st}: FAIL in {_fmt_hms_ms(elapsed_ms)} — {failure.get('error')} | Fix: {failure.get('suggestion')}"
                    )
                else:
                    if st in {
                        "paralinguistics",
                        "affect_and_assemble",
                    } and self.stats.config_snapshot.get("transcribe_failed"):
                        self.corelog.warn(f"  - {st}: SKIPPED (transcribe_failed)")
                    else:
                        elapsed_ms = float(self.stats.stage_timings_ms.get(st, 0.0))
                        self.corelog.info(f"  - {st}: PASS in {_fmt_hms_ms(elapsed_ms)}")
        except Exception:
            pass

        self.checkpoints.create_checkpoint(
            input_audio_path,
            ProcessingStage.COMPLETE,
            manifest,
            progress=100.0,
        )
        return manifest

