"""Core orchestration logic for the DiaRemot audio analysis pipeline."""

from __future__ import annotations

import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..affect.intent_defaults import INTENT_LABELS_DEFAULT
from ..summaries.conversation_analysis import ConversationMetrics
from ..summaries.html_summary_generator import HTMLSummaryGenerator
from ..summaries.pdf_summary_generator import PDFSummaryGenerator
from . import speaker_diarization as _speaker_diarization
from .audio_preprocessing import AudioPreprocessor, PreprocessConfig
from .auto_tuner import AutoTuner
from .component_factory import build_component_bundle
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
from .logging_utils import CoreLogger, RunStats, StageGuard, _fmt_hms_ms
from .outputs import (
    default_affect,
    ensure_segment_keys,
    write_human_transcript,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_speakers_summary,
    write_timeline_csv,
)
from .pipeline_checkpoint_system import PipelineCheckpointManager, ProcessingStage
from .runtime_env import DEFAULT_WHISPER_MODEL, configure_local_cache_env
from .stages import PIPELINE_STAGES, PipelineState

# Backwards-compatible aliases for test hooks and StageGuard shims
DiarizationConfig = _speaker_diarization.DiarizationConfig
SpeakerDiarizer = _speaker_diarization.SpeakerDiarizer

configure_local_cache_env()

try:
    from ..affect import paralinguistics as para
except Exception:
    para = None
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
class AudioAnalysisPipelineV2:
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
        self.paralinguistics_module = para

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

    def _init_components(self, cfg: dict[str, Any]):
        """Initialize pipeline components with graceful error handling."""

        bundle = build_component_bundle(cfg, self.corelog)

        self.pp_conf = bundle.pp_conf or PreprocessConfig()
        if bundle.pre is not None:
            self.pre = bundle.pre
        else:
            try:
                self.pre = AudioPreprocessor(self.pp_conf)
            except Exception:
                self.pre = None

        diar_conf = bundle.diar_conf
        if diar_conf is None:
            diar_conf = _speaker_diarization.DiarizationConfig(
                target_sr=self.pp_conf.target_sr
            )
        self.diar_conf = diar_conf

        self.diar = bundle.diar
        if self.diar is None:
            try:
                self.diar = _speaker_diarization.SpeakerDiarizer(self.diar_conf)
            except Exception:
                self.diar = None

        self.tx = bundle.tx
        if self.tx is None:
            try:
                from .transcription_module import AudioTranscriber

                self.tx = AudioTranscriber()
            except Exception:
                self.tx = None

        self.auto_tuner = bundle.auto_tuner
        if self.auto_tuner is None:
            try:
                self.auto_tuner = AutoTuner()
            except Exception:
                self.auto_tuner = None

        self.affect = bundle.affect
        self.sed_tagger = bundle.sed_tagger

        self.html = bundle.html
        if self.html is None:
            try:
                self.html = HTMLSummaryGenerator()
            except Exception:
                self.html = None

        self.pdf = bundle.pdf
        if self.pdf is None:
            try:
                self.pdf = PDFSummaryGenerator()
            except Exception:
                self.pdf = None

        if bundle.issues:
            for issue in bundle.issues:
                if issue not in self.stats.issues:
                    self.stats.issues.append(issue)

        self.stats.models.update(
            {
                "preprocessor": getattr(self.pre, "__class__", type(self.pre)).__name__,
                "diarizer": getattr(self.diar, "__class__", type(self.diar)).__name__,
                "transcriber": getattr(self.tx, "__class__", type(self.tx)).__name__,
                "affect": getattr(self.affect, "__class__", type(self.affect)).__name__,
            }
        )

        if bundle.config_snapshot:
            self.stats.config_snapshot = bundle.config_snapshot
        else:
            self.stats.config_snapshot = {
                "target_sr": self.pp_conf.target_sr,
                "noise_reduction": cfg.get("noise_reduction", True),
                "enable_sed": bool(cfg.get("enable_sed", True)),
                "registry_path": getattr(self.diar_conf, "registry_path", None),
                "ahc_distance_threshold": getattr(
                    self.diar_conf, "ahc_distance_threshold", None
                ),
                "whisper_model": str(cfg.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                "beam_size": cfg.get("beam_size", 1),
                "temperature": cfg.get("temperature", 0.0),
                "no_speech_threshold": cfg.get("no_speech_threshold", 0.20),
                "intent_labels": cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
                "affect_backend": cfg.get("affect_backend", "onnx"),
                "affect_text_model_dir": cfg.get("affect_text_model_dir"),
                "affect_ser_model_dir": cfg.get("affect_ser_model_dir"),
                "affect_vad_model_dir": cfg.get("affect_vad_model_dir"),
                "affect_intent_model_dir": cfg.get("affect_intent_model_dir"),
                "affect_analyzer_threads": cfg.get("affect_analyzer_threads"),
                "text_emotion_model": cfg.get(
                    "text_emotion_model", "SamLowe/roberta-base-go_emotions"
                ),
                "disable_affect": bool(cfg.get("disable_affect", False)),
            }

    def _affect_hint(self, v, a, d, intent):
        try:
            if a is None or v is None:
                return "neutral-status"
            if a > 0.5 and v < 0:
                return "agitated-negative"
            if a < 0.3 and v > 0.2:
                return "calm-positive"
            return f"neutral-{intent}"
        except Exception:
            return "neutral-status"

    def _affect_unified(self, wav: np.ndarray, sr: int, text: str) -> dict[str, Any]:
        try:
            if hasattr(self.affect, "analyze"):
                res = self.affect.analyze(wav=wav, sr=sr, text=text)
                if getattr(self.affect, "issues", None):
                    for issue in self.affect.issues:
                        if issue not in self.stats.issues:
                            self.stats.issues.append(issue)
                return res or default_affect()

            # Fallback implementation
            return default_affect()

        except Exception as e:
            self.corelog.warn(f"Affect analysis failed: {e}")
            return default_affect()

    def _extract_paraling(
        self, wav: np.ndarray, sr: int, segs: list[dict[str, Any]]
    ) -> dict[int, dict[str, Any]]:
        """Extract paralinguistic features with fallback"""
        wav = np.asarray(wav, dtype=np.float32)
        results: dict[int, dict[str, Any]] = {}

        def _safe_float(value: Any) -> float | None:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        def _safe_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None

        try:
            if para and hasattr(para, "extract"):
                out = para.extract(wav, sr, segs) or []
                for i, d in enumerate(out):
                    seg = segs[i] if i < len(segs) else {}
                    start = _safe_float(seg.get("start"))
                    if start is None:
                        start = _safe_float(seg.get("start_time")) or 0.0
                    end = _safe_float(seg.get("end"))
                    if end is None:
                        end = _safe_float(seg.get("end_time")) or start
                    duration_s = _safe_float(d.get("duration_s"))
                    if duration_s is None:
                        duration_s = max(0.0, (end or 0.0) - (start or 0.0))

                    words = _safe_int(d.get("words"))
                    if words is None:
                        text = seg.get("text") or ""
                        words = len(text.split())

                    pause_ratio = _safe_float(d.get("pause_ratio"))
                    if pause_ratio is None:
                        pause_time = _safe_float(d.get("pause_time_s")) or 0.0
                        pause_ratio = (pause_time / duration_s) if duration_s > 0 else 0.0
                    pause_ratio = max(0.0, min(1.0, pause_ratio))

                    results[i] = {
                        "wpm": float(d.get("wpm", 0.0) or 0.0),
                        "duration_s": float(duration_s),
                        "words": int(words),
                        "pause_count": int(d.get("pause_count", 0) or 0),
                        "pause_time_s": float(d.get("pause_time_s", 0.0) or 0.0),
                        "pause_ratio": float(pause_ratio),
                        "f0_mean_hz": float(d.get("f0_mean_hz", 0.0) or 0.0),
                        "f0_std_hz": float(d.get("f0_std_hz", 0.0) or 0.0),
                        "loudness_rms": float(d.get("loudness_rms", 0.0) or 0.0),
                        "disfluency_count": int(d.get("disfluency_count", 0) or 0),
                        "vq_jitter_pct": float(d.get("vq_jitter_pct", 0.0) or 0.0),
                        "vq_shimmer_db": float(d.get("vq_shimmer_db", 0.0) or 0.0),
                        "vq_hnr_db": float(d.get("vq_hnr_db", 0.0) or 0.0),
                        "vq_cpps_db": float(d.get("vq_cpps_db", 0.0) or 0.0),
                    }
                return results
        except Exception as e:
            self.corelog.warn(f"[paralinguistics] fallback: {e}")

        # Lightweight fallback
        for i, s in enumerate(segs):
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(1e-6, end - start)
            txt = s.get("text") or ""
            words = max(0, len(txt.split()))
            wpm = (words / dur) * 60.0 if dur > 0 else 0.0

            i0 = int(start * sr)
            i1 = int(end * sr)
            clip_slice = wav[max(0, i0) : max(0, i1)]
            if hasattr(clip_slice, "astype"):
                clip_arr = clip_slice.astype(np.float32, copy=False)
            else:
                clip_arr = np.asarray(clip_slice, dtype=np.float32)
            clip_size = clip_arr.size if hasattr(clip_arr, "size") else len(clip_arr)
            loud = float(np.sqrt(np.mean(clip_arr**2))) if clip_size > 0 else 0.0

            results[i] = {
                "wpm": float(wpm),
                "duration_s": float(dur),
                "words": int(words),
                "pause_count": 0,
                "pause_time_s": 0.0,
                "pause_ratio": 0.0,
                "f0_mean_hz": 0.0,
                "f0_std_hz": 0.0,
                "loudness_rms": float(loud),
                "disfluency_count": 0,
                "vq_jitter_pct": 0.0,
                "vq_shimmer_db": 0.0,
                "vq_hnr_db": 0.0,
                "vq_cpps_db": 0.0,
            }
        return results

    def _write_outputs(
        self,
        input_audio_path: str,
        outp: Path,
        segments_final: list[dict[str, Any]],
        speakers_summary: list[dict[str, Any]],
        health: Any,
        turns: list[dict[str, Any]],
        overlap_stats: dict[str, Any],
        per_speaker_interrupts: dict[str, Any],
        conv_metrics: ConversationMetrics | None,
        duration_s: float,
        sed_info: dict[str, Any] | None,
    ):
        """Write all output files"""
        # Primary CSV
        write_segments_csv(outp / "diarized_transcript_with_emotion.csv", segments_final)

        # JSONL segments
        write_segments_jsonl(outp / "segments.jsonl", segments_final)

        # Timeline
        write_timeline_csv(outp / "timeline.csv", segments_final)

        # Human-readable transcript
        write_human_transcript(outp / "diarized_transcript_readable.txt", segments_final)

        # QC report
        write_qc_report(
            outp / "qc_report.json",
            self.stats,
            health,
            n_turns=len(turns),
            n_segments=len(segments_final),
            segments=segments_final,
        )

        # Speakers summary
        write_speakers_summary(outp / "speakers_summary.csv", speakers_summary)

        # HTML summary
        try:
            html_path = self.html.render_to_html(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            html_path = None
            self.corelog.warn(
                f"HTML summary skipped: {e}. Verify HTML template assets or install report dependencies."
            )

        # PDF summary
        try:
            pdf_path = self.pdf.render_to_pdf(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as e:
            pdf_path = None
            self.corelog.warn(
                f"PDF summary skipped: {e}. Ensure wkhtmltopdf/LaTeX prerequisites are installed."
            )

        self.checkpoints.create_checkpoint(
            input_audio_path,
            ProcessingStage.SUMMARY_GENERATION,
            {"html": html_path, "pdf": pdf_path},
            progress=90.0,
        )

    def process_audio_file(self, input_audio_path: str, out_dir: str) -> dict[str, Any]:
        """Main processing entry point coordinating modular stages."""

        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        self.stats.file_id = Path(input_audio_path).name

        state = PipelineState(input_audio_path=input_audio_path, out_dir=outp)

        try:
            for stage in PIPELINE_STAGES:
                with StageGuard(self.corelog, self.stats, stage.name) as guard:
                    stage.runner(self, state, guard)
        except Exception as exc:
            self.corelog.error(f"Pipeline failed with unhandled error: {exc}")
            if not state.segments_final and state.norm_tx:
                state.segments_final = [
                    ensure_segment_keys(
                        {
                            "file_id": self.stats.file_id,
                            "start": seg.get("start", 0.0),
                            "end": seg.get("end", 0.0),
                            "speaker_id": seg.get("speaker_id", "Unknown"),
                            "speaker_name": seg.get("speaker_name", "Unknown"),
                            "text": seg.get("text", ""),
                        }
                    )
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

    # Helper methods for output generation
    def _summarize_speakers(
        self,
        segments: list[dict[str, Any]],
        per_speaker_interrupts: dict[str, dict[str, Any]],
        overlap_stats: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        prof = {}
        for s in segments:
            sid = str(s.get("speaker_id", "Unknown"))
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(0.0, end - start)
            p = prof.setdefault(
                sid,
                {
                    "speaker_name": s.get("speaker_name"),
                    "total_duration": 0.0,
                    "word_count": 0,
                    "avg_wpm": 0.0,
                    "avg_valence": 0.0,
                    "avg_arousal": 0.0,
                    "avg_dominance": 0.0,
                    "interruptions_made": 0,
                    "interruptions_received": 0,
                    "overlap_ratio": 0.0,
                },
            )
            p["total_duration"] += dur
            words = len((s.get("text") or "").split())
            p["word_count"] += words

            # Update averages
            for k_src, k_dst in (
                ("valence", "avg_valence"),
                ("arousal", "avg_arousal"),
                ("dominance", "avg_dominance"),
            ):
                val = s.get(k_src, None)
                if val is not None:
                    prev = p[k_dst]
                    cnt = p.get("_n_" + k_dst, 0) + 1
                    p[k_dst] = (prev * (cnt - 1) + float(val)) / float(cnt)
                    p["_n_" + k_dst] = cnt

        # Add interrupt data
        for sid, vals in (per_speaker_interrupts or {}).items():
            p = prof.setdefault(str(sid), {})
            p["interruptions_made"] = int(vals.get("made", 0) or 0)
            p["interruptions_received"] = int(vals.get("received", 0) or 0)

        # Clean up internal counters
        for sid, p in prof.items():
            if not p.get("speaker_name"):
                p["speaker_name"] = sid
            # Remove internal counters
            keys_to_remove = [k for k in p.keys() if k.startswith("_n_")]
            for k in keys_to_remove:
                del p[k]

        return prof

    def _quick_take(self, speakers: dict[str, dict[str, Any]], duration_s: float) -> str:
        if not speakers:
            return "No speakers identified."
        most = max(speakers.items(), key=lambda kv: float(kv[1].get("total_duration", 0.0)))[1]
        tone = "neutral"
        v = float(most.get("avg_valence", 0.0))
        if v > 0.2:
            tone = "positive"
        elif v < -0.2:
            tone = "negative"
        return (
            f"{len(speakers)} speakers over {int(duration_s // 60)} min; most-active tone {tone}."
        )

    def _moments_to_check(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        arr = [(i, float(s.get("arousal", 0.0) or 0.0)) for i, s in enumerate(segments)]
        arr.sort(key=lambda kv: kv[1], reverse=True)
        picks = arr[:10]
        out = []
        for i, _ in picks:
            s = segments[i]
            out.append(
                {
                    "timestamp": float(s.get("start", 0.0) or 0.0),
                    "speaker": str(s.get("speaker_id", "Unknown")),
                    "description": (s.get("text") or "")[:180],
                    "type": "peak",
                }
            )
        out.sort(key=lambda m: m["timestamp"])
        return out

    def _action_items(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for s in segments:
            text = (s.get("text") or "").lower()
            intent = str(s.get("intent_top") or s.get("intent") or "")
            if (
                intent in {"command", "instruction", "request", "suggestion"}
                or "let's " in text
                or "we will" in text
            ):
                out.append(
                    {
                        "type": "action",
                        "text": s.get("text") or "",
                        "speaker": str(s.get("speaker_id", "Unknown")),
                        "timestamp": float(s.get("start", 0.0) or 0.0),
                        "confidence": 0.8,
                        "intent": intent or "unknown",
                    }
                )
        return out
