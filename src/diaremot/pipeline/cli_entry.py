"""Command line entry helpers for the audio pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import build_pipeline_config
from .orchestrator import (
    AudioAnalysisPipelineV2,
    clear_pipeline_cache,
    config_verify_dependencies,
)
from .runtime_env import DEFAULT_WHISPER_MODEL
from .speaker_diarization import DiarizationConfig

__all__ = ["main", "_build_arg_parser", "_args_to_config"]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-first orchestration pipeline for diarization, ASR, and affect scoring."
    )
    parser.add_argument("--input", help="Path to input audio file")
    parser.add_argument("--outdir", help="Directory to write outputs")
    parser.add_argument(
        "--registry_path",
        default=str(Path("registry") / "speaker_registry.json"),
        help="Persistent speaker registry path",
    )
    parser.add_argument(
        "--ahc_distance_threshold",
        type=float,
        default=DiarizationConfig.ahc_distance_threshold,
        help="Hierarchical clustering distance threshold",
    )
    parser.add_argument(
        "--speaker_limit",
        type=int,
        default=None,
        help="Maximum number of speakers to detect",
    )
    parser.add_argument(
        "--whisper-model",
        default=str(DEFAULT_WHISPER_MODEL),
        help="Whisper model identifier or path",
    )
    parser.add_argument(
        "--asr-backend",
        default="faster",
        choices=["faster", "whisper", "auto"],
        help="ASR backend selection",
    )
    parser.add_argument(
        "--asr-compute-type",
        default="int8",
        choices=["float32", "int8", "int8_float16"],
        help="Compute precision for ASR backend",
    )
    parser.add_argument(
        "--asr-cpu-threads",
        type=int,
        default=1,
        help="CPU threads for ASR backend",
    )
    parser.add_argument("--language", default=None, help="Force recognition language")
    parser.add_argument(
        "--language-mode",
        default="auto",
        choices=["auto", "manual"],
        help="Language detection mode",
    )
    parser.add_argument(
        "--ignore-tx-cache",
        action="store_true",
        help="Ignore transcription cache even if available",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console logging noise",
    )
    parser.add_argument(
        "--disable-affect",
        action="store_true",
        help="Disable affect (emotion/intent) analysis",
    )
    parser.add_argument(
        "--affect-backend",
        default="onnx",
        choices=["auto", "onnx"],
        help="Backend for affect analysis",
    )
    parser.add_argument(
        "--affect-text-model-dir",
        default=None,
        help="Path to text emotion model assets",
    )
    parser.add_argument(
        "--affect-intent-model-dir",
        default=None,
        help="Path to intent classifier model assets",
    )
    parser.add_argument(
        "--affect-ser-model-dir",
        default=None,
        help="Path to speech emotion recognition model assets",
    )
    parser.add_argument(
        "--affect-vad-model-dir",
        default=None,
        help="Path to valence/arousal/dominance model assets",
    )
    parser.add_argument("--beam-size", type=int, default=1, help="ASR beam search size")
    parser.add_argument("--temperature", type=float, default=0.0, help="ASR decoding temperature")
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.5,
        help="Whisper no-speech probability threshold",
    )
    parser.add_argument(
        "--noise-reduction",
        action="store_true",
        help="Enable spectral subtraction noise reduction",
    )
    parser.add_argument(
        "--enable-sed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable background sound event detection (use --no-enable-sed to skip)",
    )
    parser.add_argument(
        "--chunk-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic chunking",
    )
    parser.add_argument(
        "--chunk-threshold-minutes",
        type=float,
        default=60.0,
        help="Duration threshold before chunking",
    )
    parser.add_argument(
        "--chunk-size-minutes",
        type=float,
        default=20.0,
        help="Chunk size in minutes",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=30.0,
        help="Chunk overlap in seconds",
    )
    parser.add_argument(
        "--verify_deps",
        action="store_true",
        help="Only verify dependency availability",
    )
    parser.add_argument(
        "--strict_dependency_versions",
        action="store_true",
        help="Require minimum dependency versions during verification",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.30,
        help="Silero VAD probability threshold",
    )
    parser.add_argument(
        "--vad-min-speech-sec",
        type=float,
        default=0.8,
        help="Minimum detected speech duration",
    )
    parser.add_argument(
        "--vad-min-silence-sec",
        type=float,
        default=0.8,
        help="Minimum detected silence duration",
    )
    parser.add_argument(
        "--vad-speech-pad-sec",
        type=float,
        default=0.2,
        help="Padding around detected speech",
    )
    parser.add_argument(
        "--no-energy-fallback",
        action="store_true",
        help="Disable energy-based VAD fallback",
    )
    parser.add_argument(
        "--energy-gate-db",
        type=float,
        default=-33.0,
        help="Energy gate for fallback VAD",
    )
    parser.add_argument(
        "--energy-hop-sec",
        type=float,
        default=0.01,
        help="Energy VAD hop length",
    )
    parser.add_argument(
        "--vad-backend",
        default="auto",
        choices=["auto", "onnx"],
        help="Preferred Silero VAD backend",
    )
    parser.add_argument(
        "--asr-window-sec",
        type=int,
        default=480,
        help="Maximum Whisper window size in seconds",
    )
    parser.add_argument(
        "--asr-segment-timeout",
        type=float,
        default=300.0,
        help="Timeout for individual ASR segments",
    )
    parser.add_argument(
        "--asr-batch-timeout",
        type=float,
        default=1200.0,
        help="Timeout for batching diarization segments",
    )
    parser.add_argument(
        "--cpu-diarizer",
        action="store_true",
        help="Use CPU optimized diarizer wrapper",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached diarization/transcription artefacts before running",
    )
    return parser


def _args_to_config(args: argparse.Namespace, *, ignore_tx_cache: bool) -> dict[str, Any]:
    return {
        "registry_path": args.registry_path,
        "ahc_distance_threshold": args.ahc_distance_threshold,
        "speaker_limit": args.speaker_limit,
        "whisper_model": args.whisper_model,
        "asr_backend": args.asr_backend,
        "compute_type": args.asr_compute_type,
        "cpu_threads": int(args.asr_cpu_threads),
        "language": args.language,
        "language_mode": args.language_mode,
        "ignore_tx_cache": ignore_tx_cache,
        "quiet": bool(args.quiet),
        "disable_affect": bool(args.disable_affect),
        "affect_backend": args.affect_backend,
        "affect_text_model_dir": args.affect_text_model_dir,
        "affect_intent_model_dir": args.affect_intent_model_dir,
        "affect_ser_model_dir": args.affect_ser_model_dir,
        "affect_vad_model_dir": args.affect_vad_model_dir,
        "beam_size": args.beam_size,
        "temperature": args.temperature,
        "no_speech_threshold": args.no_speech_threshold,
        "noise_reduction": bool(args.noise_reduction),
        "enable_sed": bool(args.enable_sed),
        "auto_chunk_enabled": bool(args.chunk_enabled),
        "chunk_threshold_minutes": float(args.chunk_threshold_minutes),
        "chunk_size_minutes": float(args.chunk_size_minutes),
        "chunk_overlap_seconds": float(args.chunk_overlap_seconds),
        "vad_threshold": args.vad_threshold,
        "vad_min_speech_sec": args.vad_min_speech_sec,
        "vad_min_silence_sec": args.vad_min_silence_sec,
        "vad_speech_pad_sec": args.vad_speech_pad_sec,
        "vad_backend": args.vad_backend,
        "disable_energy_vad_fallback": bool(args.no_energy_fallback),
        "energy_gate_db": args.energy_gate_db,
        "energy_hop_sec": args.energy_hop_sec,
        "max_asr_window_sec": int(args.asr_window_sec),
        "segment_timeout_sec": float(args.asr_segment_timeout),
        "batch_timeout_sec": float(args.asr_batch_timeout),
        "cpu_diarizer": bool(args.cpu_diarizer),
    }


def _handle_cache_clear(requested: bool, *, cache_root: Path, ignore_tx_cache: bool) -> bool:
    if not requested:
        return ignore_tx_cache
    try:
        clear_pipeline_cache(cache_root)
        print("Cache cleared successfully.")
        return True
    except PermissionError:
        print("Warning: Could not fully clear cache due to permissions. Ignoring cached results.")
        return True
    except RuntimeError as exc:
        print(f"Warning: Cache clear failed: {exc}. Ignoring cached results.")
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Cache clear failed: {exc}. Ignoring cached results.")
        return True


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.verify_deps:
        ok, problems = config_verify_dependencies(bool(args.strict_dependency_versions))
        if ok:
            suffix = " with required versions" if args.strict_dependency_versions else ""
            print(f"All core dependencies are importable{suffix}.")
            return 0
        print("Dependency verification failed\n  - " + "\n  - ".join(problems))
        return 1

    if not args.input or not args.outdir:
        parser.error("--input and --outdir are required unless --verify_deps is used")

    ignore_tx_cache = _handle_cache_clear(
        args.clear_cache,
        cache_root=Path(".cache"),
        ignore_tx_cache=bool(args.ignore_tx_cache),
    )

    config = _args_to_config(args, ignore_tx_cache=ignore_tx_cache)
    pipeline = AudioAnalysisPipelineV2(build_pipeline_config(config))
    manifest = pipeline.process_audio_file(args.input, args.outdir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
