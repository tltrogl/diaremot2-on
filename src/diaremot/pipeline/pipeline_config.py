"""Configuration tables and helpers for the audio pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .runtime_env import DEFAULT_WHISPER_MODEL
from .speaker_diarization import DiarizationConfig

DEFAULT_PIPELINE_CONFIG: dict[str, Any] = {
    "registry_path": "speaker_registry.json",
    "ahc_distance_threshold": DiarizationConfig.ahc_distance_threshold,
    "speaker_limit": None,
    "whisper_model": str(DEFAULT_WHISPER_MODEL),
    "asr_backend": "faster",
    "compute_type": "int8",
    "cpu_threads": 1,
    "language": None,
    "language_mode": "auto",
    "ignore_tx_cache": False,
    "quiet": False,
    "disable_affect": False,
    "affect_backend": "onnx",
    "affect_text_model_dir": None,
    "affect_intent_model_dir": None,
    "affect_ser_model_dir": None,
    "affect_vad_model_dir": None,
    "affect_analyzer_threads": None,
    "beam_size": 1,
    "temperature": 0.0,
    "no_speech_threshold": 0.20,
    "noise_reduction": False,
    "auto_chunk_enabled": True,
    "chunk_threshold_minutes": 60.0,
    "chunk_size_minutes": 20.0,
    "chunk_overlap_seconds": 30.0,
    "vad_threshold": 0.22,
    "vad_min_speech_sec": 0.25,
    "vad_min_silence_sec": 0.25,
    "vad_speech_pad_sec": 0.2,
    "vad_backend": "auto",
    "disable_energy_vad_fallback": False,
    "energy_gate_db": -33.0,
    "energy_hop_sec": 0.01,
    "max_asr_window_sec": 480,
    "segment_timeout_sec": 300.0,
    "batch_timeout_sec": 1200.0,
    "cpu_diarizer": False,
    "validate_dependencies": False,
    "strict_dependency_versions": False,
    "cache_root": ".cache",
    "cache_roots": [],
    "log_dir": "logs",
    "checkpoint_dir": "checkpoints",
    "target_sr": 16000,
    "loudness_mode": "asr",
}


CORE_DEPENDENCY_REQUIREMENTS: dict[str, str] = {
    "numpy": "1.24",
    "scipy": "1.10",
    "librosa": "0.10",
    "soundfile": "0.12",
    "ctranslate2": "4.0",
    "faster_whisper": "1.0",
    "pandas": "2.0",
    "onnxruntime": "1.16",
    "transformers": "4.30",
}


def build_pipeline_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a pipeline configuration merged with overrides."""

    config = dict(DEFAULT_PIPELINE_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if key not in DEFAULT_PIPELINE_CONFIG and value is None:
                # Skip unknown keys that explicitly request default behaviour
                continue
            if value is not None or key in config:
                config[key] = value
    return config


def clear_pipeline_cache(cache_root: Path | None = None) -> None:
    """Remove cached diarization/transcription artefacts."""

    cache_dir = Path(cache_root) if cache_root else Path(".cache")
    if cache_dir.exists():
        import shutil

        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except PermissionError as exc:  # pragma: no cover - defensive programming
            raise RuntimeError(
                "Could not clear cache directory due to insufficient permissions"
            ) from exc
    cache_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "CORE_DEPENDENCY_REQUIREMENTS",
    "build_pipeline_config",
    "clear_pipeline_cache",
]
