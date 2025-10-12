"""Reusable helpers for pipeline diagnostics smoke tests.

This module centralises the tiny synthetic-audio generation and the
corresponding fast configuration overrides that multiple diagnostics scripts
were previously duplicating.  Keeping the logic here ensures changes to the
smoke-test waveform or pipeline configuration propagate consistently across
the health-check CLI and the comprehensive validation script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio_pipeline_core import AudioAnalysisPipelineV2

AudioFactory = Callable[[int, float], np.ndarray]

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DURATION_SECONDS = 1.0


# Shared fast-path configuration overrides for diagnostics flows.

SMOKE_TEST_PIPELINE_OVERRIDES: dict[str, Any] = {
    "whisper_model": "tiny.en",
    "noise_reduction": False,
    "beam_size": 1,
    "temperature": 0.0,
    "no_speech_threshold": 0.6,
    "asr_backend": "faster",
}


SMOKE_TEST_TRANSCRIBE_KWARGS: dict[str, Any] = {
    "model_size": "tiny.en",
    "compute_type": "int8",
    "beam_size": 1,
    "temperature": 0.0,
    "no_speech_threshold": 0.6,
}


@dataclass(slots=True)
class SmokeTestResult:
    """Structured response emitted by :func:`run_pipeline_smoke_test`."""

    success: bool
    output_dir: Path | None = None
    wav_path: Path | None = None
    run_result: dict[str, Any] | None = None
    error: str | None = None


def silence_audio_factory(sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Return a simple block of silence for ultra-fast diagnostics."""

    frames = max(int(sample_rate * duration_seconds), sample_rate)
    return np.zeros(frames, dtype=np.float32)


def burst_audio_factory(sample_rate: int, duration_seconds: float) -> np.ndarray:
    """Generate the bursty synthetic waveform used by the validation script."""

    duration_seconds = max(duration_seconds, 4.0)
    frames = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, frames, endpoint=False)
    audio = np.zeros_like(t, dtype=np.float32)

    # Add three exponentially decaying tones to mimic speech-like bursts.
    for start_time in (0.5, 2.0, 3.5):
        start_idx = int(start_time * sample_rate)
        end_idx = min(int((start_time + 1.0) * sample_rate), frames)
        if start_idx >= frames:
            break
        local_t = t[start_idx:end_idx] - start_time
        freq = 200 + 100 * np.random.random()
        burst = np.sin(2 * np.pi * freq * local_t) * np.exp(-5 * local_t)
        audio[start_idx:end_idx] = burst.astype(np.float32) * 0.1

    noise = np.random.normal(0, 0.01, frames).astype(np.float32)
    return (audio + noise).astype(np.float32)


def prepare_smoke_wav(
    target_dir: Path,
    *,
    waveform_factory: AudioFactory = silence_audio_factory,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    filename: str = "smoke.wav",
) -> Path:
    """Create a temporary WAV file containing synthetic diagnostics audio."""

    target_dir.mkdir(parents=True, exist_ok=True)
    wav_path = target_dir / filename
    audio = waveform_factory(sample_rate, duration_seconds)
    try:
        import soundfile as sf  # Local import keeps diagnostics lightweight

        sf.write(wav_path, audio, sample_rate)
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("soundfile is required to prepare diagnostics audio") from exc
    return wav_path


def run_pipeline_smoke_test(
    *,
    config_overrides: dict[str, Any] | None = None,
    tmp_dir: Path | None = None,
    output_dir: Path | None = None,
    waveform_factory: AudioFactory = silence_audio_factory,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    wav_path: Path | None = None,
) -> SmokeTestResult:
    """Execute a fast end-to-end pipeline run using synthetic audio."""

    tmp_dir = tmp_dir or Path("healthcheck_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        effective_config: dict[str, Any] = dict(SMOKE_TEST_PIPELINE_OVERRIDES)
        if config_overrides:
            effective_config.update(config_overrides)

        effective_config.setdefault("registry_path", str(tmp_dir / "registry.json"))

        wav_path = wav_path or prepare_smoke_wav(
            tmp_dir,
            waveform_factory=waveform_factory,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )

        out_dir = output_dir or (tmp_dir / "out")
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = AudioAnalysisPipelineV2(effective_config)
        run_result = pipeline.process_audio_file(str(wav_path), str(out_dir))

        return SmokeTestResult(
            success=True,
            output_dir=Path(run_result.get("out_dir", out_dir)),
            wav_path=wav_path,
            run_result=run_result,
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        return SmokeTestResult(success=False, wav_path=wav_path, error=str(exc))


__all__ = [
    "SMOKE_TEST_PIPELINE_OVERRIDES",
    "SMOKE_TEST_TRANSCRIBE_KWARGS",
    "SmokeTestResult",
    "burst_audio_factory",
    "prepare_smoke_wav",
    "run_pipeline_smoke_test",
    "silence_audio_factory",
]
