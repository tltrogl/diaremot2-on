"""Configuration objects and presets for paralinguistics extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class ParalinguisticsConfig:
    """Production-optimized configuration for paralinguistic analysis."""

    frame_ms: int = 25
    hop_ms: int = 10

    base_silence_dbfs: float = -45.0
    adaptive_silence: bool = True
    silence_floor_percentile: int = 5
    silence_margin_db: float = 8.0

    pause_min_ms: int = 200
    pause_long_ms: int = 600

    f0_min_hz: float = 65.0
    f0_max_hz: float = 400.0
    pitch_method: str = "pyin"
    pitch_frame_length: int = 2048
    pitch_hop_length: int = 512
    pitch_min_coverage: float = 0.05
    pitch_interp_max_gap_ms: int = 100

    voice_quality_enabled: bool = True
    vq_min_duration_sec: float = 0.6
    vq_min_snr_db: float = 10.0
    vq_use_parselmouth: bool = True
    vq_fallback_enabled: bool = True

    syllable_estimation: bool = True
    disfluency_detection: bool = True

    spectral_features_enabled: bool = True
    spectral_n_fft: int = 1024
    spectral_hop_length: int = 256

    use_vectorized_ops: bool = True
    enable_caching: bool = True
    parallel_processing: bool = False
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 2)

    backchannel_max_ms: int = 300

    max_audio_length_sec: float = 30.0
    enable_memory_optimization: bool = True


COMPREHENSIVE_FILLER_WORDS: Final[frozenset[str]] = frozenset(
    {
        "um",
        "uh",
        "erm",
        "er",
        "mm",
        "hmm",
        "like",
        "you know",
        "i mean",
        "sort of",
        "kinda",
        "right",
        "okay",
        "ok",
        "so",
        "well",
        "uhh",
        "umm",
        "basically",
        "literally",
        "actually",
        "totally",
        "absolutely",
        "definitely",
        "pretty much",
        "kind of",
        "you see",
        "let me see",
        "let's see",
        "anyway",
    }
)

VOWELS: Final[frozenset[str]] = frozenset("aeiouyAEIOUY")


def get_config_preset(
    preset_name: str,
    *,
    max_workers: int | None = None,
    **overrides: object,
) -> ParalinguisticsConfig:
    """Return a configuration preset tuned for the requested workload."""

    kwargs = dict(overrides)
    if max_workers is not None:
        kwargs.setdefault("max_workers", max_workers)

    if preset_name == "fast":
        return ParalinguisticsConfig(
            frame_ms=30,
            hop_ms=15,
            adaptive_silence=False,
            voice_quality_enabled=False,
            spectral_features_enabled=False,
            syllable_estimation=False,
            disfluency_detection=False,
            max_audio_length_sec=15.0,
            enable_memory_optimization=True,
            **kwargs,
        )
    if preset_name == "balanced":
        return ParalinguisticsConfig(**kwargs)
    if preset_name == "quality":
        return ParalinguisticsConfig(
            frame_ms=20,
            hop_ms=8,
            adaptive_silence=True,
            silence_floor_percentile=3,
            silence_margin_db=10.0,
            voice_quality_enabled=True,
            vq_use_parselmouth=True,
            vq_fallback_enabled=True,
            vq_min_snr_db=8.0,
            spectral_features_enabled=True,
            spectral_n_fft=2048,
            spectral_hop_length=128,
            syllable_estimation=True,
            disfluency_detection=True,
            pitch_frame_length=4096,
            pitch_hop_length=256,
            pitch_min_coverage=0.08,
            max_audio_length_sec=60.0,
            enable_memory_optimization=False,
            **kwargs,
        )
    if preset_name == "research":
        return ParalinguisticsConfig(
            frame_ms=15,
            hop_ms=5,
            adaptive_silence=True,
            silence_floor_percentile=1,
            silence_margin_db=12.0,
            voice_quality_enabled=True,
            vq_use_parselmouth=True,
            vq_fallback_enabled=True,
            vq_min_snr_db=10.0,
            spectral_features_enabled=True,
            spectral_n_fft=4096,
            spectral_hop_length=64,
            syllable_estimation=True,
            disfluency_detection=True,
            pitch_frame_length=8192,
            pitch_hop_length=128,
            pitch_min_coverage=0.1,
            max_audio_length_sec=300.0,
            enable_memory_optimization=False,
            parallel_processing=True,
            **kwargs,
        )
    raise ValueError(
        f"Unknown preset: {preset_name}. Available: fast, balanced, quality, research"
    )


__all__ = [
    "ParalinguisticsConfig",
    "COMPREHENSIVE_FILLER_WORDS",
    "VOWELS",
    "get_config_preset",
]
