"""Production-optimized paralinguistic feature extraction utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .analysis import (
    analyze_speech_patterns_v2,
    compute_overlap_and_interruptions,
    detect_backchannels_v2,
    detect_speech_anomalies_v2,
)
from .audio import (
    advanced_audio_quality_assessment,
    compute_rms_fallback,
    enhanced_pitch_statistics,
    get_cached_pitch_params,
    get_optimized_frame_params,
    optimized_pause_analysis,
    optimized_spectral_features,
    robust_pitch_extraction,
    single_pitch_extraction,
    vectorized_silence_detection,
)
from .benchmark import benchmark_performance_v2
from .cli import main
from .config import (
    COMPREHENSIVE_FILLER_WORDS,
    VOWELS,
    ParalinguisticsConfig,
    get_config_preset,
)
from .environment import (
    LIBROSA_AVAILABLE,
    PARSELMOUTH_AVAILABLE,
    SCIPY_AVAILABLE,
    call,
    librosa,
    parselmouth,
    scipy_signal,
)
from .features import compute_segment_features_v2, process_segments_batch_v2
from .text import (
    advanced_disfluency_detection,
    enhanced_syllable_estimation,
    enhanced_word_tokenization,
    vectorized_syllable_count,
)
from .validation import validate_module
from .voice import compute_voice_quality_fallback, compute_voice_quality_parselmouth


def compute_segment_features(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    text: str,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for legacy callers."""

    return compute_segment_features_v2(audio, sr, start_time, end_time, text, cfg)


def process_segments_batch(
    segments: list[tuple[np.ndarray, int, float, float, str]],
    cfg: ParalinguisticsConfig | None = None,
    progress_callback: callable | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for legacy callers."""

    return process_segments_batch_v2(segments, cfg, progress_callback)


def extract(wav: np.ndarray, sr: int, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paralinguistic features per segment for pipeline consumption."""

    out: list[dict[str, Any]] = []
    cfg = ParalinguisticsConfig()
    total = max(0.0, float(len(wav) / max(1, sr)))

    for seg in segs:
        try:
            start = float(seg.get("start", seg.get("start_time", 0.0)) or 0.0)
            end = float(seg.get("end", seg.get("end_time", start)) or start)
            start = max(0.0, min(start, total))
            end = max(start, min(end, total))
            text = seg.get("text") or ""
            duration_s = max(0.0, end - start)
            words = len(text.split())
            feats = compute_segment_features_v2(wav, sr, start, end, text, cfg)
            out.append(
                {
                    "wpm": feats.get("wpm", 0.0),
                    "duration_s": duration_s,
                    "words": words,
                    "pause_count": int(feats.get("pause_count", 0)),
                    "pause_time_s": float(feats.get("pause_total_sec", 0.0) or 0.0),
                    "pause_ratio": float(feats.get("pause_ratio", 0.0) or 0.0),
                    "f0_mean_hz": float(feats.get("pitch_med_hz", 0.0) or 0.0),
                    "f0_std_hz": float(feats.get("pitch_iqr_hz", 0.0) or 0.0),
                    "loudness_rms": float(feats.get("loudness_dbfs_med", 0.0) or 0.0),
                    "disfluency_count": int(
                        feats.get("filler_count", 0)
                        + feats.get("repetition_count", 0)
                        + feats.get("false_start_count", 0)
                    ),
                    "vq_jitter_pct": float(feats.get("vq_jitter_pct", 0.0) or 0.0),
                    "vq_shimmer_db": float(feats.get("vq_shimmer_db", 0.0) or 0.0),
                    "vq_hnr_db": float(feats.get("vq_hnr_db", 0.0) or 0.0),
                    "vq_cpps_db": float(feats.get("vq_cpps_db", 0.0) or 0.0),
                    "vq_voiced_ratio": float(feats.get("vq_voiced_ratio", 0.0) or 0.0),
                    "vq_spectral_slope_db": float(feats.get("vq_spectral_slope_db", 0.0) or 0.0),
                    "vq_reliable": bool(feats.get("vq_reliable", False)),
                    "vq_note": str(feats.get("vq_note", "")),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            out.append({"error": str(exc)})

    return out


__all__ = [
    "compute_segment_features_v2",
    "process_segments_batch_v2",
    "compute_segment_features",
    "process_segments_batch",
    "extract",
    "ParalinguisticsConfig",
    "get_config_preset",
    "analyze_speech_patterns_v2",
    "detect_speech_anomalies_v2",
    "detect_backchannels_v2",
    "compute_overlap_and_interruptions",
    "benchmark_performance_v2",
    "validate_module",
    "advanced_audio_quality_assessment",
    "compute_rms_fallback",
    "enhanced_pitch_statistics",
    "get_cached_pitch_params",
    "get_optimized_frame_params",
    "optimized_pause_analysis",
    "optimized_spectral_features",
    "robust_pitch_extraction",
    "single_pitch_extraction",
    "vectorized_silence_detection",
    "advanced_disfluency_detection",
    "enhanced_syllable_estimation",
    "enhanced_word_tokenization",
    "vectorized_syllable_count",
    "compute_voice_quality_parselmouth",
    "compute_voice_quality_fallback",
    "COMPREHENSIVE_FILLER_WORDS",
    "VOWELS",
    "LIBROSA_AVAILABLE",
    "SCIPY_AVAILABLE",
    "PARSELMOUTH_AVAILABLE",
    "librosa",
    "parselmouth",
    "scipy_signal",
    "call",
    "main",
]

__version__ = "2.2.0"
__author__ = "Paralinguistics Research Team"
__description__ = "Production-optimized paralinguistic feature extraction with enhanced CPU performance"
