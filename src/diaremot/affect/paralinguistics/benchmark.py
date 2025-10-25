"""Benchmark helpers for the paralinguistics subsystem."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import ParalinguisticsConfig
from .environment import LIBROSA_AVAILABLE, PARSELMOUTH_AVAILABLE, SCIPY_AVAILABLE
from .features import compute_segment_features_v2


def benchmark_performance_v2(
    test_audio: np.ndarray,
    sr: int,
    test_text: str = "This is a test sentence for benchmarking.",
    iterations: int = 10,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    """Benchmark paralinguistic feature extraction throughput."""

    cfg = cfg or ParalinguisticsConfig()
    print(f"Running paralinguistics benchmark with {iterations} iterations...")

    times: list[float] = []
    feature_counts: list[int] = []

    try:
        _ = compute_segment_features_v2(test_audio, sr, 0.0, len(test_audio) / sr, test_text, cfg)
    except Exception as exc:
        print(f"Warm-up run failed: {exc}")
        return {"error": "Benchmark failed during warm-up"}

    for _ in range(iterations):
        start_time = time.time()
        try:
            features = compute_segment_features_v2(test_audio, sr, 0.0, len(test_audio) / sr, test_text, cfg)
            processing_time = time.time() - start_time
            times.append(processing_time)
            valid_features = sum(
                1
                for value in features.values()
                if value is not None and not (isinstance(value, float) and np.isnan(value))
            )
            feature_counts.append(valid_features)
        except Exception as exc:
            print(f"Benchmark iteration failed: {exc}")
            times.append(np.nan)
            feature_counts.append(0)

    valid_times = [t for t in times if not np.isnan(t)]
    valid_features = [f for f, t in zip(feature_counts, times, strict=False) if not np.isnan(t)]

    if not valid_times:
        return {"error": "All benchmark iterations failed"}

    results = {
        "iterations_completed": len(valid_times),
        "iterations_failed": iterations - len(valid_times),
        "success_rate": len(valid_times) / iterations,
        "mean_time_sec": float(np.mean(valid_times)),
        "std_time_sec": float(np.std(valid_times)),
        "min_time_sec": float(np.min(valid_times)),
        "max_time_sec": float(np.max(valid_times)),
        "median_time_sec": float(np.median(valid_times)),
        "mean_features": float(np.mean(valid_features)),
        "feature_consistency": float(np.std(valid_features)),
        "performance_rating": _get_performance_rating(np.mean(valid_times), len(test_audio) / sr),
        "audio_duration_sec": len(test_audio) / sr,
        "audio_samples": len(test_audio),
        "sample_rate": sr,
        "config_summary": {
            "voice_quality_enabled": cfg.voice_quality_enabled,
            "spectral_features_enabled": cfg.spectral_features_enabled,
            "parallel_processing": cfg.parallel_processing,
            "use_parselmouth": cfg.vq_use_parselmouth and PARSELMOUTH_AVAILABLE,
        },
        "libraries_available": {
            "librosa": LIBROSA_AVAILABLE,
            "scipy": SCIPY_AVAILABLE,
            "parselmouth": PARSELMOUTH_AVAILABLE,
        },
    }

    return results


def _get_performance_rating(processing_time: float, audio_duration: float) -> str:
    ratio = processing_time / audio_duration
    if ratio < 0.1:
        return "excellent"
    if ratio < 0.2:
        return "good"
    if ratio < 0.5:
        return "acceptable"
    if ratio < 1.0:
        return "slow"
    return "very_slow"


__all__ = ["benchmark_performance_v2"]
