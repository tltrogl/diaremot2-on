"""Core feature extraction routines for paralinguistics."""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .audio import (
    advanced_audio_quality_assessment,
    compute_rms_fallback,
    enhanced_pitch_statistics,
    get_optimized_frame_params,
    optimized_pause_analysis,
    optimized_spectral_features,
    robust_pitch_extraction,
    vectorized_silence_detection,
)
from .config import ParalinguisticsConfig
from .environment import LIBROSA_AVAILABLE, PARSELMOUTH_AVAILABLE, librosa
from .text import (
    advanced_disfluency_detection,
    enhanced_word_tokenization,
    vectorized_syllable_count,
)
from .voice import compute_voice_quality_fallback, compute_voice_quality_parselmouth


def compute_segment_features_v2(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    text: str,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    """Compute the full set of paralinguistic metrics for a transcript segment."""

    cfg = cfg or ParalinguisticsConfig()

    duration = max(1e-6, float(end_time - start_time))
    start_idx = max(0, int(start_time * sr))
    end_idx = min(len(audio), int(end_time * sr))

    if cfg.enable_memory_optimization:
        segment_audio = audio[start_idx:end_idx].copy().astype(np.float32)
    else:
        segment_audio = audio[start_idx:end_idx].astype(np.float32, copy=False)

    flags: dict[str, Any] = {
        "processing_version": "2.1.0",
        "duration_sec": duration,
        "audio_samples": len(segment_audio),
    }

    words = enhanced_word_tokenization(text) if text else ()
    word_count = len(words)
    flags["word_count"] = word_count

    if duration < 0.5:
        wpm = sps = np.nan
        flags["very_short_segment"] = True
    else:
        wpm = (60.0 * word_count) / duration
        if cfg.syllable_estimation and words:
            syllable_count = vectorized_syllable_count(words)
            sps = (60.0 * syllable_count) / duration
            flags["syllable_count"] = syllable_count
        else:
            sps = np.nan

    if cfg.disfluency_detection and words:
        filler_count, repetition_count, false_start_count = advanced_disfluency_detection(words, text)
        total_disfluencies = filler_count + repetition_count + false_start_count
        disfluency_rate = (100.0 * total_disfluencies) / max(1, word_count)
        flags["total_disfluencies"] = total_disfluencies
    else:
        filler_count = repetition_count = false_start_count = 0
        disfluency_rate = 0.0

    if segment_audio.size == 0:
        return _get_empty_features_v2(wpm, sps, filler_count, repetition_count, false_start_count, disfluency_rate, flags)

    frame_length, hop_length = get_optimized_frame_params(sr, cfg.frame_ms, cfg.hop_ms)
    flags["frame_params"] = {"frame_length": frame_length, "hop_length": hop_length}

    try:
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(y=segment_audio, frame_length=frame_length, hop_length=hop_length, center=True)[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-12)
        else:
            rms_db = compute_rms_fallback(segment_audio, frame_length, hop_length)

        if rms_db.size == 0:
            loudness_dbfs_med = loudness_dr_db = loudness_over_floor_db = np.nan
            pause_count = pause_total_sec = pause_ratio = pause_short_count = pause_long_count = 0
            floor_db = -60.0
            flags["empty_rms"] = True
        else:
            if cfg.adaptive_silence:
                floor_db = float(np.percentile(rms_db, cfg.silence_floor_percentile))
                silence_threshold = floor_db + cfg.silence_margin_db
                flags["adaptive_threshold"] = {"floor_db": floor_db, "threshold_db": silence_threshold}
            else:
                floor_db = cfg.base_silence_dbfs
                silence_threshold = floor_db

            loudness_dbfs_med = float(np.median(rms_db))
            loudness_dr_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
            loudness_over_floor_db = float(loudness_dbfs_med - floor_db)

            silence_mask = vectorized_silence_detection(rms_db, silence_threshold, cfg.enable_memory_optimization)
            (
                pause_count,
                pause_total_sec,
                pause_ratio,
                pause_short_count,
                pause_long_count,
            ) = optimized_pause_analysis(silence_mask, sr, hop_length, cfg)

            flags["pause_analysis"] = {
                "silence_frames": int(np.sum(silence_mask)),
                "total_frames": len(silence_mask),
            }
    except Exception as exc:
        flags["loudness_analysis_error"] = str(exc)
        loudness_dbfs_med = loudness_dr_db = loudness_over_floor_db = np.nan
        pause_count = pause_total_sec = pause_ratio = pause_short_count = pause_long_count = 0

    try:
        f0, voiced_flag = robust_pitch_extraction(segment_audio, sr, cfg)
        if f0.size > 0 and np.any(voiced_flag):
            times = np.arange(len(f0)) * (hop_length / sr)
            pitch_med_hz, pitch_iqr_hz, pitch_slope_hzps = enhanced_pitch_statistics(f0, voiced_flag, times, cfg)
            flags["pitch_analysis"] = {
                "voiced_frames": int(np.sum(voiced_flag)),
                "total_frames": len(f0),
                "coverage": float(np.mean(voiced_flag)),
            }
        else:
            pitch_med_hz = pitch_iqr_hz = pitch_slope_hzps = np.nan
            flags["no_pitch_detected"] = True
    except Exception as exc:
        flags["pitch_analysis_error"] = str(exc)
        pitch_med_hz = pitch_iqr_hz = pitch_slope_hzps = np.nan

    spectral_centroid_med_hz = spectral_centroid_iqr_hz = spectral_flatness_med = np.nan
    if cfg.spectral_features_enabled:
        try:
            (
                spectral_centroid_med_hz,
                spectral_centroid_iqr_hz,
                spectral_flatness_med,
            ) = optimized_spectral_features(segment_audio, sr, cfg)
            flags["spectral_analysis"] = "completed"
        except Exception as exc:
            flags["spectral_analysis_error"] = str(exc)

    if cfg.voice_quality_enabled and duration >= cfg.vq_min_duration_sec:
        try:
            snr_db, is_reliable, quality_status = advanced_audio_quality_assessment(segment_audio, sr)
            if is_reliable and snr_db >= cfg.vq_min_snr_db:
                if cfg.vq_use_parselmouth and PARSELMOUTH_AVAILABLE:
                    voice_quality = compute_voice_quality_parselmouth(segment_audio, sr, cfg)
                    method = "parselmouth"
                else:
                    voice_quality = compute_voice_quality_fallback(segment_audio, sr, cfg)
                    method = "fallback"

                vq_jitter_pct = voice_quality.get("jitter_pct", 0.0)
                vq_shimmer_db = voice_quality.get("shimmer_db", 0.0)
                vq_hnr_db = voice_quality.get("hnr_db", 0.0)
                vq_cpps_db = voice_quality.get("cpps_db", 0.0)
                vq_voiced_ratio = voice_quality.get("voiced_ratio", 0.0)
                vq_spectral_slope_db = voice_quality.get("spectral_slope_db", 0.0)

                vq_reliable = bool(is_reliable and snr_db >= cfg.vq_min_snr_db and vq_voiced_ratio >= 0.5)
                vq_note = f"{quality_status}_voiced_{vq_voiced_ratio:.2f}"
                flags["voice_quality"] = {
                    "method": method,
                    "snr_db": snr_db,
                    "quality_status": quality_status,
                }
            else:
                if cfg.vq_fallback_enabled:
                    voice_quality = compute_voice_quality_fallback(segment_audio, sr, cfg)
                    vq_jitter_pct = voice_quality.get("jitter_pct", 0.0)
                    vq_shimmer_db = voice_quality.get("shimmer_db", 0.0)
                    vq_hnr_db = voice_quality.get("hnr_db", 0.0)
                    vq_cpps_db = voice_quality.get("cpps_db", 0.0)
                    vq_voiced_ratio = voice_quality.get("voiced_ratio", 0.0)
                    vq_spectral_slope_db = voice_quality.get("spectral_slope_db", 0.0)
                    vq_reliable = False
                    vq_note = f"unreliable_{quality_status}_snr_{snr_db:.1f}dB"
                else:
                    vq_jitter_pct = vq_shimmer_db = vq_hnr_db = vq_cpps_db = 0.0
                    vq_voiced_ratio = vq_spectral_slope_db = 0.0
                    vq_reliable = False
                    vq_note = "disabled_fallback"

                flags["voice_quality"] = {
                    "method": "unreliable",
                    "snr_db": snr_db,
                    "quality_status": quality_status,
                }
        except Exception as exc:
            flags["voice_quality_error"] = str(exc)
            vq_jitter_pct = vq_shimmer_db = vq_hnr_db = vq_cpps_db = 0.0
            vq_voiced_ratio = vq_spectral_slope_db = 0.0
            vq_reliable = False
            vq_note = f"analysis_failed_{str(exc)[:50]}"
    else:
        vq_jitter_pct = vq_shimmer_db = vq_hnr_db = vq_cpps_db = 0.0
        vq_voiced_ratio = vq_spectral_slope_db = 0.0
        vq_reliable = False
        if not cfg.voice_quality_enabled:
            vq_note = "disabled"
        else:
            vq_note = f"too_short_{duration:.2f}s"
        flags["voice_quality"] = "disabled_or_too_short"

    features = {
        "wpm": round(float(wpm), 2) if not np.isnan(wpm) else np.nan,
        "sps": round(float(sps), 2) if not np.isnan(sps) else np.nan,
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "false_start_count": int(false_start_count),
        "disfluency_rate": float(disfluency_rate),
        "pause_count": int(pause_count),
        "pause_total_sec": float(pause_total_sec),
        "pause_ratio": float(pause_ratio),
        "pause_short_count": int(pause_short_count),
        "pause_long_count": int(pause_long_count),
        "pitch_med_hz": float(pitch_med_hz) if not np.isnan(pitch_med_hz) else np.nan,
        "pitch_iqr_hz": float(pitch_iqr_hz) if not np.isnan(pitch_iqr_hz) else np.nan,
        "pitch_slope_hzps": float(pitch_slope_hzps) if not np.isnan(pitch_slope_hzps) else np.nan,
        "loudness_dbfs_med": float(loudness_dbfs_med) if not np.isnan(loudness_dbfs_med) else np.nan,
        "loudness_dr_db": float(loudness_dr_db) if not np.isnan(loudness_dr_db) else np.nan,
        "loudness_over_floor_db": float(loudness_over_floor_db) if not np.isnan(loudness_over_floor_db) else np.nan,
        "spectral_centroid_med_hz": float(spectral_centroid_med_hz) if not np.isnan(spectral_centroid_med_hz) else np.nan,
        "spectral_centroid_iqr_hz": float(spectral_centroid_iqr_hz) if not np.isnan(spectral_centroid_iqr_hz) else np.nan,
        "spectral_flatness_med": float(spectral_flatness_med) if not np.isnan(spectral_flatness_med) else np.nan,
        "vq_jitter_pct": float(vq_jitter_pct),
        "vq_shimmer_db": float(vq_shimmer_db),
        "vq_hnr_db": float(vq_hnr_db),
        "vq_cpps_db": float(vq_cpps_db),
        "vq_voiced_ratio": float(vq_voiced_ratio),
        "vq_spectral_slope_db": float(vq_spectral_slope_db),
        "vq_reliable": bool(vq_reliable),
        "vq_note": str(vq_note),
        "paralinguistics_flags_json": json.dumps(flags, default=str, separators=(",", ":")),
    }

    return features


def _get_empty_features_v2(
    wpm: float,
    sps: float,
    filler_count: int,
    repetition_count: int,
    false_start_count: int,
    disfluency_rate: float,
    flags: dict[str, Any],
) -> dict[str, Any]:
    """Return placeholders when audio is unavailable."""

    flags["empty_audio"] = True
    return {
        "wpm": round(float(wpm), 2) if not np.isnan(wpm) else np.nan,
        "sps": round(float(sps), 2) if not np.isnan(sps) else np.nan,
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "false_start_count": int(false_start_count),
        "disfluency_rate": float(disfluency_rate),
        "pause_count": 0,
        "pause_total_sec": 0.0,
        "pause_ratio": 0.0,
        "pause_short_count": 0,
        "pause_long_count": 0,
        "pitch_med_hz": np.nan,
        "pitch_iqr_hz": np.nan,
        "pitch_slope_hzps": np.nan,
        "loudness_dbfs_med": np.nan,
        "loudness_dr_db": np.nan,
        "loudness_over_floor_db": np.nan,
        "spectral_centroid_med_hz": np.nan,
        "spectral_centroid_iqr_hz": np.nan,
        "spectral_flatness_med": np.nan,
        "vq_jitter_pct": 0.0,
        "vq_shimmer_db": 0.0,
        "vq_hnr_db": 0.0,
        "vq_cpps_db": 0.0,
        "vq_voiced_ratio": 0.0,
        "vq_spectral_slope_db": 0.0,
        "vq_reliable": False,
        "vq_note": "no_audio",
        "paralinguistics_flags_json": json.dumps(flags, default=str, separators=(",", ":")),
    }


def _get_error_features_v2(error_msg: str) -> dict[str, Any]:
    flags = {"processing_error": error_msg, "error_timestamp": time.time()}
    return {
        "wpm": np.nan,
        "sps": np.nan,
        "filler_count": 0,
        "repetition_count": 0,
        "false_start_count": 0,
        "disfluency_rate": 0.0,
        "pause_count": 0,
        "pause_total_sec": 0.0,
        "pause_ratio": 0.0,
        "pause_short_count": 0,
        "pause_long_count": 0,
        "pitch_med_hz": np.nan,
        "pitch_iqr_hz": np.nan,
        "pitch_slope_hzps": np.nan,
        "loudness_dbfs_med": np.nan,
        "loudness_dr_db": np.nan,
        "loudness_over_floor_db": np.nan,
        "spectral_centroid_med_hz": np.nan,
        "spectral_centroid_iqr_hz": np.nan,
        "spectral_flatness_med": np.nan,
        "vq_jitter_pct": 0.0,
        "vq_shimmer_db": 0.0,
        "vq_hnr_db": 0.0,
        "vq_cpps_db": 0.0,
        "vq_voiced_ratio": 0.0,
        "vq_spectral_slope_db": 0.0,
        "vq_reliable": False,
        "vq_note": "processing_error",
        "paralinguistics_flags_json": json.dumps(flags, default=str, separators=(",", ":")),
    }


def process_segments_batch_v2(
    segments: list[tuple[np.ndarray, int, float, float, str]],
    cfg: ParalinguisticsConfig | None = None,
    progress_callback: callable | None = None,
) -> list[dict[str, Any]]:
    """Batch process segments, optionally in parallel."""

    cfg = cfg or ParalinguisticsConfig()
    results: list[dict[str, Any]] = []
    total_segments = len(segments)
    processed_count = 0

    if cfg.parallel_processing and total_segments >= cfg.max_workers:
        with ThreadPoolExecutor(max_workers=min(cfg.max_workers, total_segments)) as executor:
            futures = [
                executor.submit(compute_segment_features_v2, audio, sr, start, end, text, cfg)
                for audio, sr, start, end, text in segments
            ]
            for idx, future in enumerate(futures):
                try:
                    results.append(future.result(timeout=30))
                    processed_count += 1
                    if progress_callback:
                        progress_callback(processed_count, total_segments)
                except Exception as exc:
                    warnings.warn(f"Segment {idx} processing failed: {exc}")
                    results.append(_get_error_features_v2(str(exc)))
    else:
        for idx, (seg_audio, seg_sr, start, end, seg_text) in enumerate(segments):
            try:
                results.append(compute_segment_features_v2(seg_audio, seg_sr, start, end, seg_text, cfg))
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count, total_segments)
            except Exception as exc:
                warnings.warn(f"Segment {idx} processing failed: {exc}")
                results.append(_get_error_features_v2(str(exc)))

    return results


__all__ = [
    "compute_segment_features_v2",
    "process_segments_batch_v2",
]
