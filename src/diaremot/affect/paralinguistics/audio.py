"""Audio-centric helpers for paralinguistic feature extraction."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np

from .config import ParalinguisticsConfig
from .environment import LIBROSA_AVAILABLE, SCIPY_AVAILABLE, librosa, scipy_signal


def get_optimized_frame_params(sr: int, frame_ms: int, hop_ms: int) -> tuple[int, int]:
    """Return frame and hop lengths tuned for the current sample rate."""

    frame_length = max(1, int(sr * frame_ms / 1000))
    hop_length = max(1, int(sr * hop_ms / 1000))
    return frame_length, hop_length


def vectorized_silence_detection(rms_db: np.ndarray, threshold_db: float, memory_efficient: bool) -> np.ndarray:
    """Return a boolean mask for frames considered silent."""

    if rms_db.size == 0:
        return np.zeros_like(rms_db, dtype=bool)

    silence_mask = rms_db < threshold_db

    if silence_mask.size > 7 and SCIPY_AVAILABLE:
        kernel_size = min(7, max(3, silence_mask.size // 20))
        if kernel_size >= 3:
            if memory_efficient and silence_mask.size > 1000:
                chunk_size = 500
                smoothed = np.empty_like(silence_mask, dtype=bool)
                for idx in range(0, len(silence_mask), chunk_size):
                    end_idx = min(idx + chunk_size, len(silence_mask))
                    chunk = silence_mask[idx:end_idx].astype(np.uint8)
                    smoothed_chunk = scipy_signal.medfilt(chunk, kernel_size)
                    smoothed[idx:end_idx] = smoothed_chunk.astype(bool)
                silence_mask = smoothed
            else:
                silence_mask = scipy_signal.medfilt(silence_mask.astype(np.uint8), kernel_size).astype(bool)

    return silence_mask


def optimized_pause_analysis(
    silence_mask: np.ndarray, sr: int, hop_length: int, cfg: ParalinguisticsConfig
) -> tuple[int, float, float, int, int]:
    """Return pause statistics derived from a silence mask."""

    if silence_mask.size == 0:
        return 0, 0.0, 0.0, 0, 0

    min_pause_frames = max(1, int((cfg.pause_min_ms / 1000.0) * sr / hop_length))
    long_pause_frames = max(1, int((cfg.pause_long_ms / 1000.0) * sr / hop_length))

    padded = np.concatenate(([False], silence_mask, [False]))
    diff = np.diff(padded.astype(int))

    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]

    if len(run_starts) != len(run_ends) or len(run_starts) == 0:
        return 0, 0.0, 0.0, 0, 0

    run_lengths = run_ends - run_starts
    valid_mask = run_lengths >= min_pause_frames
    valid_lengths = run_lengths[valid_mask]

    if len(valid_lengths) == 0:
        return 0, 0.0, 0.0, 0, 0

    long_mask = valid_lengths >= long_pause_frames
    pause_count = len(valid_lengths)
    pause_long_count = int(np.sum(long_mask))
    pause_short_count = pause_count - pause_long_count

    total_pause_frames = int(np.sum(valid_lengths))
    pause_total_sec = (total_pause_frames * hop_length) / sr
    total_duration_sec = (len(silence_mask) * hop_length) / sr
    pause_ratio = pause_total_sec / max(1e-6, total_duration_sec)

    return pause_count, pause_total_sec, pause_ratio, pause_short_count, pause_long_count


@lru_cache(maxsize=32)
def get_cached_pitch_params(sr: int, cfg: ParalinguisticsConfig) -> dict[str, Any]:
    """Return cached parameters for pitch extraction."""

    return {
        "fmin": cfg.f0_min_hz,
        "fmax": cfg.f0_max_hz,
        "frame_length": cfg.pitch_frame_length,
        "hop_length": cfg.pitch_hop_length,
        "center": True,
        "fill_na": np.nan,
    }


def robust_pitch_extraction(audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Extract pitch using PYIN/YIN with chunked fallbacks."""

    if not LIBROSA_AVAILABLE or audio.size == 0:
        return np.array([]), np.array([])

    if len(audio) > sr * cfg.max_audio_length_sec:
        warnings.warn(
            f"Audio longer than {cfg.max_audio_length_sec}s, chunking for memory efficiency",
            RuntimeWarning,
            stacklevel=2,
        )
        chunk_size = int(sr * cfg.max_audio_length_sec)
        overlap = chunk_size // 4

        f0_chunks: list[np.ndarray] = []
        voiced_chunks: list[np.ndarray] = []

        for start in range(0, len(audio), chunk_size - overlap):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]

            if len(chunk) < sr * 0.1:
                continue

            f0_chunk, voiced_chunk = single_pitch_extraction(chunk, sr, cfg)

            if f0_chunk.size > 0:
                f0_chunks.append(f0_chunk)
                voiced_chunks.append(voiced_chunk)

        if f0_chunks:
            return np.concatenate(f0_chunks), np.concatenate(voiced_chunks)
        return np.array([]), np.array([])

    return single_pitch_extraction(audio, sr, cfg)


def single_pitch_extraction(audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Single pass pitch extraction with graceful fallbacks."""

    if not LIBROSA_AVAILABLE:
        return np.array([]), np.array([])

    params = get_cached_pitch_params(sr, cfg)

    try:
        if cfg.pitch_method == "pyin":
            f0, voiced_flag, _voiced_probs = librosa.pyin(audio, **params)
            return f0, voiced_flag
        f0 = librosa.yin(audio, **params)
        voiced_flag = ~np.isnan(f0)
        return f0, voiced_flag
    except Exception as exc:
        warnings.warn(f"Primary pitch extraction failed: {exc}, trying fallback", RuntimeWarning, stacklevel=2)
        try:
            f0 = librosa.yin(audio, fmin=cfg.f0_min_hz, fmax=cfg.f0_max_hz)
            voiced_flag = ~np.isnan(f0)
            return f0, voiced_flag
        except Exception as inner_exc:
            warnings.warn(f"All pitch extraction methods failed: {inner_exc}", RuntimeWarning, stacklevel=2)
            return np.array([]), np.array([])


def enhanced_pitch_statistics(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray | None,
    cfg: ParalinguisticsConfig,
) -> tuple[float, float, float]:
    """Return robust pitch statistics with outlier handling."""

    if f0.size == 0 or np.sum(voiced_flag) == 0:
        return np.nan, np.nan, np.nan

    coverage = np.mean(voiced_flag)
    if coverage < cfg.pitch_min_coverage:
        return np.nan, np.nan, np.nan

    voiced_f0 = f0[voiced_flag]

    q25, q75 = np.percentile(voiced_f0, [25, 75])
    iqr = q75 - q25

    if iqr > 0:
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        clean_mask = (voiced_f0 >= lower_bound) & (voiced_f0 <= upper_bound)
        if np.sum(clean_mask) > len(voiced_f0) * 0.5:
            voiced_f0 = voiced_f0[clean_mask]

    median_f0 = float(np.median(voiced_f0))
    q25_clean, q75_clean = np.percentile(voiced_f0, [25, 75])
    iqr_f0 = float(q75_clean - q25_clean)

    slope_f0 = np.nan
    if times is not None and len(voiced_f0) >= 10:
        try:
            if len(voiced_f0) > 100:
                indices = np.linspace(0, len(voiced_f0) - 1, 50, dtype=int)
                times_sub = times[voiced_flag][indices]
                f0_sub = voiced_f0[indices]
            else:
                times_sub = times[voiced_flag] if len(times) == len(f0) else np.arange(len(voiced_f0))
                f0_sub = voiced_f0

            if len(f0_sub) >= 3:
                slopes = []
                step = max(1, len(f0_sub) // 10)
                for idx in range(0, len(f0_sub) - step, step):
                    dt = times_sub[idx + step] - times_sub[idx]
                    if dt > 0:
                        df = f0_sub[idx + step] - f0_sub[idx]
                        slopes.append(df / dt)

                if slopes:
                    slope_f0 = float(np.median(slopes))
        except Exception:
            pass

    return median_f0, iqr_f0, slope_f0


def optimized_spectral_features(audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig) -> tuple[float, float, float]:
    """Compute spectral centroid/flatness statistics with fallbacks."""

    if not LIBROSA_AVAILABLE or audio.size == 0:
        return np.nan, np.nan, np.nan

    try:
        hop_length = cfg.spectral_hop_length
        n_fft = cfg.spectral_n_fft

        if cfg.enable_memory_optimization and len(audio) > sr * 10:
            chunk_duration = 5.0
            chunk_samples = int(sr * chunk_duration)
            centroids: list[float] = []
            flatnesses: list[float] = []

            for start in range(0, len(audio), chunk_samples):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]

                if len(chunk) < n_fft:
                    continue

                spec_cent = librosa.feature.spectral_centroid(
                    y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True
                )[0]
                spec_flat = librosa.feature.spectral_flatness(
                    y=chunk, n_fft=n_fft, hop_length=hop_length, center=True
                )[0]

                if spec_cent.size > 0:
                    centroids.extend(spec_cent)
                if spec_flat.size > 0:
                    flatnesses.extend(spec_flat)

            if centroids and flatnesses:
                centroids_arr = np.array(centroids)
                flatness_arr = np.array(flatnesses)
            else:
                return np.nan, np.nan, np.nan
        else:
            centroids_arr = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True
            )[0]
            flatness_arr = librosa.feature.spectral_flatness(
                y=audio, n_fft=n_fft, hop_length=hop_length, center=True
            )[0]

        if centroids_arr.size > 0:
            cent_med = float(np.median(centroids_arr))
            if centroids_arr.size > 4:
                q25, q75 = np.percentile(centroids_arr, [25, 75])
                cent_iqr = float(q75 - q25)
            else:
                cent_iqr = 0.0
        else:
            cent_med = cent_iqr = np.nan

        flat_med = float(np.median(flatness_arr)) if flatness_arr.size > 0 else np.nan
        return cent_med, cent_iqr, flat_med
    except Exception as exc:
        warnings.warn(f"Spectral feature computation failed: {exc}", RuntimeWarning, stacklevel=2)
        return np.nan, np.nan, np.nan


def advanced_audio_quality_assessment(audio: np.ndarray, sr: int) -> tuple[float, bool, str]:
    """Estimate SNR and reliability of a segment for voice quality features."""

    if audio.size == 0:
        return -60.0, False, "empty"

    duration = len(audio) / sr
    if duration < 0.1:
        return -40.0, False, "too_short"

    try:
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return -60.0, False, "silent"

        frame_size = min(sr // 20, len(audio) // 20)
        if frame_size < 64:
            return -30.0, False, "insufficient_length"

        hop_size = frame_size // 2
        frame_energies = [np.mean(audio[idx : idx + frame_size] ** 2) for idx in range(0, len(audio) - frame_size, hop_size)]

        if len(frame_energies) < 5:
            return -30.0, False, "insufficient_frames"

        energies = np.array(frame_energies)
        noise_floor = np.percentile(energies, 5)
        signal_level = np.percentile(energies, 85)

        if noise_floor <= 0:
            snr_db = 10.0
        else:
            snr_db = 10 * np.log10((signal_level + 1e-12) / (noise_floor + 1e-12))

        clipping_ratio = np.mean(np.abs(audio) > 0.95)
        dynamic_range_db = 10 * np.log10((np.max(energies) + 1e-12) / (noise_floor + 1e-12))

        reliable = (
            duration >= 0.2
            and snr_db >= 6.0
            and clipping_ratio < 0.03
            and dynamic_range_db >= 12.0
        )

        if not reliable:
            if snr_db < 6.0:
                status = f"low_snr_{snr_db:.1f}dB"
            elif clipping_ratio >= 0.03:
                status = f"clipping_{clipping_ratio:.2%}"
            elif dynamic_range_db < 12.0:
                status = f"low_dr_{dynamic_range_db:.1f}dB"
            else:
                status = "short_duration"
        else:
            status = f"reliable_snr_{snr_db:.1f}dB"

        return float(snr_db), reliable, status
    except Exception as exc:
        warnings.warn(f"Audio quality assessment failed: {exc}", RuntimeWarning, stacklevel=2)
        return 5.0, False, "analysis_error"


def compute_rms_fallback(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Compute RMS using numpy when librosa is unavailable."""

    if audio.size == 0:
        return np.array([])

    frame_length = min(frame_length, len(audio))
    if frame_length < 1:
        return np.array([])

    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    rms_values = np.zeros(num_frames)

    for idx in range(num_frames):
        start_idx = idx * hop_length
        end_idx = min(start_idx + frame_length, len(audio))
        if end_idx > start_idx:
            frame = audio[start_idx:end_idx]
            rms_values[idx] = np.sqrt(np.mean(frame**2))

    return 20 * np.log10(rms_values + 1e-12)


__all__ = [
    "get_optimized_frame_params",
    "vectorized_silence_detection",
    "optimized_pause_analysis",
    "get_cached_pitch_params",
    "robust_pitch_extraction",
    "single_pitch_extraction",
    "enhanced_pitch_statistics",
    "optimized_spectral_features",
    "advanced_audio_quality_assessment",
    "compute_rms_fallback",
]
