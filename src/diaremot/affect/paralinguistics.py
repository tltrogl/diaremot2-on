"""Production-optimized paralinguistic feature extraction utilities.

Provides advanced speech metrics such as words per minute (WPM) and syllables
per second (SPS). Both metrics are stored with two-decimal precision. The
module targets CPU-only deployments and includes enhanced voice-quality
analysis.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

# Core imports with graceful fallbacks
try:
    import librosa
    import librosa.feature

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available - paralinguistic features will be limited")

try:
    from scipy import signal as scipy_signal

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - some voice quality features disabled")

# High-quality voice analysis (Praat algorithms via Python)
try:
    import parselmouth
    from parselmouth.praat import call

    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

# Suppress performance-impacting warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")

# ============================================================================
# Enhanced Configuration for Production Pipeline
# ============================================================================


@dataclass(frozen=True)
class ParalinguisticsConfig:
    """Production-optimized configuration for paralinguistic analysis"""

    # Frame processing (balanced speed vs accuracy)
    frame_ms: int = 25  # Optimal for speech analysis
    hop_ms: int = 10  # Standard hop for most features

    # Adaptive silence detection
    base_silence_dbfs: float = -45.0  # Conservative base threshold
    adaptive_silence: bool = True
    silence_floor_percentile: int = 5  # Use 5th percentile as noise floor
    silence_margin_db: float = 8.0  # Margin above noise floor

    # Optimized pause detection
    pause_min_ms: int = 200  # Minimum meaningful pause
    pause_long_ms: int = 600  # Long pause threshold

    # Robust pitch analysis
    f0_min_hz: float = 65.0  # Lower bound for human speech
    f0_max_hz: float = 400.0  # Upper bound for human speech
    pitch_method: str = "pyin"  # Most robust method
    pitch_frame_length: int = 2048
    pitch_hop_length: int = 512
    pitch_min_coverage: float = 0.05  # Minimum voiced content
    pitch_interp_max_gap_ms: int = 100  # Gap interpolation limit

    # Enhanced voice quality analysis
    voice_quality_enabled: bool = True
    vq_min_duration_sec: float = 0.6  # Minimum segment duration (conversational)
    vq_min_snr_db: float = 10.0  # Minimum SNR for reliability
    vq_use_parselmouth: bool = True  # Prefer Praat algorithms
    vq_fallback_enabled: bool = True  # Always provide fallback

    # Text analysis features
    syllable_estimation: bool = True
    disfluency_detection: bool = True

    # Optimized spectral features
    spectral_features_enabled: bool = True
    spectral_n_fft: int = 1024  # Balanced resolution vs speed
    spectral_hop_length: int = 256

    # CPU performance optimizations
    use_vectorized_ops: bool = True
    enable_caching: bool = True
    parallel_processing: bool = False  # Conservative default
    # Dynamically scale workers with available CPU cores (override via config)
    max_workers: int = field(default_factory=lambda: os.cpu_count() or 2)

    # FIXED: Missing backchannel detection parameter
    backchannel_max_ms: int = 300  # Max duration for backchannel classification

    # Memory optimization
    max_audio_length_sec: float = 30.0  # Chunk longer audio
    enable_memory_optimization: bool = True


# Enhanced feature sets
COMPREHENSIVE_FILLER_WORDS = frozenset(
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

VOWELS = frozenset("aeiouyAEIOUY")

# ============================================================================
# Optimized Core Functions with Enhanced CPU Performance
# ============================================================================


@lru_cache(maxsize=64)
def _get_optimized_frame_params(sr: int, frame_ms: int, hop_ms: int) -> tuple[int, int]:
    """CPU-optimized frame/hop calculation with power-of-2 alignment"""
    frame = max(64, int(sr * frame_ms / 1000.0))
    hop = max(32, int(sr * hop_ms / 1000.0))

    # Ensure FFT-friendly sizes (powers of 2)
    frame = int(2 ** np.ceil(np.log2(frame)))
    hop = min(hop, frame // 4)  # Prevent excessive overlap

    return frame, hop


def _vectorized_silence_detection_v2(
    rms_db: np.ndarray, silence_threshold: float, memory_efficient: bool = True
) -> np.ndarray:
    """Enhanced vectorized silence detection with memory optimization"""

    if rms_db.size == 0:
        return np.array([], dtype=bool)

    # Primary silence mask (vectorized)
    silence_mask = rms_db < silence_threshold

    # Apply smoothing for spurious detection reduction
    if rms_db.size > 7 and SCIPY_AVAILABLE:
        kernel_size = min(7, max(3, rms_db.size // 20))
        if kernel_size >= 3:
            # Use median filter for noise immunity
            if memory_efficient and rms_db.size > 1000:
                # Process in chunks for large arrays
                chunk_size = 500
                smoothed = np.empty_like(silence_mask, dtype=bool)
                for i in range(0, len(silence_mask), chunk_size):
                    end_idx = min(i + chunk_size, len(silence_mask))
                    chunk = silence_mask[i:end_idx].astype(np.uint8)
                    smoothed_chunk = scipy_signal.medfilt(chunk, kernel_size)
                    smoothed[i:end_idx] = smoothed_chunk.astype(bool)
                silence_mask = smoothed
            else:
                silence_mask = scipy_signal.medfilt(
                    silence_mask.astype(np.uint8), kernel_size
                ).astype(bool)

    return silence_mask


def _optimized_pause_analysis(
    silence_mask: np.ndarray, sr: int, hop_length: int, cfg: ParalinguisticsConfig
) -> tuple[int, float, float, int, int]:
    """CPU-optimized pause analysis with vectorized operations"""

    if silence_mask.size == 0:
        return 0, 0.0, 0.0, 0, 0

    # Frame count thresholds
    min_pause_frames = max(1, int((cfg.pause_min_ms / 1000.0) * sr / hop_length))
    long_pause_frames = max(1, int((cfg.pause_long_ms / 1000.0) * sr / hop_length))

    # Efficient run-length encoding using diff
    padded = np.concatenate(([False], silence_mask, [False]))
    diff = np.diff(padded.astype(int))

    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]

    if len(run_starts) != len(run_ends) or len(run_starts) == 0:
        return 0, 0.0, 0.0, 0, 0

    # Vectorized length calculation and filtering
    run_lengths = run_ends - run_starts
    valid_mask = run_lengths >= min_pause_frames
    valid_lengths = run_lengths[valid_mask]

    if len(valid_lengths) == 0:
        return 0, 0.0, 0.0, 0, 0

    # Statistics computation (vectorized)
    long_mask = valid_lengths >= long_pause_frames
    pause_count = len(valid_lengths)
    pause_long_count = int(np.sum(long_mask))
    pause_short_count = pause_count - pause_long_count

    # Time calculations
    total_pause_frames = int(np.sum(valid_lengths))
    pause_total_sec = (total_pause_frames * hop_length) / sr
    total_duration_sec = (len(silence_mask) * hop_length) / sr
    pause_ratio = pause_total_sec / max(1e-6, total_duration_sec)

    return (
        pause_count,
        pause_total_sec,
        pause_ratio,
        pause_short_count,
        pause_long_count,
    )


@lru_cache(maxsize=32)
def _get_cached_pitch_params(sr: int, cfg: ParalinguisticsConfig) -> dict:
    """Cached pitch extraction parameters for CPU optimization"""
    return {
        "fmin": cfg.f0_min_hz,
        "fmax": cfg.f0_max_hz,
        "frame_length": cfg.pitch_frame_length,
        "hop_length": cfg.pitch_hop_length,
        "center": True,
        "fill_na": np.nan,  # Explicit NaN handling
    }


def _robust_pitch_extraction_v2(
    audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced robust pitch extraction with multiple fallbacks"""

    if not LIBROSA_AVAILABLE or audio.size == 0:
        return np.array([]), np.array([])

    # Memory optimization for long audio
    if len(audio) > sr * cfg.max_audio_length_sec:
        warnings.warn(
            f"Audio longer than {cfg.max_audio_length_sec}s, chunking for memory efficiency"
        )
        # Process in overlapping chunks
        chunk_size = int(sr * cfg.max_audio_length_sec)
        overlap = chunk_size // 4

        f0_chunks = []
        voiced_chunks = []

        for start in range(0, len(audio), chunk_size - overlap):
            end = min(start + chunk_size, len(audio))
            chunk = audio[start:end]

            if len(chunk) < sr * 0.1:  # Skip very short chunks
                continue

            f0_chunk, voiced_chunk = _single_pitch_extraction(chunk, sr, cfg)

            if f0_chunk.size > 0:
                f0_chunks.append(f0_chunk)
                voiced_chunks.append(voiced_chunk)

        if f0_chunks:
            return np.concatenate(f0_chunks), np.concatenate(voiced_chunks)
        else:
            return np.array([]), np.array([])
    else:
        return _single_pitch_extraction(audio, sr, cfg)


def _single_pitch_extraction(
    audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Single-chunk pitch extraction with fallback methods"""
    if not LIBROSA_AVAILABLE:
        return np.array([]), np.array([])

    params = _get_cached_pitch_params(sr, cfg)

    try:
        # Primary method: PYIN (most robust)
        if cfg.pitch_method == "pyin":
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, **params)
            return f0, voiced_flag
        else:
            # FIXED: Corrected fallback method comment
            # Fallback: YIN method
            f0 = librosa.yin(audio, **params)
            voiced_flag = ~np.isnan(f0)
            return f0, voiced_flag

    except Exception as e:
        warnings.warn(f"Primary pitch extraction failed: {e}, trying fallback")

        try:
            # Secondary fallback: basic YIN
            f0 = librosa.yin(audio, fmin=cfg.f0_min_hz, fmax=cfg.f0_max_hz)
            voiced_flag = ~np.isnan(f0)
            return f0, voiced_flag

        except Exception as e2:
            warnings.warn(f"All pitch extraction methods failed: {e2}")
            return np.array([]), np.array([])


def _enhanced_pitch_statistics(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray | None,
    cfg: ParalinguisticsConfig,
) -> tuple[float, float, float]:
    """Enhanced pitch statistics with robust outlier handling"""

    if f0.size == 0 or np.sum(voiced_flag) == 0:
        return np.nan, np.nan, np.nan

    # Voice coverage validation
    coverage = np.mean(voiced_flag)
    if coverage < cfg.pitch_min_coverage:
        return np.nan, np.nan, np.nan

    # Extract voiced segments with outlier filtering
    voiced_f0 = f0[voiced_flag]

    # Robust outlier removal (IQR method)
    q25, q75 = np.percentile(voiced_f0, [25, 75])
    iqr = q75 - q25

    if iqr > 0:
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        clean_mask = (voiced_f0 >= lower_bound) & (voiced_f0 <= upper_bound)

        if np.sum(clean_mask) > len(voiced_f0) * 0.5:  # Keep majority of data
            voiced_f0 = voiced_f0[clean_mask]

    # Robust statistics
    median_f0 = float(np.median(voiced_f0))
    q25_clean, q75_clean = np.percentile(voiced_f0, [25, 75])
    iqr_f0 = float(q75_clean - q25_clean)

    # Enhanced slope estimation
    slope_f0 = np.nan
    if times is not None and len(voiced_f0) >= 10:
        try:
            # Robust slope using Theil-Sen estimator approximation
            if len(voiced_f0) > 100:
                # Subsample for efficiency
                indices = np.linspace(0, len(voiced_f0) - 1, 50, dtype=int)
                times_sub = times[voiced_flag][indices]
                f0_sub = voiced_f0[indices]
            else:
                times_sub = (
                    times[voiced_flag] if len(times) == len(f0) else np.arange(len(voiced_f0))
                )
                f0_sub = voiced_f0

            if len(f0_sub) >= 3:
                # Use robust regression (median of slopes)
                slopes = []
                step = max(1, len(f0_sub) // 10)
                for i in range(0, len(f0_sub) - step, step):
                    dt = times_sub[i + step] - times_sub[i]
                    if dt > 0:
                        df = f0_sub[i + step] - f0_sub[i]
                        slopes.append(df / dt)

                if slopes:
                    slope_f0 = float(np.median(slopes))
        except Exception:
            pass

    return median_f0, iqr_f0, slope_f0


def _optimized_spectral_features(
    audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig
) -> tuple[float, float, float]:
    """CPU-optimized spectral feature computation"""

    if not LIBROSA_AVAILABLE or audio.size == 0:
        return np.nan, np.nan, np.nan

    try:
        hop_length = cfg.spectral_hop_length
        n_fft = cfg.spectral_n_fft

        # Efficient spectral computation with memory optimization
        if cfg.enable_memory_optimization and len(audio) > sr * 10:
            # Process in chunks for long audio
            chunk_duration = 5.0  # seconds
            chunk_samples = int(sr * chunk_duration)

            centroids = []
            flatnesses = []

            for start in range(0, len(audio), chunk_samples):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]

                if len(chunk) < n_fft:
                    continue

                # Spectral centroid
                spec_cent = librosa.feature.spectral_centroid(
                    y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True
                )[0]

                # Spectral flatness
                spec_flat = librosa.feature.spectral_flatness(
                    y=chunk, n_fft=n_fft, hop_length=hop_length, center=True
                )[0]

                if spec_cent.size > 0:
                    centroids.extend(spec_cent)
                if spec_flat.size > 0:
                    flatnesses.extend(spec_flat)

            if centroids and flatnesses:
                centroids = np.array(centroids)
                flatnesses = np.array(flatnesses)
            else:
                return np.nan, np.nan, np.nan
        else:
            # Direct computation for shorter audio
            centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, center=True
            )[0]

            flatnesses = librosa.feature.spectral_flatness(
                y=audio, n_fft=n_fft, hop_length=hop_length, center=True
            )[0]

        # Robust statistics with outlier handling
        if centroids.size > 0:
            cent_med = float(np.median(centroids))
            if centroids.size > 4:
                q25, q75 = np.percentile(centroids, [25, 75])
                cent_iqr = float(q75 - q25)
            else:
                cent_iqr = 0.0
        else:
            cent_med = cent_iqr = np.nan

        if flatnesses.size > 0:
            flat_med = float(np.median(flatnesses))
        else:
            flat_med = np.nan

        return cent_med, cent_iqr, flat_med

    except Exception as e:
        warnings.warn(f"Spectral feature computation failed: {e}")
        return np.nan, np.nan, np.nan


# ============================================================================
# Enhanced Voice Quality Analysis with Latest Parselmouth Techniques
# ============================================================================


def _advanced_audio_quality_assessment(audio: np.ndarray, sr: int) -> tuple[float, bool, str]:
    """Advanced audio quality assessment for voice quality reliability"""

    if audio.size == 0:
        return -60.0, False, "empty"

    duration = len(audio) / sr
    if duration < 0.1:
        return -40.0, False, "too_short"

    # Enhanced SNR estimation using spectral methods
    try:
        # RMS-based energy analysis
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return -60.0, False, "silent"

        # Frame-based analysis for noise floor estimation
        frame_size = min(sr // 20, len(audio) // 20)  # 50ms frames
        if frame_size < 64:
            return -30.0, False, "insufficient_length"

        hop_size = frame_size // 2
        frame_energies = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i : i + frame_size]
            energy = np.mean(frame**2)
            frame_energies.append(energy)

        if len(frame_energies) < 5:
            return -30.0, False, "insufficient_frames"

        energies = np.array(frame_energies)

        # Advanced noise floor estimation (5th percentile)
        noise_floor = np.percentile(energies, 5)
        signal_level = np.percentile(energies, 85)  # Voice activity level

        if noise_floor <= 0:
            snr_db = 10.0  # Default reasonable value
        else:
            snr_db = 10 * np.log10((signal_level + 1e-12) / (noise_floor + 1e-12))

        # Clipping detection
        clipping_ratio = np.mean(np.abs(audio) > 0.95)

        # Dynamic range assessment
        dynamic_range_db = 10 * np.log10((np.max(energies) + 1e-12) / (noise_floor + 1e-12))

        # Reliability assessment with enhanced criteria
        reliable = (
            duration >= 0.2  # Minimum duration
            and snr_db >= 6.0  # Adequate signal-to-noise ratio
            and clipping_ratio < 0.03  # Minimal clipping
            and dynamic_range_db >= 12.0  # Sufficient dynamic range
        )

        # Quality status
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

    except Exception as e:
        warnings.warn(f"Audio quality assessment failed: {e}")
        return 5.0, False, "analysis_error"


def _compute_voice_quality_parselmouth_v2(
    audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig
) -> dict[str, float]:
    """Enhanced voice quality analysis using latest Parselmouth techniques"""

    if not PARSELMOUTH_AVAILABLE:
        return _compute_voice_quality_fallback_v2(audio, sr, cfg)

    try:
        # Create Sound object with optimal settings
        sound = parselmouth.Sound(audio, sampling_frequency=sr)

        # Enhanced pitch analysis for voice quality
        pitch = call(sound, "To Pitch", 0.0, cfg.f0_min_hz, cfg.f0_max_hz)

        # Point Process with optimized settings
        point_process = call(sound, "To PointProcess (periodic, cc)", cfg.f0_min_hz, cfg.f0_max_hz)

        results = {}

        # Jitter analysis (multiple measures)
        try:
            jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            results["jitter_pct"] = float(jitter_local * 100) if not np.isnan(jitter_local) else 0.0
        except Exception:
            results["jitter_pct"] = 0.0

        # Shimmer analysis (enhanced measures)
        try:
            shimmer_local_db = call(
                [sound, point_process],
                "Get shimmer (local_dB)",
                0.0,
                0.0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            results["shimmer_db"] = (
                float(shimmer_local_db) if not np.isnan(shimmer_local_db) else 0.0
            )
        except Exception:
            results["shimmer_db"] = 0.0

        # Harmonics-to-Noise Ratio (enhanced)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, cfg.f0_min_hz, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0.0, 0.0)
            results["hnr_db"] = float(hnr) if not np.isnan(hnr) else 0.0
        except Exception:
            results["hnr_db"] = 0.0

        # CPPS (Cepstral Peak Prominence Smoothed) - latest method
        try:
            # Enhanced CPPS computation with optimized parameters
            spectrum = call(sound, "To Spectrum", True)
            cepstrum = call(spectrum, "To PowerCepstrum")
            cpps = call(
                cepstrum,
                "Get CPPS",
                False,
                0.02,
                0.0005,
                cfg.f0_min_hz,
                cfg.f0_max_hz,
                0.05,
                "Parabolic",
                0.001,
                0.05,
                "Straight",
                "Robust",
            )
            results["cpps_db"] = float(cpps) if not np.isnan(cpps) else 0.0
        except Exception:
            # Fallback CPPS approximation
            try:
                spectrum = call(sound, "To Spectrum", True)
                cepstrum = call(spectrum, "To PowerCepstrum")
                cpps_simple = call(
                    cepstrum,
                    "Get CPPS",
                    False,
                    0.02,
                    0.0005,
                    cfg.f0_min_hz,
                    cfg.f0_max_hz,
                    0.05,
                )
                results["cpps_db"] = float(cpps_simple) if not np.isnan(cpps_simple) else 0.0
            except Exception:
                # Use HNR as CPPS approximation
                results["cpps_db"] = max(0.0, results.get("hnr_db", 0.0) - 2.0)

        # Advanced voicing measures
        try:
            f0_values = pitch.selected_array["frequency"]
            voiced_frames = ~np.isnan(f0_values) if f0_values.size > 0 else np.array([])
            results["voiced_ratio"] = (
                float(np.mean(voiced_frames)) if voiced_frames.size > 0 else 0.0
            )
        except Exception:
            results["voiced_ratio"] = 0.0

        # Additional voice quality measures (if computational resources allow)
        try:
            # Spectral slope (breathiness indicator)
            ltas = call(sound, "To Ltas", 100)
            slope = call(ltas, "Get slope", 0, 1000, 4000, "dB")
            results["spectral_slope_db"] = float(slope) if not np.isnan(slope) else 0.0
        except Exception:
            results["spectral_slope_db"] = 0.0

        return results

    except Exception as e:
        warnings.warn(f"Enhanced Parselmouth analysis failed: {e}")
        return _compute_voice_quality_fallback_v2(audio, sr, cfg)


def _compute_voice_quality_fallback_v2(
    audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig
) -> dict[str, float]:
    """Enhanced fallback voice quality estimation with improved algorithms"""

    try:
        # Enhanced pitch tracking
        f0, voiced_flag = _robust_pitch_extraction_v2(audio, sr, cfg)

        if f0.size == 0:
            return {
                "jitter_pct": 0.0,
                "shimmer_db": 0.0,
                "hnr_db": 0.0,
                "cpps_db": 0.0,
                "voiced_ratio": 0.0,
                "spectral_slope_db": 0.0,
            }

        # Voiced ratio
        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size > 0 else 0.0

        # Enhanced jitter estimation (period-to-period variation)
        jitter_pct = 0.0
        if np.sum(voiced_flag) > 10:
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 3:
                # Convert F0 to periods
                periods = 1.0 / (voiced_f0 + 1e-12)
                # Calculate relative jitter
                if len(periods) > 1:
                    period_diffs = np.abs(np.diff(periods))
                    mean_period = np.mean(periods)
                    jitter_pct = float((np.mean(period_diffs) / mean_period) * 100)
                    jitter_pct = np.clip(jitter_pct, 0.0, 25.0)  # Reasonable bounds

        # Enhanced shimmer estimation (amplitude variation)
        shimmer_db = 0.0
        if LIBROSA_AVAILABLE and audio.size > sr * 0.1:  # At least 100ms
            try:
                # Frame-based amplitude analysis
                hop_length = min(256, len(audio) // 20)
                rms_frames = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

                if len(rms_frames) > 5:
                    # Convert to dB and calculate variation
                    rms_db = librosa.amplitude_to_db(rms_frames + 1e-12)
                    # Robust shimmer: median absolute deviation of amplitude differences
                    db_diffs = np.diff(rms_db)
                    shimmer_db = float(np.median(np.abs(db_diffs - np.median(db_diffs))))
                    shimmer_db = np.clip(shimmer_db, 0.0, 15.0)
            except Exception:
                pass

        # Enhanced HNR estimation using spectral regularity
        hnr_db = 0.0
        if LIBROSA_AVAILABLE:
            try:
                # Use spectral contrast as HNR proxy
                contrast = librosa.feature.spectral_contrast(
                    y=audio, sr=sr, hop_length=512, n_bands=6, fmin=cfg.f0_min_hz
                )
                if contrast.size > 0:
                    # Average contrast across frequency bands and time
                    avg_contrast = np.mean(contrast)
                    hnr_db = float(np.clip(avg_contrast, 0.0, 40.0))

                # Alternative: spectral regularity measure
                if hnr_db == 0.0:
                    stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=1024))
                    if stft_mag.size > 0:
                        # Harmonic regularity approximation
                        spectral_peaks = np.max(stft_mag, axis=0)
                        spectral_means = np.mean(stft_mag, axis=0)
                        regularity = spectral_peaks / (spectral_means + 1e-12)
                        hnr_db = float(
                            np.clip(20 * np.log10(np.mean(regularity) + 1e-12), 0.0, 35.0)
                        )
            except Exception:
                hnr_db = 8.0  # Conservative default

        # CPPS approximation from spectral characteristics
        cpps_db = 0.0
        if LIBROSA_AVAILABLE and hnr_db > 0:
            try:
                # Cepstral analysis approximation
                stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=2048))
                if stft_mag.shape[0] > 100:  # Sufficient frequency resolution
                    # Log spectrum for cepstral domain
                    log_spec = np.log(stft_mag + 1e-12)
                    # Simple cepstral peak prominence approximation
                    cepstrum_frames = []
                    for frame in range(log_spec.shape[1]):
                        frame_spec = log_spec[:, frame]
                        # Real cepstrum approximation (DCT)
                        if SCIPY_AVAILABLE:
                            cepstrum = scipy_signal.fftconvolve(
                                frame_spec, frame_spec[::-1], mode="full"
                            )
                            if len(cepstrum) > 50:
                                # Look for cepstral peak in expected F0 range
                                quefrency_range = slice(20, min(100, len(cepstrum) // 2))
                                peak_val = np.max(cepstrum[quefrency_range])
                                cepstrum_frames.append(peak_val)

                    if cepstrum_frames:
                        cpps_db = float(np.clip(np.mean(cepstrum_frames) * 0.1, 0.0, 35.0))
            except Exception:
                pass

            if cpps_db == 0.0:
                # Simple approximation from HNR
                cpps_db = max(0.0, hnr_db - 3.0)

        # Spectral slope estimation
        spectral_slope_db = 0.0
        if LIBROSA_AVAILABLE:
            try:
                # Compute long-term average spectrum (LTAS)
                stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=2048))
                ltas = np.mean(stft_mag, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

                # Focus on 1-4 kHz range for spectral slope (breathiness indicator)
                freq_mask = (freqs >= 1000) & (freqs <= 4000)
                if np.sum(freq_mask) > 10:
                    slope_freqs = freqs[freq_mask]
                    slope_mag = ltas[freq_mask]

                    # Linear regression in log domain
                    if np.any(slope_mag > 0):
                        log_mag = np.log10(slope_mag + 1e-12)
                        log_freq = np.log10(slope_freqs)

                        # Simple slope calculation
                        if len(log_freq) > 2:
                            slope = (log_mag[-1] - log_mag[0]) / (log_freq[-1] - log_freq[0])
                            spectral_slope_db = float(slope * 20)  # Convert to dB/decade
            except Exception:
                pass

        # Clamp all values to reasonable ranges
        results = {
            "jitter_pct": float(np.clip(jitter_pct, 0.0, 25.0)),
            "shimmer_db": float(np.clip(shimmer_db, 0.0, 15.0)),
            "hnr_db": float(np.clip(hnr_db, 0.0, 40.0)),
            "cpps_db": float(np.clip(cpps_db, 0.0, 35.0)),
            "voiced_ratio": float(np.clip(voiced_ratio, 0.0, 1.0)),
            "spectral_slope_db": float(np.clip(spectral_slope_db, -30.0, 10.0)),
        }

        return results

    except Exception as e:
        warnings.warn(f"Enhanced fallback voice quality failed: {e}")
        return {
            "jitter_pct": 0.0,
            "shimmer_db": 0.0,
            "hnr_db": 0.0,
            "cpps_db": 0.0,
            "voiced_ratio": 0.0,
            "spectral_slope_db": 0.0,
        }


# ============================================================================
# Enhanced Text Analysis with Advanced NLP Techniques
# ============================================================================


@lru_cache(maxsize=256)
def _enhanced_word_tokenization(text: str) -> tuple[str, ...]:
    """Enhanced word tokenization preserving contractions and handling edge cases"""
    if not text:
        return ()

    # Enhanced tokenization preserving meaningful punctuation
    words = []
    current_word = []

    for char in text.lower():
        if char.isalnum():
            current_word.append(char)
        elif char in ("'", "-") and current_word:  # Contractions and hyphenated words
            current_word.append(char)
        else:
            if current_word:
                word = "".join(current_word)
                # Filter out very short fragments unless they're meaningful
                if len(word) > 1 or word in ("i", "a"):
                    words.append(word)
                current_word = []

    # Handle final word
    if current_word:
        word = "".join(current_word)
        if len(word) > 1 or word in ("i", "a"):
            words.append(word)

    return tuple(words)


def _advanced_disfluency_detection(words: tuple[str, ...], raw_text: str) -> tuple[int, int, int]:
    """Advanced disfluency detection with improved pattern recognition"""

    if not words:
        return 0, 0, 0

    words_array = np.array(words)
    text_lower = f" {raw_text.lower()} "

    # Enhanced filler detection
    filler_count = 0

    # Multi-word filler phrases
    multi_word_fillers = [
        "you know",
        "i mean",
        "sort of",
        "kind of",
        "let me see",
        "let's see",
    ]
    for phrase in multi_word_fillers:
        filler_count += text_lower.count(f" {phrase} ")

    # Single-word fillers (vectorized)
    single_word_fillers = COMPREHENSIVE_FILLER_WORDS - {
        phrase for phrase in multi_word_fillers if " " in phrase
    }
    for filler in single_word_fillers:
        if " " not in filler:  # Ensure single word
            filler_count += int(np.sum(words_array == filler))

    # Enhanced repetition detection
    repetition_count = 0
    if len(words) > 1:
        # Consecutive identical words (excluding articles and short function words)
        exclude_from_repetition = {"the", "a", "an", "to", "of", "in", "on", "at", "by"}

        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]

            if (
                current_word == next_word
                and len(current_word) > 2
                and current_word not in exclude_from_repetition
            ):
                repetition_count += 1

        # Non-consecutive repetition within short spans (2-3 words apart)
        for i in range(len(words) - 3):
            if words[i] == words[i + 2] and len(words[i]) > 3:
                # Check if it's likely a repair (not just natural repetition)
                middle_word = words[i + 1]
                if middle_word in COMPREHENSIVE_FILLER_WORDS or len(middle_word) <= 2:
                    repetition_count += 1

    # Enhanced false start detection
    false_start_count = 0

    # Pattern-based detection (dashes, ellipses)
    punctuation_patterns = [" - ", " -- ", "... "]
    for pattern in punctuation_patterns:
        false_start_count += text_lower.count(pattern)

    # Linguistic pattern detection (pronoun restarts)
    restart_patterns = [
        r" i - i ",
        r" we - we ",
        r" they - they ",
        r" you - you ",
        r" he - he ",
        r" she - she ",
        r" it - it ",
    ]

    import re

    for pattern in restart_patterns:
        matches = re.findall(pattern, text_lower)
        false_start_count += len(matches)

    # Word-level false starts (abrupt topic changes)
    if len(words) >= 4:
        for i in range(len(words) - 3):
            # Look for pattern: word1 word2 - word3 word4 (where word1 != word3)
            if (
                i + 3 < len(words)
                and len(words[i]) > 2
                and len(words[i + 2]) > 2
                and words[i] != words[i + 2]
                and words[i + 1] in {"um", "uh", "er"}
            ):  # Filler between attempts
                false_start_count += 1

    return int(filler_count), int(repetition_count), int(false_start_count)


@lru_cache(maxsize=512)
def _enhanced_syllable_estimation(word: str) -> int:
    """Enhanced syllable estimation with improved rules"""
    if not word or len(word) < 1:
        return 1

    word_lower = word.lower().strip(".,!?;:")

    if len(word_lower) == 1:
        return 1

    # Count vowel groups
    vowel_runs = 0
    prev_was_vowel = False

    # Special handling for common endings
    silent_e = word_lower.endswith("e") and len(word_lower) > 2

    for i, char in enumerate(word_lower):
        is_vowel = char in VOWELS

        # Special cases for diphthongs and vowel combinations
        if is_vowel:
            if not prev_was_vowel:
                vowel_runs += 1
            elif i < len(word_lower) - 1:
                # Handle some diphthongs as single syllables
                if char + word_lower[i - 1] in {
                    "ai",
                    "au",
                    "ea",
                    "ee",
                    "ei",
                    "ie",
                    "oa",
                    "oo",
                    "ou",
                    "ui",
                }:
                    pass  # Don't increment for diphthongs

        prev_was_vowel = is_vowel

    # Adjust for silent 'e'
    if silent_e and vowel_runs > 1:
        vowel_runs -= 1

    # Handle special endings that add syllables
    syllable_endings = ["tion", "sion", "cial", "tial", "ious", "eous"]
    for ending in syllable_endings:
        if word_lower.endswith(ending):
            vowel_runs += 1
            break

    # Ensure minimum of 1 syllable
    return max(1, vowel_runs)


def _vectorized_syllable_count_v2(words: tuple[str, ...]) -> int:
    """Enhanced vectorized syllable counting"""
    if not words:
        return 0

    return sum(_enhanced_syllable_estimation(word) for word in words)


# ============================================================================
# Main Enhanced Feature Computation Pipeline
# ============================================================================


def compute_segment_features_v2(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    text: str,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    """
    Production-optimized paralinguistic feature extraction with enhanced capabilities

    Returns comprehensive feature set with improved accuracy and reliability:

    Text-based features:
    - wpm, sps: Speech rate (words/syllables per minute, rounded to two decimals)
    - filler_count, repetition_count, false_start_count: Disfluency counts
    - disfluency_rate: Overall disfluency percentage

    Audio-based features:
    - pause_count, pause_total_sec, pause_ratio, pause_short_count, pause_long_count: Pause analysis
    - pitch_med_hz, pitch_iqr_hz, pitch_slope_hzps: Pitch characteristics
    - loudness_dbfs_med, loudness_dr_db, loudness_over_floor_db: Loudness analysis
    - spectral_centroid_med_hz, spectral_centroid_iqr_hz, spectral_flatness_med: Spectral features

    Voice quality (enhanced):
    - vq_jitter_pct, vq_shimmer_db, vq_hnr_db, vq_cpps_db: Standard voice quality
    - vq_voiced_ratio, vq_spectral_slope_db: Additional measures
    - vq_reliable, vq_note: Quality assessment

    Metadata:
    - paralinguistics_flags_json: Processing flags and diagnostics
    """

    cfg = cfg or ParalinguisticsConfig()

    # Input validation and preprocessing
    duration = max(1e-6, float(end_time - start_time))
    start_idx = max(0, int(start_time * sr))
    end_idx = min(len(audio), int(end_time * sr))

    # Memory-efficient audio slicing
    if cfg.enable_memory_optimization:
        segment_audio = audio[start_idx:end_idx].copy().astype(np.float32)
    else:
        segment_audio = audio[start_idx:end_idx].astype(np.float32, copy=False)

    # Processing flags and metadata
    flags = {
        "processing_version": "2.1.0",
        "duration_sec": duration,
        "audio_samples": len(segment_audio),
    }

    # ============= ENHANCED TEXT-BASED FEATURES =============

    # Advanced word tokenization
    words = _enhanced_word_tokenization(text) if text else ()
    word_count = len(words)
    flags["word_count"] = word_count

    # Speech rate computation
    if duration < 0.5:
        wpm = sps = np.nan
        flags["very_short_segment"] = True
    else:
        wpm = (60.0 * word_count) / duration

        if cfg.syllable_estimation and words:
            syllable_count = _vectorized_syllable_count_v2(words)
            sps = (60.0 * syllable_count) / duration
            flags["syllable_count"] = syllable_count
        else:
            sps = np.nan

    # Advanced disfluency detection
    if cfg.disfluency_detection and words:
        filler_count, repetition_count, false_start_count = _advanced_disfluency_detection(
            words, text
        )
        total_disfluencies = filler_count + repetition_count + false_start_count
        disfluency_rate = (100.0 * total_disfluencies) / max(1, word_count)
        flags["total_disfluencies"] = total_disfluencies
    else:
        filler_count = repetition_count = false_start_count = 0
        disfluency_rate = 0.0

    # ============= ENHANCED AUDIO-BASED FEATURES =============

    if segment_audio.size == 0:
        return _get_empty_features_v2(
            wpm,
            sps,
            filler_count,
            repetition_count,
            false_start_count,
            disfluency_rate,
            flags,
        )

    # Optimized frame parameters
    frame_length, hop_length = _get_optimized_frame_params(sr, cfg.frame_ms, cfg.hop_ms)
    flags["frame_params"] = {"frame_length": frame_length, "hop_length": hop_length}

    # ============= ENHANCED LOUDNESS AND PAUSE ANALYSIS =============

    try:
        # CPU-optimized RMS computation
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(
                y=segment_audio,
                frame_length=frame_length,
                hop_length=hop_length,
                center=True,
            )[0]
            rms_db = librosa.amplitude_to_db(rms + 1e-12)
        else:
            rms_db = _compute_rms_fallback_v2(segment_audio, frame_length, hop_length)

        if rms_db.size == 0:
            loudness_dbfs_med = loudness_dr_db = loudness_over_floor_db = np.nan
            pause_count = pause_total_sec = pause_ratio = pause_short_count = pause_long_count = 0
            floor_db = -60.0
            flags["empty_rms"] = True
        else:
            # Enhanced adaptive silence threshold
            if cfg.adaptive_silence:
                floor_db = float(np.percentile(rms_db, cfg.silence_floor_percentile))
                silence_threshold = floor_db + cfg.silence_margin_db
                flags["adaptive_threshold"] = {
                    "floor_db": floor_db,
                    "threshold_db": silence_threshold,
                }
            else:
                floor_db = cfg.base_silence_dbfs
                silence_threshold = floor_db

            # Enhanced loudness statistics
            loudness_dbfs_med = float(np.median(rms_db))
            loudness_dr_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
            loudness_over_floor_db = float(loudness_dbfs_med - floor_db)

            # Optimized pause analysis
            silence_mask = _vectorized_silence_detection_v2(
                rms_db, silence_threshold, cfg.enable_memory_optimization
            )
            (
                pause_count,
                pause_total_sec,
                pause_ratio,
                pause_short_count,
                pause_long_count,
            ) = _optimized_pause_analysis(silence_mask, sr, hop_length, cfg)

            flags["pause_analysis"] = {
                "silence_frames": int(np.sum(silence_mask)),
                "total_frames": len(silence_mask),
            }

    except Exception as e:
        flags["loudness_analysis_error"] = str(e)
        loudness_dbfs_med = loudness_dr_db = loudness_over_floor_db = np.nan
        pause_count = pause_total_sec = pause_ratio = pause_short_count = pause_long_count = 0
        floor_db = -60.0

    # ============= ENHANCED PITCH ANALYSIS =============

    try:
        # Enhanced robust pitch extraction
        f0, voiced_flag = _robust_pitch_extraction_v2(segment_audio, sr, cfg)

        if f0.size > 0 and np.any(voiced_flag):
            # Time array for slope calculation
            times = np.arange(len(f0)) * (hop_length / sr)

            # Enhanced pitch statistics
            pitch_med_hz, pitch_iqr_hz, pitch_slope_hzps = _enhanced_pitch_statistics(
                f0, voiced_flag, times, cfg
            )

            flags["pitch_analysis"] = {
                "voiced_frames": int(np.sum(voiced_flag)),
                "total_frames": len(f0),
                "coverage": float(np.mean(voiced_flag)),
            }
        else:
            pitch_med_hz = pitch_iqr_hz = pitch_slope_hzps = np.nan
            flags["no_pitch_detected"] = True

    except Exception as e:
        flags["pitch_analysis_error"] = str(e)
        pitch_med_hz = pitch_iqr_hz = pitch_slope_hzps = np.nan

    # ============= ENHANCED SPECTRAL FEATURES =============

    if cfg.spectral_features_enabled:
        try:
            (
                spectral_centroid_med_hz,
                spectral_centroid_iqr_hz,
                spectral_flatness_med,
            ) = _optimized_spectral_features(segment_audio, sr, cfg)
            flags["spectral_analysis"] = "completed"
        except Exception as e:
            flags["spectral_analysis_error"] = str(e)
            spectral_centroid_med_hz = spectral_centroid_iqr_hz = spectral_flatness_med = np.nan

    # ============= ENHANCED VOICE QUALITY ANALYSIS =============

    # Voice quality analysis with enhanced error handling
    if cfg.voice_quality_enabled and duration >= cfg.vq_min_duration_sec:
        try:
            # Audio quality assessment
            snr_db, is_reliable, quality_status = _advanced_audio_quality_assessment(
                segment_audio, sr
            )

            if is_reliable and snr_db >= cfg.vq_min_snr_db:
                # Use Parselmouth if available and preferred
                if cfg.vq_use_parselmouth and PARSELMOUTH_AVAILABLE:
                    voice_quality = _compute_voice_quality_parselmouth_v2(segment_audio, sr, cfg)
                else:
                    voice_quality = _compute_voice_quality_fallback_v2(segment_audio, sr, cfg)

                # Extract individual metrics
                vq_jitter_pct = voice_quality.get("jitter_pct", 0.0)
                vq_shimmer_db = voice_quality.get("shimmer_db", 0.0)
                vq_hnr_db = voice_quality.get("hnr_db", 0.0)
                vq_cpps_db = voice_quality.get("cpps_db", 0.0)
                vq_voiced_ratio = voice_quality.get("voiced_ratio", 0.0)
                vq_spectral_slope_db = voice_quality.get("spectral_slope_db", 0.0)

                # Reliability gate includes voiced coverage
                vq_reliable = bool(
                    is_reliable
                    and (snr_db >= cfg.vq_min_snr_db)
                    and (voice_quality.get("voiced_ratio", 0.0) >= 0.5)
                )
                vq_note = f"{quality_status}_voiced_{voice_quality.get('voiced_ratio', 0.0):.2f}"

                flags["voice_quality"] = {
                    "method": (
                        "parselmouth"
                        if (cfg.vq_use_parselmouth and PARSELMOUTH_AVAILABLE)
                        else "fallback"
                    ),
                    "snr_db": snr_db,
                    "quality_status": quality_status,
                }
            else:
                # Fallback for unreliable audio
                if cfg.vq_fallback_enabled:
                    voice_quality = _compute_voice_quality_fallback_v2(segment_audio, sr, cfg)

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

        except Exception as e:
            flags["voice_quality_error"] = str(e)
            vq_jitter_pct = vq_shimmer_db = vq_hnr_db = vq_cpps_db = 0.0
            vq_voiced_ratio = vq_spectral_slope_db = 0.0
            vq_reliable = False
            vq_note = f"analysis_failed_{str(e)[:50]}"
    else:
        # Voice quality disabled or segment too short
        vq_jitter_pct = vq_shimmer_db = vq_hnr_db = vq_cpps_db = 0.0
        vq_voiced_ratio = vq_spectral_slope_db = 0.0
        vq_reliable = False

        if not cfg.voice_quality_enabled:
            vq_note = "disabled"
        else:
            vq_note = f"too_short_{duration:.2f}s"

        flags["voice_quality"] = "disabled_or_too_short"

    # ============= FEATURE ASSEMBLY AND OUTPUT =============

    # Compile all features into final output
    features = {
        # Text-based features
        "wpm": round(float(wpm), 2) if not np.isnan(wpm) else np.nan,
        "sps": round(float(sps), 2) if not np.isnan(sps) else np.nan,
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "false_start_count": int(false_start_count),
        "disfluency_rate": float(disfluency_rate),
        # Pause and timing features
        "pause_count": int(pause_count),
        "pause_total_sec": float(pause_total_sec),
        "pause_ratio": float(pause_ratio),
        "pause_short_count": int(pause_short_count),
        "pause_long_count": int(pause_long_count),
        # Pitch features
        "pitch_med_hz": float(pitch_med_hz) if not np.isnan(pitch_med_hz) else np.nan,
        "pitch_iqr_hz": float(pitch_iqr_hz) if not np.isnan(pitch_iqr_hz) else np.nan,
        "pitch_slope_hzps": (float(pitch_slope_hzps) if not np.isnan(pitch_slope_hzps) else np.nan),
        # Loudness features
        "loudness_dbfs_med": (
            float(loudness_dbfs_med) if not np.isnan(loudness_dbfs_med) else np.nan
        ),
        "loudness_dr_db": (float(loudness_dr_db) if not np.isnan(loudness_dr_db) else np.nan),
        "loudness_over_floor_db": (
            float(loudness_over_floor_db) if not np.isnan(loudness_over_floor_db) else np.nan
        ),
        # Spectral features
        "spectral_centroid_med_hz": (
            float(spectral_centroid_med_hz) if not np.isnan(spectral_centroid_med_hz) else np.nan
        ),
        "spectral_centroid_iqr_hz": (
            float(spectral_centroid_iqr_hz) if not np.isnan(spectral_centroid_iqr_hz) else np.nan
        ),
        "spectral_flatness_med": (
            float(spectral_flatness_med) if not np.isnan(spectral_flatness_med) else np.nan
        ),
        # Voice quality features
        "vq_jitter_pct": float(vq_jitter_pct),
        "vq_shimmer_db": float(vq_shimmer_db),
        "vq_hnr_db": float(vq_hnr_db),
        "vq_cpps_db": float(vq_cpps_db),
        "vq_voiced_ratio": float(vq_voiced_ratio),
        "vq_spectral_slope_db": float(vq_spectral_slope_db),
        "vq_reliable": bool(vq_reliable),
        "vq_note": str(vq_note),
        # Metadata and diagnostics
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
    """Return empty feature set when audio is unavailable"""

    flags["empty_audio"] = True

    return {
        # Text-based features (still available)
        "wpm": round(float(wpm), 2) if not np.isnan(wpm) else np.nan,
        "sps": round(float(sps), 2) if not np.isnan(sps) else np.nan,
        "filler_count": int(filler_count),
        "repetition_count": int(repetition_count),
        "false_start_count": int(false_start_count),
        "disfluency_rate": float(disfluency_rate),
        # Audio-based features (all NaN/zero)
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


def _compute_rms_fallback_v2(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """CPU-optimized RMS computation fallback when librosa is unavailable"""

    if audio.size == 0:
        return np.array([])

    # Ensure frame length doesn't exceed audio length
    frame_length = min(frame_length, len(audio))

    if frame_length < 1:
        return np.array([])

    # Vectorized frame-based RMS computation
    num_frames = max(1, (len(audio) - frame_length) // hop_length + 1)
    rms_values = np.zeros(num_frames)

    for i in range(num_frames):
        start_idx = i * hop_length
        end_idx = min(start_idx + frame_length, len(audio))

        if end_idx > start_idx:
            frame = audio[start_idx:end_idx]
            rms_values[i] = np.sqrt(np.mean(frame**2))

    # Convert to dB
    return 20 * np.log10(rms_values + 1e-12)


# ============================================================================
# Enhanced Batch Processing and Pipeline Integration
# ============================================================================


def process_segments_batch_v2(
    segments: list[tuple[np.ndarray, int, float, float, str]],
    cfg: ParalinguisticsConfig | None = None,
    progress_callback: callable | None = None,
) -> list[dict[str, Any]]:
    """
    Batch process multiple segments with CPU optimization and progress tracking

    Args:
        segments: List of (audio, sr, start_time, end_time, text) tuples
        cfg: Configuration object
        progress_callback: Optional callback for progress updates

    Returns:
        List of feature dictionaries
    """

    cfg = cfg or ParalinguisticsConfig()
    results = []

    # Progress tracking
    total_segments = len(segments)
    processed_count = 0

    # CPU optimization: use threading for I/O bound operations
    if cfg.parallel_processing and total_segments >= cfg.max_workers:
        with ThreadPoolExecutor(max_workers=min(cfg.max_workers, total_segments)) as executor:
            # Submit all tasks
            futures = []
            for i, (audio, sr, start_time, end_time, text) in enumerate(segments):
                future = executor.submit(
                    compute_segment_features_v2,
                    audio,
                    sr,
                    start_time,
                    end_time,
                    text,
                    cfg,
                )
                futures.append((i, future))

            # Collect results in order
            segment_results = [None] * total_segments
            for i, future in futures:
                try:
                    segment_results[i] = future.result(timeout=30)  # 30s timeout per segment
                    processed_count += 1

                    if progress_callback:
                        progress_callback(processed_count, total_segments)

                except Exception as e:
                    warnings.warn(f"Segment {i} processing failed: {e}")
                    segment_results[i] = _get_error_features_v2(str(e))

            results = [r for r in segment_results if r is not None]

    else:
        # Sequential processing
        for i, (audio, sr, start_time, end_time, text) in enumerate(segments):
            try:
                features = compute_segment_features_v2(audio, sr, start_time, end_time, text, cfg)
                results.append(features)
                processed_count += 1

                if progress_callback:
                    progress_callback(processed_count, total_segments)

            except Exception as e:
                warnings.warn(f"Segment {i} processing failed: {e}")
                results.append(_get_error_features_v2(str(e)))

    return results


def _get_error_features_v2(error_msg: str) -> dict[str, Any]:
    """Return error feature set when processing fails"""

    flags = {"processing_error": error_msg, "error_timestamp": time.time()}

    return {
        # All features set to NaN or 0
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


# ============================================================================
# Advanced Feature Analysis and Utilities
# ============================================================================


def analyze_speech_patterns_v2(features_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Advanced speech pattern analysis across multiple segments

    Args:
        features_list: List of feature dictionaries from multiple segments

    Returns:
        Aggregate speech pattern analysis
    """

    if not features_list:
        return {"error": "No features provided"}

    # Extract numeric features for analysis
    numeric_features = [
        "wpm",
        "sps",
        "disfluency_rate",
        "pause_ratio",
        "pitch_med_hz",
        "pitch_iqr_hz",
        "loudness_dbfs_med",
        "vq_jitter_pct",
        "vq_shimmer_db",
        "vq_hnr_db",
    ]

    analysis = {}

    for feature_name in numeric_features:
        values = []
        for features in features_list:
            val = features.get(feature_name, np.nan)
            if not np.isnan(val) and val != 0.0:
                values.append(val)

        if values:
            analysis[f"{feature_name}_mean"] = float(np.mean(values))
            analysis[f"{feature_name}_std"] = float(np.std(values))
            analysis[f"{feature_name}_median"] = float(np.median(values))
            analysis[f"{feature_name}_min"] = float(np.min(values))
            analysis[f"{feature_name}_max"] = float(np.max(values))
        else:
            analysis[f"{feature_name}_mean"] = np.nan
            analysis[f"{feature_name}_std"] = np.nan
            analysis[f"{feature_name}_median"] = np.nan
            analysis[f"{feature_name}_min"] = np.nan
            analysis[f"{feature_name}_max"] = np.nan

    # Count-based features
    count_features = [
        "filler_count",
        "repetition_count",
        "false_start_count",
        "pause_count",
    ]

    for feature_name in count_features:
        values = [features.get(feature_name, 0) for features in features_list]
        analysis[f"{feature_name}_total"] = int(np.sum(values))
        analysis[f"{feature_name}_mean"] = float(np.mean(values))

    # Voice quality reliability
    reliable_segments = sum(1 for f in features_list if f.get("vq_reliable", False))
    analysis["voice_quality_reliability"] = reliable_segments / len(features_list)

    # Overall speech characteristics
    analysis["total_segments"] = len(features_list)
    analysis["analysis_timestamp"] = time.time()

    return analysis


def detect_speech_anomalies_v2(
    features: dict[str, Any], reference_stats: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Detect speech anomalies and unusual patterns

    Args:
        features: Feature dictionary from segment analysis
        reference_stats: Optional reference statistics for comparison

    Returns:
        Anomaly detection results
    """

    anomalies = {"flags": [], "severity_score": 0.0, "details": {}}

    # Speech rate anomalies
    wpm = features.get("wpm", np.nan)
    if not np.isnan(wpm):
        if wpm < 80:
            anomalies["flags"].append("very_slow_speech")
            anomalies["severity_score"] += 1.0
        elif wpm > 300:
            anomalies["flags"].append("very_fast_speech")
            anomalies["severity_score"] += 1.0

        anomalies["details"]["speech_rate"] = wpm

    # Disfluency anomalies
    disfluency_rate = features.get("disfluency_rate", 0.0)
    if disfluency_rate > 15.0:
        anomalies["flags"].append("high_disfluency")
        anomalies["severity_score"] += 1.5
    elif disfluency_rate > 8.0:
        anomalies["flags"].append("elevated_disfluency")
        anomalies["severity_score"] += 0.5

    # Pause pattern anomalies
    pause_ratio = features.get("pause_ratio", 0.0)
    if pause_ratio > 0.5:
        anomalies["flags"].append("excessive_pausing")
        anomalies["severity_score"] += 1.0
    elif pause_ratio < 0.05:
        anomalies["flags"].append("minimal_pausing")
        anomalies["severity_score"] += 0.5

    # Voice quality anomalies
    if features.get("vq_reliable", False):
        jitter = features.get("vq_jitter_pct", 0.0)
        shimmer = features.get("vq_shimmer_db", 0.0)
        hnr = features.get("vq_hnr_db", 0.0)

        if jitter > 1.5:
            anomalies["flags"].append("high_jitter")
            anomalies["severity_score"] += 0.8

        if shimmer > 1.0:
            anomalies["flags"].append("high_shimmer")
            anomalies["severity_score"] += 0.8

        if hnr < 10.0:
            anomalies["flags"].append("low_hnr")
            anomalies["severity_score"] += 0.6

    # Pitch anomalies
    pitch_med = features.get("pitch_med_hz", np.nan)
    if not np.isnan(pitch_med):
        if pitch_med < 100 or pitch_med > 350:
            anomalies["flags"].append("unusual_pitch_range")
            anomalies["severity_score"] += 0.5

    # Overall severity classification
    if anomalies["severity_score"] >= 3.0:
        anomalies["overall_severity"] = "high"
    elif anomalies["severity_score"] >= 1.5:
        anomalies["overall_severity"] = "moderate"
    elif anomalies["severity_score"] >= 0.5:
        anomalies["overall_severity"] = "mild"
    else:
        anomalies["overall_severity"] = "normal"

    return anomalies


# ============================================================================
# Enhanced Backchannel Detection with CPU Optimization
# ============================================================================


def detect_backchannels_v2(
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    cfg: ParalinguisticsConfig | None = None,
) -> list[dict[str, Any]]:
    """
    Enhanced backchannel detection using audio and linguistic cues

    Args:
        audio: Full audio array
        sr: Sample rate
        segments: List of segment dictionaries with start/end times and text
        cfg: Configuration object

    Returns:
        List of backchannel detection results
    """

    cfg = cfg or ParalinguisticsConfig()
    backchannels = []

    # Backchannel keywords and patterns
    backchannel_keywords = {
        "mm",
        "mmm",
        "mhmm",
        "uh-huh",
        "yeah",
        "yep",
        "yes",
        "okay",
        "ok",
        "right",
        "sure",
        "exactly",
        "absolutely",
        "definitely",
        "totally",
        "i see",
        "oh",
        "wow",
        "really",
        "interesting",
        "hmm",
    }

    # Process each segment
    for i, segment in enumerate(segments):
        text = segment.get("text", "").lower().strip()
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", 0.0)
        duration_ms = (end_time - start_time) * 1000

        # Skip if too long for backchannel
        if duration_ms > cfg.backchannel_max_ms:
            continue

        # Text-based backchannel detection
        words = _enhanced_word_tokenization(text)
        is_backchannel_text = False

        if len(words) <= 3:  # Short utterances only
            # Check for exact matches
            text_clean = " ".join(words)
            if text_clean in backchannel_keywords:
                is_backchannel_text = True

            # Check for partial matches in short utterances
            elif len(words) == 1 and any(kw in text_clean for kw in ["mm", "oh", "ah", "um"]):
                is_backchannel_text = True

        # Audio-based confirmation
        _is_backchannel_audio = False
        confidence_audio = 0.0

        if is_backchannel_text:
            try:
                # Extract segment audio
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                segment_audio = audio[start_idx:end_idx]

                if len(segment_audio) > 0:
                    # Check energy characteristics (backchannels are typically low energy)
                    rms_energy = np.sqrt(np.mean(segment_audio**2))

                    # Check pitch characteristics (backchannels often have falling pitch)
                    if LIBROSA_AVAILABLE:
                        f0, voiced_flag = _robust_pitch_extraction_v2(segment_audio, sr, cfg)

                        if len(f0) > 3 and np.sum(voiced_flag) > 2:
                            voiced_f0 = f0[voiced_flag]
                            pitch_trend = (
                                np.polyfit(range(len(voiced_f0)), voiced_f0, 1)[0]
                                if len(voiced_f0) > 2
                                else 0
                            )

                            # Backchannel indicators:
                            # - Falling pitch (negative slope)
                            # - Moderate energy
                            # - Short duration

                            audio_indicators = 0

                            if pitch_trend < -5:  # Falling pitch
                                audio_indicators += 1

                            if 0.001 < rms_energy < 0.1:  # Moderate energy
                                audio_indicators += 1

                            if duration_ms < 800:  # Short duration
                                audio_indicators += 1

                            confidence_audio = audio_indicators / 3.0
                            _is_backchannel_audio = confidence_audio >= 0.5

            except Exception as e:
                warnings.warn(f"Audio analysis for backchannel failed: {e}")

        # Combined confidence score
        if is_backchannel_text:
            confidence_text = 1.0 if text_clean in backchannel_keywords else 0.7
            confidence_combined = (confidence_text + confidence_audio) / 2.0

            backchannels.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_ms": duration_ms,
                    "text": text,
                    "is_backchannel": True,
                    "confidence_text": confidence_text,
                    "confidence_audio": confidence_audio,
                    "confidence_combined": confidence_combined,
                    "method": "text_audio_combined",
                }
            )

    return backchannels


# ============================================================================
# Performance Monitoring and Benchmarking
# ============================================================================


def benchmark_performance_v2(
    test_audio: np.ndarray,
    sr: int,
    test_text: str = "This is a test sentence for benchmarking.",
    iterations: int = 10,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    """
    Comprehensive performance benchmarking for the paralinguistics module

    Args:
        test_audio: Audio array for testing
        sr: Sample rate
        test_text: Text for testing
        iterations: Number of benchmark iterations
        cfg: Configuration object

    Returns:
        Benchmark results and performance metrics
    """

    cfg = cfg or ParalinguisticsConfig()

    print(f"Running paralinguistics benchmark with {iterations} iterations...")

    # Timing storage
    times = []
    _memory_usage = []
    feature_counts = []

    # Warm-up run
    try:
        _ = compute_segment_features_v2(test_audio, sr, 0.0, len(test_audio) / sr, test_text, cfg)
    except Exception as e:
        print(f"Warm-up run failed: {e}")
        return {"error": "Benchmark failed during warm-up"}

    # Main benchmark loop
    for i in range(iterations):
        start_time = time.time()

        try:
            features = compute_segment_features_v2(
                test_audio, sr, 0.0, len(test_audio) / sr, test_text, cfg
            )

            end_time = time.time()
            processing_time = end_time - start_time
            times.append(processing_time)

            # Count non-null features
            valid_features = sum(
                1
                for v in features.values()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            )
            feature_counts.append(valid_features)

        except Exception as e:
            print(f"Benchmark iteration {i} failed: {e}")
            times.append(np.nan)
            feature_counts.append(0)

    # Filter out failed iterations
    valid_times = [t for t in times if not np.isnan(t)]
    valid_features = [f for f, t in zip(feature_counts, times, strict=False) if not np.isnan(t)]

    if not valid_times:
        return {"error": "All benchmark iterations failed"}

    # Performance statistics
    results = {
        "iterations_completed": len(valid_times),
        "iterations_failed": iterations - len(valid_times),
        "success_rate": len(valid_times) / iterations,
        # Timing statistics
        "mean_time_sec": float(np.mean(valid_times)),
        "std_time_sec": float(np.std(valid_times)),
        "min_time_sec": float(np.min(valid_times)),
        "max_time_sec": float(np.max(valid_times)),
        "median_time_sec": float(np.median(valid_times)),
        # Feature statistics
        "mean_features": float(np.mean(valid_features)),
        "feature_consistency": float(np.std(valid_features)),
        # Performance ratings
        "performance_rating": _get_performance_rating(np.mean(valid_times), len(test_audio) / sr),
        # System information
        "audio_duration_sec": len(test_audio) / sr,
        "audio_samples": len(test_audio),
        "sample_rate": sr,
        "config_summary": {
            "voice_quality_enabled": cfg.voice_quality_enabled,
            "spectral_features_enabled": cfg.spectral_features_enabled,
            "parallel_processing": cfg.parallel_processing,
            "use_parselmouth": cfg.vq_use_parselmouth and PARSELMOUTH_AVAILABLE,
        },
        # Library availability
        "libraries_available": {
            "librosa": LIBROSA_AVAILABLE,
            "scipy": SCIPY_AVAILABLE,
            "parselmouth": PARSELMOUTH_AVAILABLE,
        },
    }

    return results


def _get_performance_rating(processing_time: float, audio_duration: float) -> str:
    """Get performance rating based on processing time vs audio duration"""

    ratio = processing_time / audio_duration

    if ratio < 0.1:
        return "excellent"
    elif ratio < 0.2:
        return "good"
    elif ratio < 0.5:
        return "acceptable"
    elif ratio < 1.0:
        return "slow"
    else:
        return "very_slow"


# ============================================================================
# Enhanced Configuration Management and Presets
# ============================================================================


def get_config_preset(preset_name: str, *, max_workers: int | None = None) -> ParalinguisticsConfig:
    """
    Get predefined configuration presets for different use cases

    Args:
        preset_name: Name of preset ("fast", "balanced", "quality", "research")
        max_workers: Optional override for parallel worker count

    Returns:
        Configured ParalinguisticsConfig object
    """

    kwargs = {"max_workers": max_workers} if max_workers is not None else {}

    if preset_name == "fast":
        # Optimized for speed, reduced accuracy
        return ParalinguisticsConfig(
            # Reduced frame resolution
            frame_ms=30,
            hop_ms=15,
            # Simplified silence detection
            adaptive_silence=False,
            base_silence_dbfs=-40.0,
            # Minimal voice quality analysis
            voice_quality_enabled=False,
            spectral_features_enabled=False,
            # Disabled expensive features
            syllable_estimation=False,
            disfluency_detection=True,  # Keep for basic analysis
            # CPU optimizations
            use_vectorized_ops=True,
            enable_caching=True,
            parallel_processing=False,  # Overhead not worth it for fast mode
            enable_memory_optimization=True,
            # Relaxed quality requirements
            vq_min_snr_db=3.0,
            pitch_min_coverage=0.02,
            **kwargs,
        )

    elif preset_name == "balanced":
        # Default balanced configuration (same as default)
        return ParalinguisticsConfig(**kwargs)

    elif preset_name == "quality":
        # Optimized for accuracy, slower processing
        return ParalinguisticsConfig(
            # Higher resolution
            frame_ms=20,
            hop_ms=8,
            # Enhanced silence detection
            adaptive_silence=True,
            silence_floor_percentile=3,
            silence_margin_db=10.0,
            # Full voice quality analysis
            voice_quality_enabled=True,
            vq_use_parselmouth=True,
            vq_fallback_enabled=True,
            vq_min_snr_db=8.0,
            # Enhanced spectral analysis
            spectral_features_enabled=True,
            spectral_n_fft=2048,
            spectral_hop_length=128,
            # All text features enabled
            syllable_estimation=True,
            disfluency_detection=True,
            # Higher quality pitch analysis
            pitch_frame_length=4096,
            pitch_hop_length=256,
            pitch_min_coverage=0.08,
            # Relaxed processing constraints
            max_audio_length_sec=60.0,
            enable_memory_optimization=False,
            **kwargs,
        )

    elif preset_name == "research":
        # Maximum feature extraction for research purposes
        return ParalinguisticsConfig(
            # Maximum resolution
            frame_ms=15,
            hop_ms=5,
            # Comprehensive silence detection
            adaptive_silence=True,
            silence_floor_percentile=1,
            silence_margin_db=12.0,
            # Full voice quality with both methods
            voice_quality_enabled=True,
            vq_use_parselmouth=True,
            vq_fallback_enabled=True,
            vq_min_snr_db=10.0,
            # Maximum spectral resolution
            spectral_features_enabled=True,
            spectral_n_fft=4096,
            spectral_hop_length=64,
            # All features enabled
            syllable_estimation=True,
            disfluency_detection=True,
            # High-quality pitch analysis
            pitch_frame_length=8192,
            pitch_hop_length=128,
            pitch_min_coverage=0.1,
            # No processing limits
            max_audio_length_sec=300.0,
            enable_memory_optimization=False,
            parallel_processing=True,
            max_workers=max_workers if max_workers is not None else 4,
        )

    else:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: fast, balanced, quality, research"
        )


# ============================================================================
# Main Entry Point and CLI Interface
# ============================================================================


def main():
    """Main entry point for command-line usage and testing"""

    import argparse

    parser = argparse.ArgumentParser(description="Paralinguistic Feature Extraction")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--text", type=str, help="Corresponding text")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, help="End time (seconds)")
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality", "research"],
        help="Configuration preset",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Load configuration
    cfg = get_config_preset(args.preset)

    if args.benchmark:
        # Generate test audio for benchmarking
        duration = 10.0  # 10 seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        # Generate complex test signal (speech-like)
        test_audio = (
            0.5 * np.sin(2 * np.pi * 200 * t)  # Fundamental
            + 0.3 * np.sin(2 * np.pi * 400 * t)  # Harmonic
            + 0.1 * np.random.normal(0, 1, len(t))  # Noise
        ).astype(np.float32)

        test_text = (
            "This is a test sentence for benchmarking the paralinguistic feature extraction system."
        )

        results = benchmark_performance_v2(test_audio, sr, test_text, cfg=cfg)

        print("\n" + "=" * 60)
        print("PARALINGUISTICS BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"Mean Processing Time: {results.get('mean_time_sec', 0):.3f}s")
        print(f"Performance Rating: {results.get('performance_rating', 'unknown')}")
        print(f"Features Extracted: {results.get('mean_features', 0):.0f}")

        libs = results.get("libraries_available", {})
        print("\nLibrary Status:")
        print(f"  Librosa: {Y if libs.get('librosa') else 'N'}")
        print(f"  SciPy: {Y if libs.get('scipy') else 'N'}")
        print(f"  Parselmouth: {Y if libs.get('parselmouth') else 'N'}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")

    elif args.audio:
        try:
            # Load audio file
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(args.audio, sr=None)
            else:
                raise ImportError("librosa required for audio file loading")

            # Determine end time
            end_time = args.end if args.end else len(audio) / sr

            # Load text
            text = args.text or ""
            if not text and args.text:
                with open(args.text) as f:
                    text = f.read().strip()

            # Process segment
            print(f"Processing audio segment: {args.start:.2f}s to {end_time:.2f}s")
            start_time = time.time()

            features = compute_segment_features_v2(audio, sr, args.start, end_time, text, cfg)

            processing_time = time.time() - start_time

            # Display results
            print(f"\nProcessing completed in {processing_time:.3f}s")
            print("\nFeature Summary:")
            print("-" * 40)

            # Group features by category
            text_features = [
                "wpm",
                "sps",
                "filler_count",
                "repetition_count",
                "false_start_count",
                "disfluency_rate",
            ]
            audio_features = [
                "pause_count",
                "pause_ratio",
                "pitch_med_hz",
                "loudness_dbfs_med",
            ]
            voice_features = [
                "vq_jitter_pct",
                "vq_shimmer_db",
                "vq_hnr_db",
                "vq_reliable",
            ]

            print("Text Analysis:")
            for feat in text_features:
                if feat in features:
                    val = features[feat]
                    if isinstance(val, float) and not np.isnan(val):
                        print(f"  {feat}: {val:.2f}")
                    elif isinstance(val, int):
                        print(f"  {feat}: {val}")

            print("\nAudio Analysis:")
            for feat in audio_features:
                if feat in features:
                    val = features[feat]
                    if isinstance(val, float) and not np.isnan(val):
                        print(f"  {feat}: {val:.2f}")

            print("\nVoice Quality:")
            for feat in voice_features:
                if feat in features:
                    val = features[feat]
                    if feat == "vq_reliable":
                        print(f"  {feat}: {val}")
                    elif isinstance(val, float):
                        print(f"  {feat}: {val:.2f}")

            # Save results if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(features, f, indent=2, default=str)
                print(f"\nFull results saved to: {args.output}")

        except Exception as e:
            print(f"Error processing audio: {e}")
            return 1

    else:
        print("Please specify --audio for processing or --benchmark for testing")
        print("Use --help for full usage information")
        return 1

    return 0


# ============================================================================
# Module Validation and Self-Test
# ============================================================================


def validate_module():
    """
    Validate module functionality and dependencies

    Returns:
        Dict with validation results
    """

    validation = {"status": "unknown", "dependencies": {}, "features": {}, "errors": []}

    # Check dependencies
    validation["dependencies"]["librosa"] = LIBROSA_AVAILABLE
    validation["dependencies"]["scipy"] = SCIPY_AVAILABLE
    validation["dependencies"]["parselmouth"] = PARSELMOUTH_AVAILABLE
    validation["dependencies"]["numpy"] = True  # Always available

    # Test basic functionality
    try:
        # Create test data
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = 0.1 * np.sin(2 * np.pi * 150 * t)
        test_text = "Hello world test"

        # Test configuration
        cfg = ParalinguisticsConfig()
        validation["features"]["config_creation"] = True

        # Test feature extraction
        features = compute_segment_features_v2(test_audio, sr, 0.0, duration, test_text, cfg)
        validation["features"]["basic_extraction"] = True
        validation["features"]["feature_count"] = len(
            [
                k
                for k, v in features.items()
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
        )

        # Test presets
        for preset in ["fast", "balanced", "quality", "research"]:
            try:
                _preset_cfg = get_config_preset(preset)
                validation["features"][f"preset_{preset}"] = True
            except Exception as e:
                validation["errors"].append(f"Preset {preset} failed: {e}")
                validation["features"][f"preset_{preset}"] = False

        # Test batch processing
        segments = [
            (test_audio, sr, 0.0, 1.0, "test one"),
            (test_audio, sr, 1.0, 2.0, "test two"),
        ]
        batch_results = process_segments_batch_v2(segments, cfg)
        validation["features"]["batch_processing"] = len(batch_results) == 2

        validation["status"] = "success"

    except Exception as e:
        validation["status"] = "failed"
        validation["errors"].append(f"Validation failed: {e}")

    return validation


# ============================================================================
# Export and Module Interface
# ============================================================================


# Stable API aliases for pipeline compatibility
def compute_segment_features(
    audio: np.ndarray,
    sr: int,
    start_time: float,
    end_time: float,
    text: str,
    cfg: ParalinguisticsConfig | None = None,
) -> dict[str, Any]:
    return compute_segment_features_v2(audio, sr, start_time, end_time, text, cfg)


def process_segments_batch(
    segments: list[tuple[np.ndarray, int, float, float, str]],
    cfg: ParalinguisticsConfig | None = None,
    progress_callback: callable | None = None,
) -> list[dict[str, Any]]:
    return process_segments_batch_v2(segments, cfg, progress_callback)


def compute_overlap_and_interruptions(
    segments: list[dict[str, Any]],
    min_overlap_sec: float = 0.05,
    interruption_gap_sec: float = 0.15,
) -> dict[str, Any]:
    if not segments:
        return {
            "overlap_total_sec": 0.0,
            "overlap_ratio": 0.0,
            "by_speaker": {},
            "interruptions": [],
        }

    norm: list[tuple[float, float, str, dict[str, Any]]] = []
    for seg in segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start) or start)
        if end < start:
            start, end = end, start
        speaker = seg.get("speaker_id") or seg.get("speaker") or "unknown"
        norm.append((start, end, str(speaker), seg))

    norm.sort(key=lambda x: x[0])

    total_start = norm[0][0]
    total_end = max(e for _, e, _, _ in norm)
    total_dur = max(1e-6, total_end - total_start)

    overlap_total = 0.0
    by_speaker: dict[str, dict[str, Any]] = {}
    interruptions: list[dict[str, Any]] = []

    j = 0
    for i in range(len(norm)):
        si, ei, spk_i, _ = norm[i]
        while j < i and norm[j][1] <= si:
            j += 1
        for k in range(j, i):
            sk, ek, spk_k, _ = norm[k]
            start = max(si, sk)
            end = min(ei, ek)
            ov = end - start
            if ov >= min_overlap_sec:
                overlap_total += ov
                for spk in (spk_i, spk_k):
                    slot = by_speaker.setdefault(spk, {"overlap_sec": 0.0, "interruptions": 0})
                    slot["overlap_sec"] += ov

                if spk_i != spk_k:
                    later = (spk_i, si) if si > sk else (spk_k, sk)
                    earlier = (spk_k, sk) if si > sk else (spk_i, si)
                    if 0.0 <= (later[1] - earlier[1]) <= interruption_gap_sec:
                        by_speaker.setdefault(later[0], {"overlap_sec": 0.0, "interruptions": 0})[
                            "interruptions"
                        ] += 1
                        interruptions.append(
                            {
                                "at": float(later[1]),
                                "interrupter": later[0],
                                "interrupted": earlier[0],
                                "overlap_sec": float(ov),
                            }
                        )

    return {
        "overlap_total_sec": float(overlap_total),
        "overlap_ratio": float(overlap_total / total_dur) if total_dur > 0 else 0.0,
        "by_speaker": by_speaker,
        "interruptions": interruptions,
    }


def extract(wav: np.ndarray, sr: int, segs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute paralinguistic features per segment for pipeline consumption.

    Returns list aligned with `segs` including core keys and advanced voice-quality metrics.
    """
    out: list[dict[str, Any]] = []
    cfg = ParalinguisticsConfig()
    total = max(0.0, float(len(wav) / max(1, sr)))
    for s in segs:
        try:
            start = float(s.get("start", s.get("start_time", 0.0)) or 0.0)
            end = float(s.get("end", s.get("end_time", start)) or start)
            start = max(0.0, min(start, total))
            end = max(start, min(end, total))
            text = s.get("text") or ""
            duration_s = max(0.0, end - start)
            words = len(text.split())
            feats = compute_segment_features_v2(wav, sr, start, end, text, cfg)
            out.append(
                {
                    "wpm": feats.get("wpm"),
                    "duration_s": duration_s,
                    "words": words,
                    "pause_count": feats.get("pause_count"),
                    "pause_time_s": feats.get("pause_total_sec"),
                    "pause_ratio": feats.get("pause_ratio"),
                    "f0_mean_hz": feats.get("pitch_med_hz"),
                    "f0_std_hz": feats.get("pitch_iqr_hz"),
                    "loudness_rms": feats.get("loudness_dbfs_med"),
                    "disfluency_count": int(feats.get("filler_count", 0))
                    + int(feats.get("repetition_count", 0))
                    + int(feats.get("false_start_count", 0)),
                    "vq_jitter_pct": feats.get("vq_jitter_pct"),
                    "vq_shimmer_db": feats.get("vq_shimmer_db"),
                    "vq_hnr_db": feats.get("vq_hnr_db"),
                    "vq_cpps_db": feats.get("vq_cpps_db"),
                    "vq_voiced_ratio": feats.get("vq_voiced_ratio"),
                    "vq_spectral_slope_db": feats.get("vq_spectral_slope_db"),
                    "vq_reliable": feats.get("vq_reliable"),
                    "vq_note": feats.get("vq_note"),
                }
            )
        except Exception as e:
            out.append({"error": str(e)})
    return out


__all__ = [
    # Core functions (v2 + stable aliases)
    "compute_segment_features_v2",
    "process_segments_batch_v2",
    "compute_segment_features",
    "process_segments_batch",
    # Configuration
    "ParalinguisticsConfig",
    "get_config_preset",
    # Analysis utilities
    "analyze_speech_patterns_v2",
    "detect_speech_anomalies_v2",
    "detect_backchannels_v2",
    "compute_overlap_and_interruptions",
    # Performance and validation
    "benchmark_performance_v2",
    "validate_module",
    # Constants
    "COMPREHENSIVE_FILLER_WORDS",
    "LIBROSA_AVAILABLE",
    "SCIPY_AVAILABLE",
    "PARSELMOUTH_AVAILABLE",
]

# Version information
__version__ = "2.1.0"
__author__ = "Paralinguistics Research Team"
__description__ = (
    "Production-optimized paralinguistic feature extraction with enhanced CPU performance"
)

if __name__ == "__main__":
    import sys

    sys.exit(main())
