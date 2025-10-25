"""Voice-quality feature extraction helpers."""

from __future__ import annotations

import warnings

import numpy as np

from .audio import robust_pitch_extraction
from .config import ParalinguisticsConfig
from .environment import (
    LIBROSA_AVAILABLE,
    PARSELMOUTH_AVAILABLE,
    SCIPY_AVAILABLE,
    call,
    librosa,
    parselmouth,
    scipy_signal,
)


def compute_voice_quality_parselmouth(audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig) -> dict[str, float]:
    """Compute Praat-derived voice quality metrics when Parselmouth is available."""

    if not PARSELMOUTH_AVAILABLE:
        return compute_voice_quality_fallback(audio, sr, cfg)

    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = call(sound, "To Pitch", 0.0, cfg.f0_min_hz, cfg.f0_max_hz)
        point_process = call(sound, "To PointProcess (periodic, cc)", cfg.f0_min_hz, cfg.f0_max_hz)

        results: dict[str, float] = {}

        try:
            jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            results["jitter_pct"] = float(jitter_local * 100) if not np.isnan(jitter_local) else 0.0
        except Exception:
            results["jitter_pct"] = 0.0

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
            results["shimmer_db"] = float(shimmer_local_db) if not np.isnan(shimmer_local_db) else 0.0
        except Exception:
            results["shimmer_db"] = 0.0

        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, cfg.f0_min_hz, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0.0, 0.0)
            results["hnr_db"] = float(hnr) if not np.isnan(hnr) else 0.0
        except Exception:
            results["hnr_db"] = 0.0

        try:
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
                results["cpps_db"] = max(0.0, results.get("hnr_db", 0.0) - 2.0)

        try:
            f0_values = pitch.selected_array["frequency"]
            voiced_frames = ~np.isnan(f0_values) if f0_values.size > 0 else np.array([])
            results["voiced_ratio"] = float(np.mean(voiced_frames)) if voiced_frames.size > 0 else 0.0
        except Exception:
            results["voiced_ratio"] = 0.0

        try:
            ltas = call(sound, "To Spectrum", True).values if hasattr(sound, "values") else None
            if ltas is None:
                raise RuntimeError("unable to compute LTAS")
        except Exception:
            ltas = None

        results.setdefault("spectral_slope_db", 0.0)
        return results
    except Exception as exc:
        warnings.warn(f"Parselmouth voice quality failed: {exc}", RuntimeWarning, stacklevel=2)
        return compute_voice_quality_fallback(audio, sr, cfg)


def compute_voice_quality_fallback(audio: np.ndarray, sr: int, cfg: ParalinguisticsConfig) -> dict[str, float]:
    """Estimate voice quality without Parselmouth using numpy/librosa."""

    try:
        f0, voiced_flag = robust_pitch_extraction(audio, sr, cfg)

        if f0.size == 0:
            return {
                "jitter_pct": 0.0,
                "shimmer_db": 0.0,
                "hnr_db": 0.0,
                "cpps_db": 0.0,
                "voiced_ratio": 0.0,
                "spectral_slope_db": 0.0,
            }

        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag.size > 0 else 0.0

        jitter_pct = 0.0
        if np.sum(voiced_flag) > 10:
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 3:
                periods = 1.0 / (voiced_f0 + 1e-12)
                if len(periods) > 1:
                    period_diffs = np.abs(np.diff(periods))
                    mean_period = np.mean(periods)
                    jitter_pct = float((np.mean(period_diffs) / mean_period) * 100)
                    jitter_pct = np.clip(jitter_pct, 0.0, 25.0)

        shimmer_db = 0.0
        if LIBROSA_AVAILABLE and audio.size > sr * 0.1:
            try:
                hop_length = min(256, len(audio) // 20)
                rms_frames = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
                if len(rms_frames) > 5:
                    rms_db = librosa.amplitude_to_db(rms_frames + 1e-12)
                    db_diffs = np.diff(rms_db)
                    shimmer_db = float(np.median(np.abs(db_diffs - np.median(db_diffs))))
                    shimmer_db = np.clip(shimmer_db, 0.0, 15.0)
            except Exception:
                pass

        hnr_db = 0.0
        if LIBROSA_AVAILABLE:
            try:
                contrast = librosa.feature.spectral_contrast(
                    y=audio, sr=sr, hop_length=512, n_bands=6, fmin=cfg.f0_min_hz
                )
                if contrast.size > 0:
                    avg_contrast = np.mean(contrast)
                    hnr_db = float(np.clip(avg_contrast, 0.0, 40.0))

                if hnr_db == 0.0:
                    stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=1024))
                    if stft_mag.size > 0:
                        spectral_peaks = np.max(stft_mag, axis=0)
                        spectral_means = np.mean(stft_mag, axis=0)
                        regularity = spectral_peaks / (spectral_means + 1e-12)
                        hnr_db = float(np.clip(20 * np.log10(np.mean(regularity) + 1e-12), 0.0, 35.0))
            except Exception:
                hnr_db = 8.0

        cpps_db = 0.0
        if LIBROSA_AVAILABLE and hnr_db > 0:
            try:
                stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=2048))
                if stft_mag.shape[0] > 100 and SCIPY_AVAILABLE:
                    log_spec = np.log(stft_mag + 1e-12)
                    cepstrum_frames = []
                    for frame in range(log_spec.shape[1]):
                        frame_spec = log_spec[:, frame]
                        cepstrum = scipy_signal.fftconvolve(frame_spec, frame_spec[::-1], mode="full")
                        if len(cepstrum) > 50:
                            quefrency_range = slice(20, min(100, len(cepstrum) // 2))
                            peak_val = np.max(cepstrum[quefrency_range])
                            cepstrum_frames.append(peak_val)
                    if cepstrum_frames:
                        cpps_db = float(np.clip(np.mean(cepstrum_frames) * 0.1, 0.0, 35.0))
            except Exception:
                pass

            if cpps_db == 0.0:
                cpps_db = max(0.0, hnr_db - 3.0)

        spectral_slope_db = 0.0
        if LIBROSA_AVAILABLE:
            try:
                stft_mag = np.abs(librosa.stft(audio, hop_length=512, n_fft=2048))
                ltas = np.mean(stft_mag, axis=1)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
                freq_mask = (freqs >= 1000) & (freqs <= 4000)
                if np.sum(freq_mask) > 10:
                    slope_freqs = freqs[freq_mask]
                    slope_mag = ltas[freq_mask]
                    if np.any(slope_mag > 0):
                        log_mag = np.log10(slope_mag + 1e-12)
                        log_freq = np.log10(slope_freqs)
                        if len(log_freq) > 2:
                            slope = (log_mag[-1] - log_mag[0]) / (log_freq[-1] - log_freq[0])
                            spectral_slope_db = float(slope * 20)
            except Exception:
                pass

        return {
            "jitter_pct": float(np.clip(jitter_pct, 0.0, 25.0)),
            "shimmer_db": float(np.clip(shimmer_db, 0.0, 15.0)),
            "hnr_db": float(np.clip(hnr_db, 0.0, 40.0)),
            "cpps_db": float(np.clip(cpps_db, 0.0, 35.0)),
            "voiced_ratio": float(np.clip(voiced_ratio, 0.0, 1.0)),
            "spectral_slope_db": float(np.clip(spectral_slope_db, -30.0, 10.0)),
        }
    except Exception as exc:
        warnings.warn(f"Enhanced fallback voice quality failed: {exc}", RuntimeWarning, stacklevel=2)
        return {
            "jitter_pct": 0.0,
            "shimmer_db": 0.0,
            "hnr_db": 0.0,
            "cpps_db": 0.0,
            "voiced_ratio": 0.0,
            "spectral_slope_db": 0.0,
        }


__all__ = [
    "compute_voice_quality_parselmouth",
    "compute_voice_quality_fallback",
]
