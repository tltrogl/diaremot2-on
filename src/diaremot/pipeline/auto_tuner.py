"""Adaptive heuristics to tune diarization and ASR knobs per recording."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # Parselmouth is optional during unit tests
    from .audio_preprocessing import AudioHealth
except Exception:  # pragma: no cover - defensive import fallback
    AudioHealth = Any  # type: ignore


@dataclass
class AutoTuneResult:
    """Container describing recommended adjustments."""

    diarization: dict[str, float] = field(default_factory=dict)
    asr: dict[str, float | int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diarization": dict(self.diarization),
            "asr": dict(self.asr),
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
        }


class AutoTuner:
    """Simple rule-based tuner grounded in audio health metrics."""

    def __init__(
        self,
        *,
        min_vad_threshold: float = 0.12,
        max_vad_threshold: float = 0.38,
        min_no_speech: float = 0.32,
        max_no_speech: float = 0.65,
    ) -> None:
        self.min_vad_threshold = float(min_vad_threshold)
        self.max_vad_threshold = float(max_vad_threshold)
        self.min_no_speech = float(min_no_speech)
        self.max_no_speech = float(max_no_speech)

    def recommend(
        self,
        *,
        health: AudioHealth | None,
        audio: np.ndarray,
        sr: int,
        diar_config: Mapping[str, Any] | Any,
        asr_config: Mapping[str, Any],
    ) -> AutoTuneResult:
        """Return recommended adjustments for the current recording."""

        result = AutoTuneResult()

        if audio is None or not sr:
            return result
        length = getattr(audio, "size", None)
        if length is None:
            try:
                length = len(audio)  # type: ignore[arg-type]
            except Exception:
                length = 0
        if not length:
            return result

        # Guard against NaNs and extreme values
        arr = np.asarray(audio, dtype=np.float32)
        if hasattr(arr, "tolist"):
            base_iter = arr.tolist()
        else:
            base_iter = list(arr)
        finite = [float(x) for x in base_iter if math.isfinite(float(x))]
        if not finite:
            return result
        abs_values = [abs(x) for x in finite]
        peak = max(abs_values) if abs_values else 0.0
        mean_square = sum(x * x for x in finite) / max(1, len(finite))
        rms = math.sqrt(mean_square) if mean_square > 0.0 else 0.0
        noise_floor = _percentile(abs_values, 10.0)
        crest = 0.0
        if rms > 0.0 and peak > 0.0:
            crest = 20.0 * math.log10(max(peak, 1e-5) / max(rms, 1e-6))

        peak_dbfs = 20.0 * math.log10(max(peak, 1e-5))
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
        noise_dbfs = 20.0 * math.log10(max(noise_floor, 1e-6))

        result.metrics.update(
            {
                "peak_dbfs": _round3(peak_dbfs),
                "rms_dbfs": _round3(rms_dbfs),
                "noise_floor_dbfs": _round3(noise_dbfs),
                "crest_factor_db": _round3(crest),
            }
        )

        snr_db = _safe_float(getattr(health, "snr_db", None))
        silence_ratio = _safe_float(getattr(health, "silence_ratio", None))
        dynamic_range = _safe_float(getattr(health, "dynamic_range_db", None))
        clipping = bool(getattr(health, "clipping_detected", False))

        if snr_db is not None:
            result.metrics["snr_db"] = _round3(snr_db, digits=2)
        if silence_ratio is not None:
            result.metrics["silence_ratio"] = _round3(silence_ratio)
        if dynamic_range is not None:
            result.metrics["dynamic_range_db"] = _round3(dynamic_range, digits=2)
        if clipping:
            result.notes.append("clipping_detected")

        base_vad = _safe_float(getattr(diar_config, "vad_threshold", None))
        base_min_speech = _safe_float(getattr(diar_config, "vad_min_speech_sec", None))
        base_min_silence = _safe_float(getattr(diar_config, "vad_min_silence_sec", None))
        base_pad = _safe_float(getattr(diar_config, "speech_pad_sec", None))

        # --- Diarization heuristics ---
        if snr_db is not None:
            if snr_db < 6.0 and base_vad is not None:
                target = max(self.min_vad_threshold, base_vad - 0.12)
                if _changed(base_vad, target):
                    result.diarization["vad_threshold"] = _round3(target)
                    result.notes.append("snr<6_lower_vad")
                if base_min_speech is not None:
                    new_min_speech = max(0.30, min(base_min_speech, 0.45))
                    if _changed(base_min_speech, new_min_speech):
                        result.diarization["vad_min_speech_sec"] = _round3(new_min_speech)
                        result.notes.append("shorten_min_speech_low_snr")
                if base_pad is not None:
                    new_pad = min(0.30, max(base_pad, 0.22))
                    if _changed(base_pad, new_pad):
                        result.diarization["speech_pad_sec"] = _round3(new_pad)
            elif snr_db < 12.0 and base_vad is not None:
                target = max(self.min_vad_threshold, base_vad - 0.08)
                if _changed(base_vad, target):
                    result.diarization["vad_threshold"] = _round3(target)
                    result.notes.append("snr<12_lower_vad")
                if base_pad is not None:
                    new_pad = min(0.24, max(base_pad, 0.18))
                    if _changed(base_pad, new_pad):
                        result.diarization["speech_pad_sec"] = _round3(new_pad)
            elif snr_db > 26.0 and base_vad is not None:
                target = min(self.max_vad_threshold, max(base_vad, 0.30))
                if _changed(base_vad, target):
                    result.diarization["vad_threshold"] = _round3(target)
                    result.notes.append("snr>26_raise_vad")

        if silence_ratio is not None and base_min_silence is not None:
            if silence_ratio > 0.65:
                new_min_silence = min(1.2, max(base_min_silence, 0.6))
                if _changed(base_min_silence, new_min_silence):
                    result.diarization["vad_min_silence_sec"] = _round3(new_min_silence)
                    result.notes.append("high_silence_extend_min_silence")
            elif silence_ratio < 0.25:
                new_min_silence = max(0.35, min(base_min_silence, 0.55))
                if _changed(base_min_silence, new_min_silence):
                    result.diarization["vad_min_silence_sec"] = _round3(new_min_silence)
                    result.notes.append("dense_speech_shorten_min_silence")

        if dynamic_range is not None and dynamic_range < 12.0 and base_min_speech is not None:
            new_min_speech = max(0.28, min(base_min_speech, 0.40))
            if _changed(base_min_speech, new_min_speech):
                result.diarization["vad_min_speech_sec"] = _round3(new_min_speech)
                result.notes.append("low_dynamic_range_shorter_min_speech")

        if clipping and base_pad is not None:
            new_pad = min(0.28, max(base_pad, 0.20))
            if _changed(base_pad, new_pad):
                result.diarization["speech_pad_sec"] = _round3(new_pad)

        # --- ASR heuristics ---
        beam_size = _safe_int(asr_config.get("beam_size"))
        no_speech_threshold = _safe_float(asr_config.get("no_speech_threshold"))

        if snr_db is not None and snr_db < 10.0:
            if beam_size is not None and beam_size < 2:
                result.asr["beam_size"] = 2
                result.notes.append("low_snr_raise_beam")
            if no_speech_threshold is not None:
                new_no_speech = max(
                    self.min_no_speech, min(no_speech_threshold - 0.1, self.max_no_speech)
                )
                if _changed(no_speech_threshold, new_no_speech):
                    result.asr["no_speech_threshold"] = _round3(new_no_speech)
                    result.notes.append("low_snr_lower_no_speech")
        elif silence_ratio is not None and silence_ratio > 0.7:
            if no_speech_threshold is not None:
                new_no_speech = min(self.max_no_speech, max(no_speech_threshold, 0.55))
                if _changed(no_speech_threshold, new_no_speech):
                    result.asr["no_speech_threshold"] = _round3(new_no_speech)
                    result.notes.append("high_silence_raise_no_speech")
        elif snr_db is not None and snr_db > 28.0:
            if beam_size is not None and beam_size > 1:
                result.asr["beam_size"] = 1
                result.notes.append("high_snr_restore_beam")

        if clipping and no_speech_threshold is not None:
            new_no_speech = max(self.min_no_speech, min(no_speech_threshold, 0.52))
            if _changed(no_speech_threshold, new_no_speech):
                result.asr["no_speech_threshold"] = _round3(new_no_speech)
                result.notes.append("clipping_guard_no_speech")

        if not result.notes:
            result.notes.append("noop")

        return result


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _changed(before: float, after: float, *, tol: float = 1e-3) -> bool:
    return abs(float(before) - float(after)) > tol


def _round3(value: float, *, digits: int = 3) -> float:
    return float(round(float(value), digits))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, float(percentile)))
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_vals[lower]
    lower_val = sorted_vals[lower]
    upper_val = sorted_vals[upper]
    weight = rank - lower
    return lower_val + (upper_val - lower_val) * weight


__all__ = ["AutoTuner", "AutoTuneResult"]
