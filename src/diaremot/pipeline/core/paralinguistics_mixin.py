"""Paralinguistics feature extraction helper mixin."""

from __future__ import annotations

import importlib
import importlib.util
import math
from types import ModuleType
from typing import Any

import numpy as np


def _load_paralinguistics() -> ModuleType | None:
    spec = importlib.util.find_spec("diaremot.affect.paralinguistics")
    if spec is None:
        return None
    return importlib.import_module("diaremot.affect.paralinguistics")


para = _load_paralinguistics()


class ParalinguisticsMixin:
    def _extract_paraling(self, wav: np.ndarray, sr: int, segs: list[dict[str, Any]]):
        wav = np.asarray(wav, dtype=np.float32)
        results: dict[int, dict[str, Any]] = {}

        def _safe_float(value: Any) -> float | None:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        def _safe_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                try:
                    return int(float(value))
                except (TypeError, ValueError):
                    return None

        try:
            extract_fn = getattr(para, "extract", None) if para else None
            if extract_fn:
                out = extract_fn(wav, sr, segs) or []
                for i, d in enumerate(out):
                    seg = segs[i] if i < len(segs) else {}
                    start = _safe_float(seg.get("start"))
                    if start is None:
                        start = _safe_float(seg.get("start_time")) or 0.0
                    end = _safe_float(seg.get("end"))
                    if end is None:
                        end = _safe_float(seg.get("end_time")) or start
                    duration_s = _safe_float(d.get("duration_s"))
                    if duration_s is None:
                        duration_s = max(0.0, (end or 0.0) - (start or 0.0))

                    words = _safe_int(d.get("words"))
                    if words is None:
                        text = seg.get("text") or ""
                        words = len(text.split())

                    pause_ratio = _safe_float(d.get("pause_ratio"))
                    if pause_ratio is None:
                        pause_time = _safe_float(d.get("pause_time_s")) or 0.0
                        pause_ratio = (pause_time / duration_s) if duration_s > 0 else 0.0
                    pause_ratio = max(0.0, min(1.0, pause_ratio))

                    results[i] = {
                        "wpm": float(d.get("wpm", 0.0) or 0.0),
                        "duration_s": float(duration_s),
                        "words": int(words),
                        "pause_count": int(d.get("pause_count", 0) or 0),
                        "pause_time_s": float(d.get("pause_time_s", 0.0) or 0.0),
                        "pause_ratio": float(pause_ratio),
                        "f0_mean_hz": float(d.get("f0_mean_hz", 0.0) or 0.0),
                        "f0_std_hz": float(d.get("f0_std_hz", 0.0) or 0.0),
                        "loudness_rms": float(d.get("loudness_rms", 0.0) or 0.0),
                        "disfluency_count": int(d.get("disfluency_count", 0) or 0),
                        "vq_jitter_pct": float(d.get("vq_jitter_pct", 0.0) or 0.0),
                        "vq_shimmer_db": float(d.get("vq_shimmer_db", 0.0) or 0.0),
                        "vq_hnr_db": float(d.get("vq_hnr_db", 0.0) or 0.0),
                        "vq_cpps_db": float(d.get("vq_cpps_db", 0.0) or 0.0),
                    }
                return results
        except Exception as exc:  # pragma: no cover - best effort fallback
            self.corelog.warn(f"[paralinguistics] fallback: {exc}")

        for i, s in enumerate(segs):
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(1e-6, end - start)
            txt = s.get("text") or ""
            words = max(0, len(txt.split()))
            wpm = (words / dur) * 60.0 if dur > 0 else 0.0

            i0 = int(start * sr)
            i1 = int(end * sr)
            clip_slice = wav[max(0, i0) : max(0, i1)]
            if hasattr(clip_slice, "astype"):
                clip_arr = clip_slice.astype(np.float32, copy=False)
            else:
                clip_arr = np.asarray(clip_slice, dtype=np.float32)
            clip_size = clip_arr.size if hasattr(clip_arr, "size") else len(clip_arr)
            loud = float(np.sqrt(np.mean(clip_arr**2))) if clip_size > 0 else 0.0

            results[i] = {
                "wpm": float(wpm),
                "duration_s": float(dur),
                "words": int(words),
                "pause_count": 0,
                "pause_time_s": 0.0,
                "pause_ratio": 0.0,
                "f0_mean_hz": 0.0,
                "f0_std_hz": 0.0,
                "loudness_rms": float(loud),
                "disfluency_count": 0,
                "vq_jitter_pct": 0.0,
                "vq_shimmer_db": 0.0,
                "vq_hnr_db": 0.0,
                "vq_cpps_db": 0.0,
            }
        return results
