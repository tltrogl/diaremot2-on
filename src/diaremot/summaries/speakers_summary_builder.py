"""
speakers_summary_builder.py — Enriched rollups from diarized_transcript_with_emotion.csv

Input: segments list with per-segment data
Output: List of speaker summary dicts

Notes:
- Interruptions/overlap are provided via per_speaker_interrupts parameter
- We aggregate text emotions using the FULL 28-class distribution when present (preferred)
- avg_wpm = (total_words / total_duration_minutes). WPM from segments (if present) is used only as a backup
"""

from __future__ import annotations

import json
import statistics
from typing import Any

GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def _float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _safe_json(s):
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _words(text: str) -> int:
    if not text:
        return 0
    return sum(1 for tok in str(text).strip().split() if tok)


def _median(vals: list[float], default: float = 0.0) -> float:
    vals = [v for v in vals if v is not None]
    try:
        return float(statistics.median(vals)) if vals else default
    except Exception:
        return default


def _top_k(d: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _normalize(d: dict[str, float]) -> dict[str, float]:
    s = sum(d.values()) or 1.0
    return {k: (v / s) for k, v in d.items()}


def build_speakers_summary(
    segments: list[dict[str, Any]],
    per_speaker_interrupts: dict[str, dict[str, Any]],
    overlap_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build speaker summary from segments data (matches core interface)
    Returns list of dicts suitable for CSV writing by core
    """
    # Accumulators per speaker
    agg: dict[str, dict[str, Any]] = {}

    # Read segments
    for row in segments:
        sid = row.get("speaker_id") or "Unknown"
        name = row.get("speaker_name") or ""
        start = _float(row.get("start"))
        end = _float(row.get("end"))
        dur = max(0.0, end - start)
        text = row.get("text", "")
        words = _words(text)

        # Init speaker bucket
        S = agg.setdefault(
            sid,
            {
                "speaker_id": sid,
                "speaker_name": name,
                "durations": [],
                "words": 0,
                "per_seg_wpm": [],
                "valence": [],
                "arousal": [],
                "dominance": [],
                "ser_votes": {},  # majority for dominant_speech_emotion
                "text_full_sum": {k: 0.0 for k in GOEMOTIONS_LABELS},
                "text_top_sum": {},  # fallback if no full
                "f0_mean_vals": [],
                "f0_std_vals": [],
                "loudness_vals": [],
                "pause_total_sec": 0.0,
                "pause_count": 0,
                # optional overlap/interruptions (may be merged later)
                "interruptions_made": 0,
                "interruptions_received": 0,
                "overlap_ratio": None,
                # voice quality accumulators
                "vq_jitter": [],
                "vq_shimmer": [],
                "vq_hnr": [],
                "vq_cpps": [],
                "vq_hint_votes": {},
            },
        )

        # Core accumulations
        S["durations"].append(dur)
        S["words"] += words

        if row.get("wpm") not in (None, ""):
            try:
                S["per_seg_wpm"].append(_float(row["wpm"]))
            except Exception:
                pass

        # V/A/D
        for k in ("valence", "arousal", "dominance"):
            if row.get(k) not in (None, ""):
                S[k].append(_float(row[k]))

        # SER top label vote (optional)
        emo_top = row.get("emotion_top") or ""
        if emo_top:
            S["ser_votes"][emo_top] = S["ser_votes"].get(emo_top, 0) + 1

        # Text emotions (prefer full 28-class)
        full = _safe_json(row.get("text_emotions_full_json"))
        if isinstance(full, dict):
            for lbl, sc in full.items():
                if lbl in S["text_full_sum"]:
                    S["text_full_sum"][lbl] += _float(sc)
        else:
            top5 = _safe_json(row.get("text_emotions_top5_json"))
            if isinstance(top5, list):
                for item in top5:
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        S["text_top_sum"][item["label"]] = S["text_top_sum"].get(
                            item["label"], 0.0
                        ) + _float(item["score"])
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        lbl, sc = item[0], item[1]
                        S["text_top_sum"][str(lbl)] = S["text_top_sum"].get(str(lbl), 0.0) + _float(
                            sc
                        )

        # Paralinguistics
        if row.get("f0_mean_hz") not in (None, ""):
            S["f0_mean_vals"].append(_float(row["f0_mean_hz"]))
        if row.get("f0_std_hz") not in (None, ""):
            S["f0_std_vals"].append(_float(row["f0_std_hz"]))
        if row.get("loudness_rms") not in (None, ""):
            S["loudness_vals"].append(_float(row["loudness_rms"]))
        # Voice-quality metrics per segment
        if row.get("vq_jitter_pct") not in (None, ""):
            S["vq_jitter"].append(_float(row["vq_jitter_pct"]))
        if row.get("vq_shimmer_db") not in (None, ""):
            S["vq_shimmer"].append(_float(row["vq_shimmer_db"]))
        if row.get("vq_hnr_db") not in (None, ""):
            S["vq_hnr"].append(_float(row["vq_hnr_db"]))
        if row.get("vq_cpps_db") not in (None, ""):
            S["vq_cpps"].append(_float(row["vq_cpps_db"]))
        if row.get("voice_quality_hint") not in (None, ""):
            _h = str(row.get("voice_quality_hint")).strip()
            if _h:
                S["vq_hint_votes"][_h] = S["vq_hint_votes"].get(_h, 0) + 1
        if row.get("pause_time_s") not in (None, ""):
            S["pause_total_sec"] += _float(row["pause_time_s"])
        if row.get("pause_count") not in (None, ""):
            S["pause_count"] += int(_float(row["pause_count"]))

        # Optional: interruptions/overlap if segments carried them
        if row.get("interruptions_made") not in (None, ""):
            S["interruptions_made"] += int(_float(row["interruptions_made"]))
        if row.get("interruptions_received") not in (None, ""):
            S["interruptions_received"] += int(_float(row["interruptions_received"]))
        if row.get("overlap_ratio") not in (None, ""):
            # keep last or max; here we keep max seen
            val = _float(row["overlap_ratio"])
            S["overlap_ratio"] = val if S["overlap_ratio"] is None else max(S["overlap_ratio"], val)

    # Merge interruption data from per_speaker_interrupts
    for speaker_id, int_data in (per_speaker_interrupts or {}).items():
        if speaker_id in agg:
            S = agg[speaker_id]
            S["interruptions_made"] = int(int_data.get("made", S["interruptions_made"]))
            S["interruptions_received"] = int(int_data.get("received", S["interruptions_received"]))
            overlap_sec = float(int_data.get("overlap_sec", 0.0))
            total_dur = sum(S["durations"]) or 1.0
            S["overlap_ratio"] = overlap_sec / total_dur

    # Emit rows
    rows: list[dict[str, Any]] = []
    for sid, S in agg.items():
        total_dur = sum(S["durations"])
        total_min = total_dur / 60.0 if total_dur > 0 else 0.0
        # primary WPM from words/time; fallback to median of per-segment WPM
        avg_wpm = (S["words"] / total_min) if total_min > 0 else (_median(S["per_seg_wpm"], 0.0))

        # Dominant speech emotion by votes
        dominant_speech = (
            max(S["ser_votes"], key=S["ser_votes"].get) if S["ser_votes"] else "neutral"
        )

        # Text emotions aggregation
        if any(v > 0 for v in S["text_full_sum"].values()):
            text_full = _normalize(S["text_full_sum"])
        elif S["text_top_sum"]:
            text_full = _normalize(
                {k: v for k, v in S["text_top_sum"].items() if k in GOEMOTIONS_LABELS}
                or {"neutral": 1.0}
            )
        else:
            text_full = {"neutral": 1.0}
        text_top5 = [{"label": k, "score": float(v)} for k, v in _top_k(text_full, 5)]
        dominant_text = text_top5[0]["label"] if text_top5 else "neutral"

        # Loudness DR (simple robust spread): 20*log10((p95-p05)+eps) proxy—if we had RMS per frame we could do better.
        # Here we approximate DR via IQR proxy from per-seg RMS; scaled to dB-ish units.
        loud_vals = sorted([v for v in S["loudness_vals"] if v is not None])
        if len(loud_vals) >= 4:
            q1 = loud_vals[int(0.25 * len(loud_vals))]
            q3 = loud_vals[int(0.75 * len(loud_vals))]
            dr = max(0.0, (q3 - q1) * 20.0)  # heuristic
        else:
            dr = 0.0
        loud_med = _median(loud_vals, 0.0)

        pause_total = float(S["pause_total_sec"])
        pause_ratio = (pause_total / total_dur) if total_dur > 0 else 0.0

        rows.append(
            {
                "speaker_id": sid,
                "speaker_name": S["speaker_name"],
                "total_duration": round(total_dur, 3),
                "word_count": int(S["words"]),
                "avg_wpm": round(avg_wpm, 1),
                "avg_valence": round(_median(S["valence"], 0.0), 3),
                "avg_arousal": round(_median(S["arousal"], 0.0), 3),
                "avg_dominance": round(_median(S["dominance"], 0.0), 3),
                "dominant_speech_emotion": dominant_speech,
                "dominant_text_emotion": dominant_text,
                "interruptions_made": int(S["interruptions_made"]),
                "interruptions_received": int(S["interruptions_received"]),
                "overlap_ratio": (
                    round(S["overlap_ratio"], 3) if (S["overlap_ratio"] is not None) else ""
                ),
                "f0_mean_hz": round(_median(S["f0_mean_vals"], 0.0), 2),
                "f0_std_hz": round(_median(S["f0_std_vals"], 0.0), 2),
                "loudness_rms_med": round(loud_med, 3),
                "loudness_dr_db": round(dr, 2),
                "pause_total_sec": round(pause_total, 2),
                "pause_count": int(S["pause_count"]),
                "pause_ratio": round(pause_ratio, 3),
                # Voice-quality rollups (median)
                "vq_jitter_pct_med": round(_median(S["vq_jitter"], 0.0), 3),
                "vq_shimmer_db_med": round(_median(S["vq_shimmer"], 0.0), 3),
                "vq_hnr_db_med": round(_median(S["vq_hnr"], 0.0), 3),
                "vq_cpps_db_med": round(_median(S["vq_cpps"], 0.0), 3),
                "voice_quality_hint": (
                    max(S["vq_hint_votes"], key=S["vq_hint_votes"].get)
                    if S["vq_hint_votes"]
                    else ""
                ),
                "text_emotions_top5_json": json.dumps(text_top5, ensure_ascii=False),
                "text_emotions_full_json": json.dumps(text_full, ensure_ascii=False),
            }
        )

    return rows
