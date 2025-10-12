"""Affect analysis and assembly stage."""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from typing import Any

import numpy as np

from ..logging_utils import StageGuard
from ..outputs import ensure_segment_keys
from .base import PipelineState

__all__ = ["run"]


def _estimate_snr_db_from_noise(noise_score: Any) -> float | None:
    """Convert a PANNs noise score into an approximate SNR in dB.

    The raw ``noise_score`` returned by :class:`PANNSEventTagger` is a sum of
    clip-wise probabilities for labels that are considered "noise-like". In
    practice the value tends to fall within ``[0, ~2]`` for speech recordings.

    We map that scalar onto a coarse signal-to-noise ratio estimate using a
    logarithmic curve so that small increases in noise probability have a
    noticeable impact while still saturating gracefully for very noisy clips.
    The heuristic below assumes ~35 dB SNR for pristine audio and rolls off
    toward 0 dB as ``noise_score`` grows. Results are clamped to ``[-5, 35]``
    so downstream consumers always receive a finite float.
    """

    try:
        score = float(noise_score)
    except (TypeError, ValueError):
        return None

    if score <= 0.0:
        return 35.0

    snr = 35.0 - 20.0 * math.log10(1.0 + 10.0 * score)
    if snr < -5.0:
        return -5.0
    if snr > 35.0:
        return 35.0
    return snr


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    segments_final: list[dict[str, Any]] = []

    if pipeline.stats.config_snapshot.get("transcribe_failed"):
        state.segments_final = segments_final
        guard.done(segments=0)
        return

    sed_payload = state.sed_info or {}
    timeline_events = []
    if isinstance(sed_payload, dict):
        timeline_events = sed_payload.get("timeline_events") or []
    timeline_index = _build_timeline_index(timeline_events)

    for idx, seg in enumerate(state.norm_tx):
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        i0 = int(start * state.sr)
        i1 = int(end * state.sr)
        clip = state.y[max(0, i0) : max(0, i1)] if len(state.y) > 0 else np.array([])
        text = seg.get("text") or ""

        aff = pipeline._affect_unified(clip, state.sr, text)
        pm = state.para_metrics.get(idx, {})

        duration_s = _coerce_positive_float(pm.get("duration_s"))
        if duration_s is None:
            duration_s = max(0.0, end - start)

        words = _coerce_int(pm.get("words"))
        if words is None:
            words = len(text.split())

        pause_ratio = _coerce_positive_float(pm.get("pause_ratio"))
        if pause_ratio is None:
            pause_time = _coerce_positive_float(pm.get("pause_time_s")) or 0.0
            pause_ratio = (pause_time / duration_s) if duration_s > 0 else 0.0
        pause_ratio = max(0.0, min(1.0, pause_ratio))

        vad = aff.get("vad", {})
        speech_emotion = aff.get("speech_emotion", {})
        text_emotions = aff.get("text_emotions", {})
        intent = aff.get("intent", {})

        row = {
            "file_id": pipeline.stats.file_id,
            "start": start,
            "end": end,
            "speaker_id": seg.get("speaker_id"),
            "speaker_name": seg.get("speaker_name"),
            "text": text,
            "valence": float(vad.get("valence", 0.0)) if vad.get("valence") is not None else None,
            "arousal": float(vad.get("arousal", 0.0)) if vad.get("arousal") is not None else None,
            "dominance": (
                float(vad.get("dominance", 0.0)) if vad.get("dominance") is not None else None
            ),
            "emotion_top": speech_emotion.get("top", "neutral"),
            "emotion_scores_json": json.dumps(
                speech_emotion.get("scores_8class", {"neutral": 1.0}), ensure_ascii=False
            ),
            "text_emotions_top5_json": json.dumps(
                text_emotions.get("top5", [{"label": "neutral", "score": 1.0}]),
                ensure_ascii=False,
            ),
            "text_emotions_full_json": json.dumps(
                text_emotions.get("full_28class", {"neutral": 1.0}), ensure_ascii=False
            ),
            "intent_top": intent.get("top", "status_update"),
            "intent_top3_json": json.dumps(intent.get("top3", []), ensure_ascii=False),
            "low_confidence_ser": bool(speech_emotion.get("low_confidence_ser", False)),
            "vad_unstable": bool(state.vad_unstable),
            "affect_hint": aff.get("affect_hint", "neutral-status"),
            "asr_logprob_avg": seg.get("asr_logprob_avg"),
            "snr_db": seg.get("snr_db"),
            "wpm": pm.get("wpm", 0.0),
            "duration_s": duration_s,
            "words": words,
            "pause_ratio": pause_ratio,
            "pause_count": pm.get("pause_count", 0),
            "pause_time_s": pm.get("pause_time_s", 0.0),
            "f0_mean_hz": pm.get("f0_mean_hz", 0.0),
            "f0_std_hz": pm.get("f0_std_hz", 0.0),
            "loudness_rms": pm.get("loudness_rms", 0.0),
            "disfluency_count": pm.get("disfluency_count", 0),
            "vq_jitter_pct": pm.get("vq_jitter_pct"),
            "vq_shimmer_db": pm.get("vq_shimmer_db"),
            "vq_hnr_db": pm.get("vq_hnr_db"),
            "vq_cpps_db": pm.get("vq_cpps_db"),
            "voice_quality_hint": pm.get("vq_note"),
            "error_flags": seg.get("error_flags", ""),
        }

        events_top = []
        snr_db_sed = None
        if isinstance(sed_payload, dict) and sed_payload:
            events_top = sed_payload.get("top") or []
            row["noise_tag"] = sed_payload.get("dominant_label")
            snr_db_sed = _estimate_snr_db_from_noise(sed_payload.get("noise_score"))

        if timeline_index is not None:
            overlaps = _intersect_events(start, end, timeline_index)
            if overlaps:
                events_top = _topk_by_overlap(overlaps, k=3)
                snr_from_events = _estimate_snr_from_events(overlaps, max(1e-6, row.get("duration_s", end - start)))
                if snr_from_events is not None:
                    snr_db_sed = snr_from_events
            try:
                row["events_top3_json"] = json.dumps(events_top[:3], ensure_ascii=False)
            except (TypeError, ValueError):
                row["events_top3_json"] = "[]"
        elif events_top:
            try:
                row["events_top3_json"] = json.dumps(events_top[:3], ensure_ascii=False)
            except (TypeError, ValueError):
                row["events_top3_json"] = "[]"

        if snr_db_sed is not None:
            row["snr_db_sed"] = snr_db_sed

        segments_final.append(ensure_segment_keys(row))

    state.segments_final = segments_final

    if timeline_index is not None and isinstance(state.turns, list):
        for turn in state.turns:
            if not isinstance(turn, dict):
                continue
            try:
                start = float(turn.get("start", turn.get("start_time", 0.0)) or 0.0)
                end = float(turn.get("end", turn.get("end_time", start)) or start)
            except (TypeError, ValueError):
                continue
            overlaps = _intersect_events(start, end, timeline_index)
            if overlaps:
                turn["events_top3"] = _topk_by_overlap(overlaps, k=3)
                snr_turn = _estimate_snr_from_events(overlaps, max(1e-6, end - start))
                if snr_turn is not None:
                    turn["snr_db_sed"] = snr_turn
            elif "events_top3" not in turn:
                turn["events_top3"] = []

    guard.done(segments=len(segments_final))


def _coerce_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0:
        return None
    return result


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _build_timeline_index(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not events:
        return None
    try:
        sorted_events = sorted(events, key=lambda ev: float(ev.get("start", 0.0)))
    except Exception:
        return None
    starts = [float(ev.get("start", 0.0)) for ev in sorted_events]
    return {"events": sorted_events, "starts": starts}


def _intersect_events(start: float, end: float, index: dict[str, Any]) -> list[dict[str, Any]]:
    events = index.get("events") or []
    starts = index.get("starts") or []
    if not events or not starts or end <= start:
        return []
    overlaps: list[dict[str, Any]] = []
    idx = max(0, bisect_left(starts, start) - 1)
    while idx < len(events):
        event = events[idx]
        ev_start = float(event.get("start", 0.0))
        ev_end = float(event.get("end", ev_start))
        if ev_start >= end:
            break
        if ev_end <= start:
            idx += 1
            continue
        overlap_start = max(start, ev_start)
        overlap_end = min(end, ev_end)
        overlap = overlap_end - overlap_start
        if overlap > 0:
            score = float(event.get("score", 0.0))
            overlaps.append(
                {
                    "label": event.get("label"),
                    "score": score,
                    "start": ev_start,
                    "end": ev_end,
                    "overlap": overlap,
                    "weight": overlap * max(0.0, score),
                }
            )
        idx += 1
    return overlaps


def _topk_by_overlap(overlaps: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    if not overlaps:
        return []
    aggregates: dict[str, dict[str, float | str]] = {}
    for item in overlaps:
        label = str(item.get("label", "unknown"))
        slot = aggregates.setdefault(label, {"label": label, "overlap": 0.0, "weight": 0.0, "score": 0.0})
        overlap = float(item.get("overlap", 0.0))
        weight = float(item.get("weight", 0.0))
        slot["overlap"] = float(slot["overlap"]) + overlap
        slot["weight"] = float(slot["weight"]) + weight
    for slot in aggregates.values():
        overlap = float(slot.get("overlap", 0.0))
        weight = float(slot.get("weight", 0.0))
        slot["score"] = (weight / overlap) if overlap > 0 else 0.0
    ranked = sorted(
        aggregates.values(),
        key=lambda item: (float(item.get("weight", 0.0)), float(item.get("score", 0.0))),
        reverse=True,
    )
    return [
        {
            "label": entry.get("label"),
            "score": float(entry.get("score", 0.0)),
            "overlap": float(entry.get("overlap", 0.0)),
        }
        for entry in ranked[:k]
    ]


def _estimate_snr_from_events(overlaps: list[dict[str, Any]], duration: float) -> float | None:
    if not overlaps or duration <= 0:
        return None
    total_overlap = sum(float(item.get("overlap", 0.0)) for item in overlaps)
    total_weight = sum(float(item.get("weight", 0.0)) for item in overlaps)
    if total_overlap <= 0:
        return None
    density = min(1.0, total_overlap / duration)
    weighted = min(1.0, total_weight / duration)
    composite = min(1.0, 0.5 * density + 0.5 * weighted)
    snr = 35.0 - 35.0 * composite
    if snr < -5.0:
        return -5.0
    if snr > 35.0:
        return 35.0
    return snr
