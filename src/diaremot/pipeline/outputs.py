"""Output helpers for the DiaRemot pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .logging_utils import RunStats, _make_json_safe

SEGMENT_COLUMNS = [
    "file_id",
    "start",
    "end",
    "speaker_id",
    "speaker_name",
    "text",
    "valence",
    "arousal",
    "dominance",
    "emotion_top",
    "emotion_scores_json",
    "text_emotions_top5_json",
    "text_emotions_full_json",
    "intent_top",
    "intent_top3_json",
    "events_top3_json",
    "noise_tag",
    "asr_logprob_avg",
    "snr_db",
    "snr_db_sed",
    "wpm",
    "duration_s",
    "words",
    "pause_ratio",
    "low_confidence_ser",
    "vad_unstable",
    "affect_hint",
    "pause_count",
    "pause_time_s",
    "f0_mean_hz",
    "f0_std_hz",
    "loudness_rms",
    "disfluency_count",
    "error_flags",
    "vq_jitter_pct",
    "vq_shimmer_db",
    "vq_hnr_db",
    "vq_cpps_db",
    "voice_quality_hint",
]


def default_affect() -> dict[str, Any]:
    ser_scores = {"neutral": 1.0}
    text_full = {"neutral": 1.0}
    return {
        "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        "speech_emotion": {
            "top": "neutral",
            "scores_8class": ser_scores,
            "low_confidence_ser": True,
        },
        "text_emotions": {
            "top5": [{"label": "neutral", "score": 1.0}],
            "full_28class": text_full,
        },
        "intent": {
            "top": "status_update",
            "top3": [
                {"label": "status_update", "score": 1.0},
                {"label": "small_talk", "score": 0.0},
                {"label": "opinion", "score": 0.0},
            ],
        },
        "affect_hint": "neutral-status",
    }


def ensure_segment_keys(seg: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "events_top3_json": "[]",
        "low_confidence_ser": False,
        "vad_unstable": False,
        "error_flags": "",
    }
    for key in SEGMENT_COLUMNS:
        if key not in seg:
            seg[key] = defaults.get(key, None)
    return seg


def write_segments_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in SEGMENT_COLUMNS})


def write_segments_jsonl(path: Path, segments: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps(segment, ensure_ascii=False) + "\n")


def write_timeline_csv(path: Path, segments: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start", "end", "speaker_id"])
        for segment in segments:
            writer.writerow(
                [
                    segment.get("start", 0.0),
                    segment.get("end", 0.0),
                    segment.get("speaker_id", ""),
                ]
            )


def write_qc_report(
    path: Path,
    stats: RunStats,
    health: Any,
    *,
    n_turns: int,
    n_segments: int,
    segments: list[dict[str, Any]],
) -> None:
    payload = {
        "run_id": stats.run_id,
        "file_id": stats.file_id,
        "schema_version": stats.schema_version,
        "stage_timings_ms": stats.stage_timings_ms,
        "stage_counts": stats.stage_counts,
        "warnings": stats.warnings,
        "errors": getattr(stats, "errors", []),
        "failures": getattr(stats, "failures", []),
        "models": stats.models,
        "config_snapshot": stats.config_snapshot,
        "audio_health": {
            "snr_db": float(getattr(health, "snr_db", 0.0)) if health else None,
            "silence_ratio": float(getattr(health, "silence_ratio", 0.0)) if health else None,
            "clipping_detected": (
                bool(getattr(health, "clipping_detected", False)) if health else None
            ),
            "dynamic_range_db": float(getattr(health, "dynamic_range_db", 0.0)) if health else None,
        },
        "counts": {"turns": int(n_turns), "segments": int(n_segments)},
    }

    try:

        def _avg(key: str) -> float | None:
            values = []
            for seg in segments or []:
                value = seg.get(key)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except Exception:
                    continue
            return float(sum(values) / len(values)) if values else None

        payload["voice_quality_summary"] = {
            "vq_jitter_pct_avg": _avg("vq_jitter_pct"),
            "vq_shimmer_db_avg": _avg("vq_shimmer_db"),
            "vq_hnr_db_avg": _avg("vq_hnr_db"),
            "vq_cpps_db_avg": _avg("vq_cpps_db"),
        }
    except Exception:  # pragma: no cover - best effort aggregate
        pass

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_speakers_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    headers = sorted({key for row in rows for key in row.keys()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, None) for key in headers})


__all__ = [
    "SEGMENT_COLUMNS",
    "default_affect",
    "ensure_segment_keys",
    "write_segments_csv",
    "write_segments_jsonl",
    "write_timeline_csv",
    "write_qc_report",
    "write_speakers_summary",
]
