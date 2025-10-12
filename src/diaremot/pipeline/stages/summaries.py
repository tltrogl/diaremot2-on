"""Summary and reporting stages."""

from __future__ import annotations

from ...summaries.conversation_analysis import (
    ConversationMetrics,
    analyze_conversation_flow,
)
from ...summaries.speakers_summary_builder import build_speakers_summary
from ..logging_utils import StageGuard
from .base import PipelineState

__all__ = [
    "run_overlap",
    "run_conversation",
    "run_speaker_rollups",
    "run_outputs",
]


def run_overlap(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    overlap_stats = {"overlap_total_sec": 0.0, "overlap_ratio": 0.0}
    per_speaker = {}
    try:
        module = getattr(pipeline, "paralinguistics_module", None)
        if module and hasattr(module, "compute_overlap_and_interruptions"):
            overlap = module.compute_overlap_and_interruptions(state.turns) or {}
        else:
            overlap = {}
        overlap_stats = {
            "overlap_total_sec": float(overlap.get("overlap_total_sec", 0.0)),
            "overlap_ratio": float(overlap.get("overlap_ratio", 0.0)),
        }

        per_speaker_map: dict[str, dict[str, float | int]] = {}
        made_provided: set[str] = set()

        def _ensure_entry(key: str) -> dict[str, float | int]:
            return per_speaker_map.setdefault(key, {"made": 0, "received": 0, "overlap_sec": 0.0})

        for speaker_id, values in (overlap.get("by_speaker") or {}).items():
            if not isinstance(values, dict):
                continue
            key = str(speaker_id)
            slot = _ensure_entry(key)
            slot["overlap_sec"] = float(values.get("overlap_sec", 0.0) or 0.0)

            made_raw = values.get("made", values.get("interruptions"))
            try:
                made_val = int(float(made_raw))
            except (TypeError, ValueError):
                made_val = 0
            slot["made"] = made_val
            made_provided.add(key)

        for item in overlap.get("interruptions", []) or []:
            if not isinstance(item, dict):
                continue

            interrupter = item.get("interrupter")
            if interrupter not in (None, ""):
                key = str(interrupter)
                slot = _ensure_entry(key)
                if key not in made_provided:
                    slot["made"] = int(slot.get("made", 0)) + 1

            interrupted = item.get("interrupted")
            if interrupted in (None, ""):
                continue
            slot = _ensure_entry(str(interrupted))
            slot["received"] = int(slot.get("received", 0)) + 1

        per_speaker = per_speaker_map
    except (AttributeError, RuntimeError, ValueError) as exc:
        pipeline.corelog.warn(
            "[overlap] skipped: "
            f"{exc}. Install paralinguistics extras or validate overlap feature inputs."
        )
    state.overlap_stats = overlap_stats
    state.per_speaker_interrupts = per_speaker
    guard.done()


def run_conversation(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    try:
        metrics = analyze_conversation_flow(state.segments_final, state.duration_s)
        pipeline.corelog.event(
            "conversation_analysis",
            "metrics",
            balance=metrics.turn_taking_balance,
            pace=metrics.conversation_pace_turns_per_min,
            coherence=metrics.topic_coherence_score,
        )
        state.conv_metrics = metrics
    except (RuntimeError, ValueError, ZeroDivisionError) as exc:
        pipeline.corelog.warn(
            f"Conversation analysis failed: {exc}. Falling back to neutral conversational metrics."
        )
        try:
            state.conv_metrics = ConversationMetrics(
                turn_taking_balance=0.5,
                interruption_rate_per_min=0.0,
                avg_turn_duration_sec=0.0,
                conversation_pace_turns_per_min=0.0,
                silence_ratio=0.0,
                speaker_dominance={},
                response_latency_stats={},
                topic_coherence_score=0.0,
                energy_flow=[],
            )
        except (TypeError, ValueError):
            state.conv_metrics = None
    guard.done()


def run_speaker_rollups(
    pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard
) -> None:
    try:
        summary = build_speakers_summary(
            state.segments_final, state.per_speaker_interrupts, state.overlap_stats
        )
        if isinstance(summary, dict):
            summary = [dict(v, speaker_id=k) for k, v in summary.items()]
        elif not isinstance(summary, list):
            summary = []
        state.speakers_summary = summary
    except (RuntimeError, ValueError, TypeError) as exc:
        pipeline.corelog.warn(
            "Speaker rollups failed: "
            f"{exc}. Inspect segment records or disable speaker summary generation."
        )
        state.speakers_summary = []
    guard.done(count=len(state.speakers_summary))


def run_outputs(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    pipeline._write_outputs(
        state.input_audio_path,
        state.out_dir,
        state.segments_final,
        state.speakers_summary,
        state.health,
        state.turns,
        state.overlap_stats,
        state.per_speaker_interrupts,
        state.conv_metrics,
        state.duration_s,
        state.sed_info,
    )

    if state.cache_dir:
        try:
            (state.cache_dir / ".done").write_text("ok", encoding="utf-8")
        except OSError:
            pass

    guard.done()
