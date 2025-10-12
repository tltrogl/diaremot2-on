"""Automatic speech recognition stage."""

from __future__ import annotations

import subprocess
import time
from typing import TYPE_CHECKING, Any

from ..logging_utils import StageGuard
from ..pipeline_checkpoint_system import ProcessingStage
from .base import PipelineState
from .utils import atomic_write_json

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def _placeholder_segment(seg: dict[str, Any]) -> Any:
    return type(
        "TranscriptionSegment",
        (),
        {
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "text": "[Transcription unavailable]",
            "speaker_id": seg["speaker_id"],
            "speaker_name": seg["speaker_name"],
            "asr_logprob_avg": None,
            "snr_db": None,
        },
    )()


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    tx_in: list[dict[str, Any]] = []
    speaker_name_map: dict[str, str] = {}
    for turn in state.turns:
        start = float(turn.get("start", turn.get("start_time", 0.0)) or 0.0)
        end = float(turn.get("end", turn.get("end_time", start + 0.5)) or (start + 0.5))
        speaker_id = str(turn.get("speaker"))
        speaker_name = turn.get("speaker_name") or speaker_id
        speaker_name_map[speaker_id] = speaker_name
        tx_in.append(
            {
                "start_time": start,
                "end_time": end,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
            }
        )

    tx_out: list[Any] = []
    if state.resume_tx and state.tx_cache:
        tx_out = []
        guard.done(segments=len(state.tx_cache.get("segments", []) or []))
    else:
        try:
            tx_out = pipeline.tx.transcribe_segments(state.y, state.sr, tx_in) or []
        except (
            RuntimeError,
            TimeoutError,
            subprocess.CalledProcessError,
        ) as exc:
            pipeline.corelog.warn(
                "Transcription failed: "
                f"{exc}; generating placeholder segments. Verify faster-whisper setup or tune --asr-segment-timeout."
            )
            tx_out = [_placeholder_segment(seg) for seg in tx_in]

        guard.done(segments=len(tx_out))

    norm_tx: list[dict[str, Any]] = []
    if state.resume_tx and state.tx_cache and (state.tx_cache.get("segments") is not None):
        for segment in state.tx_cache.get("segments", []):
            norm_tx.append(
                {
                    "start": float(
                        segment.get("start", 0.0) or segment.get("start_time", 0.0) or 0.0
                    ),
                    "end": float(segment.get("end", 0.0) or segment.get("end_time", 0.0) or 0.0),
                    "speaker_id": segment.get("speaker_id"),
                    "speaker_name": segment.get("speaker_name"),
                    "text": segment.get("text", ""),
                    "asr_logprob_avg": segment.get("asr_logprob_avg"),
                    "snr_db": segment.get("snr_db"),
                    "error_flags": segment.get("error_flags", ""),
                }
            )
    else:
        for item in tx_out:
            if hasattr(item, "__dict__"):
                payload = item.__dict__
            elif isinstance(item, dict):
                payload = item
            else:
                payload = {}
            norm_tx.append(
                {
                    "start": float(payload.get("start_time", payload.get("start", 0.0)) or 0.0),
                    "end": float(payload.get("end_time", payload.get("end", 0.0)) or 0.0),
                    "speaker_id": payload.get("speaker_id"),
                    "speaker_name": payload.get("speaker_name"),
                    "text": payload.get("text", ""),
                    "asr_logprob_avg": payload.get("asr_logprob_avg"),
                    "snr_db": payload.get("snr_db"),
                    "error_flags": "",
                }
            )

    state.tx_out = tx_out
    state.norm_tx = norm_tx

    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.TRANSCRIPTION,
        norm_tx,
        progress=60.0,
    )

    if state.cache_dir:
        try:
            atomic_write_json(
                state.cache_dir / "tx.json",
                {
                    "version": pipeline.cache_version,
                    "audio_sha16": state.audio_sha16,
                    "pp_signature": state.pp_sig,
                    "segments": norm_tx,
                    "saved_at": time.time(),
                },
            )
        except OSError as exc:
            pipeline.corelog.warn(
                f"[cache] tx.json write failed: {exc}. Check disk space and cache directory permissions."
            )
