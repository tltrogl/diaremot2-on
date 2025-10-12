"""Speaker diarization stage."""

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


def _fallback_turn(duration_s: float) -> dict[str, Any]:
    return {
        "start": 0.0,
        "end": duration_s,
        "speaker": "Speaker_1",
        "speaker_name": "Speaker_1",
    }


def _jsonable_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for turn in turns:
        try:
            tcopy = dict(turn)
        except (TypeError, ValueError, AttributeError):
            continue
        emb = tcopy.get("embedding")
        try:
            import numpy as _np

            if isinstance(emb, _np.ndarray):
                tcopy["embedding"] = emb.tolist()
            elif hasattr(emb, "tolist"):
                tcopy["embedding"] = emb.tolist()
            elif emb is not None and not isinstance(emb, list | float | int | str | bool):
                tcopy["embedding"] = None
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            tcopy["embedding"] = None
        out.append(tcopy)
    return out


def _speaker_metrics(turns: list[dict[str, Any]]) -> tuple[int, int]:
    speakers = {
        str(turn.get("speaker", "")).strip()
        for turn in turns
        if isinstance(turn, dict) and turn.get("speaker")
    }
    return len(speakers), len(turns)


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    turns: list[dict[str, Any]] = []
    duration_s = state.duration_s

    if (
        state.resume_tx
        and not state.diar_cache
        and state.tx_cache
        and state.tx_cache.get("segments")
    ):
        tx_segments = state.tx_cache.get("segments", []) or []
        for entry in tx_segments:
            try:
                start = float(entry.get("start", entry.get("start_time", 0.0)) or 0.0)
                end = float(entry.get("end", entry.get("end_time", 0.0)) or 0.0)
            except Exception:
                start, end = 0.0, 0.0
            if end < start:
                start, end = end, start
            speaker_id = str(entry.get("speaker_id", entry.get("speaker", "Speaker_1")))
            speaker_name = entry.get("speaker_name") or speaker_id
            turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker_id,
                    "speaker_name": speaker_name,
                }
            )
        if not turns:
            turns.append(_fallback_turn(duration_s))
        state.vad_unstable = False
        speakers_est, turn_count = _speaker_metrics(turns)
        pipeline.corelog.info(
            f"[diarize] resume (tx cache) estimated {speakers_est} speakers across {turn_count} turns"
        )
        guard.done(turns=turn_count, speakers_est=speakers_est)
    elif state.resume_diar and state.diar_cache:
        turns = state.diar_cache.get("turns", []) or []
        if not turns:
            pipeline.corelog.warn("Cached diar.json has 0 turns; proceeding with fallback")
            turns = [_fallback_turn(duration_s)]
        vad_toggles = 0
        try:
            vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
        except (TypeError, AttributeError):
            vad_toggles = 0
        state.vad_unstable = (vad_toggles / max(1, int(duration_s / 60) or 1)) > 60
        speakers_est, turn_count = _speaker_metrics(turns)
        pipeline.corelog.info(
            f"[diarize] resume (diar cache) estimated {speakers_est} speakers across {turn_count} turns"
        )
        guard.done(turns=turn_count, speakers_est=speakers_est)
    else:
        try:
            turns = pipeline.diar.diarize_audio(state.y, state.sr) or []
        except (
            RuntimeError,
            ValueError,
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            pipeline.corelog.warn(
                "Diarization failed: "
                f"{exc}; reverting to single-speaker assumption. Verify ECAPA/pyannote assets."
            )
            turns = []
        if not turns:
            pipeline.corelog.warn("Diarizer returned 0 turns; using fallback")
            turns = [_fallback_turn(duration_s)]

        vad_toggles = 0
        try:
            vad_toggles = sum(1 for t in turns if t.get("is_boundary_flip"))
        except (TypeError, AttributeError):
            vad_toggles = 0
        state.vad_unstable = (vad_toggles / max(1, int(duration_s / 60) or 1)) > 60
        speakers_est, turn_count = _speaker_metrics(turns)
        pipeline.corelog.info(
            f"[diarize] estimated {speakers_est} speakers across {turn_count} turns"
        )
        guard.done(turns=turn_count, speakers_est=speakers_est)

        if state.cache_dir:
            try:
                atomic_write_json(
                    state.cache_dir / "diar.json",
                    {
                        "version": pipeline.cache_version,
                        "audio_sha16": state.audio_sha16,
                        "pp_signature": state.pp_sig,
                        "turns": _jsonable_turns(turns),
                        "saved_at": time.time(),
                    },
                )
            except OSError as exc:
                pipeline.corelog.warn(
                    f"[cache] diar.json write failed: {exc}. Ensure cache directory is writable."
                )

    if not turns:
        turns = [_fallback_turn(duration_s)]

    state.turns = turns
    pipeline.checkpoints.create_checkpoint(
        state.input_audio_path,
        ProcessingStage.DIARIZATION,
        turns,
        progress=30.0,
    )
