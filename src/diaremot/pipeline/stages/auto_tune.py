"""Stage that adapts diarization/ASR knobs based on audio health."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..auto_tuner import AutoTuner
from ..logging_utils import StageGuard
from .base import PipelineState

__all__ = ["run"]


def _ensure_auto_tuner(pipeline: AudioAnalysisPipelineV2) -> AutoTuner:
    tuner = getattr(pipeline, "auto_tuner", None)
    if tuner is None:
        tuner = AutoTuner()
        pipeline.auto_tuner = tuner
    return tuner


def _get_asr_config(pipeline: AudioAnalysisPipelineV2) -> dict[str, Any]:
    tx = getattr(pipeline, "tx", None)
    async_tx = getattr(tx, "_async_transcriber", None)
    if async_tx is not None and hasattr(async_tx, "config"):
        return async_tx.config  # type: ignore[return-value]
    if tx is not None and hasattr(tx, "config"):
        return tx.config  # type: ignore[return-value]
    return {}


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    tuner = _ensure_auto_tuner(pipeline)
    diar_conf = getattr(pipeline, "diar_conf", None)
    asr_config = _get_asr_config(pipeline)

    result = tuner.recommend(
        health=getattr(state, "health", None),
        audio=getattr(state, "y", None),
        sr=int(getattr(state, "sr", 0) or 0),
        diar_config=diar_conf or {},
        asr_config=asr_config,
    )

    applied: dict[str, dict[str, dict[str, Any]]] = {
        "diarization": {},
        "asr": {},
    }

    if diar_conf is not None and result.diarization:
        for key, new_value in result.diarization.items():
            if not hasattr(diar_conf, key):
                continue
            previous = getattr(diar_conf, key)
            if previous is not None and abs(float(previous) - float(new_value)) <= 1e-3:
                continue
            try:
                setattr(diar_conf, key, new_value)
            except Exception:
                continue
            applied["diarization"][key] = {
                "previous": previous,
                "new": new_value,
            }
            if key == "vad_threshold" and hasattr(pipeline, "diar"):
                try:
                    pipeline.diar.vad.threshold = float(new_value)
                except Exception:
                    pass
            if key == "speech_pad_sec" and hasattr(pipeline, "diar"):
                try:
                    pipeline.diar.vad.speech_pad_sec = float(new_value)
                except Exception:
                    pass

    if asr_config and result.asr:
        for key, new_value in result.asr.items():
            previous = asr_config.get(key)
            if previous is not None and previous == new_value:
                continue
            asr_config[key] = new_value
            applied["asr"][key] = {
                "previous": previous,
                "new": new_value,
            }

    state.tuning_summary = {
        "result": result.to_dict(),
        "applied": applied,
    }
    state.tuning_history.append(deepcopy(state.tuning_summary))

    # Persist into run stats for downstream diagnostics
    snapshot = getattr(pipeline.stats, "config_snapshot", {})
    snapshot.setdefault("auto_tune", {})
    snapshot["auto_tune"] = {
        "metrics": result.metrics,
        "notes": list(result.notes),
        "applied": applied,
    }
    pipeline.stats.config_snapshot = snapshot

    if any(applied.values()):
        pipeline.corelog.info(
            "[auto_tune] applied adjustments: %s",
            {k: v for k, v in applied.items() if v},
        )
        pipeline.corelog.event(
            "auto_tune",
            "applied",
            metrics=result.metrics,
            notes=result.notes,
            applied=applied,
        )
    else:
        pipeline.corelog.info("[auto_tune] no adjustments required")
        pipeline.corelog.event("auto_tune", "noop", metrics=result.metrics, notes=result.notes)

    guard.done()
