"""Paralinguistics extraction stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..logging_utils import StageGuard
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    metrics: dict[int, dict[str, object]] = {}
    if not pipeline.stats.config_snapshot.get("transcribe_failed"):
        wav = np.asarray(state.y, dtype=np.float32)
        tmp_metrics = pipeline._extract_paraling(wav, state.sr, state.norm_tx)
        if isinstance(tmp_metrics, dict):
            metrics = tmp_metrics
    state.para_metrics = metrics
    guard.done(count=len(metrics))
