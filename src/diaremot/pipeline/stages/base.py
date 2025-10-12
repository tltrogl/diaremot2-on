"""Shared state and definitions for pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if not hasattr(np, "array"):
    raise RuntimeError("NumPy is required for pipeline stage execution")


@dataclass
class PipelineState:
    """Mutable state passed between pipeline stages."""

    input_audio_path: str
    out_dir: Path
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sr: int = 16000
    health: Any = None
    sed_info: dict[str, Any] | None = None
    tx_out: list[Any] = field(default_factory=list)
    norm_tx: list[dict[str, Any]] = field(default_factory=list)
    speakers_summary: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    segments_final: list[dict[str, Any]] = field(default_factory=list)
    conv_metrics: Any = None
    duration_s: float = 0.0
    overlap_stats: dict[str, Any] = field(default_factory=dict)
    per_speaker_interrupts: dict[str, Any] = field(default_factory=dict)
    audio_sha16: str = ""
    pp_sig: dict[str, Any] = field(default_factory=dict)
    cache_dir: Path | None = None
    diar_cache: dict[str, Any] | None = None
    tx_cache: dict[str, Any] | None = None
    resume_diar: bool = False
    resume_tx: bool = False
    vad_unstable: bool = False
    para_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)
    dependency_summary: dict[str, Any] | None = None
    tuning_summary: dict[str, Any] = field(default_factory=dict)
    tuning_history: list[dict[str, Any]] = field(default_factory=list)


StageRunner = Callable[["AudioAnalysisPipelineV2", PipelineState, Any], None]


@dataclass(frozen=True)
class StageDefinition:
    name: str
    runner: StageRunner


__all__ = ["PipelineState", "StageDefinition", "StageRunner"]
