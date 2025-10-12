"""Dependency health stage."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import dependency_health_summary
from ..logging_utils import StageGuard
from .base import PipelineState

if TYPE_CHECKING:
    from ..orchestrator import AudioAnalysisPipelineV2

__all__ = ["run"]


def run(pipeline: AudioAnalysisPipelineV2, state: PipelineState, guard: StageGuard) -> None:
    # Skip the dependency summary when validation isn't requested. The config flag
    # `validate_dependencies` is already used in AudioAnalysisPipelineV2 to decide
    # whether the upfront verification runs; honouring it here prevents the stage
    # from blocking long runs with repeated import checks.
    if not pipeline.cfg.get("validate_dependencies", False):
        pipeline.corelog.info("[dependency_check] skipped (validate_dependencies=false)")
        guard.done(skipped=1)
        return

    dep_summary = dependency_health_summary()
    unhealthy = [k for k, v in dep_summary.items() if v.get("status") != "ok"]
    pipeline.corelog.event("dependency_check", "summary", unhealthy=unhealthy)
    if unhealthy:
        pipeline.corelog.warn(f"Dependency issues detected: {unhealthy}")
        for key in unhealthy:
            issue = dep_summary[key].get("issue")
            if issue:
                pipeline.stats.warnings.append(f"dep:{key}: {issue}")

    try:
        pipeline.stats.config_snapshot["dependency_ok"] = len(unhealthy) == 0
        pipeline.stats.config_snapshot["dependency_summary"] = dep_summary
    except Exception:
        pass

    state.dependency_summary = dep_summary
    guard.done(unhealthy=len(unhealthy))
