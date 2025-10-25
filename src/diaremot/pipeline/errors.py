"""Unified error types for the DiaRemot pipeline.

The orchestration layer historically raised heterogeneous exceptions that
were hard to correlate with a specific pipeline stage.  This module provides a
compact hierarchy that captures stage metadata and a serialisable context
payload so that callers (CLI, REST, notebooks) can render actionable error
messages.  The exceptions intentionally keep the public API surface narrow to
avoid leaking implementation details from individual stages.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PipelineError",
    "StageExecutionError",
    "ConfigurationError",
    "DependencyError",
    "attach_context",
    "coerce_stage_error",
]


@dataclass(slots=True)
class PipelineError(RuntimeError):
    """Base class for pipeline level failures.

    Attributes
    ----------
    message:
        Human readable description of the failure.
    stage:
        Optional stage identifier (``None`` for configuration level issues).
    context:
        JSON serialisable dictionary with granular diagnostics.  Callers can
        surface this directly to end users or persist it for incident review.
    cause:
        Underlying exception (kept for debugging, not included in ``__str__``).
    """

    message: str
    stage: str | None = None
    context: MutableMapping[str, Any] = field(default_factory=dict)
    cause: Exception | None = None

    def __post_init__(self) -> None:  # pragma: no cover - defensive programming
        if self.context is None:
            self.context = {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.stage:
            return f"[{self.stage}] {self.message}"
        return self.message


class StageExecutionError(PipelineError):
    """Error raised when a specific stage fails to execute."""


class ConfigurationError(PipelineError):
    """Raised when configuration validation fails."""


class DependencyError(PipelineError):
    """Raised when runtime dependencies are missing or incompatible."""


def attach_context(
    error: PipelineError,
    context: Mapping[str, Any] | None,
) -> PipelineError:
    """Merge ``context`` into ``error.context`` preserving existing keys."""

    if not context:
        return error
    error.context.update(context)
    return error


def coerce_stage_error(
    stage: str,
    message: str,
    *,
    context: Mapping[str, Any] | None = None,
    cause: Exception | None = None,
) -> StageExecutionError:
    """Create :class:`StageExecutionError` with a rich context payload."""

    payload: MutableMapping[str, Any] = {}
    if context:
        payload.update(context)
    if cause:
        payload.setdefault("cause", repr(cause))
    return StageExecutionError(message=message, stage=stage, context=payload, cause=cause)
