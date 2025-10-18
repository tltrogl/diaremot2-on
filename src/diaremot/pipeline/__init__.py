"""Core pipeline orchestration and helper components."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_SUBMODULES: dict[str, str] = {
    "audio_pipeline_core": "diaremot.pipeline.audio_pipeline_core",
    "audio_preprocessing": "diaremot.pipeline.audio_preprocessing",
    "cli_entry": "diaremot.pipeline.cli_entry",
    "cpu_optimized_diarizer": "diaremot.pipeline.cpu_optimized_diarizer",
    "config": "diaremot.pipeline.config",
    "component_factory": "diaremot.pipeline.component_factory",
    "logging_utils": "diaremot.pipeline.logging_utils",
    "orchestrator": "diaremot.pipeline.orchestrator",
    "outputs": "diaremot.pipeline.outputs",
    "pipeline_checkpoint_system": "diaremot.pipeline.pipeline_checkpoint_system",
    "pipeline_diagnostic": "diaremot.pipeline.pipeline_diagnostic",
    "pipeline_healthcheck": "diaremot.pipeline.pipeline_healthcheck",
    "run_pipeline": "diaremot.pipeline.run_pipeline",
    "speaker_diarization": "diaremot.pipeline.speaker_diarization",
    "transcription_module": "diaremot.pipeline.transcription_module",
    "validate_system_complete": "diaremot.pipeline.validate_system_complete",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()))
