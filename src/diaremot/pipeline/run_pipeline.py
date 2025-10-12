"""Compatibility shims for legacy ``diaremot.pipeline.run_pipeline`` imports."""

from __future__ import annotations

from . import audio_pipeline_core as _core
from .cache_env import configure_local_cache_env

if "_DIAREMOT_CACHE_ENV_CONFIGURED" not in globals():
    configure_local_cache_env()
    _DIAREMOT_CACHE_ENV_CONFIGURED = True

AudioAnalysisPipelineV2 = _core.AudioAnalysisPipelineV2
DEFAULT_PIPELINE_CONFIG = _core.DEFAULT_PIPELINE_CONFIG


def build_pipeline_config(overrides=None):
    return _core.build_pipeline_config(overrides)


def run_pipeline(input_path, outdir, *, config=None, clear_cache=False):
    return _core.run_pipeline(
        input_path,
        outdir,
        config=config,
        clear_cache=clear_cache,
    )


def resume(checkpoint_path, *, outdir=None, config=None, allow_reprocess=False):
    if allow_reprocess:
        import warnings

        warnings.warn(
            "allow_reprocess flag is ignored by audio_pipeline_core.resume; continuing without reprocessing.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _core.resume(
        checkpoint_path,
        outdir=outdir,
        config=config,
    )


def diagnostics(require_versions=False):
    return _core.diagnostics(require_versions=require_versions)


def verify_dependencies(strict=False):
    return _core.verify_dependencies(strict=strict)


def clear_pipeline_cache(cache_root=None):
    return _core.clear_pipeline_cache(cache_root)


__all__ = [
    "AudioAnalysisPipelineV2",
    "DEFAULT_PIPELINE_CONFIG",
    "build_pipeline_config",
    "run_pipeline",
    "resume",
    "diagnostics",
    "verify_dependencies",
    "clear_pipeline_cache",
]
