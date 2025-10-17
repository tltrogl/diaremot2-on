"""
DiaRemot: Enhanced Speech ML Analysis Package
Optimized lazy-loading with comprehensive integration
"""

__version__ = "2.2.0"
__author__ = "DiaRemot ML Team"

import importlib
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

from .io.download_utils import download_file

# Module caches for lazy loading
_cached_modules = {}
_logger = logging.getLogger(__name__)

_LEGACY_MODULE_ALIASES = {
    "audio_pipeline_core": "diaremot.pipeline.audio_pipeline_core",
    "audio_preprocessing": "diaremot.pipeline.audio_preprocessing",
    "cpu_optimized_diarizer": "diaremot.pipeline.cpu_optimized_diarizer",
    "pipeline_checkpoint_system": "diaremot.pipeline.pipeline_checkpoint_system",
    "pipeline_diagnostic": "diaremot.pipeline.pipeline_diagnostic",
    "pipeline_healthcheck": "diaremot.pipeline.pipeline_healthcheck",
    "run_pipeline": "diaremot.pipeline.run_pipeline",
    "speaker_diarization": "diaremot.pipeline.speaker_diarization",
    "transcription_module": "diaremot.pipeline.transcription_module",
    "validate_system_complete": "diaremot.pipeline.validate_system_complete",
    "emotion_analysis": "diaremot.affect.emotion_analysis",
    "emotion_analyzer": "diaremot.affect.emotion_analyzer",
    "intent_defaults": "diaremot.affect.intent_defaults",
    "paralinguistics": "diaremot.affect.paralinguistics",
    "sed_panns": "diaremot.affect.sed_panns",
    "conversation_analysis": "diaremot.summaries.conversation_analysis",
    "html_summary_generator": "diaremot.summaries.html_summary_generator",
    "pdf_summary_generator": "diaremot.summaries.pdf_summary_generator",
    "speakers_summary_builder": "diaremot.summaries.speakers_summary_builder",
    "download_utils": "diaremot.io.download_utils",
    "speaker_registry_manager": "diaremot.io.speaker_registry_manager",
    "onnx_utils": "diaremot.io.onnx_utils",
}


def _install_legacy_alias(name: str, target: str) -> None:
    fullname = f"{__name__}.{name}"
    if fullname in sys.modules:
        return

    proxy = types.ModuleType(fullname)
    proxy.__doc__ = f"Compatibility alias for {target}"

    def _load_module():
        module = importlib.import_module(target)
        sys.modules[fullname] = module
        return module

    def _proxy_getattr(attr: str):
        module = _load_module()
        return getattr(module, attr)

    def _proxy_dir():
        module = _load_module()
        return dir(module)

    proxy.__getattr__ = _proxy_getattr  # type: ignore[attr-defined]
    proxy.__dir__ = _proxy_dir  # type: ignore[attr-defined]
    sys.modules[fullname] = proxy


for _legacy_name, _target in _LEGACY_MODULE_ALIASES.items():
    _install_legacy_alias(_legacy_name, _target)


def _get_or_create_logger():
    """Get package logger with appropriate configuration"""
    logger = logging.getLogger("diaremot")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_preprocessor(config: dict[str, Any] | None = None):
    """Get AudioPreprocessor with configuration"""
    if "preprocessor" not in _cached_modules:
        try:
            from .pipeline.audio_preprocessing import (
                AudioPreprocessor,
                PreprocessConfig,
            )

            _cached_modules["preprocessor"] = (AudioPreprocessor, PreprocessConfig)
            _get_or_create_logger().info("AudioPreprocessor loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load preprocessing: {e}")
            raise

    AudioPreprocessor, PreprocessConfig = _cached_modules["preprocessor"]

    if config:
        preprocess_config = PreprocessConfig(**config)
    else:
        preprocess_config = PreprocessConfig()

    return AudioPreprocessor(preprocess_config)


def get_diarizer(config: dict[str, Any] | None = None):
    """Get SpeakerDiarizer with configuration and registry integration"""
    if "diarizer" not in _cached_modules:
        try:
            from .pipeline.speaker_diarization import (
                DiarizationConfig,
                SpeakerDiarizer,
            )

            _cached_modules["diarizer"] = (SpeakerDiarizer, DiarizationConfig)
            _get_or_create_logger().info("SpeakerDiarizer loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load diarization: {e}")
            raise

    SpeakerDiarizer, DiarizationConfig = _cached_modules["diarizer"]

    if config:
        diar_config = DiarizationConfig(**config)
    else:
        diar_config = DiarizationConfig()

    return SpeakerDiarizer(diar_config)


def get_transcriber(config: dict[str, Any] | None = None):
    """Get AudioTranscriber with CPU optimization"""
    if "transcriber" not in _cached_modules:
        try:
            from .pipeline.transcription_module import (
                AudioTranscriber,
                create_transcriber,
            )

            _cached_modules["transcriber"] = (AudioTranscriber, create_transcriber)
            _get_or_create_logger().info("AudioTranscriber loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load transcription: {e}")
            raise

    AudioTranscriber, create_transcriber = _cached_modules["transcriber"]

    # Force CPU-only configuration
    cpu_config = {"device": "cpu", "compute_type": "float32", **(config or {})}

    # Default model selection: allow overriding via env var DIAREMOT_MODEL
    try:
        if "model_size" not in cpu_config:
            env_model = os.getenv("DIAREMOT_MODEL")
            if env_model:
                cpu_config["model_size"] = env_model
    except Exception:
        pass

    return create_transcriber(**cpu_config)


def get_pipeline(config: dict[str, Any] | None = None):
    """Get enhanced AudioAnalysisPipelineV2 with full integration"""
    if "pipeline" not in _cached_modules:
        try:
            from .pipeline.audio_pipeline_core import AudioAnalysisPipelineV2

            _cached_modules["pipeline"] = AudioAnalysisPipelineV2
            _get_or_create_logger().info("AudioAnalysisPipelineV2 loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load pipeline core: {e}")
            raise

    AudioAnalysisPipelineV2 = _cached_modules["pipeline"]
    return AudioAnalysisPipelineV2(config)


def get_registry_manager(registry_path: str | None = None):
    """Get thread-safe speaker registry manager"""
    if "registry" not in _cached_modules:
        try:
            from .io.speaker_registry_manager import SpeakerRegistryManager

            _cached_modules["registry"] = SpeakerRegistryManager
            _get_or_create_logger().info("SpeakerRegistryManager loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load registry manager: {e}")
            raise

    SpeakerRegistryManager = _cached_modules["registry"]
    resolved = (
        registry_path
        or os.getenv("DIAREMOT_REGISTRY_PATH")
        or str(Path.cwd() / "registry" / "speaker_registry.json")
    )
    return SpeakerRegistryManager(resolved)


def get_checkpoint_manager(checkpoint_dir: str | None = None):
    """Get pipeline checkpoint manager for resume functionality"""
    if "checkpoint" not in _cached_modules:
        try:
            from .pipeline.pipeline_checkpoint_system import (
                PipelineCheckpointManager,
            )

            _cached_modules["checkpoint"] = PipelineCheckpointManager
            _get_or_create_logger().info("PipelineCheckpointManager loaded successfully")
        except ImportError as e:
            _get_or_create_logger().error(f"Failed to load checkpoint system: {e}")
            raise

    PipelineCheckpointManager = _cached_modules["checkpoint"]
    resolved = (
        checkpoint_dir or os.getenv("DIAREMOT_CHECKPOINT_DIR") or str(Path.cwd() / "checkpoints")
    )
    return PipelineCheckpointManager(resolved)


def create_integrated_pipeline(
    config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Create fully integrated pipeline with all components"""
    config = config or {}

    # Initialize core components
    pipeline = get_pipeline(config)
    registry_manager = get_registry_manager(config.get("registry_path"))
    checkpoint_manager = get_checkpoint_manager(config.get("checkpoint_dir"))

    # Component status
    components = {
        "pipeline": pipeline,
        "registry_manager": registry_manager,
        "checkpoint_manager": checkpoint_manager,
        "preprocessor_available": True,
        "diarizer_available": True,
        "transcriber_available": True,
    }

    return pipeline, components


# Package health check
def validate_system() -> dict[str, Any]:
    """Comprehensive system validation"""
    status = {
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "version": __version__,
        "components": {},
        "critical_issues": [],
        "warnings": [],
    }

    # Test each component
    components_to_test = [
        ("preprocessor", lambda: get_preprocessor()),
        ("diarizer", lambda: get_diarizer()),
        ("transcriber", lambda: get_transcriber()),
        ("pipeline", lambda: get_pipeline()),
        ("registry", lambda: get_registry_manager()),
        ("checkpoint", lambda: get_checkpoint_manager()),
    ]

    for name, loader in components_to_test:
        try:
            component = loader()
            status["components"][name] = {
                "available": True,
                "type": type(component).__name__,
                "module": getattr(component, "__module__", "unknown"),
            }
        except Exception as e:
            status["components"][name] = {"available": False, "error": str(e)}
            status["critical_issues"].append(f"{name}: {e}")

    # Overall health
    available_count = sum(1 for c in status["components"].values() if c["available"])
    total_count = len(status["components"])

    if available_count == total_count:
        status["health"] = "excellent"
    elif available_count >= total_count * 0.8:
        status["health"] = "good"
    elif available_count >= total_count * 0.5:
        status["health"] = "degraded"
    else:
        status["health"] = "critical"

    return status


# Enhanced logging during import
_init_logger = _get_or_create_logger()
_init_logger.info(f"DiaRemot v{__version__} package initialized")

# Package-level exports
__all__ = [
    "get_preprocessor",
    "get_diarizer",
    "get_transcriber",
    "get_pipeline",
    "get_registry_manager",
    "get_checkpoint_manager",
    "create_integrated_pipeline",
    "validate_system",
    "download_file",
    "__version__",
]
