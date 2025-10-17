"""Helpers for constructing pipeline dependencies with granular error handling."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from ..affect.emotion_analyzer import EmotionIntentAnalyzer
from ..affect.intent_defaults import INTENT_LABELS_DEFAULT
from ..affect.sed_panns import PANNSEventTagger, SEDConfig  # type: ignore
from ..summaries.html_summary_generator import HTMLSummaryGenerator
from ..summaries.pdf_summary_generator import PDFSummaryGenerator
from . import speaker_diarization as _speaker_diarization
from .audio_preprocessing import AudioPreprocessor, PreprocessConfig
from .auto_tuner import AutoTuner
from .logging_utils import CoreLogger
from .runtime_env import DEFAULT_WHISPER_MODEL, WINDOWS_MODELS_ROOT

T = TypeVar("T")


@dataclass
class ComponentInit(Generic[T]):
    """Capture the outcome of a single component initialisation."""

    instance: T | None
    config: Any | None = None
    issues: list[str] = field(default_factory=list)


@dataclass
class ComponentBundle:
    """Container for pipeline components and captured metadata."""

    pp_conf: PreprocessConfig
    diar_conf: _speaker_diarization.DiarizationConfig
    pre: AudioPreprocessor | None
    diar: _speaker_diarization.SpeakerDiarizer | None
    tx: Any
    auto_tuner: AutoTuner | None
    affect: EmotionIntentAnalyzer | None
    sed_tagger: Any
    html: HTMLSummaryGenerator | None
    pdf: PDFSummaryGenerator | None
    models: dict[str, str] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def build_component_bundle(
    cfg: dict[str, Any] | None,
    corelog: CoreLogger,
) -> ComponentBundle:
    """Create the core pipeline components with per-component error isolation."""

    settings: dict[str, Any] = dict(cfg or {})

    pp_result = _init_preprocessor(settings, corelog)
    pp_conf = pp_result.config or PreprocessConfig()

    diar_result = _init_diarizer(settings, pp_conf, corelog)
    diar_conf = diar_result.config or _speaker_diarization.DiarizationConfig(
        target_sr=pp_conf.target_sr
    )

    tx_result = _init_transcriber(settings, corelog)
    auto_result = _init_auto_tuner(corelog)

    affect_kwargs = _build_affect_kwargs(settings)
    affect_result = _init_affect(settings, affect_kwargs, corelog)

    sed_result = _init_sed(settings, corelog)
    html_result, pdf_result = _init_report_generators(corelog)

    issues = _collect_issues(
        pp_result,
        diar_result,
        tx_result,
        auto_result,
        affect_result,
        sed_result,
        html_result,
        pdf_result,
    )

    models = {
        "preprocessor": _model_name(pp_result.instance),
        "diarizer": _model_name(diar_result.instance),
        "transcriber": _model_name(tx_result.instance),
        "affect": _model_name(affect_result.instance),
    }

    config_snapshot = _build_config_snapshot(
        settings,
        pp_conf,
        diar_conf,
        affect_kwargs,
    )

    return ComponentBundle(
        pp_conf=pp_conf,
        diar_conf=diar_conf,
        pre=pp_result.instance,
        diar=diar_result.instance,
        tx=tx_result.instance,
        auto_tuner=auto_result.instance,
        affect=affect_result.instance,
        sed_tagger=sed_result.instance,
        html=html_result.instance,
        pdf=pdf_result.instance,
        models=models,
        config_snapshot=config_snapshot,
        issues=issues,
    )


def _init_preprocessor(
    settings: dict[str, Any],
    corelog: CoreLogger,
) -> ComponentInit[AudioPreprocessor]:
    denoise_mode = "spectral_sub_soft" if settings.get("noise_reduction", True) else "none"
    pp_conf = PreprocessConfig(
        target_sr=settings.get("target_sr", 16000),
        denoise=denoise_mode,
        loudness_mode=settings.get("loudness_mode", "asr"),
        auto_chunk_enabled=settings.get("auto_chunk_enabled", True),
        chunk_threshold_minutes=settings.get("chunk_threshold_minutes", 60.0),
        chunk_size_minutes=settings.get("chunk_size_minutes", 20.0),
        chunk_overlap_seconds=settings.get("chunk_overlap_seconds", 30.0),
    )

    try:
        pre = AudioPreprocessor(pp_conf)
        return ComponentInit(instance=pre, config=pp_conf)
    except Exception as exc:  # pragma: no cover - defensive
        corelog.error("[preprocessor] initialization failed: %s", exc)
        return ComponentInit(
            instance=None,
            config=pp_conf,
            issues=[f"preprocessor initialization failed: {exc}"],
        )


def _init_diarizer(
    settings: dict[str, Any],
    pp_conf: PreprocessConfig,
    corelog: CoreLogger,
) -> ComponentInit[_speaker_diarization.SpeakerDiarizer]:
    diar_conf = _build_diarization_config(settings, pp_conf)

    try:
        diar = _speaker_diarization.SpeakerDiarizer(diar_conf)
    except Exception as exc:  # pragma: no cover - defensive
        corelog.error("[diarizer] initialization failed: %s", exc)
        return ComponentInit(
            instance=None,
            config=diar_conf,
            issues=[f"diarizer initialization failed: {exc}"],
        )

    if bool(settings.get("cpu_diarizer", False)):
        try:
            from .cpu_optimized_diarizer import (  # type: ignore
                CPUOptimizationConfig,
                CPUOptimizedSpeakerDiarizer,
            )

            cpu_conf = CPUOptimizationConfig(max_speakers=diar_conf.speaker_limit)
            diar = CPUOptimizedSpeakerDiarizer(diar, cpu_conf)
            corelog.info("[diarizer] using CPU-optimized wrapper")
        except Exception as exc:  # pragma: no cover - defensive
            corelog.warn("[diarizer] CPU wrapper unavailable, using baseline: %s", exc)

    return ComponentInit(instance=diar, config=diar_conf)


def _init_transcriber(settings: dict[str, Any], corelog: CoreLogger) -> ComponentInit[Any]:
    from .transcription_module import AudioTranscriber

    transcriber_config = {
        "model_size": str(settings.get("whisper_model", DEFAULT_WHISPER_MODEL)),
        "language": settings.get("language", None),
        "beam_size": settings.get("beam_size", 1),
        "temperature": settings.get("temperature", 0.0),
        "compression_ratio_threshold": settings.get("compression_ratio_threshold", 2.5),
        "log_prob_threshold": settings.get("log_prob_threshold", -1.0),
        "no_speech_threshold": settings.get("no_speech_threshold", 0.20),
        "condition_on_previous_text": settings.get("condition_on_previous_text", False),
        "word_timestamps": settings.get("word_timestamps", True),
        "max_asr_window_sec": settings.get("max_asr_window_sec", 480),
        "vad_min_silence_ms": settings.get("vad_min_silence_ms", 1800),
        "language_mode": settings.get("language_mode", "auto"),
        "compute_type": settings.get("compute_type", None),
        "cpu_threads": settings.get("cpu_threads", None),
        "asr_backend": settings.get("asr_backend", "auto"),
        "segment_timeout_sec": settings.get("segment_timeout_sec", 300.0),
        "batch_timeout_sec": settings.get("batch_timeout_sec", 1200.0),
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_DEVICE"] = "cpu"

    try:
        tx = AudioTranscriber(**transcriber_config)
        return ComponentInit(instance=tx)
    except Exception as exc:  # pragma: no cover - defensive
        corelog.error("[transcriber] initialization failed: %s", exc)
        return ComponentInit(
            instance=None,
            issues=[f"transcriber initialization failed: {exc}"],
        )


def _init_auto_tuner(corelog: CoreLogger) -> ComponentInit[AutoTuner]:
    try:
        tuner = AutoTuner()
        return ComponentInit(instance=tuner)
    except Exception as exc:  # pragma: no cover - defensive
        corelog.error("[auto_tuner] initialization failed: %s", exc)
        return ComponentInit(
            instance=None,
            issues=[f"auto_tuner initialization failed: {exc}"],
        )


def _build_affect_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "text_emotion_model": settings.get(
            "text_emotion_model", "SamLowe/roberta-base-go_emotions"
        ),
        "intent_labels": settings.get("intent_labels", INTENT_LABELS_DEFAULT),
        "affect_backend": settings.get("affect_backend", "onnx"),
        "affect_text_model_dir": settings.get("affect_text_model_dir"),
        "affect_ser_model_dir": settings.get("affect_ser_model_dir"),
        "affect_vad_model_dir": settings.get("affect_vad_model_dir"),
        "affect_intent_model_dir": settings.get("affect_intent_model_dir"),
        "analyzer_threads": settings.get("affect_analyzer_threads"),
        "disable_downloads": settings.get("disable_downloads"),
        "model_dir": settings.get("affect_model_dir"),
    }


def _init_affect(
    settings: dict[str, Any],
    affect_kwargs: dict[str, Any],
    corelog: CoreLogger,
) -> ComponentInit[EmotionIntentAnalyzer | None]:
    if settings.get("disable_affect"):
        return ComponentInit(instance=None, config=affect_kwargs)

    backend = affect_kwargs.get("affect_backend")
    if backend is not None:
        affect_kwargs = dict(affect_kwargs, affect_backend=str(backend))

    normalized_kwargs = dict(affect_kwargs)
    normalized_kwargs["affect_text_model_dir"] = _normalize_model_dir(
        normalized_kwargs.get("affect_text_model_dir")
    )
    normalized_kwargs["affect_ser_model_dir"] = _normalize_model_dir(
        normalized_kwargs.get("affect_ser_model_dir")
    )
    normalized_kwargs["affect_vad_model_dir"] = _normalize_model_dir(
        normalized_kwargs.get("affect_vad_model_dir")
    )
    normalized_kwargs["affect_intent_model_dir"] = _normalize_model_dir(
        normalized_kwargs.get("affect_intent_model_dir")
    )

    try:
        affect = EmotionIntentAnalyzer(**normalized_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        corelog.error("[affect] initialization failed: %s", exc)
        return ComponentInit(
            instance=None,
            config=normalized_kwargs,
            issues=[f"affect initialization failed: {exc}"],
        )

    issues = []
    if getattr(affect, "issues", None):
        issues.extend(str(issue) for issue in affect.issues)

    return ComponentInit(instance=affect, config=normalized_kwargs, issues=issues)


def _init_sed(settings: dict[str, Any], corelog: CoreLogger) -> ComponentInit[Any]:
    if not bool(settings.get("enable_sed", True)):
        corelog.info(
            "[sed] background sound event detection disabled; tagger will not be initialised."
        )
        return ComponentInit(instance=None)

    try:
        if PANNSEventTagger is not None:
            sed = PANNSEventTagger(SEDConfig() if SEDConfig else None)
        else:  # pragma: no cover - defensive
            sed = None
    except Exception as exc:  # pragma: no cover - defensive
        corelog.warn(
            "[sed] initialization failed: %s. Background tagging will emit empty results.",
            exc,
        )
        return ComponentInit(
            instance=None,
            issues=["background_sed assets unavailable; emitting empty tag summary"],
        )

    if sed is None or not getattr(sed, "available", False):
        return ComponentInit(
            instance=sed,
            issues=["background_sed assets unavailable; emitting empty tag summary"],
        )

    return ComponentInit(instance=sed)


def _init_report_generators(corelog: CoreLogger) -> tuple[
    ComponentInit[HTMLSummaryGenerator], ComponentInit[PDFSummaryGenerator]
]:
    html_result: ComponentInit[HTMLSummaryGenerator]
    pdf_result: ComponentInit[PDFSummaryGenerator]

    try:
        html_result = ComponentInit(instance=HTMLSummaryGenerator())
    except Exception as exc:  # pragma: no cover - defensive
        corelog.warn("[html] generator unavailable: %s", exc)
        html_result = ComponentInit(
            instance=None,
            issues=["HTML summary generator unavailable"],
        )

    try:
        pdf_result = ComponentInit(instance=PDFSummaryGenerator())
    except Exception as exc:  # pragma: no cover - defensive
        corelog.warn("[pdf] generator unavailable: %s", exc)
        pdf_result = ComponentInit(
            instance=None,
            issues=["PDF summary generator unavailable"],
        )

    return html_result, pdf_result


def _collect_issues(*results: ComponentInit[Any]) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    for res in results:
        for issue in res.issues:
            if issue and issue not in seen:
                seen.add(issue)
                collected.append(issue)
    return collected


def _model_name(obj: Any) -> str:
    if obj is None:
        return "NoneType"
    return getattr(obj, "__class__", type(obj)).__name__


def _normalize_model_dir(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return os.fspath(value)
    except TypeError:
        return str(value)


def _build_diarization_config(
    settings: dict[str, Any],
    pp_conf: PreprocessConfig,
) -> _speaker_diarization.DiarizationConfig:
    registry_path = settings.get(
        "registry_path", str(Path("registry") / "speaker_registry.json")
    )
    if not Path(registry_path).is_absolute():
        registry_path = str(Path.cwd() / registry_path)

    ecapa_path = settings.get("ecapa_model_path")
    search_paths = [
        ecapa_path,
        WINDOWS_MODELS_ROOT / "ecapa_tdnn.onnx" if WINDOWS_MODELS_ROOT else None,
        Path("models") / "ecapa_tdnn.onnx",
        Path("..") / "models" / "ecapa_tdnn.onnx",
        Path("..") / "diaremot" / "models" / "ecapa_tdnn.onnx",
        Path("..") / ".." / "models" / "ecapa_tdnn.onnx",
        Path("models") / "Diarization" / "ecapa-onnx" / "ecapa_tdnn.onnx",
    ]

    resolved_path = None
    for candidate in search_paths:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.is_absolute():
            candidate_path = Path.cwd() / candidate_path
        if candidate_path.exists():
            resolved_path = str(candidate_path.resolve())
            break

    return _speaker_diarization.DiarizationConfig(
        target_sr=pp_conf.target_sr,
        registry_path=registry_path,
        ahc_distance_threshold=settings.get("ahc_distance_threshold", 0.15),
        speaker_limit=settings.get("speaker_limit", None),
        clustering_backend=str(settings.get("clustering_backend", "ahc")),
        min_speakers=settings.get("min_speakers", None),
        max_speakers=settings.get("max_speakers", None),
        ecapa_model_path=resolved_path,
        vad_backend=settings.get("vad_backend", "auto"),
        vad_threshold=settings.get(
            "vad_threshold",
            _speaker_diarization.DiarizationConfig.vad_threshold,
        ),
        vad_min_speech_sec=settings.get(
            "vad_min_speech_sec",
            _speaker_diarization.DiarizationConfig.vad_min_speech_sec,
        ),
        vad_min_silence_sec=settings.get(
            "vad_min_silence_sec",
            _speaker_diarization.DiarizationConfig.vad_min_silence_sec,
        ),
        speech_pad_sec=settings.get(
            "vad_speech_pad_sec",
            _speaker_diarization.DiarizationConfig.speech_pad_sec,
        ),
        allow_energy_vad_fallback=not bool(
            settings.get("disable_energy_vad_fallback", False)
        ),
        energy_gate_db=settings.get(
            "energy_gate_db",
            _speaker_diarization.DiarizationConfig.energy_gate_db,
        ),
        energy_hop_sec=settings.get(
            "energy_hop_sec",
            _speaker_diarization.DiarizationConfig.energy_hop_sec,
        ),
    )


def _build_config_snapshot(
    settings: dict[str, Any],
    pp_conf: PreprocessConfig,
    diar_conf: _speaker_diarization.DiarizationConfig,
    affect_kwargs: dict[str, Any],
) -> dict[str, Any]:
    affect_backend = affect_kwargs.get("affect_backend")
    if affect_backend is not None:
        affect_backend = str(affect_backend)

    return {
        "target_sr": pp_conf.target_sr,
        "noise_reduction": settings.get("noise_reduction", True),
        "enable_sed": bool(settings.get("enable_sed", True)),
        "registry_path": diar_conf.registry_path,
        "ahc_distance_threshold": diar_conf.ahc_distance_threshold,
        "whisper_model": str(settings.get("whisper_model", DEFAULT_WHISPER_MODEL)),
        "beam_size": settings.get("beam_size", 1),
        "temperature": settings.get("temperature", 0.0),
        "no_speech_threshold": settings.get("no_speech_threshold", 0.20),
        "intent_labels": settings.get("intent_labels", INTENT_LABELS_DEFAULT),
        "affect_backend": affect_backend,
        "affect_text_model_dir": _normalize_model_dir(
            affect_kwargs.get("affect_text_model_dir")
        ),
        "affect_ser_model_dir": _normalize_model_dir(
            affect_kwargs.get("affect_ser_model_dir")
        ),
        "affect_vad_model_dir": _normalize_model_dir(
            affect_kwargs.get("affect_vad_model_dir")
        ),
        "affect_intent_model_dir": _normalize_model_dir(
            affect_kwargs.get("affect_intent_model_dir")
        ),
        "affect_analyzer_threads": affect_kwargs.get("analyzer_threads"),
        "text_emotion_model": affect_kwargs["text_emotion_model"],
        "disable_affect": bool(settings.get("disable_affect", False)),
    }

