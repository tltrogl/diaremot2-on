"""Factory mixins for assembling heavy-weight pipeline components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import speaker_diarization as _speaker_diarization
from ...affect.emotion_analyzer import EmotionIntentAnalyzer
from ...affect.intent_defaults import INTENT_LABELS_DEFAULT
from ...affect.sed_panns import PANNSEventTagger, SEDConfig  # type: ignore
from ..audio_preprocessing import AudioPreprocessor, PreprocessConfig
from ..auto_tuner import AutoTuner
from ..errors import coerce_stage_error
from ..runtime_env import DEFAULT_WHISPER_MODEL, WINDOWS_MODELS_ROOT
from ...summaries.html_summary_generator import HTMLSummaryGenerator
from ...summaries.pdf_summary_generator import PDFSummaryGenerator
from ..transcription_module import AudioTranscriber

# Local imports that are heavy should live inside functions to keep module load
# time minimal.  The mixin therefore only references lightweight shims here and
# defers expensive imports to the call sites.


class ComponentFactoryMixin:
    """Encapsulates component bootstrap logic used by the pipeline executor."""

    def _init_components(self, cfg: dict[str, Any]) -> None:
        self.pre = None
        self.diar = None
        self.tx = None
        self.affect = None
        self.sed_tagger = None
        self.html = None
        self.pdf = None
        self.auto_tuner = None

        affect_kwargs: dict[str, Any] = {
            "text_emotion_model": cfg.get("text_emotion_model", "SamLowe/roberta-base-go_emotions"),
            "intent_labels": cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
            "affect_backend": cfg.get("affect_backend", "onnx"),
            "affect_text_model_dir": cfg.get("affect_text_model_dir"),
            "affect_ser_model_dir": cfg.get("affect_ser_model_dir"),
            "affect_vad_model_dir": cfg.get("affect_vad_model_dir"),
            "affect_intent_model_dir": cfg.get("affect_intent_model_dir"),
            "analyzer_threads": cfg.get("affect_analyzer_threads"),
            "disable_downloads": cfg.get("disable_downloads"),
            "model_dir": cfg.get("affect_model_dir"),
        }

        try:
            denoise_mode = "spectral_sub_soft" if cfg.get("noise_reduction", True) else "none"
            self.pp_conf = PreprocessConfig(
                target_sr=cfg.get("target_sr", 16000),
                denoise=denoise_mode,
                loudness_mode=cfg.get("loudness_mode", "asr"),
                auto_chunk_enabled=cfg.get("auto_chunk_enabled", True),
                chunk_threshold_minutes=cfg.get("chunk_threshold_minutes", 60.0),
                chunk_size_minutes=cfg.get("chunk_size_minutes", 20.0),
                chunk_overlap_seconds=cfg.get("chunk_overlap_seconds", 30.0),
            )
            self.pre = AudioPreprocessor(self.pp_conf)

            registry_path = cfg.get("registry_path", str(Path("registry") / "speaker_registry.json"))
            if not Path(registry_path).is_absolute():
                registry_path = str(Path.cwd() / registry_path)

            ecapa_path = cfg.get("ecapa_model_path")
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
            ecapa_path = resolved_path

            self.diar_conf = _speaker_diarization.DiarizationConfig(
                target_sr=self.pp_conf.target_sr,
                registry_path=registry_path,
                ahc_distance_threshold=cfg.get("ahc_distance_threshold", 0.15),
                speaker_limit=cfg.get("speaker_limit", None),
                clustering_backend=str(cfg.get("clustering_backend", "ahc")),
                min_speakers=cfg.get("min_speakers", None),
                max_speakers=cfg.get("max_speakers", None),
                ecapa_model_path=ecapa_path,
                vad_backend=cfg.get("vad_backend", "auto"),
                vad_threshold=cfg.get("vad_threshold", _speaker_diarization.DiarizationConfig.vad_threshold),
                vad_min_speech_sec=cfg.get("vad_min_speech_sec", _speaker_diarization.DiarizationConfig.vad_min_speech_sec),
                vad_min_silence_sec=cfg.get("vad_min_silence_sec", _speaker_diarization.DiarizationConfig.vad_min_silence_sec),
                speech_pad_sec=cfg.get("vad_speech_pad_sec", _speaker_diarization.DiarizationConfig.speech_pad_sec),
                allow_energy_vad_fallback=not bool(cfg.get("disable_energy_vad_fallback", False)),
                energy_gate_db=cfg.get("energy_gate_db", _speaker_diarization.DiarizationConfig.energy_gate_db),
                energy_hop_sec=cfg.get("energy_hop_sec", _speaker_diarization.DiarizationConfig.energy_hop_sec),
            )

            self.diar = _speaker_diarization.SpeakerDiarizer(self.diar_conf)
            if bool(cfg.get("cpu_diarizer", False)):
                try:
                    from ..cpu_optimized_diarizer import (
                        CPUOptimizationConfig,
                        CPUOptimizedSpeakerDiarizer,
                    )

                    cpu_conf = CPUOptimizationConfig(max_speakers=self.diar_conf.speaker_limit)
                    self.diar = CPUOptimizedSpeakerDiarizer(self.diar, cpu_conf)
                    self.corelog.info("[diarizer] using CPU-optimized wrapper")
                except Exception as exc:  # pragma: no cover - optional component
                    self.corelog.warn(f"[diarizer] CPU wrapper unavailable, using baseline: {exc}")

            transcriber_config = {
                "model_size": str(cfg.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                "language": cfg.get("language", None),
                "beam_size": cfg.get("beam_size", 1),
                "temperature": cfg.get("temperature", 0.0),
                "compression_ratio_threshold": cfg.get("compression_ratio_threshold", 2.5),
                "log_prob_threshold": cfg.get("log_prob_threshold", -1.0),
                "no_speech_threshold": cfg.get("no_speech_threshold", 0.20),
                "condition_on_previous_text": cfg.get("condition_on_previous_text", False),
                "word_timestamps": cfg.get("word_timestamps", True),
                "max_asr_window_sec": cfg.get("max_asr_window_sec", 480),
                "vad_min_silence_ms": cfg.get("vad_min_silence_ms", 1800),
                "language_mode": cfg.get("language_mode", "auto"),
                "compute_type": cfg.get("compute_type", None),
                "cpu_threads": cfg.get("cpu_threads", None),
                "asr_backend": cfg.get("asr_backend", "auto"),
                "local_first": cfg.get("local_first", True),
                "segment_timeout_sec": cfg.get("segment_timeout_sec", 300.0),
                "batch_timeout_sec": cfg.get("batch_timeout_sec", 1200.0),
            }

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["TORCH_DEVICE"] = "cpu"
            self.tx = AudioTranscriber(**transcriber_config)
            self.auto_tuner = AutoTuner()

            def _normalize_model_dir(value: Any) -> str | None:
                if value in (None, ""):
                    return None
                try:
                    return os.fspath(value)
                except TypeError:
                    return str(value)

            affect_backend = cfg.get("affect_backend", "onnx")
            if affect_backend is not None:
                affect_backend = str(affect_backend)
            affect_text_model_dir = _normalize_model_dir(cfg.get("affect_text_model_dir"))
            affect_ser_model_dir = _normalize_model_dir(cfg.get("affect_ser_model_dir"))
            affect_vad_model_dir = _normalize_model_dir(cfg.get("affect_vad_model_dir"))
            affect_intent_model_dir = _normalize_model_dir(cfg.get("affect_intent_model_dir"))
            affect_analyzer_threads = cfg.get("affect_analyzer_threads")

            if cfg.get("disable_affect"):
                self.affect = None
            else:
                self.affect = EmotionIntentAnalyzer(
                    text_emotion_model=cfg.get("text_emotion_model", "SamLowe/roberta-base-go_emotions"),
                    intent_labels=cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
                    affect_backend=affect_backend,
                    affect_text_model_dir=affect_text_model_dir,
                    affect_ser_model_dir=affect_ser_model_dir,
                    affect_vad_model_dir=affect_vad_model_dir,
                    affect_intent_model_dir=affect_intent_model_dir,
                    analyzer_threads=affect_analyzer_threads,
                    disable_downloads=cfg.get("disable_downloads"),
                    model_dir=cfg.get("affect_model_dir"),
                )
                if getattr(self.affect, "issues", None):
                    self.stats.issues.extend(self.affect.issues)

            affect_kwargs.update(
                {
                    "affect_backend": affect_backend,
                    "affect_text_model_dir": affect_text_model_dir,
                    "affect_ser_model_dir": affect_ser_model_dir,
                    "affect_vad_model_dir": affect_vad_model_dir,
                    "affect_intent_model_dir": affect_intent_model_dir,
                    "analyzer_threads": affect_analyzer_threads,
                }
            )

            sed_enabled = bool(cfg.get("enable_sed", True))
            self.sed_tagger = None
            if sed_enabled:
                try:
                    if PANNSEventTagger is not None:
                        self.sed_tagger = PANNSEventTagger(SEDConfig() if SEDConfig else None)
                except Exception as exc:  # pragma: no cover - best effort fallback
                    self.sed_tagger = None
                    self.corelog.warn(
                        "[sed] initialization failed: %s. Background tagging will emit empty results.",
                        exc,
                    )
            else:
                self.corelog.info(
                    "[sed] background sound event detection disabled; tagger will not be initialised."
                )
            if sed_enabled and (
                self.sed_tagger is None or not getattr(self.sed_tagger, "available", False)
            ):
                self.stats.issues.append("background_sed assets unavailable; emitting empty tag summary")

            self.html = HTMLSummaryGenerator()
            self.pdf = PDFSummaryGenerator()

            self.stats.models.update(
                {
                    "preprocessor": getattr(self.pre, "__class__", type(self.pre)).__name__,
                    "diarizer": getattr(self.diar, "__class__", type(self.diar)).__name__,
                    "transcriber": getattr(self.tx, "__class__", type(self.tx)).__name__,
                    "affect": getattr(self.affect, "__class__", type(self.affect)).__name__,
                }
            )

            self.stats.config_snapshot = {
                "target_sr": self.pp_conf.target_sr,
                "noise_reduction": cfg.get("noise_reduction", True),
                "enable_sed": sed_enabled,
                "registry_path": self.diar_conf.registry_path,
                "ahc_distance_threshold": self.diar_conf.ahc_distance_threshold,
                "whisper_model": str(cfg.get("whisper_model", DEFAULT_WHISPER_MODEL)),
                "beam_size": cfg.get("beam_size", 1),
                "temperature": cfg.get("temperature", 0.0),
                "no_speech_threshold": cfg.get("no_speech_threshold", 0.20),
                "intent_labels": cfg.get("intent_labels", INTENT_LABELS_DEFAULT),
                "affect_backend": affect_backend,
                "affect_text_model_dir": affect_text_model_dir,
                "affect_ser_model_dir": affect_ser_model_dir,
                "affect_vad_model_dir": affect_vad_model_dir,
                "affect_intent_model_dir": affect_intent_model_dir,
                "affect_analyzer_threads": affect_analyzer_threads,
                "text_emotion_model": cfg.get("text_emotion_model", "SamLowe/roberta-base-go_emotions"),
                "disable_affect": bool(cfg.get("disable_affect", False)),
            }

        except Exception as exc:  # pragma: no cover - defensive fallback
            self.corelog.error(f"Component initialization error: {exc}")
            try:
                if getattr(self, "pre", None) is None:
                    self.pre = AudioPreprocessor(PreprocessConfig())
            except Exception:
                self.pre = None
            try:
                if getattr(self, "diar", None) is None:
                    self.diar = _speaker_diarization.SpeakerDiarizer(
                        _speaker_diarization.DiarizationConfig(target_sr=16000)
                    )
            except Exception:
                self.diar = None
            try:
                if getattr(self, "tx", None) is None:
                    self.tx = AudioTranscriber()
            except Exception:
                self.tx = None
            try:
                if getattr(self, "affect", None) is None and not cfg.get("disable_affect"):
                    self.affect = EmotionIntentAnalyzer(**affect_kwargs)
            except Exception:
                self.affect = None
            try:
                if getattr(self, "pdf", None) is None:
                    self.pdf = PDFSummaryGenerator()
            except Exception:
                pass
            try:
                if getattr(self, "auto_tuner", None) is None:
                    self.auto_tuner = AutoTuner()
            except Exception:
                self.auto_tuner = None
            raise coerce_stage_error(
                "dependency_check",
                "Component initialisation failed",
                context={"config": cfg},
                cause=exc,
            )
