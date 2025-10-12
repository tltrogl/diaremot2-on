# transcription_module.py
# Optimized, async-enabled transcription module with modern Python patterns
# Designed for high-throughput pipeline integration

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# Suppress ML library warnings early
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Force CPU-only before imports
os.environ.update(
    {
        "CUDA_VISIBLE_DEVICES": "",
        "TORCH_DEVICE": "cpu",
        "FORCE_CPU": "1",
        "OMP_NUM_THREADS": "1",  # Prevent oversubscription
        "TOKENIZERS_PARALLELISM": "false",  # Prevent tokenizer warnings
    }
)

# Encourage stable CPU-only behavior on Windows/CPU-only setups
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Prefer generic ISA on CPUs without AVX/AVX2; users can override externally
os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")


# Backend availability checks
class BackendAvailability:
    """Singleton to track available backends"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._check_backends()
        return cls._instance

    def _check_backends(self):
        import importlib.util as _util

        # Librosa
        try:
            import librosa

            self.librosa = librosa
            self.has_librosa = True
        except Exception:
            self.has_librosa = False
            self.librosa = None

        # Avoid importing faster-whisper/ctranslate2/torch eagerly
        self.has_faster_whisper = False
        self.WhisperModel = None  # type: ignore[attr-defined]
        if _util.find_spec("faster_whisper") is not None:
            try:
                from faster_whisper import WhisperModel as _FWWhisperModel  # type: ignore

                self.has_faster_whisper = True
                self.WhisperModel = _FWWhisperModel
            except Exception:
                self.has_faster_whisper = False
                self.WhisperModel = None

        # Avoid importing openai-whisper/torch eagerly
        self.has_openai_whisper = _util.find_spec("whisper") is not None
        self.openai_whisper = None


backends = BackendAvailability()


# Optimized audio resampling
@lru_cache(maxsize=16)
def _get_resampling_kernel(orig_sr: int, target_sr: int, length: int) -> np.ndarray:
    """Cached resampling kernel generation for common SR pairs"""
    if backends.has_librosa:
        return backends.librosa.resample(
            np.zeros(min(1024, length), dtype=np.float32),
            orig_sr=orig_sr,
            target_sr=target_sr,
        )
    return np.array([])


def resample_audio_fast(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Optimized audio resampling with caching and fast paths"""
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    # Fast paths for common ratios
    ratio = target_sr / orig_sr
    if ratio == 0.5 and orig_sr == 32000:  # 32k -> 16k
        return audio[::2].astype(np.float32)
    elif ratio == 2.0 and orig_sr == 8000:  # 8k -> 16k
        return np.repeat(audio, 2).astype(np.float32)

    # Use librosa if available, fallback otherwise
    if backends.has_librosa:
        return backends.librosa.resample(
            audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
        )

    # Simple linear interpolation fallback
    new_length = int(len(audio) * ratio)
    old_indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(old_indices, np.arange(len(audio)), audio).astype(np.float32)


@dataclass
class TranscriptionSegment:
    """Enhanced segment with validation and serialization"""

    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker_id: str | None = None
    speaker_name: str | None = None
    words: list[dict[str, Any]] | None = None
    language: str | None = None
    language_probability: float | None = None
    processing_time: float | None = None
    model_used: str | None = None
    asr_logprob_avg: float | None = None
    snr_db: float | None = None

    def __post_init__(self):
        """Validate and normalize fields"""
        self.start_time = max(0.0, float(self.start_time))
        self.end_time = max(self.start_time, float(self.end_time))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.text = str(self.text).strip()

        # Normalize speaker fields
        if self.speaker_id and not self.speaker_name:
            self.speaker_name = self.speaker_id

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "words": self.words,
            "language": self.language,
            "language_probability": self.language_probability,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "asr_logprob_avg": self.asr_logprob_avg,
            "snr_db": self.snr_db,
            "duration": self.duration,
        }


@dataclass
class BatchingConfig:
    """Configuration for intelligent batching"""

    enabled: bool = True
    min_segments_threshold: int = 12  # Reduced from 24
    short_segment_max_sec: float = 8.0  # Reduced from 10.0
    batch_silence_sec: float = 0.3  # Reduced silence padding
    min_segment_sec: float = 0.05  # Reduced minimum
    max_batch_duration_sec: float = 300  # 5-minute batches max

    # Batch size optimization
    target_batch_size_sec: float = 60.0  # Target 1-minute batches
    max_segments_per_batch: int = 50


class ModelManager:
    """Centralized model management with lazy loading and memory optimization"""

    def __init__(self):
        self._models: dict[str, Any] = {}
        self._model_configs: dict[str, dict[str, Any]] = {}
        self._load_locks: dict[str, asyncio.Lock] = {}
        self.logger = logging.getLogger(__name__ + ".ModelManager")
        # Collect load errors for diagnostics
        self.last_errors: dict[str, str] = {}

    @asynccontextmanager
    async def get_model(self, model_key: str, config: dict[str, Any]):
        """Async context manager for model access with automatic cleanup"""
        lock = self._load_locks.get(model_key)
        if lock is None:
            lock = asyncio.Lock()
            self._load_locks[model_key] = lock

        async with lock:
            if self._should_reload_model(model_key, config):
                await self._load_model(model_key, config)

        try:
            yield self._models[model_key]
        finally:
            # Model stays loaded for reuse within session
            pass

    def _should_reload_model(self, model_key: str, config: dict[str, Any]) -> bool:
        existing_model = self._models.get(model_key)
        if existing_model is None:
            return True

        previous_config = self._model_configs.get(model_key)
        if previous_config is None:
            return True

        critical_keys = (
            "model_size",
            "compute_type",
            "asr_backend",
            "cpu_threads",
        )

        for key in critical_keys:
            if previous_config.get(key) != config.get(key):
                return True

        return False

    async def _load_model(self, model_key: str, config: dict[str, Any]):
        """Load model in thread pool to avoid blocking"""

        def _load():
            pref = str(config.get("asr_backend", "auto")).lower()

            last_error: Exception | None = None

            def _try_faster() -> Any | None:
                nonlocal last_error
                if not backends.has_faster_whisper:
                    return None
                try:
                    return self._load_faster_whisper(config)
                except Exception as exc:  # pragma: no cover - runtime fallback guard
                    last_error = exc
                    self.logger.warning("Faster-Whisper load failed: %s", exc)
                    return None

            def _try_openai() -> Any | None:
                nonlocal last_error
                if not backends.has_openai_whisper:
                    return None
                try:
                    return self._load_openai_whisper(config)
                except Exception as exc:  # pragma: no cover - runtime fallback guard
                    last_error = exc
                    self.logger.warning("OpenAI Whisper load failed: %s", exc)
                    return None

            if pref == "openai":
                model = _try_openai() or _try_faster()
            elif pref == "faster":
                model = _try_faster() or _try_openai()
            else:  # auto
                model = _try_faster() or _try_openai()

            if model is None:
                if last_error is not None:
                    raise RuntimeError("No transcription backend available") from last_error
                raise RuntimeError("No transcription backend available")
            return model

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader") as executor:
            model = await loop.run_in_executor(executor, _load)
            self._models[model_key] = model
            self._model_configs[model_key] = config.copy()

    def _load_faster_whisper(self, config: dict[str, Any]) -> Any:
        """Load faster-whisper model with optimized settings"""
        try:
            # Ensure the class symbol is available for fallback attempts
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            self.last_errors["faster_whisper_import"] = str(e)
            raise
        compute_type = str(config.get("compute_type", "float32")).lower()
        if compute_type not in ("float32", "int8", "int8_float16", "float16"):
            compute_type = "float32"

        model_size = config["model_size"]

        # Check if it's a local path first
        _local_only = False
        try:
            if isinstance(model_size, str) and Path(str(model_size)).exists():
                _local_only = True
                self.logger.info(f"Using local model path: {model_size}")
        except Exception:
            _local_only = False

        # Handle HuggingFace model paths for faster-whisper
        if not _local_only and "/" in model_size and "faster-whisper" in model_size:
            # This is a HuggingFace model path like "deepdml/faster-whisper-large-v3-turbo-ct2"
            _local_only = False

        model_kwargs = {
            "device": "cpu",
            "compute_type": compute_type,
            "cpu_threads": config.get("cpu_threads", 1),
            "download_root": None,
            "local_files_only": _local_only,
        }

        try:
            model = WhisperModel(model_size, **model_kwargs)
            self.logger.info(f"Loaded faster-whisper: {model_size}")
            return model
        except Exception as e:
            self.last_errors["faster_whisper_load"] = str(e)
            self.logger.warning(f"Failed to load {model_size}, trying fallback models: {e}")

            # Try fallback models in order of preference
            fallback_models = [
                "large-v3",
                "large-v2",
                "medium",
                "small",
                "base",
                "tiny",
            ]
            for fallback in fallback_models:
                try:
                    self.logger.info(f"Trying fallback model: {fallback}")
                    model = WhisperModel(fallback, **model_kwargs)
                    self.logger.info(f"Successfully loaded fallback faster-whisper: {fallback}")
                    return model
                except Exception as fallback_e:
                    self.last_errors[f"faster_whisper_load:{fallback}"] = str(fallback_e)
                    self.logger.warning(f"Fallback {fallback} failed: {fallback_e}")
                    continue

            # If all faster-whisper models fail, raise the original error
            raise e

    def _load_openai_whisper(self, config: dict[str, Any]) -> Any:
        """Load OpenAI whisper with CPU optimization"""
        model_name = self._map_to_openai_model(config["model_size"])
        try:
            import whisper as openai_whisper  # lazy import; may require torch

            model = openai_whisper.load_model(model_name, device="cpu")
            self.logger.info(f"Loaded OpenAI whisper: {model_name}")
            return model
        except Exception as e:
            self.last_errors["openai_whisper_load"] = str(e)
            self.last_errors["openai_whisper_load"] = str(e)
            self.logger.warning(
                f"OpenAI whisper load failed for '{model_name}': {e}; trying 'tiny'"
            )
            # Try explicit tiny as a last resort
            import whisper as openai_whisper

            tiny = openai_whisper.load_model("tiny", device="cpu")
            self.logger.info("Loaded OpenAI whisper fallback: tiny")
            return tiny

    @staticmethod
    def _map_to_openai_model(model_size: str) -> str:
        """Map model names to OpenAI whisper equivalents"""
        # Handle HuggingFace faster-whisper model paths
        if "/" in model_size and "faster-whisper" in model_size:
            # Extract the base model name from HuggingFace path
            if "turbo" in model_size.lower():
                return "large-v3"
            elif "large-v3" in model_size.lower():
                return "large-v3"
            elif "large-v2" in model_size.lower():
                return "large-v2"
            elif "medium" in model_size.lower():
                return "medium"
            elif "small" in model_size.lower():
                return "small"
            elif "base" in model_size.lower():
                return "base"
            elif "tiny" in model_size.lower():
                return "tiny"

        name_map = {
            "turbo": "large-v3",
            "large-v3": "large-v3",
            "large-v2": "large-v2",
            "large": "large",
            "medium": "medium",
            "small": "small",
            "base": "base",
            "tiny": "tiny",
        }

        for key, value in name_map.items():
            if key in model_size.lower():
                return value
        return "large-v3"  # Default


class AsyncTranscriber:
    """High-performance async transcription engine"""

    def __init__(
        self,
        model_size: str = "large-v3",  # Use standard model instead of custom
        language: str | None = None,
        beam_size: int = 1,
        temperature: float | list[float] = 0.0,
        compression_ratio_threshold: float = 2.4,
        log_prob_threshold: float = -1.0,
        no_speech_threshold: float = 0.20,  # Balanced silence filtering
        condition_on_previous_text: bool = False,
        word_timestamps: bool = True,
        max_asr_window_sec: int = 480,  # 8-minute windows
        vad_min_silence_ms: int = 1200,  # Reduced for better segmentation
        language_mode: str = "auto",
        batching_config: BatchingConfig | None = None,
        max_workers: int = 2,  # Conservative threading
        segment_timeout_sec: float = 120.0,
        batch_timeout_sec: float = 600.0,
        max_concurrent_segments: int = 2,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
        asr_backend: str | None = None,
        model_concurrency: int | None = None,
    ):
        cpu_threads_value = 1
        if cpu_threads is not None:
            try:
                cpu_threads_value = max(1, int(cpu_threads))
            except Exception:
                cpu_threads_value = 1

        self.config = {
            "model_size": model_size,
            "language": language,
            "beam_size": beam_size,
            "temperature": temperature,
            "compression_ratio_threshold": compression_ratio_threshold,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "condition_on_previous_text": condition_on_previous_text,
            "word_timestamps": word_timestamps,
            "max_asr_window_sec": max_asr_window_sec,
            "vad_min_silence_ms": vad_min_silence_ms,
            "language_mode": language_mode,
            "cpu_threads": cpu_threads_value,  # Single thread per model to avoid conflicts
            "segment_timeout_sec": float(segment_timeout_sec),
            "batch_timeout_sec": float(batch_timeout_sec),
            "max_concurrent_segments": int(max_concurrent_segments),
        }
        if compute_type is not None:
            try:
                self.config["compute_type"] = str(compute_type)
            except Exception:
                pass
        if asr_backend is not None:
            try:
                self.config["asr_backend"] = str(asr_backend)
            except Exception:
                pass
        try:
            model_concurrency_value = int(model_concurrency) if model_concurrency is not None else 1
        except Exception:
            model_concurrency_value = 1
        if model_concurrency_value < 1:
            model_concurrency_value = 1
        self.config["model_concurrency"] = model_concurrency_value
        # Fallback diagnostics
        self._fallback_triggered: bool = False
        self._fallback_reason: str | None = None
        self._fallback_to: str | None = None
        self._last_failures: dict[str, str] = {}

        self.batching = batching_config or BatchingConfig()
        self.model_manager = ModelManager()
        self._max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="transcriber"
        )
        self._executor_lock = threading.Lock()
        self._model_concurrency = model_concurrency_value
        self._model_semaphore = asyncio.Semaphore(self._model_concurrency)

        self.logger = logging.getLogger(__name__ + ".AsyncTranscriber")
        self._stats = {"segments_processed": 0, "batches_processed": 0}

    def _reset_executor(self) -> None:
        """Recreate the worker pool when tasks get stuck."""

        with self._executor_lock:
            old_executor = self.executor
            try:
                old_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                self.logger.warning("Failed to cleanly shutdown executor after timeout: %s", exc)
            self.executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="transcriber"
            )

    async def transcribe_segments(
        self, audio_16k_mono: np.ndarray, sr: int, diar_segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Main async transcription with intelligent batching"""

        if sr != 16000:
            audio_16k_mono = resample_audio_fast(audio_16k_mono, sr, 16000)
            sr = 16000

        # Ensure mono
        if audio_16k_mono.ndim > 1:
            audio_16k_mono = audio_16k_mono.mean(axis=1)

        audio_16k_mono = audio_16k_mono.astype(np.float32)

        # Preprocessing and validation
        valid_segments = self._preprocess_segments(diar_segments, len(audio_16k_mono) / sr)

        if not valid_segments:
            self.logger.warning("No valid segments after preprocessing")
            return []

        # Language pinning optimization
        if self.config["language_mode"] == "pin" and not self.config["language"]:
            await self._pin_language(audio_16k_mono, sr, valid_segments)

        # Batching decision with enhanced logic
        batch_groups = self._create_batch_groups(valid_segments)

        # Process all batches concurrently
        all_results = []
        tasks = []

        for batch_type, segments in batch_groups.items():
            if not segments:
                continue

            if batch_type == "batch":
                task = self._process_batch_concurrent(audio_16k_mono, sr, segments)
            else:  # individual
                task = self._process_individual_concurrent(audio_16k_mono, sr, segments)

            tasks.append(task)

        # Await all processing tasks
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results and handle exceptions
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing failed: {result}")
                continue
            all_results.extend(result)

        # Sort by start time and return
        all_results.sort(key=lambda x: x.start_time)
        self._stats["segments_processed"] += len(all_results)

        return all_results

    def _preprocess_segments(
        self, segments: list[dict[str, Any]], audio_duration: float
    ) -> list[dict[str, Any]]:
        """Enhanced segment preprocessing with overlap resolution"""
        if not segments:
            return []

        # Normalize segment format
        normalized = []
        for seg in segments:
            start = float(seg.get("start_time", seg.get("start", 0.0)))
            end = float(seg.get("end_time", seg.get("end", start + 1.0)))

            # Validate timing
            start = max(0.0, start)
            end = min(audio_duration, max(start + self.batching.min_segment_sec, end))

            if end - start < self.batching.min_segment_sec:
                continue

            normalized.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "speaker_id": seg.get(
                        "speaker_id",
                        seg.get("speaker", f"Speaker_{len(normalized) + 1}"),
                    ),
                    "speaker_name": seg.get(
                        "speaker_name",
                        seg.get("speaker", f"Speaker_{len(normalized) + 1}"),
                    ),
                }
            )

        # Sort and resolve overlaps
        normalized.sort(key=lambda x: x["start_time"])
        resolved = []

        for seg in normalized:
            if resolved:
                prev = resolved[-1]
                if seg["start_time"] < prev["end_time"]:
                    # Resolve overlap by splitting at midpoint
                    midpoint = (prev["end_time"] + seg["start_time"]) / 2
                    prev["end_time"] = midpoint
                    seg["start_time"] = midpoint

            if seg["end_time"] - seg["start_time"] >= self.batching.min_segment_sec:
                resolved.append(seg)

        return resolved

    async def _pin_language(self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]):
        """Pin language using longest segment for consistency"""
        if not segments:
            return

        # Find longest segment
        longest = max(segments, key=lambda s: s["end_time"] - s["start_time"])

        # Transcribe just this segment for language detection
        pinning_result = await self._transcribe_single_segment(
            audio, sr, longest, force_detection=True
        )

        if (
            pinning_result
            and pinning_result.language
            and (pinning_result.language_probability or 0.0) >= 0.75
        ):
            self.config["language"] = pinning_result.language
            self.logger.info(f"Language pinned to: {pinning_result.language}")

    def _create_batch_groups(
        self, segments: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Create optimized batching groups based on segment characteristics"""
        if not self.batching.enabled or len(segments) < self.batching.min_segments_threshold:
            return {"individual": segments}

        short_segments = []
        long_segments = []

        for seg in segments:
            duration = seg["end_time"] - seg["start_time"]
            if duration <= self.batching.short_segment_max_sec:
                short_segments.append(seg)
            else:
                long_segments.append(seg)

        # Create efficient batches from short segments
        batched_groups = []
        if short_segments:
            current_batch = []
            current_duration = 0.0

            for seg in short_segments:
                seg_duration = seg["end_time"] - seg["start_time"]

                # Check batch limits
                if current_batch and (
                    current_duration + seg_duration > self.batching.max_batch_duration_sec
                    or len(current_batch) >= self.batching.max_segments_per_batch
                ):
                    batched_groups.append(current_batch)
                    current_batch = []
                    current_duration = 0.0

                current_batch.append(seg)
                current_duration += seg_duration

            if current_batch:
                batched_groups.append(current_batch)

        # Return grouped segments
        groups = {"individual": long_segments}
        if batched_groups:
            for i, batch in enumerate(batched_groups):
                groups[f"batch_{i}"] = batch

        return groups

    async def _process_batch_concurrent(
        self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Process batch of segments concurrently with optimized audio concatenation"""
        if not segments:
            return []

        # Build concatenated audio with minimal silence padding
        silence_samples = int(self.batching.batch_silence_sec * sr)
        spacer = np.zeros(silence_samples, dtype=np.float32)

        concat_parts = []
        segment_boundaries = []
        current_time = 0.0

        for seg in segments:
            start_idx = int(seg["start_time"] * sr)
            end_idx = int(seg["end_time"] * sr)
            audio_segment = audio[start_idx:end_idx]

            if len(audio_segment) == 0:
                continue

            segment_boundaries.append(
                {
                    "original": seg,
                    "concat_start": current_time,
                    "concat_end": current_time + len(audio_segment) / sr,
                }
            )

            concat_parts.extend([audio_segment, spacer])
            current_time += (len(audio_segment) + len(spacer)) / sr

        if not concat_parts:
            return []

        # Transcribe concatenated audio
        concat_audio = np.concatenate(concat_parts)

        async with self.model_manager.get_model("primary", self.config) as model:
            try:
                batch_result = await asyncio.wait_for(
                    self._run_transcription(model, concat_audio, sr),
                    timeout=self.config.get("batch_timeout_sec", 600.0),
                )
            except TimeoutError:
                self._last_failures["batch_timeout"] = (
                    f"Batch transcription exceeded {self.config.get('batch_timeout_sec', 600.0)}s"
                )
                self.logger.error(self._last_failures["batch_timeout"])
                return await self._process_individual_concurrent(audio, sr, segments)

        if not batch_result:
            # Fallback to individual processing
            return await self._process_individual_concurrent(audio, sr, segments)

        # Distribute words back to original segments
        results = self._distribute_batch_results(batch_result, segment_boundaries)
        self._stats["batches_processed"] += 1

        return results

    async def _process_individual_concurrent(
        self, audio: np.ndarray, sr: int, segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Process segments individually with controlled concurrency"""
        requested = max(1, int(self.config.get("max_concurrent_segments", 2)))
        concurrency = min(requested, self._model_concurrency)
        semaphore = asyncio.Semaphore(concurrency)

        async def process_one(seg):
            async with semaphore:
                try:
                    return await self._transcribe_single_segment(audio, sr, seg)
                except TimeoutError:
                    self.logger.warning(
                        f"Segment {seg.get('start_time', '?')}-{seg.get('end_time', '?')}s timed out after {self.config.get('segment_timeout_sec', 120.0)}s"
                    )
                    return None

        tasks = [process_one(seg) for seg in segments]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and None results
        valid_results = []
        for result in results:
            if isinstance(result, TranscriptionSegment):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.warning(f"Segment transcription failed: {result}")

        return valid_results

    async def _transcribe_single_segment(
        self,
        audio: np.ndarray,
        sr: int,
        segment: dict[str, Any],
        force_detection: bool = False,
    ) -> TranscriptionSegment | None:
        """Transcribe single segment with chunking for long audio"""

        start_time = segment["start_time"]
        end_time = segment["end_time"]
        duration = end_time - start_time

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)
        audio_segment = audio[start_idx:end_idx]

        if len(audio_segment) == 0:
            return None

        # Direct transcription for short segments
        if duration <= self.config["max_asr_window_sec"]:
            async with self.model_manager.get_model("primary", self.config) as model:
                result = await self._run_transcription(model, audio_segment, sr, force_detection)

            if result:
                result.start_time = start_time
                result.end_time = end_time
                result.speaker_id = segment.get("speaker_id")
                result.speaker_name = segment.get("speaker_name")

                # Adjust word timestamps
                if result.words:
                    for word in result.words:
                        word["start"] += start_time
                        word["end"] += start_time

            else:
                self.logger.debug(
                    "Segment transcription returned none (start=%.2fs, end=%.2fs, duration=%.2fs)",
                    start_time,
                    end_time,
                    duration,
                )
            return result

        # Chunked processing for long segments
        return await self._transcribe_long_segment(audio_segment, sr, segment)

    async def _transcribe_long_segment(
        self, audio_segment: np.ndarray, sr: int, segment: dict[str, Any]
    ) -> TranscriptionSegment | None:
        """Handle long segments with overlapping chunks for continuity"""
        duration = len(audio_segment) / sr
        chunk_size = self.config["max_asr_window_sec"]
        overlap_sec = 2.0  # 2-second overlap for context

        chunks = []
        cursor = 0.0

        while cursor < duration:
            chunk_start = max(0.0, cursor - (overlap_sec if chunks else 0.0))
            chunk_end = min(duration, cursor + chunk_size)

            start_idx = int(chunk_start * sr)
            end_idx = int(chunk_end * sr)
            chunk_audio = audio_segment[start_idx:end_idx]

            async with self.model_manager.get_model("primary", self.config) as model:
                chunk_result = await self._run_transcription(model, chunk_audio, sr)

            if chunk_result and chunk_result.text.strip():
                # Adjust timing to original segment coordinates
                chunk_result.start_time = segment["start_time"] + chunk_start
                chunk_result.end_time = segment["start_time"] + chunk_end
                chunks.append(chunk_result)

            cursor += chunk_size - overlap_sec

        # Merge chunks
        if not chunks:
            self.logger.debug(
                "Chunked transcription produced no chunks (start=%.2fs, end=%.2fs, duration=%.2fs)",
                segment["start_time"],
                segment["end_time"],
                segment["end_time"] - segment["start_time"],
            )
            return None

        # Simple concatenation for now - could be enhanced with smart merging
        merged_text = " ".join(chunk.text.strip() for chunk in chunks if chunk.text.strip())
        merged_words = []
        merged_logprobs = []

        for chunk in chunks:
            if chunk.words:
                merged_words.extend(chunk.words)
            if chunk.asr_logprob_avg is not None:
                merged_logprobs.append(chunk.asr_logprob_avg)

        return TranscriptionSegment(
            start_time=segment["start_time"],
            end_time=segment["end_time"],
            text=merged_text,
            confidence=(float(np.exp(np.mean(merged_logprobs))) if merged_logprobs else 1.0),
            speaker_id=segment.get("speaker_id"),
            speaker_name=segment.get("speaker_name"),
            words=merged_words if merged_words else None,
            language=chunks[0].language if chunks else None,
            language_probability=chunks[0].language_probability if chunks else None,
            asr_logprob_avg=(float(np.mean(merged_logprobs)) if merged_logprobs else None),
            model_used=chunks[0].model_used if chunks else "chunked",
        )

    async def _run_transcription(
        self, model: Any, audio: np.ndarray, sr: int, force_detection: bool = False
    ) -> TranscriptionSegment | None:
        """Run actual transcription in thread pool"""

        def _transcribe():
            # Prefer faster-whisper when model is a WhisperModel instance
            try:
                if (
                    backends.has_faster_whisper
                    and getattr(backends, "WhisperModel", None) is not None
                    and isinstance(model, backends.WhisperModel)
                ):
                    return self._faster_whisper_transcribe(model, audio, sr, force_detection)

                if backends.has_openai_whisper:
                    return self._openai_whisper_transcribe(model, audio, sr)

                return None
            except Exception as e:
                # Capture common cause signatures (e.g., sox_io backend errors)
                msg = str(e)
                self._last_failures["primary_exception"] = msg
                if "sox_io" in msg.lower():
                    self._fallback_reason = "torchaudio sox_io backend unavailable"
                else:
                    self._fallback_reason = f"primary backend exception: {msg[:180]}"
                self._fallback_triggered = True
                return None

        loop = asyncio.get_event_loop()
        t0 = time.time()
        timeout = float(self.config.get("segment_timeout_sec", 120.0))
        async with self._model_semaphore:
            future = loop.run_in_executor(self.executor, _transcribe)
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
            except TimeoutError:
                self._last_failures["segment_timeout"] = f"Transcription exceeded {timeout}s"
                self._fallback_triggered = True
                if not self._fallback_reason:
                    self._fallback_reason = self._last_failures["segment_timeout"]
                self.logger.warning(
                    "Transcription worker timed out after %.1fs; resetting executor",
                    timeout,
                )
                if not future.done():
                    future.cancel()
                self._reset_executor()
                return None
        if isinstance(result, TranscriptionSegment):
            try:
                result.processing_time = float(time.time() - t0)
            except Exception:
                pass
        return result

    def _faster_whisper_transcribe(
        self, model: Any, audio: np.ndarray, sr: int, force_detection: bool = False
    ) -> TranscriptionSegment | None:
        """Optimized faster-whisper transcription"""

        language = None if force_detection else self.config["language"]

        # Heuristic: faster-whisper VAD can fail on very short or low-energy audio
        # causing internal "max() arg is an empty sequence". Prefer to disable
        # VAD for short clips, and retry without VAD on VAD-related failures.
        duration = float(len(audio) / sr) if sr else 0.0
        vad_min_sil_ms = int(self.config.get("vad_min_silence_ms", 1800))
        is_short = duration < max(3.0, vad_min_sil_ms / 1000.0)
        is_quiet = False
        try:
            # Cheap loudness proxy
            is_quiet = float(np.max(np.abs(audio))) < 1e-4
        except Exception:
            pass

        def _call_transcribe(use_vad: bool, threshold_value: float):
            kw = dict(
                language=language,
                beam_size=self.config["beam_size"],
                temperature=self.config["temperature"],
                compression_ratio_threshold=self.config["compression_ratio_threshold"],
                log_prob_threshold=self.config["log_prob_threshold"],
                no_speech_threshold=threshold_value,
                condition_on_previous_text=self.config["condition_on_previous_text"],
                word_timestamps=self.config["word_timestamps"],
            )
            if use_vad:
                kw.update(
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": self.config["vad_min_silence_ms"],
                        "min_speech_duration_ms": 250,
                        "speech_pad_ms": 300,
                    },
                )
            else:
                kw.update(vad_filter=False)

            return model.transcribe(audio, **kw)

        # First attempt: honor VAD unless audio is short/quiet
        prefer_vad = not (is_short or is_quiet)
        base_threshold = float(self.config["no_speech_threshold"])

        attempts: list[tuple[bool, float]] = []
        seen: set[tuple[bool, float]] = set()

        def _add_attempt(use_vad: bool, threshold_value: float) -> None:
            key = (use_vad, round(threshold_value, 4))
            if key not in seen:
                attempts.append((use_vad, threshold_value))
                seen.add(key)

        _add_attempt(prefer_vad, base_threshold)
        if prefer_vad:
            _add_attempt(False, base_threshold)

        if base_threshold > 0.05:
            lowered = max(0.0, min(0.15, base_threshold - 0.1))
            _add_attempt(False, lowered)
            if prefer_vad:
                _add_attempt(True, lowered)

        if base_threshold > 0.0:
            _add_attempt(False, 0.0)
            if prefer_vad:
                _add_attempt(True, 0.0)

        segments = ()
        info = None
        text_parts: list[str] = []
        words: list[dict[str, Any]] = []
        logprobs: list[float] = []
        last_error: Exception | None = None

        def _collect(_segments):
            tparts: list[str] = []
            wlist: list[dict[str, Any]] = []
            lps: list[float] = []
            for seg in _segments:
                text = getattr(seg, "text", "").strip()
                if text:
                    tparts.append(text)
                    if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                        lps.append(seg.avg_logprob)
                    if self.config["word_timestamps"] and hasattr(seg, "words") and seg.words:
                        for word in seg.words:
                            wlist.append(
                                {
                                    "word": getattr(word, "word", ""),
                                    "start": getattr(word, "start", 0.0),
                                    "end": getattr(word, "end", 0.0),
                                    "probability": getattr(word, "probability", 1.0),
                                }
                            )
            return tparts, wlist, lps

        for idx, (use_vad, threshold_value) in enumerate(attempts, start=1):
            self.logger.debug(
                "ASR attempt %s/%s (vad_filter=%s, no_speech_threshold=%.2f, duration=%.2fs)",
                idx,
                len(attempts),
                use_vad,
                threshold_value,
                duration,
            )
            try:
                segments, info = _call_transcribe(use_vad, threshold_value)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if use_vad and ("empty sequence" in msg or "vad" in msg or "max()" in msg):
                    self.logger.info(
                        "Retrying faster-whisper without VAD due to VAD error (threshold=%.2f)",
                        threshold_value,
                    )
                else:
                    self.logger.warning(
                        "faster-whisper attempt failed (vad=%s threshold=%.2f): %s",
                        use_vad,
                        threshold_value,
                        e,
                    )
                continue

            text_parts, words, logprobs = _collect(segments)
            if text_parts:
                if (use_vad != prefer_vad) or (abs(threshold_value - base_threshold) > 1e-6):
                    self.logger.debug(
                        "faster-whisper succeeded with vad_filter=%s no_speech_threshold=%.2f",
                        use_vad,
                          threshold_value,
                      )
                last_error = None
                break
            else:
                self.logger.debug(
                    "faster-whisper returned empty transcript (vad_filter=%s, threshold=%.2f)",
                    use_vad,
                    threshold_value,
                )
                continue
        else:
            if last_error is not None:
                self.logger.error(f"faster-whisper transcription failed: {last_error}")
            return None
        if not text_parts:
            self.logger.debug(
                "faster-whisper produced no text after %s attempts (duration=%.2fs)",
                len(attempts),
                duration,
            )
            return None

        return TranscriptionSegment(
            start_time=0.0,
            end_time=len(audio) / sr,
            text=" ".join(text_parts),
            confidence=float(np.exp(np.mean(logprobs))) if logprobs else 1.0,
            words=words if words else None,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            asr_logprob_avg=float(np.mean(logprobs)) if logprobs else None,
            model_used="faster-whisper",
        )

    def _openai_whisper_transcribe(
        self, model: Any, audio: np.ndarray, sr: int
    ) -> TranscriptionSegment | None:
        """OpenAI whisper transcription with optimization"""
        try:
            result = model.transcribe(
                audio.astype(np.float32),
                language=self.config["language"],
                word_timestamps=self.config["word_timestamps"],
                temperature=(
                    self.config["temperature"]
                    if isinstance(self.config["temperature"], (int, float))
                    else 0.0
                ),
                fp16=False,
                verbose=False,
            )

            text = result.get("text", "").strip()
            if not text:
                return None

            words = []
            if self.config["word_timestamps"] and "segments" in result:
                for segment in result["segments"]:
                    if "words" in segment:
                        for word in segment["words"]:
                            words.append(
                                {
                                    "word": word.get("word", ""),
                                    "start": float(word.get("start", 0.0)),
                                    "end": float(word.get("end", 0.0)),
                                    "probability": float(word.get("probability", 1.0)),
                                }
                            )

            return TranscriptionSegment(
                start_time=0.0,
                end_time=len(audio) / sr,
                text=text,
                confidence=1.0,
                words=words if words else None,
                language=result.get("language"),
                language_probability=None,
                model_used="openai-whisper",
            )

        except Exception as e:
            self.logger.error(f"OpenAI whisper transcription failed: {e}")
            return None

    def _distribute_batch_results(
        self, batch_result: TranscriptionSegment, boundaries: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Distribute batched transcription results back to original segments"""

        if not batch_result.words:
            # Fallback: split text proportionally
            return self._distribute_text_proportionally(batch_result, boundaries)

        results = []

        for boundary in boundaries:
            seg = boundary["original"]
            concat_start = boundary["concat_start"]
            concat_end = boundary["concat_end"]

            # Find words within this segment's time range
            segment_words = []
            segment_text_parts = []

            for word in batch_result.words:
                word_start = word.get("start", 0.0)
                word_end = word.get("end", word_start)

                # Check if word overlaps with segment
                if word_start < concat_end and word_end > concat_start:
                    # Adjust word timing to original coordinates
                    adjusted_word = word.copy()
                    time_offset = seg["start_time"] - concat_start
                    adjusted_word["start"] = word_start + time_offset
                    adjusted_word["end"] = word_end + time_offset

                    segment_words.append(adjusted_word)

                    word_text = word.get("word", "").strip()
                    if word_text:
                        segment_text_parts.append(word_text)

            # Create segment result
            segment_text = " ".join(segment_text_parts).strip()
            if not segment_text:
                segment_text = "[silence]"

            results.append(
                TranscriptionSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=segment_text,
                    confidence=batch_result.confidence,
                    speaker_id=seg.get("speaker_id"),
                    speaker_name=seg.get("speaker_name"),
                    words=segment_words if segment_words else None,
                    language=batch_result.language,
                    language_probability=batch_result.language_probability,
                    model_used=f"{batch_result.model_used}-batched",
                    asr_logprob_avg=batch_result.asr_logprob_avg,
                )
            )

        return results

    def _distribute_text_proportionally(
        self, batch_result: TranscriptionSegment, boundaries: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Fallback text distribution when word timestamps unavailable"""

        if not batch_result.text.strip():
            return []

        # Simple proportional distribution based on segment duration
        total_duration = sum(b["concat_end"] - b["concat_start"] for b in boundaries)
        words = batch_result.text.split()

        results = []
        word_index = 0

        for boundary in boundaries:
            seg = boundary["original"]
            duration = boundary["concat_end"] - boundary["concat_start"]

            # Calculate word allocation
            word_count = max(1, int(len(words) * (duration / total_duration)))
            segment_words = words[word_index : word_index + word_count]
            word_index += word_count

            segment_text = " ".join(segment_words)
            if not segment_text:
                segment_text = "[silence]"

            results.append(
                TranscriptionSegment(
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=segment_text,
                    confidence=batch_result.confidence,
                    speaker_id=seg.get("speaker_id"),
                    speaker_name=seg.get("speaker_name"),
                    words=None,
                    language=batch_result.language,
                    language_probability=batch_result.language_probability,
                    model_used=f"{batch_result.model_used}-distributed",
                    asr_logprob_avg=batch_result.asr_logprob_avg,
                )
            )

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get comprehensive model and configuration info"""
        backend = "none"
        if backends.has_faster_whisper:
            backend = "faster-whisper"
        elif backends.has_openai_whisper:
            backend = "openai-whisper"

        info = {
            "model_size": self.config["model_size"],
            "backend": backend,
            "device": "cpu",
            "compute_type": self.config.get("compute_type", "float32"),
            "language": self.config["language"] or "auto",
            "language_mode": self.config["language_mode"],
            "batching_enabled": self.batching.enabled,
            "max_workers": self.executor._max_workers,
            "word_timestamps": self.config["word_timestamps"],
            "vad_enabled": True,
            "stats": self._stats,
        }
        # Attach fallback diagnostics and load errors
        try:
            info.update(
                {
                    "fallback_triggered": self._fallback_triggered,
                    "fallback_reason": self._fallback_reason,
                    "fallback_to": self._fallback_to,
                    "load_errors": getattr(self.model_manager, "last_errors", {}),
                    "runtime_failures": getattr(self, "_last_failures", {}),
                }
            )
        except Exception:
            pass
        return info

    async def validate_backend(self) -> dict[str, Any]:
        """Comprehensive backend validation"""
        validation = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "faster_whisper_available": backends.has_faster_whisper,
            "openai_whisper_available": backends.has_openai_whisper,
            "librosa_available": backends.has_librosa,
            "active_backend": None,
            "backend_functional": False,
            "error": None,
            "model_loadable": False,
            "test_transcription": False,
        }

        try:
            # Test model loading
            async with self.model_manager.get_model("validation", self.config) as model:
                validation["model_loadable"] = True
                validation["active_backend"] = (
                    "faster-whisper" if backends.has_faster_whisper else "openai-whisper"
                )

                # Test transcription with minimal audio
                test_audio = np.random.normal(0, 0.01, 1600).astype(
                    np.float32
                )  # 100ms of low-level noise
                result = await self._run_transcription(model, test_audio, 16000)

                validation["backend_functional"] = result is not None
                validation["test_transcription"] = result is not None

        except Exception as e:
            validation["error"] = str(e)

        return validation

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics"""
        return {
            "segments_processed": self._stats["segments_processed"],
            "batches_processed": self._stats["batches_processed"],
            "batching_enabled": self.batching.enabled,
            "average_batch_size": (
                self._stats["segments_processed"] / max(1, self._stats["batches_processed"])
                if self._stats["batches_processed"] > 0
                else 0
            ),
            "executor_threads": self.executor._max_workers,
            "models_loaded": len(self.model_manager._models),
        }

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        self.executor.shutdown(wait=True)


# Synchronous wrapper for backward compatibility
class AudioTranscriber:
    """Synchronous wrapper around AsyncTranscriber for backward compatibility"""

    def __init__(self, **kwargs):
        self._async_transcriber = AsyncTranscriber(**kwargs)
        self._loop = None

    def _get_loop(self):
        """Get or create event loop"""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def transcribe_segments(
        self, audio_16k_mono: np.ndarray, sr: int, diar_segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Synchronous transcription method"""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_transcriber.transcribe_segments(audio_16k_mono, sr, diar_segments)
        )

    def transcribe(
        self, audio_16k_mono: np.ndarray, sr: int, diar_segments: list[dict[str, Any]]
    ) -> list[TranscriptionSegment]:
        """Backward compatibility alias"""
        return self.transcribe_segments(audio_16k_mono, sr, diar_segments)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information"""
        return self._async_transcriber.get_model_info()

    def validate_backend(self) -> dict[str, Any]:
        """Validate backend"""
        loop = self._get_loop()
        return loop.run_until_complete(self._async_transcriber.validate_backend())

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics"""
        return self._async_transcriber.get_performance_stats()


# Utility functions for SNR estimation and audio quality
_SNR_CACHE: dict = {}


@lru_cache(maxsize=256)
def estimate_snr_db_cached(audio_hash: str, audio_shape: tuple) -> float:
    """Return cached SNR if available, else conservative default."""
    return _SNR_CACHE.get((audio_hash, audio_shape), 15.0)


def estimate_snr_db(audio: np.ndarray) -> float:
    """Fast SNR estimation with caching"""
    if audio.size == 0:
        return float("nan")

    # Create hash for caching
    audio_hash = hashlib.blake2s(audio.tobytes()).hexdigest()[:16]

    try:
        # Fast SNR estimation
        audio = audio.astype(np.float32)
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        noise_floor = float(np.percentile(np.abs(audio), 5) + 1e-12)  # 5th percentile as noise

        if rms <= noise_floor:
            return -20.0  # Very low SNR

        snr = 20.0 * math.log10(rms / noise_floor)
        snr = max(-30.0, min(60.0, snr))  # Clamp to reasonable range
        try:
            _SNR_CACHE[(audio_hash, audio.shape)] = snr
        except Exception:
            pass
        return snr

    except Exception:
        return 10.0  # Default fallback


# Factory function with configuration optimization
def create_transcriber(
    model_size: str = "large-v3",  # Use standard model
    enable_async: bool = False,
    enable_batching: bool = True,
    max_workers: int = 2,
    **kwargs,
) -> AudioTranscriber | AsyncTranscriber:
    """
    Factory function to create optimized transcriber

    Args:
        model_size: Whisper model to use
        enable_async: Return AsyncTranscriber instead of sync wrapper
        enable_batching: Enable intelligent batching
        max_workers: Maximum concurrent transcription workers
        **kwargs: Additional configuration options

    Returns:
        Configured transcriber instance
    """

    # Optimize batching config
    batching_config = BatchingConfig(enabled=enable_batching)
    if not enable_batching:
        batching_config.enabled = False

    # Create async transcriber with optimized settings
    _ = kwargs.pop("device", None)

    transcriber_kwargs = {
        "model_size": model_size,
        "batching_config": batching_config,
        "max_workers": max_workers,
        **kwargs,
    }

    async_transcriber = AsyncTranscriber(**transcriber_kwargs)

    if enable_async:
        return async_transcriber
    else:
        # Create a sync wrapper with equivalent config
        return AudioTranscriber(**transcriber_kwargs)


# Legacy compatibility aliases
CorrectedTranscriber = AudioTranscriber


# Enhanced error handling and logging
class TranscriptionError(Exception):
    """Custom exception for transcription errors"""

    pass


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Setup optimized logging for transcription module"""
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


# System information and diagnostics
def get_system_capabilities() -> dict[str, Any]:
    """Get comprehensive system capability information"""
    import multiprocessing
    import platform

    capabilities = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "backends": {
            "faster_whisper": backends.has_faster_whisper,
            "openai_whisper": backends.has_openai_whisper,
            "librosa": backends.has_librosa,
        },
        "environment": {
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "not set"),
            "torch_device": os.getenv("TORCH_DEVICE", "not set"),
            "omp_num_threads": os.getenv("OMP_NUM_THREADS", "not set"),
        },
    }

    # Get library versions if available
    try:
        if backends.has_librosa:
            capabilities["versions"] = {"librosa": backends.librosa.__version__}
    except Exception:
        pass

    try:
        if backends.has_faster_whisper:
            import faster_whisper

            capabilities.setdefault("versions", {})["faster_whisper"] = getattr(
                faster_whisper, "__version__", "unknown"
            )
    except Exception:
        pass

    return capabilities


# Performance benchmarking
async def benchmark_transcription(
    duration_sec: float = 10.0,
    sample_rate: int = 16000,
    num_segments: int = 5,
    model_size: str = "tiny",
) -> dict[str, Any]:
    """Benchmark transcription performance with synthetic audio"""

    # Generate test audio
    samples = int(duration_sec * sample_rate)
    test_audio = np.random.normal(0, 0.1, samples).astype(np.float32)

    # Generate test segments
    segment_duration = duration_sec / num_segments
    test_segments = []

    for i in range(num_segments):
        start = i * segment_duration
        end = (i + 1) * segment_duration
        test_segments.append(
            {
                "start_time": start,
                "end_time": end,
                "speaker_id": f"Speaker_{i % 2 + 1}",
                "speaker_name": f"Speaker_{i % 2 + 1}",
            }
        )

    # Benchmark with batching enabled
    async with AsyncTranscriber(
        model_size=model_size, batching_config=BatchingConfig(enabled=True)
    ) as transcriber:
        start_time = time.time()
        results_batched = await transcriber.transcribe_segments(
            test_audio, sample_rate, test_segments
        )
        batched_time = time.time() - start_time

    # Benchmark with batching disabled
    async with AsyncTranscriber(
        model_size=model_size, batching_config=BatchingConfig(enabled=False)
    ) as transcriber:
        start_time = time.time()
        results_individual = await transcriber.transcribe_segments(
            test_audio, sample_rate, test_segments
        )
        individual_time = time.time() - start_time

    return {
        "test_config": {
            "duration_sec": duration_sec,
            "num_segments": num_segments,
            "model_size": model_size,
        },
        "batched": {
            "processing_time": batched_time,
            "segments_returned": len(results_batched),
            "rtf": batched_time / duration_sec,  # Real-time factor
        },
        "individual": {
            "processing_time": individual_time,
            "segments_returned": len(results_individual),
            "rtf": individual_time / duration_sec,
        },
        "speedup": individual_time / batched_time if batched_time > 0 else 0.0,
    }


# Main execution and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized Transcription Module")
    parser.add_argument("--test", action="store_true", help="Run functionality test")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--model", default="tiny", help="Model size for testing")
    parser.add_argument(
        "--async", dest="enable_async", action="store_true", help="Use async interface"
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    if args.info:
        print("\n=== System Capabilities ===")
        import json

        capabilities = get_system_capabilities()
        print(json.dumps(capabilities, indent=2))

    if args.test:
        print("\n=== Functionality Test ===")

        # Create test data
        test_audio = np.random.normal(0, 0.1, 16000 * 5).astype(np.float32)  # 5 seconds
        test_segments = [
            {"start_time": 0.0, "end_time": 2.0, "speaker_id": "Speaker_1"},
            {"start_time": 2.0, "end_time": 4.0, "speaker_id": "Speaker_2"},
            {"start_time": 4.0, "end_time": 5.0, "speaker_id": "Speaker_1"},
        ]

        if args.enable_async:

            async def test_async():
                async with AsyncTranscriber(model_size=args.model) as transcriber:
                    logger.info("Testing async transcriber...")

                    # Validate backend
                    validation = await transcriber.validate_backend()
                    print(f"Backend validation: {validation}")

                    # Run transcription
                    results = await transcriber.transcribe_segments(
                        test_audio, 16000, test_segments
                    )
                    print(f"Transcribed {len(results)} segments")

                    for i, result in enumerate(results):
                        print(f"  Segment {i + 1}: {result.start_time:.1f}-{result.end_time:.1f}s")
                        print(f"    Text: {result.text}")
                        print(f"    Confidence: {result.confidence:.3f}")

            asyncio.run(test_async())
        else:
            transcriber = create_transcriber(model_size=args.model, enable_batching=True)

            logger.info("Testing sync transcriber...")
            print(f"Model info: {transcriber.get_model_info()}")

            # Validate backend
            validation = transcriber.validate_backend()
            print(f"Backend validation: {validation}")

            # Run transcription
            start_time = time.time()
            results = transcriber.transcribe_segments(test_audio, 16000, test_segments)
            elapsed = time.time() - start_time

            print(f"Transcribed {len(results)} segments in {elapsed:.2f}s")

            for i, result in enumerate(results):
                print(f"  Segment {i + 1}: {result.start_time:.1f}-{result.end_time:.1f}s")
                print(f"    Text: {result.text[:50]}{'...' if len(result.text) > 50 else ''}")
                print(f"    Confidence: {result.confidence:.3f}")

    if args.benchmark:
        print("\n=== Performance Benchmark ===")

        async def run_benchmark():
            results = await benchmark_transcription(
                duration_sec=30.0, num_segments=10, model_size=args.model
            )

            print(f"Test: {results['test_config']}")
            print(
                f"Batched: {results['batched']['processing_time']:.2f}s (RTF: {results['batched']['rtf']:.2f})"
            )
            print(
                f"Individual: {results['individual']['processing_time']:.2f}s (RTF: {results['individual']['rtf']:.2f})"
            )
            print(f"Speedup: {results['speedup']:.2f}x")

        asyncio.run(run_benchmark())

    if not any([args.test, args.benchmark, args.info]):
        print("Use --test, --benchmark, or --info. See --help for options.")
