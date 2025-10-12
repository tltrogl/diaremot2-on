"""Speaker diarization using Silero VAD and ECAPA ONNX embeddings - BATCHED VERSION.

This version batches both VAD ONNX inference and embedding extraction for massive speedups.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import scipy.signal

from ..io.onnx_utils import create_onnx_session
from ..pipeline.runtime_env import DEFAULT_MODELS_ROOT, iter_model_roots

MODEL_ROOTS = tuple(iter_model_roots()) or (DEFAULT_MODELS_ROOT,)


def iter_model_subpaths(*relative_paths: Path) -> Iterator[Path]:
    for root in MODEL_ROOTS:
        for rel in relative_paths:
            yield Path(root) / rel


# --- FIXED: sklearn AgglomerativeClustering wrapper for metric/affinity drift ---
try:
    import inspect as _inspect

    from sklearn.cluster import AgglomerativeClustering as _SkAgglo

    def _agglo(distance_threshold: float | None, **kw):
        init_sig = _inspect.signature(_SkAgglo.__init__)
        params = set(init_sig.parameters)
        wanted = {
            "n_clusters": None,
            "distance_threshold": distance_threshold,
            "linkage": kw.pop("linkage", "average"),
        }
        # FIXED: Handle sklearn API changes - newer versions use 'metric', older use 'affinity'
        if "metric" in params:
            wanted["metric"] = kw.pop("metric", kw.pop("affinity", "cosine"))
        elif "affinity" in params:
            wanted["affinity"] = kw.pop("metric", kw.pop("affinity", "cosine"))
        # Add any other valid parameters
        for k, v in kw.items():
            if k in params:
                wanted[k] = v
        return _SkAgglo(**wanted)
except Exception:
    _SkAgglo = None

    def _agglo(distance_threshold: float | None, **kw):
        raise RuntimeError("sklearn AgglomerativeClustering not available")
        return None


logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _bool_env(name: str) -> bool | None:
    """Parse a boolean environment variable."""
    val = os.getenv(name)
    if val is None:
        return None
    norm = val.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return True
    if norm in {"0", "false", "no", "off"}:
        return False
    return None


def _can_reach_host(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    """Return True when a TCP connection to ``host`` succeeds quickly."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _torch_repo_cached() -> bool:
    """Detect whether the Silero TorchHub repo already exists locally."""
    try:
        import torch.hub as hub

        hub_dir = Path(hub.get_dir())
    except Exception:
        return False
    candidates = [
        hub_dir / "snakers4_silero-vad_master",
        hub_dir / "snakers4_silero-vad_main",
        hub_dir / "silero-vad",
    ]
    return any(path.exists() for path in candidates)


def _resolve_state_shape(shape: tuple[Any, ...] | None) -> tuple[int, ...]:
    """Return a concrete hidden-state shape for Silero ONNX sessions."""
    default = (2, 1, 128)
    if not shape:
        return default
    resolved: list[int] = []
    for idx, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(int(dim))
        else:
            resolved.append(default[idx] if idx < len(default) else 1)
    if len(resolved) < len(default):
        resolved.extend(default[len(resolved) :])
    return tuple(resolved[: len(default)])


@dataclass
class DiarizationConfig:
    target_sr: int = 16000
    mono: bool = True
    vad_threshold: float = 0.22
    vad_min_speech_sec: float = 0.25
    vad_min_silence_sec: float = 0.30
    speech_pad_sec: float = 0.20
    vad_backend: str = "torch"
    embed_window_sec: float = 1.5
    embed_shift_sec: float = 0.75
    min_embedtable_sec: float = 0.6
    topk_windows: int = 3
    ahc_linkage: str = "average"
    ahc_distance_threshold: float = 0.15
    speaker_limit: int | None = None
    collar_sec: float = 0.25
    min_turn_sec: float = 1.50
    max_gap_to_merge_sec: float = 1.00
    registry_path: str = "registry/speaker_registry.json"
    auto_assign_cosine: float = 0.70
    flag_band_low: float = 0.60
    flag_band_high: float = 0.70
    ecapa_model_path: str | None = None
    allow_energy_vad_fallback: bool = True
    energy_gate_db: float = -33.0
    energy_hop_sec: float = 0.01


@dataclass
class DiarizedTurn:
    start: float
    end: float
    speaker: str
    speaker_name: str | None = None
    candidate_name: str | None = None
    needs_review: bool = False
    embedding: np.ndarray | None = None


class _SileroWrapper:
    """Silero VAD wrapper with BATCHED ONNX inference."""

    def __init__(self, threshold: float, speech_pad_sec: float = 0.05, backend: str = "auto"):
        self.threshold = float(threshold)
        self.speech_pad_sec = float(speech_pad_sec)
        self.backend_preference = (backend or "auto").lower()
        self.model = None
        self.get_speech_timestamps = None
        self.session = None
        self.input_name = None
        self.output_name = None
        self._onnx_input_name: str | None = None
        self._onnx_state_name: str | None = None
        self._onnx_sr_name: str | None = None
        self._onnx_state_output_index: int | None = None
        self._onnx_state_shape: tuple[int, ...] = (2, 1, 128)
        self._load()

    def _load(self) -> None:
        """Load Silero VAD honoring backend preference."""

        def _load_torch():
            override = _bool_env("SILERO_VAD_TORCH")
            if override is False:
                logger.info("Silero VAD Torch backend disabled via SILERO_VAD_TORCH")
                return False
            if not _torch_repo_cached():
                if override is not True and not _can_reach_host("github.com", timeout=3.0):
                    logger.info(
                        "Silero VAD TorchHub repo not cached and GitHub unreachable"
                    )
                    return False
            timeout_env = os.getenv("SILERO_TORCH_LOAD_TIMEOUT")
            try:
                timeout = float(timeout_env) if timeout_env else 30.0
            except ValueError:
                timeout = 30.0
            timeout = max(5.0, timeout)
            import torch.hub as hub

            def _hub_load():
                return hub.load(
                    "snakers4/silero-vad",
                    "silero_vad",
                    force_reload=False,
                    trust_repo=True,
                )

            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_hub_load)
                    self.model, utils = future.result(timeout=timeout)
            except FuturesTimeoutError:
                logger.warning(f"Silero VAD TorchHub load timed out after {timeout:.1f}s")
                return False
            except Exception as e:
                logger.warning(f"Silero VAD TorchHub unavailable: {e}")
                return False
            (
                self.get_speech_timestamps,
                self.save_audio,
                self.read_audio,
                self.VADIterator,
                self.collect_chunks,
            ) = utils
            self.model.eval()
            logger.info("Silero VAD PyTorch model loaded (TorchHub)")
            return True

        def _load_onnx():
            try:
                onnx_path = os.getenv("SILERO_VAD_ONNX_PATH")
                if not onnx_path:
                    candidate_paths = list(iter_model_subpaths("silero_vad.onnx"))
                    candidate_paths.extend(list(iter_model_subpaths(Path("silero") / "vad.onnx")))
                    unique_candidates: list[Path] = []
                    seen: set[str] = set()
                    for cand in candidate_paths:
                        resolved = Path(cand)
                        key = str(resolved)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_candidates.append(resolved)
                    for cand in unique_candidates:
                        if cand.exists():
                            onnx_path = str(cand)
                            break
                    if not onnx_path and unique_candidates:
                        onnx_path = str(unique_candidates[0])
                if onnx_path and Path(onnx_path).exists():
                    self.session = create_onnx_session(onnx_path, threads=1)
                    inputs = self.session.get_inputs()
                    outputs = self.session.get_outputs()
                    self.input_name = inputs[0].name if inputs else None
                    self.output_name = outputs[0].name if outputs else None
                    self._onnx_input_name = None
                    self._onnx_state_name = None
                    self._onnx_sr_name = None
                    self._onnx_state_output_index = None
                    self._onnx_state_shape = (2, 1, 128)
                    for inp in inputs:
                        lower = inp.name.lower()
                        if self._onnx_input_name is None and (
                            lower == "input" or lower.endswith("/input") or "input" in lower
                        ):
                            self._onnx_input_name = inp.name
                        elif self._onnx_state_name is None and "state" in lower:
                            self._onnx_state_name = inp.name
                            try:
                                self._onnx_state_shape = _resolve_state_shape(
                                    tuple(getattr(inp, "shape", ()))
                                )
                            except Exception:
                                self._onnx_state_shape = (2, 1, 128)
                        elif self._onnx_sr_name is None and (
                            lower == "sr" or "sample_rate" in lower or "samplerate" in lower
                        ):
                            self._onnx_sr_name = inp.name
                    if self._onnx_input_name is None and inputs:
                        self._onnx_input_name = inputs[0].name
                    if self._onnx_state_name is None:
                        for inp in inputs:
                            if "state" in inp.name.lower():
                                self._onnx_state_name = inp.name
                                self._onnx_state_shape = _resolve_state_shape(
                                    tuple(getattr(inp, "shape", ()))
                                )
                                break
                    if self._onnx_sr_name is None:
                        for inp in inputs:
                            lower = inp.name.lower()
                            if lower.startswith("sr") or "sr" == lower:
                                self._onnx_sr_name = inp.name
                                break
                    for idx, out in enumerate(outputs):
                        lower = out.name.lower()
                        if idx == 0:
                            self.output_name = out.name
                        if self._onnx_state_output_index is None and "state" in lower:
                            self._onnx_state_output_index = idx
                    if self._onnx_state_output_index is None and len(outputs) > 1:
                        self._onnx_state_output_index = 1
                    logger.info(f"Silero VAD ONNX model loaded: {onnx_path}")
                    return True
            except Exception as e:
                logger.info(f"Silero VAD ONNX unavailable: {e}")
            self.session = None
            return False

        pref = (self.backend_preference or "auto").lower()
        if pref == "onnx":
            if not _load_onnx():
                raise RuntimeError("Silero VAD ONNX requested but unavailable")
            return
        if pref == "torch":
            if not _load_torch():
                raise RuntimeError("Silero VAD Torch backend requested but unavailable")
            return
        # auto: prefer ONNX
        if _load_onnx():
            return
        override = _bool_env("SILERO_VAD_TORCH")
        should_try_torch = override is True
        if override is None:
            should_try_torch = _torch_repo_cached() or _can_reach_host("github.com", timeout=3.0)
        if should_try_torch and _load_torch():
            return
        logger.info("Silero VAD unavailable; will use fallback")

    def _detect_with_onnx_batched(
        self,
        wav: np.ndarray,
        sr: int,
        *,
        min_speech_sec: float,
        min_silence_sec: float,
    ) -> list[tuple[float, float]]:
        """BATCHED Silero ONNX inference - processes all chunks in one call."""
        if self.session is None or self._onnx_input_name is None:
            return []
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if sr != 16000:
            audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32, copy=False)
            sr = 16000
        if audio.size == 0:
            return []

        chunk_size = 512
        context_size = 64
        orig_samples = audio.shape[0]
        pad = (-orig_samples) % chunk_size
        if pad:
            audio = np.pad(audio, (0, pad), mode="constant")
        num_chunks = audio.shape[0] // chunk_size
        if num_chunks == 0:
            return []

        # BATCHED: Prepare ALL chunks at once
        batch_size = num_chunks
        state_shape = list(self._onnx_state_shape or (2, 1, 128))
        if len(state_shape) < 3:
            state_shape = [2, batch_size, 128]
        else:
            state_shape = state_shape[:3]
            state_shape[1] = batch_size

        # Create batch of windows with context
        windows = np.zeros((batch_size, context_size + chunk_size), dtype=np.float32)
        context = np.zeros((batch_size, context_size), dtype=np.float32)
        
        for i in range(num_chunks):
            offset = i * chunk_size
            chunk = audio[offset : offset + chunk_size]
            # For first chunk, context is zeros. For others, use tail of previous chunk
            if i > 0:
                context[i] = audio[max(0, offset - context_size) : offset][-context_size:]
            windows[i] = np.concatenate([context[i], chunk])

        # Initialize state for batch
        state = np.zeros(tuple(state_shape), dtype=np.float32)

        # SINGLE ONNX CALL for all chunks
        feeds: dict[str, np.ndarray] = {self._onnx_input_name: windows}
        if self._onnx_state_name:
            feeds[self._onnx_state_name] = state
        if self._onnx_sr_name:
            # Broadcast sr to match batch size
            feeds[self._onnx_sr_name] = np.full(batch_size, sr, dtype=np.int64)

        ort_outs = self.session.run(None, feeds)
        if not isinstance(ort_outs, (list, tuple)) or not ort_outs:
            return []

        logits = np.asarray(ort_outs[0], dtype=np.float32)
        
        # Process logits (batch_size, 2) or (batch_size,)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if logits.shape[-1] == 2:
            # Softmax on last dimension
            exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            denom = np.sum(exps, axis=1, keepdims=True) + 1e-9
            probs = (exps[:, 1:2] / denom).reshape(-1)
        else:
            # Sigmoid
            probs = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        
        chunk_probs = np.clip(probs, 0.0, 1.0)[:num_chunks]

        # Convert probs to segments
        chunk_duration = chunk_size / float(sr)
        total_duration = orig_samples / float(sr)
        speech_mask = chunk_probs > self.threshold
        
        segments: list[tuple[float, float]] = []
        start_idx: int | None = None
        for idx, flag in enumerate(speech_mask):
            if flag and start_idx is None:
                start_idx = idx
            elif not flag and start_idx is not None:
                end_idx = idx
                start = start_idx * chunk_duration
                end = min(end_idx * chunk_duration, total_duration)
                if end - start >= min_speech_sec:
                    segments.append((start, end))
                start_idx = None
        if start_idx is not None:
            end_idx = len(speech_mask)
            start = start_idx * chunk_duration
            end = min(end_idx * chunk_duration, total_duration)
            if end - start >= min_speech_sec:
                segments.append((start, end))

        if not segments:
            return []

        # Apply padding
        pad = float(self.speech_pad_sec)
        padded = []
        for start, end in segments:
            s = max(0.0, start - pad)
            e = min(total_duration, end + pad)
            if e - s >= min_speech_sec:
                padded.append((s, e))

        if not padded:
            return []

        return _merge_regions(padded, gap=min_silence_sec)

    def detect(
        self, wav: np.ndarray, sr: int, min_speech_sec: float, min_silence_sec: float
    ) -> list[tuple[float, float]]:
        """Run VAD and return speech regions."""
        if self.session is not None:
            try:
                return self._detect_with_onnx_batched(
                    wav,
                    sr,
                    min_speech_sec=min_speech_sec,
                    min_silence_sec=min_silence_sec,
                )
            except Exception as e:
                logger.warning(f"Silero ONNX VAD failed: {e}")
                return []
        if self.model is None or self.get_speech_timestamps is None:
            return []
        try:
            import torch

            wav_t = torch.from_numpy(wav.astype(np.float32))
            ts = self.get_speech_timestamps(
                wav_t,
                self.model,
                sampling_rate=sr,
                threshold=self.threshold,
                min_speech_duration_ms=int(float(min_speech_sec) * 1000),
                min_silence_duration_ms=int(float(min_silence_sec) * 1000),
                speech_pad_ms=int(self.speech_pad_sec * 1000),
            )
            spans = [(t["start"] / sr, t["end"] / sr) for t in ts]
            return _merge_regions(spans, gap=min_silence_sec)
        except Exception as e:
            logger.warning(f"Silero VAD failed: {e}")
            return []


class _ECAPAWrapper:
    """ECAPA-TDNN embedding extraction via ONNX Runtime."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.session = None
        self.input_name: str | None = None
        self.output_name: str | None = None
        self.model_path = Path(model_path) if model_path else None
        self._load()

    def _load(self) -> None:
        """Load ECAPA ONNX model."""
        try:
            if self.model_path and Path(self.model_path).exists():
                model_path = Path(self.model_path)
            else:
                env_path = os.getenv("ECAPA_ONNX_PATH")
                if env_path:
                    model_path = Path(env_path)
                else:
                    candidate_paths = list(
                        iter_model_subpaths(Path("ecapa_onnx") / "ecapa_tdnn.onnx")
                    )
                    candidate_paths.extend(list(iter_model_subpaths("ecapa_tdnn.onnx")))
                    unique_candidates: list[Path] = []
                    seen: set[str] = set()
                    for cand in candidate_paths:
                        resolved = Path(cand)
                        key = str(resolved)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique_candidates.append(resolved)
                    model_path = next((cand for cand in unique_candidates if cand.exists()), None)
                    if model_path is None and unique_candidates:
                        model_path = unique_candidates[0]
                    if model_path is None:
                        model_path = Path.cwd() / "models" / "ecapa_tdnn.onnx"
            self.session = create_onnx_session(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"ECAPA ONNX model loaded: {model_path}")
        except Exception as e:
            logger.error(f"ECAPA ONNX model unavailable: {e}")
            self.session = None

    def embed_batch(self, batch: list[np.ndarray], sr: int) -> list[np.ndarray | None]:
        """BATCHED embedding extraction - processes all clips in one call."""
        if self.session is None or not batch:
            return [None] * len(batch)
        try:
            # Compute log-mel spectrograms for all clips
            mel_specs = []
            max_frames = 0
            for x in batch:
                mel = librosa.feature.melspectrogram(
                    y=x,
                    sr=sr,
                    n_fft=400,
                    hop_length=160,
                    n_mels=80,
                    fmin=20,
                    fmax=sr / 2,
                )
                mel = librosa.power_to_db(mel, ref=1.0).T  # (frames, mel)
                # Apply CMVN
                try:
                    m = mel.mean(axis=0, keepdims=True)
                    s = mel.std(axis=0, keepdims=True) + 1e-8
                    mel = (mel - m) / s
                except Exception:
                    pass
                mel_specs.append(mel)
                if mel.shape[0] > max_frames:
                    max_frames = mel.shape[0]

            # Pad all to same length
            pad = np.zeros((len(batch), max_frames, mel_specs[0].shape[1]), dtype=np.float32)
            for i, mel in enumerate(mel_specs):
                pad[i, : mel.shape[0], :] = mel.astype(np.float32)
            pad = np.ascontiguousarray(pad)

            # SINGLE ONNX CALL for entire batch
            out = self.session.run([self.output_name], {self.input_name: pad})[0]
            arr = out.squeeze()
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            arr = arr.astype(np.float32, copy=False)

            # Normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
            arr = arr / norms
            return [arr[i] for i in range(arr.shape[0])]
        except Exception as e:
            logger.warning(f"ECAPA embed failed: {e}")
            return [None] * len(batch)


class SpeakerRegistry:
    def __init__(self, path: str):
        self.path = Path(path)
        self._speakers: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}
        self._use_wrapped_format: bool = False
        self._load()

    @staticmethod
    def _iso_now() -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _load(self) -> None:
        self._speakers = {}
        self._metadata = {}
        self._use_wrapped_format = False
        try:
            if not self.path.exists():
                return
            raw = self.path.read_text(encoding="utf-8")
            if not raw.strip():
                return
            data = json.loads(raw)
            if isinstance(data, dict):
                speakers = data.get("speakers")
                if isinstance(speakers, dict):
                    self._speakers = dict(speakers)
                    self._metadata = {k: v for k, v in data.items() if k != "speakers"}
                    self._use_wrapped_format = True
                else:
                    self._speakers = {k: v for k, v in data.items() if isinstance(k, str)}
            else:
                logger.warning("Registry load expected a JSON object at %s", self.path)
        except Exception as e:
            logger.warning(f"Registry load failed: {e}")
            self._speakers = {}
            self._metadata = {}
            self._use_wrapped_format = False

    def _touch_metadata(self) -> None:
        if not (self._use_wrapped_format or self._metadata):
            return
        now = self._iso_now()
        if "created_at" not in self._metadata:
            self._metadata["created_at"] = now
        self._metadata["updated_at"] = now
        if "total_speakers" in self._metadata:
            self._metadata["total_speakers"] = len(self._speakers)
        meta_block = self._metadata.get("metadata")
        if isinstance(meta_block, dict):
            if "total_speakers" in meta_block or meta_block:
                meta_block["total_speakers"] = len(self._speakers)
                self._metadata["metadata"] = meta_block
        self._use_wrapped_format = True

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_suffix(".tmp")
            payload: dict[str, Any]
            if self._use_wrapped_format or self._metadata:
                self._touch_metadata()
                payload = {**self._metadata, "speakers": self._speakers}
            else:
                payload = self._speakers
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(self.path)
        except Exception as e:
            logger.warning(f"Registry save failed: {e}")
            temp_path = self.path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def has(self, name: str) -> bool:
        return name in self._speakers

    def enroll(self, name: str, centroid: np.ndarray) -> None:
        stamp = self._iso_now()
        self._speakers[name] = {
            "centroid": (centroid.tolist() if isinstance(centroid, np.ndarray) else centroid),
            "samples": 1,
            "last_seen": stamp,
        }

    def update_centroid(self, name: str, centroid: np.ndarray, alpha: float = 0.30) -> None:
        if not self.has(name):
            self.enroll(name, centroid)
            return
        existing = self._speakers.get(name) or {}
        old = np.asarray(existing.get("centroid", []), dtype=np.float32)
        if old.size == 0:
            old = np.asarray(centroid, dtype=np.float32)
        new = np.asarray(centroid, dtype=np.float32)
        ema = (1 - alpha) * old + alpha * new
        ema = ema / (np.linalg.norm(ema) + 1e-9)
        existing["centroid"] = ema.tolist()
        existing["samples"] = int(existing.get("samples", 1)) + 1
        existing["last_seen"] = self._iso_now()
        self._speakers[name] = existing

    def match(self, centroid: np.ndarray) -> tuple[str | None, float]:
        if not self._speakers:
            return None, 0.0
        c = np.asarray(centroid, dtype=np.float32).reshape(-1)
        best, best_sim = None, -1.0
        for name, rec in self._speakers.items():
            ref = np.asarray(rec.get("centroid", []), dtype=np.float32).reshape(-1)
            if ref.size == 0:
                continue
            sim = float(np.dot(c, ref) / (np.linalg.norm(c) * np.linalg.norm(ref) + 1e-9))
            if sim > best_sim:
                best, best_sim = name, sim
        return best, best_sim


def _merge_regions(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _energy_vad_fallback(
    wav: np.ndarray, sr: int, gate_db: float, hop_sec: float
) -> list[tuple[float, float]]:
    """Simple energy-based VAD when Silero fails."""
    if wav.size == 0:
        return []
    hop = max(1, int(hop_sec * sr))
    if len(wav) < hop * 2:
        return []
    try:
        frames = librosa.util.frame(wav, frame_length=hop * 2, hop_length=hop).T
    except Exception:
        return []
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    speech = rms_db > gate_db
    regions = []
    start = None
    for i, is_speech in enumerate(speech):
        t = i * hop_sec
        if is_speech and start is None:
            start = t
        elif not is_speech and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, len(speech) * hop_sec))
    return _merge_regions(regions, gap=0.5)


class SpeakerDiarizer:
    def __init__(self, config: DiarizationConfig):
        self.config = config
        self.vad = _SileroWrapper(
            config.vad_threshold,
            config.speech_pad_sec,
            backend=getattr(config, "vad_backend", "auto"),
        )
        self.ecapa = _ECAPAWrapper(config.ecapa_model_path)
        self.registry = None
        if config.registry_path:
            try:
                self.registry = SpeakerRegistry(config.registry_path)
                logger.info(f"Registry loaded: {config.registry_path}")
            except Exception as e:
                logger.warning(f"Registry unavailable: {e}")
        self._last_turns: list[DiarizedTurn] = []

    def get_segment_embeddings(self) -> list[dict[str, Any]]:
        return [
            {"speaker": t.speaker, "embedding": t.embedding}
            for t in self._last_turns
            if t.embedding is not None
        ]

    def diarize_audio(self, wav: np.ndarray, sr: int) -> list[dict[str, Any]]:
        """Main diarization entry point with BATCHED inference."""
        self._last_turns = []
        if wav is None or wav.size == 0:
            return []
        
        # Ensure mono 16kHz
        if wav.ndim > 1:
            wav = np.mean(wav, axis=0)
        if sr != self.config.target_sr:
            wav = scipy.signal.resample_poly(wav, self.config.target_sr, sr).astype(np.float32)
            sr = self.config.target_sr
        else:
            wav = wav.astype(np.float32)
        
        # Step 1: VAD (now batched)
        speech_regions = self.vad.detect(
            wav, sr, self.config.vad_min_speech_sec, self.config.vad_min_silence_sec
        )
        
        if not speech_regions and self.config.allow_energy_vad_fallback:
            logger.info("Using energy VAD fallback")
            speech_regions = _energy_vad_fallback(
                wav, sr, self.config.energy_gate_db, self.config.energy_hop_sec
            )
        
        if not speech_regions:
            logger.warning("No speech detected by VAD")
            return []
        
        # Step 2: Extract embeddings (now batched)
        windows = self._extract_embedding_windows_batched(wav, sr, speech_regions)
        
        if len(windows) < 2:
            turn = {
                "start": speech_regions[0][0],
                "end": speech_regions[-1][1],
                "speaker": "Speaker_1",
                "speaker_name": "Speaker_1",
                "embedding": windows[0]["embedding"] if windows else None,
            }
            return [turn]
        
        # Step 3: Cluster
        embeddings = [w["embedding"] for w in windows if w["embedding"] is not None]
        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return []
        
        try:
            X = np.vstack(embeddings)
            if self.config.speaker_limit:
                clusterer = _agglo(
                    distance_threshold=None,
                    n_clusters=self.config.speaker_limit,
                    linkage=self.config.ahc_linkage,
                    metric="cosine",
                )
            else:
                clusterer = _agglo(
                    distance_threshold=self.config.ahc_distance_threshold,
                    linkage=self.config.ahc_linkage,
                    metric="cosine",
                )
            labels = clusterer.fit_predict(X)
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            labels = np.zeros(len(embeddings), dtype=int)
        
        # Attach labels
        for w, label in zip(windows, labels, strict=False):
            w["speaker"] = f"Speaker_{label + 1}"
        
        # Step 4: Build continuous segments
        turns = self._build_continuous_segments(windows, speech_regions)
        
        # Step 5: Post-process
        turns = self._merge_short_gaps(turns)
        turns = self._assign_speaker_names(turns)
        
        self._last_turns = turns
        return [self._turn_to_dict(t) for t in turns]

    def _extract_embedding_windows_batched(
        self, wav: np.ndarray, sr: int, speech_regions: list[tuple[float, float]]
    ) -> list[dict[str, Any]]:
        """BATCHED embedding extraction - extracts ALL windows then runs ONE inference."""
        # Collect ALL audio clips first
        clips = []
        window_meta = []
        
        for start_sec, end_sec in speech_regions:
            if end_sec - start_sec < self.config.min_embedtable_sec:
                continue
            cursor = start_sec
            while cursor < end_sec:
                win_end = min(cursor + self.config.embed_window_sec, end_sec)
                if win_end - cursor >= self.config.min_embedtable_sec:
                    start_idx = int(cursor * sr)
                    end_idx = int(win_end * sr)
                    clips.append(wav[start_idx:end_idx])
                    window_meta.append({'start': cursor, 'end': win_end})
                cursor += self.config.embed_shift_sec
        
        # ONE batch call for ALL clips
        if clips:
            embeddings = self.ecapa.embed_batch(clips, sr)
        else:
            embeddings = []
        
        # Map back to windows
        windows = []
        for meta, emb in zip(window_meta, embeddings):
            windows.append({
                'start': meta['start'],
                'end': meta['end'],
                'embedding': emb,
                'speaker': None,
                'region_idx': len(windows)
            })
        
        logger.info(f"Extracted {len(windows)} embeddings in single batch")
        return windows

    def _build_continuous_segments(
        self,
        windows: list[dict[str, Any]],
        speech_regions: list[tuple[float, float]],
    ) -> list[DiarizedTurn]:
        """Build continuous speaker segments from clustered windows."""
        if not windows:
            return []
        
        segments = []
        for region_start, region_end in speech_regions:
            region_windows = []
            for window in windows:
                if (
                    window.get("speaker") is not None
                    and window["start"] < region_end
                    and window["end"] > region_start
                    and window["embedding"] is not None
                ):
                    region_windows.append(
                        {
                            "start": max(window["start"], region_start),
                            "end": min(window["end"], region_end),
                            "speaker": window["speaker"],
                            "embedding": window["embedding"],
                            "duration": min(window["end"], region_end)
                            - max(window["start"], region_start),
                        }
                    )
            if not region_windows:
                continue
            
            region_windows.sort(key=lambda x: x["start"])
            
            events = []
            for w in region_windows:
                events.append(
                    {
                        "time": w["start"],
                        "type": "start",
                        "speaker": w["speaker"],
                        "embedding": w["embedding"],
                    }
                )
                events.append({"time": w["end"], "type": "end", "speaker": w["speaker"]})
            events.sort(key=lambda x: (x["time"], 0 if x["type"] == "end" else 1))
            
            active_speakers = {}
            current_time = region_start
            for event in events:
                event_time = event["time"]
                if event_time > current_time and active_speakers:
                    span_votes = {}
                    for w in region_windows:
                        if w["start"] <= current_time and w["end"] >= event_time:
                            duration = min(w["end"], event_time) - max(w["start"], current_time)
                            if duration > 0:
                                span_votes[w["speaker"]] = (
                                    span_votes.get(w["speaker"], 0) + duration
                                )
                    if span_votes:
                        dominant_speaker = max(span_votes.items(), key=lambda x: x[1])[0]
                        emb_list = [
                            w["embedding"]
                            for w in region_windows
                            if w["speaker"] == dominant_speaker
                            and w["start"] < event_time
                            and w["end"] > current_time
                            and w["embedding"] is not None
                        ]
                        speaker_embedding = None
                        if emb_list:
                            pooled = np.mean(np.vstack(emb_list), axis=0)
                            norm = np.linalg.norm(pooled)
                            speaker_embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                        segments.append(
                            DiarizedTurn(
                                start=current_time,
                                end=event_time,
                                speaker=dominant_speaker,
                                speaker_name=dominant_speaker,
                                embedding=speaker_embedding,
                            )
                        )
                
                if event["type"] == "start":
                    active_speakers[event["speaker"]] = event_time
                else:
                    active_speakers.pop(event["speaker"], None)
                current_time = event_time
            
            if active_speakers and current_time < region_end:
                span_votes = {}
                for w in region_windows:
                    if w["start"] <= current_time and w["end"] >= region_end:
                        duration = region_end - max(w["start"], current_time)
                        if duration > 0:
                            span_votes[w["speaker"]] = span_votes.get(w["speaker"], 0) + duration
                if span_votes:
                    dominant_speaker = max(span_votes.items(), key=lambda x: x[1])[0]
                    emb_list = [
                        w["embedding"]
                        for w in region_windows
                        if w["speaker"] == dominant_speaker
                        and w["start"] < region_end
                        and w["end"] > current_time
                        and w["embedding"] is not None
                    ]
                    speaker_embedding = None
                    if emb_list:
                        pooled = np.mean(np.vstack(emb_list), axis=0)
                        norm = np.linalg.norm(pooled)
                        speaker_embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                    segments.append(
                        DiarizedTurn(
                            start=current_time,
                            end=region_end,
                            speaker=dominant_speaker,
                            speaker_name=dominant_speaker,
                            embedding=speaker_embedding,
                        )
                    )
        
        if not segments:
            return []
        
        # Merge adjacent same-speaker segments
        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg.start - last.end
            if last.speaker == seg.speaker and gap <= self.config.max_gap_to_merge_sec:
                last_duration = last.end - last.start
                seg_duration = seg.end - seg.start
                if last.embedding is not None and seg.embedding is not None:
                    pooled = last.embedding * last_duration + seg.embedding * seg_duration
                    norm = np.linalg.norm(pooled)
                    last.embedding = pooled / (norm + 1e-8) if norm > 0 else pooled
                last.end = seg.end
            else:
                merged.append(seg)
        return merged

    def _merge_short_gaps(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not turns:
            return []
        merged = [turns[0]]
        for turn in turns[1:]:
            last = merged[-1]
            gap = turn.start - last.end
            if (
                last.speaker == turn.speaker
                and gap <= self.config.max_gap_to_merge_sec
                and gap >= 0
            ):
                last.end = turn.end
            else:
                merged.append(turn)
        return merged

    def _assign_speaker_names(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        if not self.registry:
            return turns
        for turn in turns:
            if turn.embedding is not None:
                name, similarity = self.registry.match(turn.embedding)
                if name and similarity >= self.config.auto_assign_cosine:
                    turn.speaker_name = name
                    turn.candidate_name = name
                    if self.config.flag_band_low <= similarity <= self.config.flag_band_high:
                        turn.needs_review = True
        return turns

    def reassign_with_registry(self, turns: list[dict[str, Any]]) -> None:
        if not self.registry:
            return
        for turn in turns:
            embedding = turn.get("embedding")
            if embedding is not None:
                name, similarity = self.registry.match(np.asarray(embedding))
                if name and similarity >= self.config.auto_assign_cosine:
                    turn["speaker_name"] = name
                    turn["candidate_name"] = name

    def _turn_to_dict(self, turn: DiarizedTurn) -> dict[str, Any]:
        return {
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.speaker,
            "speaker_name": turn.speaker_name,
            "candidate_name": turn.candidate_name,
            "needs_review": turn.needs_review,
            "embedding": (turn.embedding.tolist() if turn.embedding is not None else None),
        }
