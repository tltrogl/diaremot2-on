"""Speaker diarization using Silero VAD and ECAPA ONNX embeddings - FIXED VERSION."""

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
        # Unreachable return to satisfy static analysis
        return None

# Optional spectral clustering backend (CPU)
try:  # pragma: no cover - optional dependency
    from spectralcluster import SpectralClusterer as _SpectralClusterer  # type: ignore
except Exception:  # pragma: no cover - best effort import
    _SpectralClusterer = None  # type: ignore

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
        hub_dir / "silero-vad",  # custom mirrors
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
    # Silero VAD defaults (aligned with docs + production defaults)
    vad_threshold: float = 0.35
    vad_min_speech_sec: float = 0.80
    vad_min_silence_sec: float = 0.80
    speech_pad_sec: float = 0.10
    # VAD backend preference: 'auto' | 'torch' | 'onnx'
    # Default to 'auto' to honor ONNX-first policy and cleanly fall back.
    vad_backend: str = "auto"
    # Embedding windows
    embed_window_sec: float = 1.5
    embed_shift_sec: float = 0.75
    min_embedtable_sec: float = 0.6
    topk_windows: int = 3
    # Clustering
    ahc_linkage: str = "average"
    ahc_distance_threshold: float = 0.15
    speaker_limit: int | None = None
    clustering_backend: str = "ahc"  # 'ahc' | 'spectral'
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Post-processing
    collar_sec: float = 0.25
    min_turn_sec: float = 1.50
    max_gap_to_merge_sec: float = 1.00
    # Post-cluster merging (reduce over-fragmentation when speakers unknown)
    # Post-merge heuristics tuned to reduce micro-clusters when VAD over-splits
    post_merge_distance_threshold: float = 0.30  # cosine distance threshold to merge
    post_merge_min_speakers: int | None = None   # do not reduce below this (None=no floor)
    # Registry
    registry_path: str = "registry/speaker_registry.json"
    auto_assign_cosine: float = 0.70
    flag_band_low: float = 0.60
    flag_band_high: float = 0.70
    # Embedding model
    ecapa_model_path: str | None = None
    # Energy VAD fallback
    allow_energy_vad_fallback: bool = True
    energy_gate_db: float = -33.0
    energy_hop_sec: float = 0.01
    # Single-speaker collapse heuristic
    single_speaker_collapse: bool = True
    single_speaker_dominance: float = 0.88
    single_speaker_centroid_threshold: float = 0.20
    single_speaker_min_turns: int = 3


@dataclass
class DiarizedTurn:
    start: float
    end: float
    speaker: str
    speaker_name: str | None = None
    candidate_name: str | None = None
    needs_review: bool = False
    embedding: np.ndarray | None = None


def collapse_single_speaker_turns(
    turns: list[DiarizedTurn],
    *,
    dominance_threshold: float = 0.88,
    centroid_threshold: float = 0.08,
    min_turns: int = 3,
) -> tuple[bool, str | None, str | None]:
    """Collapse fragmented speaker labels into a single speaker when evidence strongly supports one talker."""
    if not turns or len(turns) <= 1:
        return False, None, None
    if min_turns > 1 and len(turns) < min_turns:
        return False, None, None
    speakers = {t.speaker for t in turns if t.speaker}
    if len(speakers) <= 1:
        return False, next(iter(speakers), None), None
    durations: dict[str, float] = {}
    total_duration = 0.0
    for t in turns:
        dur = max(float(t.end) - float(t.start), 0.0)
        durations[t.speaker] = durations.get(t.speaker, 0.0) + dur
        total_duration += dur
    if total_duration <= 0.0:
        return False, None, None
    dominant_speaker, dominant_duration = max(durations.items(), key=lambda item: item[1])
    dominance_ratio = dominant_duration / total_duration
    collapse_reason: str | None = None
    collapse = False
    if dominance_threshold > 0 and dominance_ratio >= dominance_threshold and len(durations) > 1:
        collapse = True
        collapse_reason = f"dominance={dominance_ratio:.2f}"
    if not collapse and centroid_threshold > 0:
        by_speaker: dict[str, list[np.ndarray]] = {}
        for t in turns:
            if t.embedding is None:
                continue
            by_speaker.setdefault(t.speaker, []).append(np.asarray(t.embedding, dtype=np.float32))
        by_speaker = {k: v for k, v in by_speaker.items() if v}
        if len(by_speaker) >= 2:
            def centroid(vecs: list[np.ndarray]) -> np.ndarray:
                arr = np.vstack(vecs)
                c = arr.mean(axis=0)
                n = np.linalg.norm(c)
                return c / (n + 1e-9)

            centroids = {spk: centroid(vecs) for spk, vecs in by_speaker.items()}
            max_distance = 0.0
            keys = list(centroids.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a = centroids[keys[i]]
                    b = centroids[keys[j]]
                    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
                    if denom <= 0:
                        continue
                    distance = 1.0 - float(np.dot(a, b) / denom)
                    if distance > max_distance:
                        max_distance = distance
            if max_distance <= centroid_threshold and len(keys) > 1:
                collapse = True
                collapse_reason = f"max_centroid_dist={max_distance:.3f}"
    if not collapse:
        return False, None, None
    canonical = dominant_speaker or "Speaker_1"
    if not canonical or canonical.lower() in {"", "none", "null"}:
        canonical = "Speaker_1"
    for turn in turns:
        turn.speaker = canonical
        turn.speaker_name = canonical
    return True, canonical, collapse_reason


class _SileroWrapper:
    """Silero VAD wrapper with optional ONNX Runtime backend.
    Prefers ONNX if an exported model is available locally, otherwise falls
    back to the TorchHub PyTorch implementation (fully CPU).
    """

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
        self._onnx_state_cache: np.ndarray | None = None
        self._onnx_context_cache: np.ndarray | None = None
        self._onnx_last_sr: int | None = None
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load Silero VAD honoring backend preference (onnx|torch|auto)."""

        # Prefer Torch by default for reliability across exports
        def _load_torch():
            override = _bool_env("SILERO_VAD_TORCH")
            if override is False:
                logger.info("Silero VAD Torch backend disabled via SILERO_VAD_TORCH")
                return False
            if not _torch_repo_cached():
                # Skip TorchHub when offline to avoid hanging on git clone
                if override is not True and not _can_reach_host("github.com", timeout=3.0):
                    logger.info(
                        "Silero VAD TorchHub repo not cached and GitHub unreachable; "
                        "falling back to energy VAD"
                    )
                    return False
            timeout_env = os.getenv("SILERO_TORCH_LOAD_TIMEOUT")
            try:
                timeout = float(timeout_env) if timeout_env else 30.0
            except ValueError:
                timeout = 30.0
            timeout = max(5.0, timeout)

            try:  # Torch is optional; skip cleanly when absent
                import torch.hub as hub  # type: ignore[attr-defined]
            except ModuleNotFoundError:
                logger.info("Silero VAD Torch backend unavailable (torch not installed)")
                return False
            except Exception as exc:  # pragma: no cover - defensive import guard
                logger.warning(f"Silero VAD Torch import failed: {exc}")
                return False

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
                logger.warning(
                    "Silero VAD TorchHub load timed out after %.1fs; using fallback",
                    timeout,
                )
                self.model = None
                self.get_speech_timestamps = None
                return False
            except Exception as e:
                logger.warning(f"Silero VAD TorchHub unavailable: {e}")
                self.model = None
                self.get_speech_timestamps = None
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
                from pathlib import Path

                from ..io.onnx_utils import create_onnx_session

                onnx_path = os.getenv("SILERO_VAD_ONNX_PATH")
                if not onnx_path:
                    candidate_paths = list(iter_model_subpaths("silero_vad.onnx"))
                    candidate_paths.extend(list(iter_model_subpaths(Path("silero") / "vad.onnx")))
                    # Some bundles ship the model inside typo'd or nested directories; scan for it.
                    for root in MODEL_ROOTS:
                        root_path = Path(root)
                        try:
                            for cand in root_path.glob("**/silero_vad.onnx"):
                                candidate_paths.append(cand)
                        except OSError:
                            continue
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
                    self._onnx_state_cache = None
                    self._onnx_context_cache = None
                    self._onnx_last_sr = None
                    logger.info(f"Silero VAD ONNX model loaded: {onnx_path}")
                    return True
            except Exception as e:
                logger.info(f"Silero VAD ONNX unavailable: {e}")
            self.session = None
            self.input_name = None
            self.output_name = None
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
        # auto: prefer ONNX when available, fall back to TorchHub only when viable
        if _load_onnx():
            return
        override = _bool_env("SILERO_VAD_TORCH")
        should_try_torch = override is True
        if override is None:
            should_try_torch = _torch_repo_cached() or _can_reach_host("github.com", timeout=3.0)
        if should_try_torch and _load_torch():
            return
        if should_try_torch:
            logger.info("Silero VAD Torch backend unavailable; proceeding without it")
        else:
            logger.info("Silero VAD Torch backend skipped (offline/disabled); using fallbacks")

    # ------------------------------------------------------------------
    def _detect_with_onnx(
        self,
        wav: np.ndarray,
        sr: int,
        *,
        min_speech_sec: float,
        min_silence_sec: float,
    ) -> list[tuple[float, float]]:
        """Streaming Silero ONNX inference that mirrors TorchHub behaviour."""
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
        batch_size = 1
        orig_samples = audio.shape[0]
        pad = (-orig_samples) % chunk_size
        if pad:
            audio = np.pad(audio, (0, pad), mode="constant")
        num_chunks = audio.shape[0] // chunk_size
        if num_chunks == 0:
            return []
        duration_sec = orig_samples / float(sr)
        logger.info(f"Starting ONNX VAD: {duration_sec:.1f}s audio → {num_chunks} chunks to process")
        state_shape = list(self._onnx_state_shape or (2, 1, 128))
        if len(state_shape) < 3:
            state_shape = [2, batch_size, 128]
        else:
            state_shape = state_shape[:3]
            state_shape[1] = batch_size
        state = np.zeros(tuple(state_shape), dtype=np.float32)
        context = np.zeros((batch_size, context_size), dtype=np.float32)
        self._onnx_state_cache = state
        self._onnx_context_cache = context
        self._onnx_last_sr = sr
        sr_array = np.array(sr, dtype=np.int64)
        chunk_probs: list[float] = []
        offset = 0
        # Log progress for long audio to help diagnose hangs
        log_every = max(100, num_chunks // 10) if num_chunks > 1000 else 0
        for chunk_idx in range(num_chunks):
            if log_every > 0 and chunk_idx > 0 and chunk_idx % log_every == 0:
                progress_pct = (chunk_idx / num_chunks) * 100
                logger.info(f"ONNX VAD progress: {chunk_idx}/{num_chunks} chunks ({progress_pct:.1f}%)")
            chunk = audio[offset : offset + chunk_size]
            offset += chunk_size
            chunk = chunk.reshape(batch_size, -1)
            window = np.concatenate([context, chunk], axis=1).astype(np.float32, copy=False)
            feeds: dict[str, np.ndarray] = {self._onnx_input_name: window}
            if self._onnx_state_name:
                feeds[self._onnx_state_name] = state
            if self._onnx_sr_name:
                feeds[self._onnx_sr_name] = sr_array
            ort_outs = self.session.run(None, feeds)
            if not isinstance(ort_outs, (list, tuple)) or not ort_outs:
                continue
            logits = np.asarray(ort_outs[0], dtype=np.float32)
            logits = np.squeeze(logits)
            if logits.ndim == 0:
                logits = logits.reshape(1)
            if logits.ndim >= 1 and logits.shape[-1] == 2 and logits.size % 2 == 0:
                logits2d = logits.reshape(-1, 2)
                exps = np.exp(logits2d - np.max(logits2d, axis=1, keepdims=True))
                denom = np.sum(exps, axis=1, keepdims=True) + 1e-9
                probs = (exps[:, 1:2] / denom).reshape(-1)
            else:
                logits1d = logits.reshape(-1)
                probs = 1.0 / (1.0 + np.exp(-logits1d))
            probs = np.clip(probs, 0.0, 1.0)
            chunk_probs.extend(probs.tolist())
            if self._onnx_state_output_index is not None and self._onnx_state_output_index < len(
                ort_outs
            ):
                try:
                    state_out = np.asarray(
                        ort_outs[self._onnx_state_output_index], dtype=np.float32
                    )
                    state = state_out.reshape(tuple(state_shape))
                except Exception:
                    state = np.zeros(tuple(state_shape), dtype=np.float32)
            context = window[:, -context_size:]
            self._onnx_state_cache = state
            self._onnx_context_cache = context
        if not chunk_probs:
            return []
        chunk_probs = np.asarray(chunk_probs, dtype=np.float32)
        if chunk_probs.size > num_chunks:
            chunk_probs = chunk_probs[:num_chunks]
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

    # ------------------------------------------------------------------
    def detect(
        self, wav: np.ndarray, sr: int, min_speech_sec: float, min_silence_sec: float
    ) -> list[tuple[float, float]]:
        """Run VAD and return speech regions."""
        if self.session is not None:
            try:
                return self._detect_with_onnx(
                    wav,
                    sr,
                    min_speech_sec=min_speech_sec,
                    min_silence_sec=min_silence_sec,
                )
            except Exception as e:  # pragma: no cover - inference issues
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
        except Exception as e:  # pragma: no cover - inference issues
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
        """Load ECAPA ONNX model from an environment-defined or default path."""
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
                    # Also honor the documented bundle path: Diarization/ecapa-onnx/ecapa_tdnn.onnx
                    candidate_paths.extend(
                        list(
                            iter_model_subpaths(
                                Path("Diarization") / "ecapa-onnx" / "ecapa_tdnn.onnx"
                            )
                        )
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
        if self.session is None or not batch:
            return [None] * len(batch)
        try:
            # Compute log-mel spectrograms (frames x mel) for each clip
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
                # Apply per-feature CMVN (mean/var norm) to match SpeechBrain ECAPA input
                try:
                    m = mel.mean(axis=0, keepdims=True)
                    s = mel.std(axis=0, keepdims=True) + 1e-8
                    mel = (mel - m) / s
                except Exception as e:
                    logger.warning(f"CMVN normalization failed for clip {len(mel_specs)}: {e}")
                mel_specs.append(mel)
                if mel.shape[0] > max_frames:
                    max_frames = mel.shape[0]
            pad = np.zeros((len(batch), max_frames, mel_specs[0].shape[1]), dtype=np.float32)
            for i, mel in enumerate(mel_specs):
                pad[i, : mel.shape[0], :] = mel.astype(np.float32)
            pad = np.ascontiguousarray(pad)
            out = self.session.run([self.output_name], {self.input_name: pad})[0]
            arr = out.squeeze()
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            arr = arr.astype(np.float32, copy=False)
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
        """Return an ISO-8601 UTC timestamp without microseconds."""
        return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _load(self) -> None:
        """Load registry data, supporting legacy flat and metadata-wrapped schemas."""
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
                    # Legacy flat dictionary schema
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
        """Atomic registry save to prevent corruption."""
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
        # Ensure 1-D vectors for cosine similarity
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
        # Holds turns from the most recent diarization run
        self._last_turns: list[DiarizedTurn] = []

    def get_segment_embeddings(self) -> list[dict[str, Any]]:
        """Return turn embeddings from the last diarization run."""
        return [
            {"speaker": t.speaker, "embedding": t.embedding}
            for t in self._last_turns
            if t.embedding is not None
        ]

    def diarize_audio(self, wav: np.ndarray, sr: int) -> list[dict[str, Any]]:
        """Main diarization entry point."""
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
        duration_min = wav.size / sr / 60.0
        logger.info(f"Starting diarization: {duration_min:.1f} min audio at {sr} Hz")
        # Step 1: VAD
        logger.info("Step 1/5: Running Voice Activity Detection...")
        speech_regions = self.vad.detect(
            wav, sr, self.config.vad_min_speech_sec, self.config.vad_min_silence_sec
        )
        logger.info(f"VAD complete: {len(speech_regions)} speech regions detected")
        # Fallback VAD if Silero failed
        if not speech_regions and self.config.allow_energy_vad_fallback:
            logger.info("Using energy VAD fallback")
            speech_regions = _energy_vad_fallback(
                wav, sr, self.config.energy_gate_db, self.config.energy_hop_sec
            )
        if not speech_regions:
            logger.warning("No speech detected by VAD")
            return []
        # Step 2: Extract embeddings
        logger.info("Step 2/5: Extracting speaker embeddings...")
        windows = self._extract_embedding_windows(wav, sr, speech_regions)
        logger.info(f"Embedding extraction complete: {len(windows)} windows extracted")
        if len(windows) < 2:
            # Single speaker case
            turn = {
                "start": speech_regions[0][0],
                "end": speech_regions[-1][1],
                "speaker": "Speaker_1",
                "speaker_name": "Speaker_1",
                "embedding": windows[0]["embedding"] if windows else None,
            }
            return [turn]
        # Step 3: Cluster
        logger.info("Step 3/5: Clustering speaker embeddings...")
        embeddings = [w["embedding"] for w in windows if w["embedding"] is not None]
        if not embeddings:
            logger.warning("No valid embeddings extracted")
            return []
        # Safeguard: warn if excessive embeddings could cause very slow clustering
        if len(embeddings) > 10000:
            logger.warning(
                f"Very large number of embeddings ({len(embeddings)}). "
                f"Clustering may take a long time. Consider reducing embed_window_sec or increasing embed_shift_sec."
            )
        try:
            X = np.vstack(embeddings)
            logger.info(f"Clustering {X.shape[0]} embeddings (dim={X.shape[1]})...")
            backend = (self.config.clustering_backend or "ahc").strip().lower()
            labels = None
            if backend == "spectral" and _SpectralClusterer is not None:
                # Spectral clustering with optional min/max bounds.
                min_c = None
                max_c = None
                if self.config.speaker_limit and int(self.config.speaker_limit) > 0:
                    min_c = max_c = int(self.config.speaker_limit)
                else:
                    if self.config.min_speakers is not None:
                        min_c = int(self.config.min_speakers)
                    if self.config.max_speakers is not None:
                        max_c = int(self.config.max_speakers)
                try:
                    spec = _SpectralClusterer(
                        min_clusters=min_c if min_c is not None else 1,
                        max_clusters=max_c if max_c is not None else None,
                        p_percentile=0.90,
                        gaussian_blur_sigma=1.0,
                    )
                    labels = spec.fit_predict(X)
                    logger.info(
                        "Spectral clustering assigned %d clusters (min=%s max=%s)",
                        int(len(set(labels))), str(min_c), str(max_c),
                    )
                except Exception as e:  # fall back to AHC on failure
                    logger.info(f"Spectral clustering failed ({e}); falling back to AHC")
                    labels = None
            if labels is None:
                if backend == "spectral" and _SpectralClusterer is None:
                    logger.info("spectralcluster not installed; falling back to AHC")
                # Default AHC
                logger.info(f"Starting AHC clustering (n={X.shape[0]}, speaker_limit={self.config.speaker_limit})...")
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
                logger.info(f"AHC clustering complete: {len(set(labels))} clusters found")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            labels = np.zeros(len(embeddings), dtype=int)
        # Attach cluster labels to windows
        for w, label in zip(windows, labels, strict=False):
            w["speaker"] = f"Speaker_{label + 1}"
        # Step 4: FIXED - Build continuous turns
        logger.info("Step 4/5: Building continuous speaker segments...")
        turns = self._build_continuous_segments(windows, speech_regions)
        logger.info(f"Built {len(turns)} initial speaker turns")
        # Step 5: Post-process
        logger.info("Step 5/5: Post-processing and merging segments...")
        turns = self._merge_short_gaps(turns)
        # Reduce micro-turn fragmentation by enforcing a minimum turn duration
        turns = self._enforce_min_turn_duration(turns)
        # Merge highly-similar speaker clusters using centroid cosine similarity
        turns = self._merge_similar_speakers(turns)
        if self.config.single_speaker_collapse:
            collapsed, canonical, reason = collapse_single_speaker_turns(
                turns,
                dominance_threshold=self.config.single_speaker_dominance,
                centroid_threshold=self.config.single_speaker_centroid_threshold,
                min_turns=self.config.single_speaker_min_turns,
            )
            if collapsed:
                msg_reason = f" ({reason})" if reason else ""
                logger.info(
                    "Collapsing diarization clusters into single speaker '%s'%s",
                    canonical,
                    msg_reason,
                )
                turns = self._merge_short_gaps(turns)
        # Final pass to stitch adjacent segments after relabeling
        turns = self._merge_short_gaps(turns)
        turns = self._assign_speaker_names(turns)
        # Store for centroid updates
        self._last_turns = turns
        final_speakers = len(set(t.speaker for t in turns))
        logger.info(f"Diarization complete: {len(turns)} turns, {final_speakers} speakers identified")
        return [self._turn_to_dict(t) for t in turns]

    def _extract_embedding_windows(
        self, wav: np.ndarray, sr: int, speech_regions: list[tuple[float, float]]
    ) -> list[dict[str, Any]]:
        """Extract overlapping windows for embedding extraction (BATCHED)."""
        # Collect all candidate clips first to run a single (or few) batched
        # ECAPA ONNX inference calls. This avoids per-window session.run() overhead.
        clips: list[np.ndarray] = []
        meta: list[tuple[float, float]] = []  # (start, end)

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
                    meta.append((cursor, win_end))
                cursor += self.config.embed_shift_sec

        if not clips:
            return []

        # Optional micro-batching for memory control on very long files
        # Environment override: DIAREMOT_ECAPA_MAX_BATCH (int)
        try:
            max_batch = int(os.getenv("DIAREMOT_ECAPA_MAX_BATCH", "512"))
            if max_batch <= 0:
                max_batch = 512
        except Exception:
            max_batch = 512

        embeddings: list[np.ndarray | None] = []
        if len(clips) <= max_batch:
            logger.info(f"Processing {len(clips)} ECAPA embeddings in single batch...")
            embeddings = self.ecapa.embed_batch(clips, sr) or []
        else:
            # Chunk to keep RAM bounded
            logger.info(f"Processing {len(clips)} ECAPA embeddings in batches of {max_batch}...")
            for i in range(0, len(clips), max_batch):
                batch_num = (i // max_batch) + 1
                total_batches = (len(clips) + max_batch - 1) // max_batch
                logger.info(f"ECAPA batch {batch_num}/{total_batches} ({i}/{len(clips)} clips)")
                batch = clips[i : i + max_batch]
                part = self.ecapa.embed_batch(batch, sr) or []
                embeddings.extend(part)

        # Map embeddings back to window structures
        windows: list[dict[str, Any]] = []
        for idx, (m, emb) in enumerate(zip(meta, embeddings)):
            start_t, end_t = m
            windows.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "embedding": emb,
                    "speaker": None,
                    "region_idx": idx,
                }
            )

        try:
            logger.info(f"ECAPA embeddings: {len(windows)} windows batched (max_batch={max_batch})")
        except Exception:
            pass

        return windows

    def _build_continuous_segments(
        self,
        windows: list[dict[str, Any]],
        speech_regions: list[tuple[float, float]],
    ) -> list[DiarizedTurn]:
        """FIXED: Build continuous speaker segments from clustered windows."""
        if not windows:
            return []
        # Create timeline events from windows (currently unused but kept for reference)
        timeline = []
        for i, window in enumerate(windows):
            if window["embedding"] is None or window.get("speaker") is None:
                continue
            speaker_id = window["speaker"]
            timeline.append(
                {
                    "time": window["start"],
                    "type": "window_start",
                    "speaker": speaker_id,
                    "embedding": window["embedding"],
                    "window_idx": i,
                }
            )
            timeline.append(
                {
                    "time": window["end"],
                    "type": "window_end",
                    "speaker": speaker_id,
                    "window_idx": i,
                }
            )
        timeline.sort(key=lambda x: (x["time"], 0 if x["type"] == "window_end" else 1))
        # Build continuous segments by processing speech regions
        segments = []
        for region_start, region_end in speech_regions:
            # Find all windows that overlap this speech region
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
            # Sort by time and resolve overlaps by voting
            region_windows.sort(key=lambda x: x["start"])
            # Create timeline for this region
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
            # Build segments within this region
            active_speakers = {}  # speaker -> last_start_time
            current_time = region_start
            for event in events:
                event_time = event["time"]
                # If there's a gap, close current segments
                if event_time > current_time and active_speakers:
                    # Vote for dominant speaker in this time span
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
                        # Aggregate embeddings for windows belonging to this span
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
                # Process event
                if event["type"] == "start":
                    active_speakers[event["speaker"]] = event_time
                else:  # "end"
                    active_speakers.pop(event["speaker"], None)
                current_time = event_time
            # Close final segments in region
            if active_speakers and current_time < region_end:
                # Vote for final segment
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
        # Merge adjacent segments from same speaker
        if not segments:
            return []
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

    def _merge_similar_speakers(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        """Consolidate speaker labels whose centroids are very similar.

        Heuristic: compute per-speaker centroids from available turn embeddings,
        iteratively merge the closest pair whose cosine distance is below
        ``post_merge_distance_threshold``. Stop when no pair remains below the
        threshold or ``post_merge_min_speakers`` is reached. Only relabel turns;
        embeddings are not recomputed beyond centroid updates.
        """
        if not turns:
            return turns
        # Collect embeddings per speaker
        by_spk: dict[str, list[np.ndarray]] = {}
        for t in turns:
            if t.embedding is None:
                continue
            by_spk.setdefault(t.speaker, []).append(np.asarray(t.embedding, dtype=np.float32))
        if len(by_spk) <= 1:
            return turns
        # Compute centroids
        def centroid(vecs: list[np.ndarray]) -> np.ndarray:
            arr = np.vstack(vecs)
            c = arr.mean(axis=0)
            n = np.linalg.norm(c) + 1e-9
            return c / n
        centroids: dict[str, np.ndarray] = {k: centroid(v) for k, v in by_spk.items() if v}
        if len(centroids) <= 1:
            return turns
        # Merge loop
        def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
            return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        min_speakers = int(self.config.post_merge_min_speakers or 0)
        base_thresh = float(self.config.post_merge_distance_threshold or 0.0)
        if base_thresh <= 0.0:
            return turns
        dynamic_thresh = base_thresh
        if len(centroids) >= max(6, min_speakers + 5):
            fallback = float(getattr(self.config, "single_speaker_centroid_threshold", 0.0) or 0.0)
            if fallback > 0.0:
                dynamic_thresh = max(dynamic_thresh, min(1.0, fallback * 2.0))
            else:
                dynamic_thresh = max(dynamic_thresh, 0.40)
        thresh = dynamic_thresh
        changed = True
        while changed and len(centroids) > max(1, min_speakers):
            changed = False
            keys = list(centroids.keys())
            best_pair = None
            best_dist = 1e9
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    d = cosine_distance(centroids[keys[i]], centroids[keys[j]])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (keys[i], keys[j])
            if best_pair is None or best_dist >= thresh:
                break
            a, b = best_pair
            # Merge label 'b' into 'a'
            for t in turns:
                if t.speaker == b:
                    t.speaker = a
                    t.speaker_name = a
            # Update centroids
            if b in by_spk:
                by_spk.setdefault(a, []).extend(by_spk[b])
                by_spk.pop(b, None)
            if a in by_spk:
                centroids[a] = centroid(by_spk[a])
            centroids.pop(b, None)
            changed = True
            if len(centroids) <= max(1, min_speakers):
                break
        # Return relabeled turns
        return turns

    def _merge_short_gaps(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        """Merge turns from same speaker separated by short gaps."""
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

    def _enforce_min_turn_duration(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        """Merge very short turns into adjacent same-speaker segments.

        This conservatively reduces rapid speaker flips caused by borderline
        VAD or clustering noise while respecting speaker identity. Only merges
        a short turn with neighbours that share the same speaker label.
        """
        if not turns:
            return []
        out: list[DiarizedTurn] = []
        min_len = float(getattr(self.config, "min_turn_sec", 0.0) or 0.0)
        for idx, cur in enumerate(turns):
            duration = float(cur.end - cur.start)
            if duration >= min_len or min_len <= 0.0:
                out.append(cur)
                continue
            # Try to merge with previous if same speaker and contiguous/nearby
            merged = False
            if out:
                prev = out[-1]
                gap = max(0.0, cur.start - prev.end)
                if prev.speaker == cur.speaker and gap <= self.config.max_gap_to_merge_sec:
                    prev.end = max(prev.end, cur.end)
                    merged = True
            # Otherwise try to merge forward with next if same speaker
            if not merged and idx + 1 < len(turns):
                nxt = turns[idx + 1]
                gap = max(0.0, nxt.start - cur.end)
                if nxt.speaker == cur.speaker and gap <= self.config.max_gap_to_merge_sec:
                    nxt.start = min(nxt.start, cur.start)
                    merged = True
            if not merged:
                out.append(cur)
        return out

    def _assign_speaker_names(self, turns: list[DiarizedTurn]) -> list[DiarizedTurn]:
        """Assign speaker names using registry if available."""
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
        """Reassign speaker names after registry updates."""
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
        """Convert DiarizedTurn to dict format expected by core."""
        return {
            "start": turn.start,
            "end": turn.end,
            "speaker": turn.speaker,
            "speaker_name": turn.speaker_name,
            "candidate_name": turn.candidate_name,
            "needs_review": turn.needs_review,
            "embedding": (turn.embedding.tolist() if turn.embedding is not None else None),
        }
