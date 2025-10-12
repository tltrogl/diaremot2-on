"""This module is optional. If panns_inference/librosa are unavailable, it
degrades gracefully and returns no tags."""

from __future__ import annotations

import csv
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..io.onnx_runtime_guard import (
    OnnxRuntimeUnavailable,
    format_unavailable_message,
    onnxruntime_available,
)
from ..io.onnx_utils import create_onnx_session
from ..pipeline.runtime_env import DEFAULT_MODELS_ROOT, iter_model_roots

MODEL_ROOTS = tuple(iter_model_roots())
if not MODEL_ROOTS:
    MODEL_ROOTS = (DEFAULT_MODELS_ROOT,)

_PANNS_SUBDIR_CANDIDATES = ("sed_panns", "panns", "panns_cnn14")
DEFAULT_PANNS_MODEL_DIR = None
for _root in MODEL_ROOTS:
    for subdir in _PANNS_SUBDIR_CANDIDATES:
        candidate = Path(_root) / subdir
        if candidate.exists():
            DEFAULT_PANNS_MODEL_DIR = candidate
            break
    if DEFAULT_PANNS_MODEL_DIR is not None:
        break
if DEFAULT_PANNS_MODEL_DIR is None:
    DEFAULT_PANNS_MODEL_DIR = Path(MODEL_ROOTS[0]) / _PANNS_SUBDIR_CANDIDATES[0]

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - env dependent
    _HAVE_LIBROSA = False

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

_HAVE_ORT = onnxruntime_available()
if not _HAVE_ORT:
    logger.info("ONNXRuntime unavailable (%s)", format_unavailable_message())

try:
    # Cheap check without importing (avoids triggering wget in panns_inference)
    import importlib.util as _ilu  # type: ignore

    _HAVE_PANNS = _ilu.find_spec("panns_inference") is not None
except Exception:
    _HAVE_PANNS = False
AudioTagging = None  # type: ignore
labels = []  # type: ignore


_ONNX_FILENAME_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("cnn14.onnx", "labels.csv"),
    ("panns_cnn14.onnx", "audioset_labels.csv"),
    ("model.onnx", "class_labels_indices.csv"),
)


_NOISE_KEYWORDS = (
    "music",
    "noise",
    "typing",
    "keyboard",
    "applause",
    "traffic",
    "engine",
    "vehicle",
    "crowd",
    "siren",
    "wind",
    "rain",
    "tv",
)


@dataclass
class SEDConfig:
    top_k: int = 3
    run_on_suspect_only: bool = True
    min_duration_sec: float = 0.25
    # Prefer a local model directory by default for offline reliability
    model_dir: Path | None = DEFAULT_PANNS_MODEL_DIR
    # Sample 5 minutes uniformly across full duration for representative coverage
    max_eval_sec: float | None = 300.0
    eval_strategy: str = "uniform"  # "head" | "uniform"


class PANNSEventTagger:
    """Lightweight wrapper for PANNs AudioSet tagging on CPU.

    Prefers an ONNX Runtime model when available, falling back to the
    original `panns_inference` PyTorch implementation.
    - Accepts 16 kHz mono audio; resamples to 32 kHz if librosa available or
      uses simple upsampling fallbacks.
    - Returns top-K labels with scores and a coarse noise score.
    """

    def __init__(self, cfg: SEDConfig | None = None, backend: str = "auto"):
        self.cfg = cfg or SEDConfig()
        if self.cfg.model_dir is not None:
            self.cfg.model_dir = Path(self.cfg.model_dir)

        backend = (backend or "auto").lower()
        if backend == "auto":
            # Prefer ONNX Runtime when both backends are available
            if _HAVE_ORT:
                backend = "onnx"
            elif _HAVE_PANNS:
                backend = "pytorch"
            else:
                backend = "none"
        self.backend = backend
        self._tagger: AudioTagging | None = None  # type: ignore
        self._session: onnxruntime.InferenceSession | None = None
        self._labels: list[str] | None = None
        self.model_paths: tuple[Path, Path] | None = None
        self.available = True
        self._warned_missing = False
        self._ensure_model()
        if not self.available:
            logger.warning(
                "PANNs event tagging unavailable: neither ONNX nor PyTorch backend could be initialized"
            )

    def _empty_result(self) -> dict[str, Any]:
        return {"top": [], "dominant_label": None, "noise_score": 0.0}

    def _emit_missing_warning(self, reason: str) -> None:
        if self._warned_missing:
            return
        self._warned_missing = True
        logger.warning(
            "[sed] assets unavailable (%s); emitting empty background tags.",
            reason,
        )

    def _gather_candidate_pairs(self, base: Path) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for model_name, label_name in _ONNX_FILENAME_CANDIDATES:
            model_path = base / model_name
            label_path = base / label_name
            if model_path.exists() and label_path.exists():
                pairs.append((model_path, label_path))
        if pairs:
            return pairs
        for model_name, label_name in _ONNX_FILENAME_CANDIDATES:
            try:
                for model_path in base.rglob(model_name):
                    label_path = model_path.parent / label_name
                    if label_path.exists():
                        pairs.append((model_path, label_path))
            except Exception:
                continue
        return pairs

    def _load_onnx_assets(self, model_path: Path, labels_path: Path) -> bool:
        try:
            self._session = create_onnx_session(model_path)
            with labels_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                labels_local: list[str] = []
                key: str | None = None
                if reader.fieldnames:
                    if "display_name" in reader.fieldnames:
                        key = "display_name"
                    else:
                        key = reader.fieldnames[0]
                if key is not None:
                    for row in reader:
                        labels_local.append(str(row.get(key, "")))
                else:
                    handle.seek(0)
                    labels_local = [line.strip() for line in handle if line.strip()]
            if not labels_local:
                raise ValueError("labels file empty")
            self._labels = labels_local
            self.model_paths = (model_path, labels_path)
            return True
        except OnnxRuntimeUnavailable as exc:
            logger.info("Failed loading ONNX model %s: %s", model_path, exc)
        except Exception as exc:
            logger.info("Failed loading ONNX model %s: %s", model_path, exc)
        self._session = None
        self._labels = None
        self.model_paths = None
        return False

    def _ensure_model(self):
        if not self.available:
            return
        if self.backend == "onnx":
            if self._session is not None and self._labels is not None:
                return
            if not _HAVE_ORT:
                self.available = False
                self._emit_missing_warning("onnxruntime unavailable")
                return

            search_roots: list[Path] = []
            seen: set[str] = set()

            def _add_candidate(path_like: Any) -> None:
                if not path_like:
                    return
                path = Path(path_like).expanduser()
                key = str(path)
                if key in seen:
                    return
                seen.add(key)
                search_roots.append(path)

            _add_candidate(self.cfg.model_dir)
            _add_candidate(DEFAULT_PANNS_MODEL_DIR)
            for root in MODEL_ROOTS:
                _add_candidate(Path(root) / "sed_panns")
                _add_candidate(Path(root) / "panns")

            env_roots = [
                os.getenv("DIAREMOT_PANNS_DIR"),
                os.getenv("DIAREMOT_MODEL_DIR"),
                os.getenv("HF_HOME"),
                os.getenv("HUGGINGFACE_HUB_CACHE"),
                os.getenv("TRANSFORMERS_CACHE"),
            ]
            # Avoid scanning entire HF/Transformers cache roots; target the specific
            # PANNs repository cache directory to keep startup fast on large caches.
            for root in [Path(p).expanduser() for p in env_roots if p]:
                _add_candidate(root / "models--qiuqiangkong--panns-tagging-onnx")

            for base in search_roots:
                if not base.exists():
                    continue
                for model_path, labels_path in self._gather_candidate_pairs(base):
                    if self._load_onnx_assets(model_path, labels_path):
                        self.available = True
                        return

            self.available = False
            self._emit_missing_warning(
                "cnn14 ONNX assets not found under "
                + ", ".join(str(path) for path in search_roots)
            )
        elif self.backend == "pytorch":
            if self._tagger is not None:
                return
            if not _HAVE_PANNS:
                self.available = False
                self._emit_missing_warning("panns_inference unavailable")
                return
            # Ensure labels exist in HOME/panns_data to avoid wget in panns_inference
            try:
                home_panns = Path.home() / "panns_data"
                home_panns.mkdir(parents=True, exist_ok=True)
                src_labels = None
                if self.cfg.model_dir:
                    for _, label_name in _ONNX_FILENAME_CANDIDATES:
                        cand = Path(self.cfg.model_dir) / label_name
                        if cand.exists():
                            src_labels = cand
                            break
                if src_labels:
                    target = home_panns / Path(src_labels).name
                    if not target.exists():
                        target.write_bytes(Path(src_labels).read_bytes())
            except Exception:
                pass

            # Resolve checkpoint file path if available
            ckpt_path: Path | None = None
            if self.cfg.model_dir and Path(self.cfg.model_dir).exists():
                for name in [
                    "Cnn14_mAP=0.431.pth",
                    "Cnn14_mAP%3D0.431.pth",
                    "Cnn14_DecisionLevelMax.pth",
                ]:
                    cand = Path(self.cfg.model_dir) / name
                    if cand.exists():
                        ckpt_path = cand
                        break
                if ckpt_path is None:
                    # Any .pth in the directory
                    for cand in Path(self.cfg.model_dir).glob("*.pth"):
                        ckpt_path = cand
                        break

            try:
                # Lazy import after labels are in place to avoid wget
                from panns_inference import AudioTagging as _AT  # type: ignore

                # Suppress verbose prints from panns_inference (e.g., checkpoint path)
                @contextmanager
                def _suppress_stdout_stderr():
                    _old_out, _old_err = sys.stdout, sys.stderr
                    try:
                        with open(os.devnull, "w") as devnull:
                            sys.stdout = devnull
                            sys.stderr = devnull
                            yield
                    finally:
                        sys.stdout = _old_out
                        sys.stderr = _old_err

                with _suppress_stdout_stderr():
                    self._tagger = _AT(
                        checkpoint_path=str(ckpt_path) if ckpt_path else None,
                        device="cpu",
                    )
            except Exception as exc:
                logger.info("Failed initializing PyTorch backend: %s", exc)
                self._tagger = None

            if self._tagger is None:
                self.available = False
                self._emit_missing_warning("PyTorch checkpoint unavailable")
            else:
                self.available = True
        else:
            self.available = False
            self._emit_missing_warning(f"unsupported backend '{self.backend}'")

    def _resample_to_32k(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if sr == 32000:
            return audio.astype(np.float32, copy=False), sr
        if _HAVE_LIBROSA:
            try:
                y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=32000)
                return y.astype(np.float32), 32000
            except Exception:
                pass
        if sr == 16000:
            y = np.repeat(audio.astype(np.float32), 2)
            return y, 32000
        ratio = 32000 / float(sr)
        new_len = int(round(len(audio) * ratio))
        if new_len <= 1:
            return audio.astype(np.float32), sr
        x_old = np.linspace(0, len(audio) - 1, len(audio), dtype=np.float32)
        x_new = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
        y = np.interp(x_new, x_old, audio).astype(np.float32)
        return y, 32000

    def tag(self, audio_16k_mono: np.ndarray, sr: int) -> dict[str, Any]:
        if not self.available:
            return self._empty_result()
        if audio_16k_mono is None or audio_16k_mono.size == 0:
            return self._empty_result()
        if (len(audio_16k_mono) / max(1, sr)) < self.cfg.min_duration_sec:
            return self._empty_result()
        self._ensure_model()
        if not self.available:
            return self._empty_result()

        # Subsample/limit audio duration to keep SED fast on long files
        y_in = audio_16k_mono
        total_sec = len(y_in) / float(sr)
        if self.cfg.max_eval_sec and total_sec > self.cfg.max_eval_sec:
            max_samples = int(self.cfg.max_eval_sec * sr)
            if self.cfg.eval_strategy == "uniform" and (self.cfg.max_eval_sec or 0) > 0:
                # Take N uniform slices totaling max_eval_sec
                slices = 5
                seg_len = max_samples // slices
                step = len(y_in) // slices
                parts = []
                for i in range(slices):
                    start = i * step
                    end = min(start + seg_len, len(y_in))
                    parts.append(y_in[start:end])
                y_in = np.concatenate(parts) if parts else y_in[:max_samples]
            else:
                # Default: head slice
                y_in = y_in[:max_samples]

        y, sr32 = self._resample_to_32k(y_in, sr)

        if self.backend == "onnx":
            if self._session is None or not self._labels:
                self._emit_missing_warning("onnx session unavailable at inference")
                return self._empty_result()
            try:
                inp = self._session.get_inputs()[0].name
                clip = self._session.run(None, {inp: y[np.newaxis, :]})[0][0]
            except Exception as exc:
                logger.info("ONNX SED inference failed: %s", exc)
                return self._empty_result()
            map_labels = self._labels
        else:
            if self._tagger is None:
                self._emit_missing_warning("panns_inference tagger unavailable")
                return self._empty_result()
            try:
                # panns_inference expects shape [B, T] at 32 kHz and returns
                # (clipwise_output[B,527], embedding[B,2048]) as numpy arrays
                cw, _emb = self._tagger.inference(y[np.newaxis, :])  # type: ignore
                clip = np.asarray(cw[0], dtype=np.float32)
            except Exception as exc:
                logger.info("PyTorch SED inference failed: %s", exc)
                return self._empty_result()
            # Prefer labels from the tagger; fallback to imported default
            map_labels = getattr(self._tagger, "labels", labels) or []  # type: ignore

        if clip.size == 0:
            return self._empty_result()
        # If labels are missing or mismatched, synthesize generic labels
        if not map_labels or len(map_labels) != clip.size:
            try:
                n = int(clip.size)
            except Exception:
                n = 0
            map_labels = [f"class_{i}" for i in range(n)]
        top_idx = clip.argsort()[-self.cfg.top_k :][::-1]
        top = [{"label": str(map_labels[i]), "score": float(clip[i])} for i in top_idx]

        noise_score = 0.0
        for i, s in enumerate(clip):
            lab = str(map_labels[i]).lower()
            if any(k in lab for k in _NOISE_KEYWORDS):
                noise_score += float(s)

        return {
            "top": top,
            "dominant_label": top[0]["label"] if top else None,
            "noise_score": float(noise_score),
        }

    @property
    def labels(self) -> list[str] | None:
        if self._labels is None:
            return None
        return list(self._labels)
