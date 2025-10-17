"""Emotion analysis utilities (ONNX-first, HF fallback).

This module adheres to DiaRemot's ONNX-preferred architecture and CPU-only
constraint. It provides text emotion (GoEmotions 28), audio SER (8-class), and
V/A/D estimates, returning fields consumed by Stage 7.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..io.onnx_utils import create_onnx_session
from ..pipeline.runtime_env import DEFAULT_MODELS_ROOT
from .intent_defaults import INTENT_LABELS_DEFAULT

# Preprocessing: strictly librosa/scipy/numpy
try:
    import librosa  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    librosa = None  # type: ignore

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.asarray([], dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    denom = np.sum(e)
    if denom <= 0:
        return np.asarray([], dtype=np.float32)
    return e / denom


def _topk_distribution(
    scores: Mapping[str, float], *, k: int = 5
) -> list[dict[str, float]]:
    items = [
        (str(label), float(score))
        for label, score in scores.items()
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    ]
    items.sort(key=lambda item: item[1], reverse=True)
    limited = items[: max(0, min(k, len(items)))]
    return [
        {"label": label, "score": float(score)}
        for label, score in limited
    ]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _sr_target() -> int:
    # Keep consistent with PreprocessConfig.target_sr in AGENTS.md
    return 16000


def _ensure_16k_mono(y: np.ndarray, sr: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim > 1:
        arr = np.mean(arr, axis=-1)
    target_sr = _sr_target()
    if librosa is not None and sr != target_sr:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32)


def _trim_max_len(y: np.ndarray, *, sr: int, max_seconds: float = 20.0) -> np.ndarray:
    if max_seconds <= 0:
        return y
    limit = int(max_seconds * sr)
    if limit <= 0 or y.size <= limit:
        return y
    return np.asarray(y[:limit], dtype=np.float32)


def _entropy(probs: Sequence[float]) -> float:
    arr = np.asarray([float(p) for p in probs if float(p) > 0.0], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(-np.sum(arr * np.log(arr)))


def _norm_entropy(probs: Mapping[str, float]) -> float:
    values = [float(v) for v in probs.values() if float(v) > 0.0]
    if not values:
        return 0.0
    ent = _entropy(values)
    max_ent = math.log(len(values)) if len(values) > 1 else 0.0
    if max_ent <= 0.0:
        return 0.0
    return float(ent / max_ent)


def _top_margin(scores: Mapping[str, float]) -> float:
    values = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    arr = np.asarray(values, dtype=np.float32)
    idx = np.argpartition(arr, -2)[-2:]
    top_two = arr[idx]
    return float(np.max(top_two) - np.min(top_two))


def _canonical_label(label: str, labels: Sequence[str]) -> str:
    if not label:
        return labels[0] if labels else ""
    lower = label.lower()
    for candidate in labels:
        if candidate.lower() == lower:
            return candidate
    return labels[0] if labels else label


def _normalize_scores(scores: Mapping[str, float], labels: Sequence[str]) -> dict[str, float]:
    base: dict[str, float] = {label: 0.0 for label in labels}
    for key, value in scores.items():
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score) or score < 0.0:
            continue
        canonical = _canonical_label(str(key), labels)
        base[canonical] = score
    total = float(sum(base.values()))
    if total > 0.0:
        return {label: float(score / total) for label, score in base.items()}
    if labels:
        base[labels[0]] = 1.0
    return base


def _ser_low_confidence(scores: Mapping[str, float]) -> bool:
    if not scores:
        return True
    values = [float(v) for v in scores.values() if math.isfinite(float(v))]
    if not values:
        return True
    top_score = max(values)
    margin = _top_margin(scores)
    entropy = _norm_entropy({k: float(v) for k, v in scores.items()})
    return top_score < 0.55 or margin < 0.18 or entropy > 0.85


def _normalize_intent_label(label: str) -> str:
    clean = (label or "status_update").strip().lower().replace(" ", "_")
    if clean == "status_update":
        clean = "status"
    clean = clean.replace("_", "-")
    return clean or "status"


# GoEmotions 28 labels (SamLowe/roberta-base-go_emotions)
GOEMOTIONS_LABELS: list[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


# Default 8-class SER labels (common mapping; can be overridden by model-specific labels)
SER8_LABELS: list[str] = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli"


def _resolve_model_dir() -> Path:
    d = os.environ.get("DIAREMOT_MODEL_DIR")
    if d:
        return Path(d).expanduser()
    return Path(DEFAULT_MODELS_ROOT)


def _resolve_component_dir(
    cli_value: str | None, env_key: str, *default_subpath: str
) -> Path:
    candidates: list[Path] = []
    if cli_value:
        candidates.append(Path(cli_value).expanduser())
    env_value = os.getenv(env_key)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    model_root = _resolve_model_dir()
    if default_subpath:
        candidates.append(model_root.joinpath(*default_subpath))
    else:
        candidates.append(model_root)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return candidates[0]


def _select_first_existing(directory: Path, names: Sequence[str]) -> Path:
    if directory.is_file() or directory.suffix.lower() == ".onnx":
        return directory
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return directory / names[0]


def _normalize_backend(value: str | None) -> str:
    if not value:
        return "auto"
    normalized = value.lower()
    if normalized not in {"auto", "onnx", "torch"}:
        return "auto"
    return normalized


def _intent_dir_has_assets(path: Path) -> bool:
    """Return True if ``path`` contains a valid local intent model bundle.

    Rules (ONNX-first and stricter to avoid mis-detecting the model root):
    - Prefer ONNX presence in this directory (model_uint8.onnx | model_int8.onnx | model.onnx).
    - Otherwise require a Transformers-style config with a valid ``model_type`` AND a tokenizer
      (tokenizer.json OR both vocab.json and merges.txt) colocated in the same directory.
    """

    if not path.exists() or not path.is_dir():
        return False

    has_onnx = any((path / name).exists() for name in ("model_uint8.onnx", "model_int8.onnx", "model.onnx"))

    cfg_path = path / "config.json"
    if not cfg_path.exists():
        return False

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        model_type = cfg.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            return False
    except Exception:
        return False

    tokenizer_present = (path / "tokenizer.json").exists() or (
        (path / "vocab.json").exists() and (path / "merges.txt").exists()
    )
    if not tokenizer_present:
        return False

    # If ONNX exports are present we require them to live alongside the tokenizer/config combo.
    if has_onnx:
        return True

    # Allow pure Transformers directories that rely on torch fallback.
    return True


def _intent_candidate_dirs(explicit: str | None) -> Iterable[Path]:
    candidates: list[Path] = []

    def _add(candidate: str | Path | None) -> None:
        if not candidate:
            return
        path = Path(candidate).expanduser()
        candidates.append(path)

    _add(explicit)
    _add(os.getenv("DIAREMOT_INTENT_MODEL_DIR"))

    model_root = os.getenv("DIAREMOT_MODEL_DIR")
    if model_root:
        root = Path(model_root).expanduser()
        _add(root)
        _add(root / "intent")
        _add(root / "bart")
        _add(root / "bart-large-mnli")
        _add(root / "facebook" / "bart-large-mnli")
        _add(root / "bart" / "facebook" / "bart-large-mnli")

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _resolve_intent_model_dir(explicit: str | None) -> str | None:
    # Explicit CLI value wins when valid, otherwise do not fall back silently
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists() and _intent_dir_has_assets(path):
            return str(path)
        return None

    # If an explicit environment override is present, require it to be valid
    env_override = os.getenv("DIAREMOT_INTENT_MODEL_DIR")
    if env_override:
        env_path = Path(env_override).expanduser()
        if _intent_dir_has_assets(env_path):
            return str(env_path)
        return None

    # Otherwise search defaults under the configured model roots
    for candidate in _intent_candidate_dirs(None):
        if _intent_dir_has_assets(candidate):
            return str(candidate)
    return None


def _find_label_index(id2label: dict[int, str], target: str) -> int | None:
    target_lower = target.lower()
    for idx, label in id2label.items():
        if str(label).lower() == target_lower:
            return int(idx)
    return None


def _ort_session(path: str):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"onnxruntime not available: {exc}") from exc

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # DiaRemot is CPU-only per AGENTS.md. Do not use GPU providers.
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, sess_options=sess_options, providers=providers)


def _maybe_import_transformers_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        return pipeline
    except Exception:
        return None


@dataclass
class EmotionOutputs:
    """Serialized affect outputs for storage layers (CSV/JSON)."""

    # Numeric affect (if available)
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    # Audio SER
    emotion_top: str = "neutral"
    emotion_scores_json: str = _json({})
    low_confidence_ser: bool = True

    # Text emotions (GoEmotions)
    text_emotions_top5_json: str = _json([])
    text_emotions_full_json: str = _json({})

    # Intent & hint
    intent_top: str = "status_update"
    intent_top3_json: str = _json([])
    affect_hint: str = "neutral-status"

    @classmethod
    def from_affect(cls, payload: Mapping[str, Any]) -> EmotionOutputs:
        def _safe_float(value: Any) -> float:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not math.isfinite(num):
                return 0.0
            return float(num)

        vad = payload.get("vad", {}) if isinstance(payload, Mapping) else {}
        ser = payload.get("speech_emotion", {}) if isinstance(payload, Mapping) else {}
        text = payload.get("text_emotions", {}) if isinstance(payload, Mapping) else {}
        intent = payload.get("intent", {}) if isinstance(payload, Mapping) else {}
        return cls(
            valence=_safe_float(vad.get("valence", 0.0)),
            arousal=_safe_float(vad.get("arousal", 0.0)),
            dominance=_safe_float(vad.get("dominance", 0.0)),
            emotion_top=str(ser.get("top", "neutral")),
            emotion_scores_json=_json(ser.get("scores_8class", {})),
            low_confidence_ser=bool(ser.get("low_confidence_ser", False)),
            text_emotions_top5_json=_json(text.get("top5", [])),
            text_emotions_full_json=_json(text.get("full_28class", {})),
            intent_top=str(intent.get("top", "status_update")),
            intent_top3_json=_json(intent.get("top3", [])),
            affect_hint=str(payload.get("affect_hint", "neutral-status")),
        )

    def to_affect(self) -> dict[str, Any]:
        def _loads(data: str, default: Any) -> Any:
            try:
                return json.loads(data)
            except (TypeError, json.JSONDecodeError):
                return default

        return {
            "vad": {
                "valence": self.valence,
                "arousal": self.arousal,
                "dominance": self.dominance,
            },
            "speech_emotion": {
                "top": self.emotion_top,
                "scores_8class": _loads(self.emotion_scores_json, {}),
                "low_confidence_ser": bool(self.low_confidence_ser),
            },
            "text_emotions": {
                "top5": _loads(self.text_emotions_top5_json, []),
                "full_28class": _loads(self.text_emotions_full_json, {}),
            },
            "intent": {
                "top": self.intent_top,
                "top3": _loads(self.intent_top3_json, []),
            },
            "affect_hint": self.affect_hint,
        }


@dataclass
class TextEmotionResult:
    top5: list[dict[str, float]]
    full: dict[str, float]


@dataclass
class SpeechEmotionResult:
    top: str
    scores: dict[str, float]
    low_confidence: bool


@dataclass
class VadEmotionResult:
    valence: float
    arousal: float
    dominance: float


@dataclass
class IntentResult:
    top: str
    top3: list[dict[str, float]]


def _default_text_result() -> TextEmotionResult:
    base = {label: 0.0 for label in GOEMOTIONS_LABELS}
    base["neutral"] = 1.0
    return TextEmotionResult(
        top5=[{"label": "neutral", "score": 1.0}],
        full=base,
    )


def _default_speech_result() -> SpeechEmotionResult:
    base = {label: 0.0 for label in SER8_LABELS}
    base["neutral"] = 1.0
    return SpeechEmotionResult(top="neutral", scores=base, low_confidence=True)


def _default_vad_result() -> VadEmotionResult:
    return VadEmotionResult(valence=0.0, arousal=0.0, dominance=0.0)


def _default_intent_result() -> IntentResult:
    return IntentResult(
        top="status_update",
        top3=[
            {"label": "status_update", "score": 1.0},
            {"label": "small_talk", "score": 0.0},
            {"label": "opinion", "score": 0.0},
        ],
    )


class OnnxTextEmotion:
    def __init__(
        self,
        model_path: str,
        labels: list[str] = GOEMOTIONS_LABELS,
        *,
        tokenizer_source: str | os.PathLike[str] | None = None,
        disable_downloads: bool = False,
    ):
        self.labels = labels
        self.sess = _ort_session(model_path)
        self.tokenizer = self._load_tokenizer(
            model_path,
            tokenizer_source=tokenizer_source,
            disable_downloads=disable_downloads,
        )

    def _load_tokenizer(
        self,
        model_path: str,
        *,
        tokenizer_source: str | os.PathLike[str] | None,
        disable_downloads: bool,
    ):
        from transformers import AutoTokenizer  # type: ignore

        candidates: list[tuple[str, dict[str, object]]] = []
        errors: list[str] = []

        if tokenizer_source:
            local_dir = Path(tokenizer_source).expanduser()
        else:
            local_dir = Path(model_path).expanduser().parent

        local_dir_str = os.fspath(local_dir)
        candidates.append((local_dir_str, {"local_files_only": True}))
        if not disable_downloads:
            candidates.append((local_dir_str, {"local_files_only": False}))
            candidates.append(("SamLowe/roberta-base-go_emotions", {}))

        for identifier, kwargs in candidates:
            try:
                return AutoTokenizer.from_pretrained(identifier, **kwargs)
            except Exception as exc:  # noqa: BLE001 - HF backend specific
                errors.append(f"{identifier}: {exc}")

        details = "; ".join(errors)
        raise RuntimeError(
            "Unable to load text emotion tokenizer; attempted candidates: " + details
        )

    def __call__(self, text: str) -> dict[str, float]:
        enc = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        inputs = {self.sess.get_inputs()[0].name: enc["input_ids"].astype(np.int64)}
        # Handle attention_mask if present
        if len(self.sess.get_inputs()) > 1 and "attention_mask" in enc:
            inputs[self.sess.get_inputs()[1].name] = enc["attention_mask"].astype(np.int64)
        # Optional token_type_ids
        if len(self.sess.get_inputs()) > 2 and "token_type_ids" in enc:
            inputs[self.sess.get_inputs()[2].name] = enc["token_type_ids"].astype(np.int64)

        out = self.sess.run(None, inputs)
        logits = out[0]
        if logits.ndim == 2:
            logits = logits[0]
        probs = _softmax(logits.astype(np.float32))
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}


class HfTextEmotionFallback:
    def __init__(self):
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            raise RuntimeError("transformers pipeline() unavailable for fallback")
        self.pipe = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
        )

    def __call__(self, text: str) -> dict[str, float]:
        out = self.pipe(text)[0]
        # HF returns list of dicts with 'label' and 'score'
        full = {d["label"].lower(): float(d["score"]) for d in out}
        # Ensure all 28 labels exist
        for lab in GOEMOTIONS_LABELS:
            full.setdefault(lab, 0.0)
        arr = np.array([full[lab] for lab in GOEMOTIONS_LABELS], dtype=np.float32)
        arr = arr / (arr.sum() + 1e-8)
        return {lab: float(arr[i]) for i, lab in enumerate(GOEMOTIONS_LABELS)}


class OnnxAudioEmotion:
    def __init__(self, model_path: str, labels: list[str] = SER8_LABELS):
        path = Path(model_path).expanduser()
        self.model_path = os.fspath(path)
        self.labels = list(labels)
        self.sess = _ort_session(self.model_path)
        input_meta = self.sess.get_inputs()[0]
        self._input_name = input_meta.name
        shape = input_meta.shape or []
        self._input_rank = len(shape)
        get_providers = getattr(self.sess, "get_providers", None)
        providers = get_providers() if callable(get_providers) else ["CPUExecutionProvider"]
        self._providers = list(providers)
        logger.info(
            "Audio SER ONNX ready via providers=%s path=%s",
            self._providers,
            self.model_path,
        )

    def _as_waveform_input(self, y: np.ndarray) -> dict[str, np.ndarray]:
        arr = y.astype(np.float32)
        if self._input_rank <= 1:
            return {self._input_name: arr}
        if self._input_rank == 2:
            return {self._input_name: arr[None, :]}
        if self._input_rank == 3:
            return {self._input_name: arr[None, None, :]}
        # Let caller decide whether to fall back to mel
        raise RuntimeError("Waveform input unsupported for rank>=4")

    def _as_mel_input(self, y: np.ndarray) -> dict[str, np.ndarray] | None:
        if librosa is None:
            return None
        sr = _sr_target()
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
        x = mel_db.astype(np.float32)[None, None, :, :]  # [1,1,64,frames]
        return {self._input_name: x}

    def __call__(self, y: np.ndarray, sr: int) -> tuple[str, dict[str, float]]:
        y = _ensure_16k_mono(y, sr)
        y = _trim_max_len(y, sr=_sr_target(), max_seconds=20.0)
        out = None
        if self._input_rank < 4:
            try:
                inputs = self._as_waveform_input(y)
                out = self.sess.run(None, inputs)
            except Exception:  # noqa: BLE001 - onnxruntime errors vary by build
                out = None
        if out is None:
            mel_inputs = self._as_mel_input(y)
            if mel_inputs is None:
                raise RuntimeError("Mel spectrogram path unavailable (librosa missing)")
            out = self.sess.run(None, mel_inputs)

        logits = np.asarray(out[0], dtype=np.float32).reshape(-1)
        probs = _softmax(logits)
        label_count = len(self.labels)
        if probs.shape[0] != label_count:
            logger.warning(
                "Audio SER logits mismatch (got=%s expected=%s) for %s",
                probs.shape[0],
                label_count,
                self.model_path,
            )
            adjusted = np.zeros(label_count, dtype=np.float32)
            limit = min(label_count, probs.shape[0])
            adjusted[:limit] = probs[:limit]
            probs = adjusted / (np.sum(adjusted) + 1e-6)
        full = {self.labels[i]: float(probs[i]) for i in range(label_count)}
        scores = _normalize_scores(full, self.labels)
        top_label = max(scores.items(), key=lambda kv: kv[1])[0]
        return top_label, scores


class OnnxVADEmotion:
    def __init__(self, model_path: str):
        self.sess = _ort_session(model_path)

    def __call__(self, y: np.ndarray, sr: int) -> tuple[float, float, float]:
        # Keep simple; feed pooled features or raw waveform
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        target_sr = _sr_target()
        if librosa is not None and sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        # Model expects rank-2 [batch, time]; feed [1, N]
        x = y.astype(np.float32)[None, :]
        inp_name = self.sess.get_inputs()[0].name
        out = self.sess.run(None, {inp_name: x})
        arr = np.array(out[0]).astype(np.float32).ravel()
        # Expect [3] -> V, A, D; clip to [-1,1]
        if arr.size >= 3:
            v, a, d = arr[:3]
        else:
            v = a = d = 0.0
        v = float(np.clip(v, -1.0, 1.0))
        a = float(np.clip(a, -1.0, 1.0))
        d = float(np.clip(d, -1.0, 1.0))
        return v, a, d


class EmotionAnalyzer:
    """
    Emotion analyzer using ONNX backends for text, speech emotion, and V/A/D outputs.

    Produces fields required by Stage 7 (affect_and_assemble):
    - valence, arousal, dominance
    - emotion_top, emotion_scores_json, low_confidence_ser
    - text_emotions_top5_json, text_emotions_full_json
    - intent_top, intent_top3_json, affect_hint
    """

    def __init__(
        self,
        model_dir: str | None = None,
        disable_downloads: bool | None = None,
        *,
        text_model_dir: str | None = None,
        ser_model_dir: str | None = None,
        vad_model_dir: str | None = None,
    ):
        base_dir = Path(model_dir).expanduser() if model_dir else _resolve_model_dir()
        self.model_dir = str(base_dir)
        self.disable_downloads = bool(disable_downloads or False)
        self.issues: list[str] = []

        self.text_model_dir = _resolve_component_dir(
            text_model_dir, "DIAREMOT_TEXT_EMO_MODEL_DIR", "text_emotions"
        )
        self.ser_model_dir = _resolve_component_dir(
            ser_model_dir, "AFFECT_SER_MODEL_DIR", "affect", "ser8"
        )
        self.vad_model_dir = _resolve_component_dir(
            vad_model_dir, "AFFECT_VAD_DIM_MODEL_DIR", "affect", "vad_dim"
        )

        # Paths
        self.path_text_onnx = str(
            _select_first_existing(
                self.text_model_dir,
                (
                    "model.int8.onnx",
                    "model.onnx",
                    "roberta-base-go_emotions.onnx",
                ),
            )
        )
        self.path_ser8_onnx = str(
            _select_first_existing(
                self.ser_model_dir,
                (
                    "model.int8.onnx",
                    "ser8.int8.onnx",
                    "model.onnx",
                    "ser_8class.onnx",
                ),
            )
        )
        self.path_vad_onnx = str(
            _select_first_existing(
                self.vad_model_dir,
                ("model.onnx", "vad_model.onnx"),
            )
        )

        # Try ONNX-only for each component (lazily initialised)
        self._text_model: OnnxTextEmotion | None = None
        self._text_fallback: HfTextEmotionFallback | None = None
        self._audio_model: OnnxAudioEmotion | None = None
        self._vad_model: OnnxVADEmotion | None = None

        # Allow explicit override from env (exported ONNX path)
        env_ser = os.getenv("DIAREMOT_SER_ONNX")
        if env_ser:
            self.path_ser8_onnx = os.fspath(Path(env_ser).expanduser())

    # Initialize lazily upon first use to avoid import overhead when unused

    # ---- Lazy initializers ----
    def _record_issue(self, message: str) -> None:
        if message not in self.issues:
            self.issues.append(message)

    def _ensure_text_model(self):
        if self._text_model is not None or self._text_fallback is not None:
            return
        try:
            self._text_model = OnnxTextEmotion(
                self.path_text_onnx,
                tokenizer_source=self.text_model_dir,
                disable_downloads=self.disable_downloads,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Text emotion ONNX unavailable: %s", exc)
            self._record_issue(
                f"Text emotion ONNX missing under {self.text_model_dir}"
            )
            if self.disable_downloads:
                self._text_fallback = None
            else:
                try:
                    self._text_fallback = HfTextEmotionFallback()
                    logger.warning("Using HuggingFace fallback for text emotion.")
                except Exception as fb_exc:  # noqa: BLE001
                    logger.warning("HF fallback unavailable: %s", fb_exc)
                    self._text_fallback = None
                    self._record_issue("Text emotion fallback unavailable; outputs neutral")

    def _ensure_audio_model(self):
        if self._audio_model is not None:
            return
        try:
            self._audio_model = OnnxAudioEmotion(self.path_ser8_onnx, labels=SER8_LABELS)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Audio SER ONNX unavailable: %s", exc)
            self._audio_model = None
        if self._audio_model is None:
            self._record_issue(
                f"Speech emotion ONNX unavailable at {self.path_ser8_onnx}"
            )

    def _ensure_vad_model(self):
        if self._vad_model is not None:
            return
        try:
            self._vad_model = OnnxVADEmotion(self.path_vad_onnx)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("V/A/D ONNX unavailable: %s", exc)
            self._vad_model = None
            self._record_issue(
                f"Valence/arousal/dominance model unavailable under {self.vad_model_dir}"
            )

    # ---- Public API ----
    def analyze_text(self, text: str) -> TextEmotionResult:
        clean = (text or "").strip()
        default_text = _default_text_result()
        if not clean:
            return default_text
        self._ensure_text_model()
        backend = self._text_model or self._text_fallback
        if backend is None:
            self._record_issue("Text emotion model unavailable; using neutral distribution")
            return default_text
        try:
            raw_scores = backend(clean)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Text emotion inference failed: %s", exc)
            self._record_issue(f"Text emotion inference failed: {exc}")
            return default_text

        normalized = _normalize_scores(raw_scores, GOEMOTIONS_LABELS)
        top5 = _topk_distribution(normalized, k=5)
        if not top5:
            normalized = dict(default_text.full)
            top5 = [dict(item) for item in default_text.top5]
        return TextEmotionResult(top5=top5, full=normalized)

    def analyze_audio(self, y: np.ndarray | None, sr: int | None) -> SpeechEmotionResult:
        if y is None or sr is None:
            return _default_speech_result()
        self._ensure_audio_model()
        if self._audio_model is None:
            self._record_issue("Speech emotion model unavailable; using neutral distribution")
            return _default_speech_result()
        try:
            top_label, raw_scores = self._audio_model(y, sr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audio SER inference failed: %s", exc)
            self._record_issue(f"Speech emotion inference failed: {exc}")
            return _default_speech_result()

        normalized = _normalize_scores(raw_scores, SER8_LABELS)
        top = _canonical_label(top_label, SER8_LABELS)
        if normalized.get(top, 0.0) <= 0.0:
            top = max(normalized.items(), key=lambda item: item[1])[0]
        low_conf = _ser_low_confidence(normalized)
        return SpeechEmotionResult(top=top, scores=normalized, low_confidence=low_conf)

    def analyze_vad_emotion(self, y: np.ndarray | None, sr: int | None) -> VadEmotionResult:
        if y is None or sr is None:
            return _default_vad_result()
        self._ensure_vad_model()
        if self._vad_model is None:
            self._record_issue("Valence/arousal/dominance model unavailable; using zeros")
            return _default_vad_result()
        try:
            v, a, d = self._vad_model(y, sr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("V/A/D inference failed: %s", exc)
            self._record_issue(f"VAD emotion inference failed: {exc}")
            return _default_vad_result()
        return VadEmotionResult(valence=float(v), arousal=float(a), dominance=float(d))

    def _make_affect_hint(self, vad: VadEmotionResult, intent_top: str) -> str:
        intent = _normalize_intent_label(intent_top)
        v = float(vad.valence)
        a = float(vad.arousal)
        if not math.isfinite(v) or not math.isfinite(a):
            return f"neutral-{intent}"
        if a > 0.55 and v < 0.0:
            return "agitated-negative"
        if a < 0.30 and v > 0.25:
            return "calm-positive"
        if v > 0.50:
            return f"bright-{intent}"
        if v < -0.35:
            return f"low-{intent}"
        return f"neutral-{intent}"

    def _build_affect_payload(
        self,
        *,
        text_res: TextEmotionResult,
        speech_res: SpeechEmotionResult,
        vad_res: VadEmotionResult,
        intent_res: IntentResult,
    ) -> dict[str, Any]:
        from ..pipeline.outputs import default_affect

        payload = default_affect()
        payload["vad"] = {
            "valence": vad_res.valence,
            "arousal": vad_res.arousal,
            "dominance": vad_res.dominance,
        }
        payload["speech_emotion"] = {
            "top": speech_res.top,
            "scores_8class": dict(speech_res.scores),
            "low_confidence_ser": speech_res.low_confidence,
        }
        payload["text_emotions"] = {
            "top5": [dict(item) for item in text_res.top5],
            "full_28class": dict(text_res.full),
        }
        payload["intent"] = {
            "top": intent_res.top,
            "top3": [dict(item) for item in intent_res.top3],
        }
        payload["affect_hint"] = self._make_affect_hint(vad_res, intent_res.top)
        return payload

    def _analyze_components(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> tuple[TextEmotionResult, SpeechEmotionResult, VadEmotionResult]:
        text_res = self.analyze_text(text)
        speech_res = self.analyze_audio(wav, sr)
        vad_res = self.analyze_vad_emotion(wav, sr)
        return text_res, speech_res, vad_res

    def analyze(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> dict[str, Any]:
        text_res, speech_res, vad_res = self._analyze_components(
            wav=wav, sr=sr, text=text
        )
        intent_res = _default_intent_result()
        return self._build_affect_payload(
            text_res=text_res,
            speech_res=speech_res,
            vad_res=vad_res,
            intent_res=intent_res,
        )

    def analyze_segment(
        self, text: str, audio_wave: np.ndarray | None, sr: int | None
    ) -> EmotionOutputs:
        """Analyze a single segment (text + audio) and serialize for storage."""

        payload = self.analyze(wav=audio_wave, sr=sr, text=text or "")
        return EmotionOutputs.from_affect(payload)


__all__ = (
    "EmotionAnalyzer",
    "EmotionOutputs",
    "GOEMOTIONS_LABELS",
    "SER8_LABELS",
)


# Back-compat alias expected by orchestrator with extended intent controls
class EmotionIntentAnalyzer(EmotionAnalyzer):
    def __init__(
        self,
        *,
        text_emotion_model: str = "SamLowe/roberta-base-go_emotions",
        intent_labels: Sequence[str] | None = None,
        affect_backend: str | None = None,
        affect_text_model_dir: str | None = None,
        affect_ser_model_dir: str | None = None,
        affect_vad_model_dir: str | None = None,
        affect_intent_model_dir: str | None = None,
        analyzer_threads: int | None = None,
        disable_downloads: bool | None = None,
        model_dir: str | None = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            disable_downloads=disable_downloads,
            text_model_dir=affect_text_model_dir,
            ser_model_dir=affect_ser_model_dir,
            vad_model_dir=affect_vad_model_dir,
        )

        self.text_emotion_model = text_emotion_model
        labels = intent_labels or INTENT_LABELS_DEFAULT
        self.intent_labels: list[str] = [str(label) for label in labels]
        self.affect_backend = _normalize_backend(affect_backend)
        self.analyzer_threads = analyzer_threads

        self.affect_text_model_dir = os.fspath(self.text_model_dir)
        self.affect_ser_model_dir = os.fspath(self.ser_model_dir)
        self.affect_vad_model_dir = os.fspath(self.vad_model_dir)

        self.affect_intent_model_dir = _resolve_intent_model_dir(affect_intent_model_dir)

        self._intent_session: object | None = None
        self._intent_tokenizer: Callable[..., dict[str, np.ndarray]] | None = None
        self._intent_config: object | None = None
        self._intent_pipeline: Callable[..., object] | None = None
        self._intent_entail_idx: int | None = None
        self._intent_contra_idx: int | None = None
        self._intent_hypothesis_template: str = "This example is {}."

    # ---- Intent helpers ----
    def _lazy_intent(self) -> None:
        backend = self.affect_backend
        if backend == "onnx":
            self._ensure_intent_onnx(strict=False)
        elif backend == "torch":
            self._ensure_intent_pipeline()
        else:
            if not self._ensure_intent_onnx(strict=False):
                self._ensure_intent_pipeline()

    def _select_onnx_model(self, model_dir: Path) -> Path | None:
        for name in ("model_uint8.onnx", "model.onnx"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        remaining = list(model_dir.glob("*.onnx"))
        return remaining[0] if remaining else None

    def _ensure_intent_onnx(self, *, strict: bool) -> bool:
        if self._intent_session is not None and self._intent_tokenizer is not None:
            return True

        model_dir_str = self.affect_intent_model_dir
        if not model_dir_str:
            if strict:
                logger.warning("Intent ONNX backend requested but no model directory is configured")
            self._record_issue("Intent model directory not configured")
            return False

        model_dir = Path(model_dir_str)
        model_path = self._select_onnx_model(model_dir)
        if model_path is None:
            if strict:
                logger.warning("Intent ONNX backend missing model.onnx in %s", model_dir)
            self._record_issue(f"Intent ONNX model missing under {model_dir}")
            return False

        threads = self.analyzer_threads or 1
        try:
            self._intent_session = create_onnx_session(model_path, threads=threads)
        except Exception as exc:  # pragma: no cover - runtime dependent
            logger.warning("Intent ONNX session unavailable: %s", exc)
            self._intent_session = None
            self._record_issue(f"Intent ONNX session unavailable: {exc}")
            return False

        try:
            from transformers import AutoConfig, AutoTokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            logger.warning("transformers unavailable for intent tokenizer: %s", exc)
            self._intent_session = None
            self._record_issue("Transformers package missing for intent tokenizer")
            return False

        try:
            # Prefer local tokenizer/config colocated with ONNX model
            self._intent_tokenizer = AutoTokenizer.from_pretrained(model_dir_str, local_files_only=True)
            self._intent_config = AutoConfig.from_pretrained(model_dir_str, local_files_only=True)
        except Exception as exc_local:  # noqa: BLE001 - environment dependent
            logger.warning("Intent tokenizer/config not found locally: %s", exc_local)
            self._record_issue(
                f"Intent tokenizer/config unavailable under {model_dir_str}: {exc_local}"
            )
            # If downloads are permitted, fall back to a known compatible config/tokenizer
            if not self.disable_downloads:
                try:
                    hf_id = DEFAULT_INTENT_MODEL  # e.g., facebook/bart-large-mnli
                    self._intent_tokenizer = AutoTokenizer.from_pretrained(hf_id)
                    self._intent_config = AutoConfig.from_pretrained(hf_id)
                    logger.warning("Using HuggingFace fallback for intent tokenizer/config: %s", hf_id)
                except Exception as exc_remote:  # noqa: BLE001
                    logger.warning("Intent fallback tokenizer/config unavailable: %s", exc_remote)
                    self._intent_session = None
                    self._intent_tokenizer = None
                    self._intent_config = None
                    self._record_issue(
                        f"Intent tokenizer/config fallback failed: {exc_remote}"
                    )
                    return False
            else:
                # Downloads disabled and no local assets
                self._intent_session = None
                self._intent_tokenizer = None
                self._intent_config = None
                return False

        id2label_raw = getattr(self._intent_config, "id2label", {})
        if isinstance(id2label_raw, dict):
            id2label = {int(k): str(v) for k, v in id2label_raw.items()}
        else:
            id2label = {int(idx): str(label) for idx, label in enumerate(id2label_raw)}
        self._intent_entail_idx = _find_label_index(id2label, "entailment")
        self._intent_contra_idx = _find_label_index(id2label, "contradiction")

        template = getattr(self._intent_config, "hypothesis_template", None)
        if isinstance(template, str) and "{}" in template:
            self._intent_hypothesis_template = template
        else:
            self._intent_hypothesis_template = "This example is {}."

        return True

    def _ensure_intent_pipeline(self) -> bool:
        if self._intent_pipeline is not None:
            return True
        pipeline = _maybe_import_transformers_pipeline()
        if pipeline is None:
            self._record_issue("Transformers pipeline unavailable for intent analysis")
            return False
        try:
            self._intent_pipeline = pipeline(
                task="zero-shot-classification",
                model=DEFAULT_INTENT_MODEL,
                multi_label=True,
            )
        except Exception as exc:  # noqa: BLE001 - HF backend specific
            logger.warning("Intent pipeline unavailable: %s", exc)
            self._intent_pipeline = None
            self._record_issue(f"Intent transformers pipeline unavailable: {exc}")
            return False
        return True

    def _intent_default_prediction(self) -> tuple[str, list[dict[str, float]]]:
        if not self.intent_labels:
            return "", []
        topn = min(3, len(self.intent_labels))
        default_labels = self.intent_labels[:topn]
        score = 1.0 / topn if topn else 0.0
        entries = [{"label": label, "score": score} for label in default_labels]
        return default_labels[0], entries

    def _infer_intent_with_onnx(self, text: str) -> tuple[str, list[dict[str, float]]]:
        if self._intent_session is None or self._intent_tokenizer is None:
            return self._intent_default_prediction()

        entail_idx = self._intent_entail_idx
        contra_idx = self._intent_contra_idx
        results: list[dict[str, float]] = []

        for label in self.intent_labels:
            hypothesis = self._intent_hypothesis_template.format(label)
            encoded = self._intent_tokenizer(
                text,
                hypothesis,
                return_tensors="np",
                truncation=True,
            )
            # Ensure ONNX inputs use int64 dtype as required by many seq2seq exports
            inputs = {name: np.asarray(value, dtype=np.int64) for name, value in encoded.items()}
            logits = self._intent_session.run(None, inputs)[0]
            arr = np.array(logits, dtype=np.float32).ravel()

            if (
                entail_idx is not None
                and contra_idx is not None
                and 0 <= entail_idx < arr.size
                and 0 <= contra_idx < arr.size
            ):
                pair = np.array([arr[contra_idx], arr[entail_idx]], dtype=np.float32)
                score = float(_softmax(pair)[-1])
            elif entail_idx is not None and 0 <= entail_idx < arr.size:
                probs = _softmax(arr)
                score = float(probs[entail_idx])
            else:
                probs = _softmax(arr)
                score = float(np.max(probs))

            results.append({"label": label, "score": score})

        results.sort(key=lambda item: item["score"], reverse=True)
        top_label = results[0]["label"] if results else ""
        return top_label, results[: min(3, len(results))]

    def _infer_intent_with_pipeline(self, text: str) -> tuple[str, list[dict[str, float]]]:
        if self._intent_pipeline is None:
            return self._intent_default_prediction()

        candidates = self.intent_labels or ["other"]
        raw = self._intent_pipeline(text, candidate_labels=candidates, multi_label=True)
        entries: list[dict[str, float]]
        if isinstance(raw, dict) and "labels" in raw and "scores" in raw:
            entries = [
                {"label": str(label), "score": float(score)}
                for label, score in zip(raw["labels"], raw["scores"])
            ]
        else:
            entries = []
            for item in raw:
                if isinstance(item, dict):
                    label = str(item.get("label", ""))
                    score = float(item.get("score", 0.0))
                else:
                    label = str(getattr(item, "label", ""))
                    score = float(getattr(item, "score", 0.0))
                entries.append({"label": label, "score": score})

        entries.sort(key=lambda item: item["score"], reverse=True)
        top_label = entries[0]["label"] if entries else ""
        return top_label, entries[: min(3, len(entries))]

    # ---- Public API extension ----
    def _infer_intent(self, text: str) -> tuple[str, list[dict[str, float]]]:
        clean_text = (text or "").strip()
        if not clean_text:
            return self._intent_default_prediction()

        self._lazy_intent()

        if self._intent_session is not None and self._intent_tokenizer is not None:
            try:
                return self._infer_intent_with_onnx(clean_text)
            except Exception as exc:  # noqa: BLE001 - runtime dependent
                logger.warning("Intent ONNX inference failed: %s", exc)

        if self._intent_pipeline is not None:
            try:
                return self._infer_intent_with_pipeline(clean_text)
            except Exception as exc:  # noqa: BLE001 - runtime dependent
                logger.warning("Intent pipeline inference failed: %s", exc)

        return self._intent_default_prediction()

    def analyze(
        self,
        *,
        wav: np.ndarray | None,
        sr: int | None,
        text: str,
    ) -> dict[str, Any]:
        text_res, speech_res, vad_res = self._analyze_components(
            wav=wav, sr=sr, text=text
        )
        intent_top, entries = self._infer_intent(text)
        if not entries:
            default_intent = _default_intent_result()
            intent_result = IntentResult(
                top=default_intent.top,
                top3=[dict(item) for item in default_intent.top3],
            )
        else:
            intent_result = IntentResult(
                top=intent_top or entries[0]["label"],
                top3=[{"label": str(item.get("label", "")), "score": float(item.get("score", 0.0))} for item in entries],
            )
        return self._build_affect_payload(
            text_res=text_res,
            speech_res=speech_res,
            vad_res=vad_res,
            intent_res=intent_result,
        )


__all__ = __all__ + ("EmotionIntentAnalyzer", "create_onnx_session")
