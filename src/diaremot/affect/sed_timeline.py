"""Timeline sound-event detection using CNN14 ONNX windows."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import librosa  # type: ignore

    _HAVE_LIBROSA = True
except Exception:  # pragma: no cover - runtime dependent
    _HAVE_LIBROSA = False

try:  # pragma: no cover - optional dependency
    from scipy.signal import medfilt

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - runtime dependent
    _HAVE_SCIPY = False

from ..io.onnx_runtime_guard import OnnxRuntimeUnavailable, ensure_onnxruntime

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TimelineArtifacts:
    csv: Path
    jsonl: Path | None
    events: list[dict[str, Any]]
    mode: str | None = None


@dataclass(slots=True)
class _FeatureInfo:
    sr: int
    hop_length: int
    win_length: int
    total_frames: int
    total_samples: int


def run_sed_timeline(
    audio_16k: np.ndarray,
    *,
    sr: int,
    cfg: Mapping[str, Any],
    out_dir: Path,
    file_id: str,
    model_paths: tuple[Path, Path] | None = None,
    labels: Iterable[str] | None = None,
) -> TimelineArtifacts | None:
    """Execute sliding-window CNN14 inference to build an events timeline."""

    if audio_16k is None or getattr(audio_16k, "size", 0) == 0:
        return None

    try:
        session = _load_session(model_paths)
    except OnnxRuntimeUnavailable as exc:  # pragma: no cover - runtime dependent
        logger.info("[sed.timeline] ONNX runtime unavailable: %s", exc)
        return None
    except FileNotFoundError as exc:
        logger.info("[sed.timeline] timeline assets missing: %s", exc)
        return None

    label_list = list(labels or _load_labels(model_paths[1] if model_paths else None))
    if not label_list:
        logger.info("[sed.timeline] no label vocabulary available; skipping timeline")
        return None

    window_sec = max(float(cfg.get("window_sec", 1.0)), 1e-3)
    hop_sec = max(float(cfg.get("hop_sec", 0.5)), 1e-3)
    enter = float(cfg.get("enter", 0.5))
    exit = float(cfg.get("exit", 0.35))
    merge_gap = max(float(cfg.get("merge_gap", 0.2)), 0.0)
    min_dur_map_raw = cfg.get("min_dur", {}) or {}
    min_dur_default = float(cfg.get("default_min_dur", 0.3))
    median_k = int(cfg.get("median_k", 5) or 1)
    if median_k < 1:
        median_k = 1
    if median_k % 2 == 0:
        median_k += 1
    batch_size = int(cfg.get("batch_size", 256) or 1)
    if batch_size < 1:
        batch_size = 1

    min_dur_map = {}
    for key, value in min_dur_map_raw.items():
        try:
            min_dur_map[str(key).lower()] = float(value)
        except (TypeError, ValueError):
            continue

    class_map = _load_classmap(cfg.get("classmap_csv"))

    y = np.asarray(audio_16k, dtype=np.float32)
    y32, sr32 = _resample_to_sr(y, sr, target=32000)
    logmel, feat_info = _compute_logmel(y32, sr32)
    frame_times, window_audio, mel_windows = _prepare_windows(
        y32, logmel, feat_info, window_sec, hop_sec
    )
    if not frame_times or (not window_audio and not mel_windows):
        logger.info("[sed.timeline] no analysis windows generated; skipping")
        return None

    # Limit total windows to keep runtime bounded on long recordings.
    # Decimate uniformly to preserve temporal ordering (coarser resolution).
    try:
        max_windows = int(cfg.get("max_windows", 6000) or 0)
    except Exception:
        max_windows = 6000
    if max_windows > 0 and len(frame_times) > max_windows:
        factor = (len(frame_times) + max_windows - 1) // max_windows
        try:
            frame_times = frame_times[::factor]
            if window_audio:
                window_audio = window_audio[::factor]
            if mel_windows:
                mel_windows = mel_windows[::factor]
            logger.info(
                f"[sed.timeline] decimated windows by {factor}x -> {len(frame_times)} frames"
            )
        except Exception:
            # If anything goes wrong, fall back to original lists
            pass

    input_name = session.get_inputs()[0].name
    inference_modes: list[tuple[str, list[np.ndarray]]] = []
    if window_audio:
        inference_modes.append(("waveform", window_audio))
    if mel_windows:
        inference_modes.append(("mel", mel_windows))

    supports_batch = _supports_batch(session)
    if not supports_batch and batch_size > 1:
        logger.info("[sed.timeline] Model input has fixed batch; falling back to single-window inference")
        batch_size = 1

    scores: np.ndarray | None = None
    used_mode: str | None = None
    mode_errors: list[str] = []
    for mode, windows in inference_modes:
        score_chunks: list[np.ndarray] = []
        try:
            if batch_size == 1:
                for idx, window in enumerate(windows):
                    preds = _run_session(session, input_name, [window], mode, allow_batch=False)
                    if preds is None:
                        raise RuntimeError("no predictions returned")
                    score_chunks.append(preds)
                    if (idx + 1) % 500 == 0 or idx + 1 == len(windows):
                        logger.info(
                            f"[sed.timeline] processed {idx + 1}/{len(windows)} {mode} windows"
                        )
            else:
                for start in range(0, len(windows), batch_size):
                    batch = windows[start : start + batch_size]
                    preds = _run_session(session, input_name, batch, mode, allow_batch=True)
                    if preds is None:
                        raise RuntimeError("no predictions returned")
                    score_chunks.append(preds)
                    processed = min(start + len(batch), len(windows))
                    if processed % 500 == 0 or processed == len(windows):
                        logger.info(
                            f"[sed.timeline] processed {processed}/{len(windows)} {mode} windows"
                        )
        except Exception as exc:  # pragma: no cover - runtime dependent
            mode_errors.append(f"{mode}:{exc}")
            continue
        if score_chunks:
            scores = np.vstack(score_chunks)
            used_mode = mode
            break

    if scores is None:
        logger.info(
            "[sed.timeline] inference failed for all candidate input layouts (%s)",
            "; ".join(mode_errors) if mode_errors else "no modes",
        )
        return None
    if scores.shape[0] != len(frame_times):
        limit = min(scores.shape[0], len(frame_times))
        scores = scores[:limit]
        frame_times = frame_times[:limit]

    filtered = _median_filter(scores, median_k)

    group_scores, group_labels = _collapse_labels(filtered, label_list, class_map)
    if group_scores.size == 0 or not group_labels:
        logger.info("[sed.timeline] label collapse produced no outputs; skipping")
        return None

    events = _build_events(
        group_scores,
        frame_times,
        group_labels,
        enter,
        exit,
        merge_gap,
        min_dur_map,
        min_dur_default,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "events_timeline.csv"
    jsonl_path = out_dir / "events.jsonl" if bool(cfg.get("write_jsonl", False)) else None

    _write_events_csv(
        csv_path,
        file_id,
        events,
        enter=enter,
        exit=exit,
        median_k=median_k,
    )

    if jsonl_path is not None:
        _write_frames_jsonl(jsonl_path, frame_times, group_scores, group_labels)

    artifacts = TimelineArtifacts(csv=csv_path, jsonl=jsonl_path, events=events, mode=used_mode)
    if used_mode and artifacts.events:
        for event in artifacts.events:
            if isinstance(event, dict):
                event.setdefault("inference_mode", used_mode)
    return artifacts


def _load_session(model_paths: tuple[Path, Path] | None):
    if not model_paths:
        raise FileNotFoundError("cnn14 ONNX model not located")
    model_path, _ = model_paths
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ort = ensure_onnxruntime()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    try:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:  # pragma: no cover - optional attribute
        pass
    providers = ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OnnxRuntimeUnavailable(str(exc), cause=exc) from exc


def _load_labels(labels_path: Path | None) -> list[str]:
    if not labels_path:
        return []
    path = Path(labels_path)
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            labels = []
            if reader.fieldnames:
                key = (
                    "display_name" if "display_name" in reader.fieldnames else reader.fieldnames[0]
                )
                for row in reader:
                    value = row.get(key)
                    if value:
                        labels.append(str(value))
            else:
                handle.seek(0)
                labels = [line.strip() for line in handle if line.strip()]
            return labels
    except Exception:  # pragma: no cover - best effort
        return []


def _load_classmap(path_value: Any) -> dict[str, str]:
    if not path_value:
        return {}
    try:
        path = Path(path_value)
    except TypeError:
        return {}
    if not path.exists():
        logger.info("[sed.timeline] class map %s missing; using raw labels", path)
        return {}

    mapping: dict[str, str] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                keys = list(row.keys())
                if len(keys) < 2:
                    continue
                raw = row.get("audioset") or row.get("label") or row.get(keys[0])
                group = row.get("group") or row.get("task") or row.get(keys[1])
                if not raw or not group:
                    continue
                mapping[str(raw).strip().lower()] = str(group).strip()
    except Exception as exc:  # pragma: no cover - best effort
        logger.info("[sed.timeline] failed loading class map %s: %s", path, exc)
        return {}
    return mapping


def _resample_to_sr(audio: np.ndarray, sr: int, *, target: int) -> tuple[np.ndarray, int]:
    if sr == target:
        return audio.astype(np.float32), sr
    if _HAVE_LIBROSA:
        try:
            y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target)
            return y.astype(np.float32), target
        except Exception:
            pass
    ratio = target / float(sr)
    new_len = int(round(len(audio) * ratio))
    if new_len <= 1:
        return audio.astype(np.float32), sr
    x_old = np.linspace(0, len(audio) - 1, len(audio), dtype=np.float32)
    x_new = np.linspace(0, len(audio) - 1, new_len, dtype=np.float32)
    y = np.interp(x_new, x_old, audio).astype(np.float32)
    return y, target


def _compute_logmel(audio: np.ndarray, sr: int) -> tuple[np.ndarray, _FeatureInfo]:
    n_fft = 1024
    hop_length = 320
    win_length = 1024
    n_mels = 64
    fmin = 50.0
    fmax = min(14000.0, sr / 2.0)

    if audio.size == 0:
        empty = np.zeros((0, n_mels), dtype=np.float32)
        return empty, _FeatureInfo(
            sr=sr, hop_length=hop_length, win_length=win_length, total_frames=0, total_samples=0
        )

    if _HAVE_LIBROSA:
        try:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                power=2.0,
                center=True,
                pad_mode="reflect",
            )
            logmel = librosa.power_to_db(mel, ref=1.0)
            logmel = logmel.T.astype(np.float32)
        except Exception as exc:  # pragma: no cover - best effort fallback
            logger.info("[sed.timeline] librosa melspectrogram failed: %s; using fallback", exc)
            logmel = _fallback_logmel(audio, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax)
    else:
        logmel = _fallback_logmel(audio, sr, n_fft, hop_length, win_length, n_mels, fmin, fmax)

    info = _FeatureInfo(
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        total_frames=int(logmel.shape[0]),
        total_samples=int(audio.shape[0]),
    )
    return logmel, info


def _fallback_logmel(
    audio: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    if audio.size == 0:
        return np.zeros((0, n_mels), dtype=np.float32)

    pad = win_length // 2
    if audio.size == 1:
        padded = np.pad(audio, (pad, pad), mode="constant")
    else:
        padded = np.pad(audio, (pad, pad), mode="reflect")
    window = np.hanning(win_length).astype(np.float32)
    frames: list[np.ndarray] = []
    limit = len(padded) - win_length
    idx = 0
    while idx <= limit:
        frame = padded[idx : idx + win_length]
        frames.append((frame * window).astype(np.float32))
        idx += hop_length
    if not frames:
        frame = np.zeros(win_length, dtype=np.float32)
        take = min(len(audio), win_length)
        frame[:take] = audio[:take]
        frames.append(frame * window)

    frame_arr = np.stack(frames, axis=0)
    fft = np.fft.rfft(frame_arr, n=n_fft, axis=1)
    power = (fft.real**2 + fft.imag**2).astype(np.float32)
    filters = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = np.maximum(power @ filters.T, 1e-10)
    logmel = 10.0 * np.log10(mel)
    return logmel.astype(np.float32)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    if fmax <= fmin:
        fmax = sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            if 0 <= k < filters.shape[1]:
                filters[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            if 0 <= k < filters.shape[1]:
                filters[m - 1, k] = (right - k) / max(1, right - center)
    return filters


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: Sequence[float]) -> np.ndarray:
    arr = np.asarray(mels, dtype=np.float64)
    return 700.0 * (10 ** (arr / 2595.0) - 1.0)


class _LazyWaveformWindows(Sequence[np.ndarray]):
    """Lazily materialize waveform windows to avoid large preallocation.

    Behaves like a sliceable sequence that yields per-window numpy arrays
    when indexed. Only the requested windows are created at access time.
    """

    def __init__(
        self,
        audio: np.ndarray,
        *,
        start_indices: list[int],
        sample_window: int,
        hop_length: int,
        sr: int,
    ) -> None:
        self._audio = np.asarray(audio, dtype=np.float32)
        self._starts = start_indices
        self._sample_window = int(sample_window)
        self._hop = int(hop_length)
        self._sr = int(sr)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._starts)

    def _get_one(self, idx: int) -> np.ndarray:
        start_idx = int(self._starts[idx])
        sample_start = int(start_idx * self._hop)
        sample_end = sample_start + self._sample_window
        clip = self._audio[sample_start:sample_end]
        pad = self._sample_window - clip.shape[0]
        if pad > 0:
            clip = np.pad(clip, (0, pad))
        return np.asarray(clip, dtype=np.float32)

    def __getitem__(self, key: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(key, slice):
            rng = range(*key.indices(len(self)))
            return [self._get_one(i) for i in rng]
        return self._get_one(int(key))


class _LazyMelWindows(Sequence[np.ndarray]):
    """Lazily materialize mel windows from a full log-mel matrix."""

    def __init__(
        self,
        logmel: np.ndarray,
        *,
        start_indices: list[int],
        frames_per_window: int,
    ) -> None:
        self._mel = np.asarray(logmel, dtype=np.float32)
        self._starts = start_indices
        self._fpw = int(frames_per_window)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._starts)

    def _get_one(self, idx: int) -> np.ndarray:
        start_idx = int(self._starts[idx])
        mel_clip = self._mel[start_idx : start_idx + self._fpw]
        if mel_clip.shape[0] < self._fpw:
            pad = self._fpw - mel_clip.shape[0]
            if mel_clip.size == 0:
                n_mels = self._mel.shape[1] if self._mel.ndim == 2 and self._mel.shape[1] > 0 else 64
                mel_clip = np.zeros((self._fpw, n_mels), dtype=np.float32)
            else:
                tail = mel_clip[-1:, :]
                mel_pad = np.repeat(tail, pad, axis=0)
                mel_clip = np.vstack([mel_clip, mel_pad])
        # CNN14 expects (mel, frames)
        return mel_clip.T.astype(np.float32)

    def __getitem__(self, key: int | slice) -> np.ndarray | list[np.ndarray]:
        if isinstance(key, slice):
            rng = range(*key.indices(len(self)))
            return [self._get_one(i) for i in rng]
        return self._get_one(int(key))


def _prepare_windows(
    audio: np.ndarray,
    logmel: np.ndarray,
    info: _FeatureInfo,
    window_sec: float,
    hop_sec: float,
) -> tuple[list[tuple[float, float]], Sequence[np.ndarray], Sequence[np.ndarray]]:
    if audio.size == 0 or info.total_frames == 0 or info.sr <= 0:
        return [], [], []

    frames_per_window = max(1, int(round(window_sec * info.sr / info.hop_length)))
    frame_step = max(1, int(round(hop_sec * info.sr / info.hop_length)))
    sample_window = max(1, int(round(window_sec * info.sr)))

    start_indices: list[int] = []
    idx = 0
    while idx < info.total_frames:
        start_indices.append(idx)
        idx += frame_step
    last_start = max(0, info.total_frames - frames_per_window)
    if not start_indices or start_indices[-1] != last_start:
        start_indices.append(last_start)
    start_indices = sorted(dict.fromkeys(start_indices))

    total_duration = info.total_samples / float(info.sr)
    frame_times: list[tuple[float, float]] = []
    for start_idx in start_indices:
        sample_start = int(start_idx * info.hop_length)
        start_time = sample_start / float(info.sr)
        end_time = min(start_time + window_sec, total_duration)
        frame_times.append((start_time, end_time))

    audio_windows: Sequence[np.ndarray] = _LazyWaveformWindows(
        audio,
        start_indices=start_indices,
        sample_window=sample_window,
        hop_length=info.hop_length,
        sr=info.sr,
    )
    mel_windows: Sequence[np.ndarray] = _LazyMelWindows(
        logmel,
        start_indices=start_indices,
        frames_per_window=frames_per_window,
    )

    return frame_times, audio_windows, mel_windows


def _supports_batch(session) -> bool:
    try:
        inputs = session.get_inputs()
        if not inputs:
            return False
        shape = inputs[0].shape
    except Exception:
        return False
    if not shape or len(shape) < 1:
        return False
    first = shape[0]
    return first in (None, "batch")


def _run_session(
    session,
    input_name: str,
    batch_samples: Sequence[np.ndarray],
    mode: str,
    *,
    allow_batch: bool,
) -> np.ndarray | None:
    if not batch_samples:
        return None

    if allow_batch:
        arr = _stack_batch(batch_samples, mode)
    else:
        arr = _stack_batch(batch_samples[:1], mode)

    if arr is None:
        return None

    try:
        preds = session.run(None, {input_name: arr})[0]
        return np.asarray(preds, dtype=np.float32)
    except Exception as exc:
        logger.info(
            "[sed.timeline] inference failed (%s mode, shape %s): %s",
            mode,
            arr.shape if isinstance(arr, np.ndarray) else "?",
            exc,
        )
        return None


def _stack_batch(batch_samples: Sequence[np.ndarray], mode: str) -> np.ndarray | None:
    if not batch_samples:
        return None

    if mode == "waveform":
        normalized: list[np.ndarray] = []
        for sample in batch_samples:
            if sample.ndim == 1:
                normalized.append(sample)
            elif sample.ndim == 2 and 1 in sample.shape:
                normalized.append(sample.reshape(-1))
            else:
                logger.warning(
                    f"[sed.timeline] Unexpected waveform window shape {sample.shape}; flattening"
                )
                normalized.append(sample.reshape(-1))
        arr = np.stack(normalized, axis=0).astype(np.float32)
    else:
        normalized = []
        for sample in batch_samples:
            if sample.ndim == 2:
                normalized.append(sample)
            elif sample.ndim == 3 and sample.shape[0] == 1:
                normalized.append(sample[0])
            else:
                logger.warning(
                    f"[sed.timeline] Unexpected mel window shape {sample.shape}; skipping"
                )
        if not normalized:
            return None
        arr = np.stack(normalized, axis=0).astype(np.float32)

    return arr


def _candidate_arrays(batch: np.ndarray, mode: str) -> Iterator[np.ndarray]:
    """Generate candidate array shapes for CNN14 inference.
    
    CNN14 expects:
    - Waveform mode: (batch, samples) - rank 2
    - Mel mode: (batch, n_mels, time_frames) or (batch, time_frames, n_mels) - rank 3
    """
    if mode == "waveform":
        # For waveform, CNN14 expects (batch, samples)
        if batch.ndim == 2:
            yield batch  # Already correct shape
        elif batch.ndim == 1:
            yield batch[np.newaxis, :]  # Add batch dimension
        elif batch.ndim == 3:
            # If somehow we have (B, 1, samples), squeeze middle dim
            yield batch.squeeze(1)
        return

    # Mel mode - CNN14 expects (batch, n_mels, time_frames)
    if batch.ndim == 3:
        yield batch  # (B, mel, frames) - most common
        # Try transposed: (B, frames, mel)
        yield np.transpose(batch, (0, 2, 1))
    elif batch.ndim == 4:
        # If we have an extra channel dim, squeeze it
        if batch.shape[1] == 1:
            yield batch.squeeze(1)  # (B, 1, mel, frames) -> (B, mel, frames)
        if batch.shape[3] == 1:
            yield batch.squeeze(3)  # (B, mel, frames, 1) -> (B, mel, frames)


def _median_filter(scores: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1 or scores.size == 0:
        return scores
    if _HAVE_SCIPY:
        try:
            return medfilt(scores, kernel_size=(kernel, 1))
        except Exception:
            pass
    pad = kernel // 2
    padded = np.pad(scores, ((pad, pad), (0, 0)), mode="edge")
    result = np.empty_like(scores)
    for idx in range(scores.shape[0]):
        window = padded[idx : idx + kernel]
        result[idx] = np.median(window, axis=0)
    return result


def _collapse_labels(
    scores: np.ndarray,
    labels: list[str],
    class_map: Mapping[str, str],
) -> tuple[np.ndarray, list[str]]:
    if scores.size == 0 or not labels:
        return np.zeros((0, 0), dtype=np.float32), []

    groups: dict[str, list[int]] = {}
    if class_map:
        normalized = {str(k).lower(): str(v) for k, v in class_map.items()}
    else:
        normalized = {}
    for idx, label in enumerate(labels):
        key = str(label)
        group = normalized.get(key.lower(), key)
        groups.setdefault(group, []).append(idx)

    if not groups:
        return np.zeros((0, 0), dtype=np.float32), []

    group_labels = list(groups.keys())
    collapsed = np.zeros((scores.shape[0], len(group_labels)), dtype=np.float32)
    for g_idx, label in enumerate(group_labels):
        indices = groups[label]
        collapsed[:, g_idx] = scores[:, indices].max(axis=1)
    return collapsed, group_labels


def _build_events(
    scores: np.ndarray,
    frames: list[tuple[float, float]],
    labels: list[str],
    enter: float,
    exit: float,
    merge_gap: float,
    min_dur_map: Mapping[str, float],
    min_dur_default: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for col, label in enumerate(labels):
        series = scores[:, col]
        active = False
        start = 0.0
        peak = 0.0
        for idx, value in enumerate(series):
            frame_start, frame_end = frames[idx]
            score_val = float(value)
            if not active and score_val >= enter:
                active = True
                start = frame_start
                peak = score_val
            elif active:
                if score_val > peak:
                    peak = score_val
                if score_val <= exit:
                    events.append(
                        {
                            "label": label,
                            "start": start,
                            "end": frame_end,
                            "score": float(max(0.0, min(1.0, peak))),
                        }
                    )
                    active = False
        if active:
            _, frame_end = frames[-1]
            events.append(
                {
                    "label": label,
                    "start": start,
                    "end": frame_end,
                    "score": float(max(0.0, min(1.0, peak))),
                }
            )

    if not events:
        return []

    min_filtered: list[dict[str, Any]] = []
    for event in events:
        label = str(event["label"])
        duration = max(0.0, float(event["end"]) - float(event["start"]))
        threshold = float(min_dur_map.get(label.lower(), min_dur_default))
        if duration + 1e-6 < threshold:
            continue
        min_filtered.append(dict(event))

    if not min_filtered:
        return []

    merged: list[dict[str, Any]] = []
    for event in sorted(min_filtered, key=lambda x: (x["start"], x["end"])):
        if not merged:
            merged.append(dict(event))
            continue
        last = merged[-1]
        if (
            str(last["label"]).lower() == str(event["label"]).lower()
            and float(event["start"]) - float(last["end"]) <= merge_gap + 1e-6
        ):
            last["end"] = max(float(last["end"]), float(event["end"]))
            last["score"] = max(float(last["score"]), float(event["score"]))
        else:
            merged.append(dict(event))

    for event in merged:
        duration = max(0.0, float(event["end"]) - float(event["start"]))
        event["duration"] = duration
        event["score"] = float(max(0.0, min(1.0, event.get("score", 0.0))))
        event["weight"] = event["score"] * duration
        event["source"] = "cnn14"

    return merged


def _write_events_csv(
    path: Path,
    file_id: str,
    events: list[dict[str, Any]],
    *,
    enter: float,
    exit: float,
    median_k: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["file_id", "start", "end", "label", "score", "source", "enter", "exit", "median_k"]
        )
        for event in events:
            writer.writerow(
                [
                    file_id,
                    f"{float(event['start']):.6f}",
                    f"{float(event['end']):.6f}",
                    event.get("label", ""),
                    f"{float(event.get('score', 0.0)):.4f}",
                    event.get("source", "cnn14"),
                    f"{float(enter):.2f}",
                    f"{float(exit):.2f}",
                    int(median_k),
                ]
            )


def _write_frames_jsonl(
    path: Path,
    frames: list[tuple[float, float]],
    scores: np.ndarray,
    labels: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, (start, end) in enumerate(frames):
            row_scores = scores[idx]
            order = np.argsort(row_scores)[::-1]
            top = [
                {"label": labels[i], "score": float(row_scores[i])}
                for i in order[: min(3, len(order))]
            ]
            payload = {
                "frame_index": idx,
                "start": float(start),
                "end": float(end),
                "topk": top,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["TimelineArtifacts", "run_sed_timeline"]
