"""PANNs CNN14 sound event detection pipeline with hysteresis merging.

The detector operates on mono 16 kHz waveforms, slices the signal into
1.0 second frames with a 0.5 second hop, and feeds each frame to the
CNN14 ONNX model. The exported model already performs STFT → log-mel
projection (64 bins, HTK=False) followed by per-example batch
normalisation. We keep the interface explicit in code comments so the
feature contract stays discoverable even though the ONNX graph handles
it internally.

The module exposes :func:`run_sed` for programmatic use and a CLI entry
point (``python -m diaremot.sed.sed_panns_onnx``). The CLI writes
``events_timeline.csv`` compatible with the rest of the pipeline and can
optionally emit a JSONL dump with frame-level posteriors for debugging.

All heavy imports (onnxruntime, librosa) are performed lazily so that
importing :mod:`diaremot.sed` remains inexpensive when the pipeline runs
without SED.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import medfilt

try:  # Optional at import time; resolved lazily in run_sed
    import soundfile as sf
except Exception as exc:  # pragma: no cover - env specific
    sf = None  # type: ignore[assignment]
    _SOUNDFILE_ERROR = exc
else:  # pragma: no cover - exercised indirectly
    _SOUNDFILE_ERROR = None

LOGGER = logging.getLogger(__name__)
_MODEL_NAME_CANDIDATES: tuple[str, ...] = ("panns_cnn14.onnx", "cnn14.onnx", "model.onnx")
_LABEL_NAME_CANDIDATES: tuple[str, ...] = (
    "audioset_labels.csv",
    "class_labels_indices.csv",
    "labels.csv",
)
_MODEL_SUBDIR_HINTS: tuple[str, ...] = ("panns", "sed_panns", "panns_cnn14")

COARSE_LABELS: tuple[str, ...] = (
    "music",
    "keyboard",
    "door",
    "tv",
    "phone",
    "vehicle",
    "siren_alarm",
    "laughter",
    "footsteps",
    "impact",
    "barking",
    "wind",
    "water",
    "crowd",
    "engine",
    "cooking",
    "typing",
    "mouse_click",
    "applause",
    "other_env",
)

_COARSE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("applause", ("applause", "clapping")),
    ("barking", ("bark", "dog")),
    (
        "cooking",
        (
            "cooking",
            "kitchen",
            "sizzle",
            "frying",
            "chopping",
            "cutting board",
            "microwave",
            "blender",
        ),
    ),
    ("door", ("door", "knock", "doorbell")),
    (
        "engine",
        ("engine", "motor", "propeller", "idling", "revving", "mechanical fan"),
    ),
    (
        "footsteps",
        (
            "footstep",
            "footsteps",
            "shoe",
            "walk",
            "walking",
            "run",
            "running",
            "sandal",
            "high-heeled",
        ),
    ),
    (
        "impact",
        (
            "impact",
            "hit",
            "slam",
            "bang",
            "thud",
            "thump",
            "smash",
            "crash",
            "clack",
            "click",
            "clunk",
            "door slam",
        ),
    ),
    (
        "keyboard",
        (
            "keyboard instrument",
            "piano",
            "synth",
            "organ",
            "harpsichord",
            "accordion",
            "keytar",
            "electric piano",
        ),
    ),
    ("laughter", ("laugh", "laughter", "chuckle", "giggle")),
    ("mouse_click", ("mouse click", "computer mouse", "mouse button")),
    (
        "music",
        (
            "music",
            "song",
            "sing",
            "guitar",
            "violin",
            "cello",
            "flute",
            "brass",
            "drum",
            "percussion",
            "orchestra",
            "band",
            "choir",
            "harp",
            "saxophone",
        ),
    ),
    (
        "phone",
        (
            "telephone",
            "phone",
            "dial tone",
            "ringtone",
            "busy signal",
            "fax",
            "texting",
        ),
    ),
    ("siren_alarm", ("siren", "alarm", "beacon", "fire alarm", "ambulance")),
    ("tv", ("television", "tv", "broadcast", "news", "sitcom")),
    (
        "typing",
        (
            "typing",
            "typewriter",
            "computer keyboard",
            "keyboarding",
            "mechanical keyboard",
        ),
    ),
    (
        "vehicle",
        (
            "vehicle",
            "car",
            "truck",
            "bus",
            "train",
            "tram",
            "subway",
            "rail",
            "motorcycle",
            "aircraft",
            "airplane",
            "helicopter",
            "boat",
            "ship",
        ),
    ),
    ("wind", ("wind", "breeze", "gust", "airflow")),
    (
        "water",
        (
            "water",
            "rain",
            "stream",
            "river",
            "ocean",
            "sea",
            "waves",
            "surf",
            "drip",
            "splash",
            "pour",
        ),
    ),
    (
        "crowd",
        (
            "crowd",
            "cheer",
            "boo",
            "stadium",
            "audience",
            "chant",
            "speech",
            "conversation",
        ),
    ),
)


@dataclass(slots=True)
class SedDebugInfo:
    """Frame-level diagnostic payload cached after :func:`run_sed`."""

    frame_times: np.ndarray
    coarse_labels: tuple[str, ...]
    coarse_posteriors: np.ndarray
    coarse_topk: list[list[tuple[str, float]]]
    fine_labels: list[str]
    fine_posteriors: np.ndarray


_LAST_DEBUG: SedDebugInfo | None = None


def _require_soundfile() -> None:
    if sf is None:  # pragma: no cover - environment specific guard
        raise RuntimeError(
            "soundfile is unavailable; install pysoundfile to enable audio decoding"
        ) from _SOUNDFILE_ERROR


def _load_audio(path: Path, sr_target: int) -> tuple[np.ndarray, int]:
    """Load mono audio and resample to ``sr_target`` when needed."""

    _require_soundfile()
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != sr_target:
        import librosa  # Local import to keep module import light

        data = librosa.resample(data, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    return data.astype(np.float32, copy=False), sr


def _frame_audio(
    waveform: np.ndarray, frame_samples: int, hop_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice waveform into overlapping frames and return (frames, start_indices)."""

    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")

    num_samples = waveform.shape[0]
    if num_samples < frame_samples:
        return np.empty((0, frame_samples), dtype=np.float32), np.empty(0, dtype=np.int64)

    starts: list[int] = []
    frames: list[np.ndarray] = []
    for start in range(0, num_samples, hop_samples):
        end = start + frame_samples
        frame = waveform[start:end]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.shape[0]))
        frames.append(frame.astype(np.float32, copy=False))
        starts.append(start)
        if end >= num_samples:
            break
    return np.stack(frames), np.asarray(starts, dtype=np.int64)


def _find_label_file(model_path: Path) -> Path | None:
    directory = model_path.parent
    for candidate in _LABEL_NAME_CANDIDATES:
        label_path = directory / candidate
        if label_path.exists():
            return label_path
    return None


def _iter_model_candidates(base: Path) -> Iterable[tuple[Path, Path]]:
    if base.is_file():
        label = _find_label_file(base)
        if label is not None:
            yield base, label
        return

    if not base.exists():
        return

    for candidate in _MODEL_NAME_CANDIDATES:
        model_path = base / candidate
        if model_path.exists():
            label = _find_label_file(model_path)
            if label is not None:
                yield model_path, label
    for subdir in _MODEL_SUBDIR_HINTS:
        sub_path = base / subdir
        if not sub_path.exists():
            continue
        yield from _iter_model_candidates(sub_path)


def _resolve_model_assets(model_path: str | None) -> tuple[Path, Path]:
    if model_path is not None:
        requested = Path(model_path)
        for candidate in _iter_model_candidates(requested):
            return candidate
        raise FileNotFoundError(
            f"Unable to locate CNN14 ONNX model under '{requested}'. "
            "Provide a directory containing the model or download models.zip from the DiaRemot release."
        )

    try:
        from ..pipeline.runtime_env import iter_model_roots
    except Exception:  # pragma: no cover - fallback when runtime_env unavailable
        roots = (Path.cwd() / "models",)
    else:
        roots = iter_model_roots()

    for root in roots:
        for candidate in _iter_model_candidates(Path(root)):
            LOGGER.debug("Found CNN14 model at %s", candidate[0])
            return candidate

    raise FileNotFoundError(
        "CNN14 ONNX model not found. Set DIAREMOT_MODEL_DIR or pass --model pointing to"
        " a directory that contains cnn14.onnx (with labels)."
    )


def _load_label_names(label_path: Path) -> list[str]:
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        key = None
        if reader.fieldnames:
            for candidate in ("display_name", "name", "label"):
                if candidate in reader.fieldnames:
                    key = candidate
                    break
            else:
                key = reader.fieldnames[0]
        labels: list[str] = []
        for row in reader:
            value = row.get(key, "") if isinstance(row, dict) else ""
            labels.append(str(value))
    if not labels:
        raise ValueError(f"Label file {label_path} is empty")
    return labels


def _assign_coarse_label(name: str) -> str:
    text = name.lower()
    for coarse, keywords in _COARSE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return coarse
    return "other_env"


def _build_coarse_mapping(label_names: Sequence[str]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for idx, name in enumerate(label_names):
        mapping[idx] = _assign_coarse_label(name)
    return mapping


def _collapse_labels(
    frame_posteriors: np.ndarray, mapping: dict[int, str], topk: int
) -> tuple[np.ndarray, tuple[str, ...], list[list[tuple[str, float]]]]:
    num_frames, _ = frame_posteriors.shape
    coarse_labels = COARSE_LABELS
    coarse_index = {label: i for i, label in enumerate(coarse_labels)}
    coarse_posteriors = np.zeros((num_frames, len(coarse_labels)), dtype=np.float32)

    for fine_idx, coarse_label in mapping.items():
        coarse_pos = coarse_index.get(coarse_label, coarse_index["other_env"])
        coarse_posteriors[:, coarse_pos] = np.maximum(
            coarse_posteriors[:, coarse_pos], frame_posteriors[:, fine_idx]
        )

    frame_topk: list[list[tuple[str, float]]] = []
    topk = int(topk)
    for frame in coarse_posteriors:
        if topk <= 0:
            frame_topk.append([])
            continue
        k = min(topk, frame.shape[0])
        top_indices = np.argpartition(-frame, k - 1)[:k]
        top_indices = top_indices[np.argsort(-frame[top_indices])]
        frame_topk.append([(coarse_labels[i], float(frame[i])) for i in top_indices])
    return coarse_posteriors, coarse_labels, frame_topk


def _hysteresis_runs(
    probabilities: Sequence[float],
    enter: float,
    exit: float,
    min_dur: float,
    merge_gap: float,
    hop_sec: float,
) -> list[tuple[int, int, float]]:
    """Run-length encode active regions using hysteresis thresholds.

    Returns a list of ``(start_frame, end_frame, peak_prob)`` tuples where
    ``end_frame`` is inclusive.
    """

    active = False
    start_idx = 0
    peak = 0.0
    runs: list[tuple[int, int, float]] = []

    for frame_idx, prob in enumerate(probabilities):
        if not active:
            if prob >= enter:
                active = True
                start_idx = frame_idx
                peak = float(prob)
        else:
            peak = max(peak, float(prob))
            if prob <= exit:
                runs.append((start_idx, frame_idx, peak))
                active = False
                peak = 0.0
    if active:
        runs.append((start_idx, len(probabilities) - 1, peak))

    filtered: list[tuple[int, int, float]] = []
    min_frames = math.ceil(min_dur / hop_sec) if hop_sec > 0 else 0
    for run in runs:
        start, end, peak = run
        frames = end - start + 1
        if frames < max(1, min_frames):
            continue
        if filtered:
            prev_start, prev_end, prev_peak = filtered[-1]
            gap_frames = start - prev_end - 1
            if gap_frames * hop_sec <= merge_gap:
                filtered[-1] = (
                    prev_start,
                    end,
                    max(prev_peak, peak),
                )
                continue
        filtered.append(run)
    return filtered


def run_sed(
    wav_path: str,
    *,
    sr_target: int = 16000,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5,
    mel_bins: int = 64,
    median_size: int = 5,
    enter_thresh: float = 0.50,
    exit_thresh: float = 0.35,
    min_dur: float = 0.30,
    merge_gap: float = 0.20,
    topk: int = 3,
    threads: int | None = None,
    model_path: str | None = None,
) -> list[dict[str, float | str]]:
    """Return merged sound events for ``wav_path``.

    Parameters
    ----------
    wav_path:
        Input audio file. Must decode to mono audio; multichannel files
        are mixed down equally.
    sr_target:
        Target sampling rate prior to inference. Audio is resampled with
        :func:`librosa.resample` when needed.
    frame_sec / hop_sec:
        Sliding window configuration in seconds. Frames shorter than a
        single hop are ignored.
    mel_bins:
        Included for configuration completeness. The CNN14 ONNX model
        internally computes 64-bin log-mel features; callers should keep
        this aligned with the exported weights.
    median_size:
        Odd window length for the temporal median filter applied to
        coarse label posterior probabilities. Set to ``0`` or ``1`` to
        disable smoothing.
    enter_thresh / exit_thresh:
        Hysteresis thresholds. A label becomes active when the smoothed
        posterior crosses ``enter_thresh`` and remains active until it
        falls below ``exit_thresh``.
    min_dur:
        Minimum event duration in seconds. Runs shorter than this after
        hysteresis are discarded.
    merge_gap:
        Merge consecutive runs for the same label when the gap between
        them is smaller than or equal to ``merge_gap`` seconds.
    topk:
        Number of coarse labels to cache per frame for debugging.
    threads:
        Optional override for the ONNX Runtime intra-op and inter-op
        thread pools. When ``None`` the library default is used.
    model_path:
        Optional directory or explicit file pointing to the CNN14 ONNX
        model. When omitted the function searches the configured model
        roots.
    """

    start_time = time.perf_counter()
    wav_path = str(wav_path)
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    LOGGER.debug(
        "run_sed(%s, sr_target=%s, frame=%.3fs, hop=%.3fs, enter=%.2f, exit=%.2f, min_dur=%.2f, merge_gap=%.2f, median=%d, topk=%d, threads=%s)",
        path,
        sr_target,
        frame_sec,
        hop_sec,
        enter_thresh,
        exit_thresh,
        min_dur,
        merge_gap,
        median_size,
        topk,
        threads,
    )

    waveform, sr = _load_audio(path, sr_target)
    if waveform.size == 0:
        LOGGER.warning("Audio %s decoded to an empty array", path)
        return []

    if mel_bins != 64:
        LOGGER.debug("mel_bins override requested: using %d bins", mel_bins)

    frame_samples = int(round(frame_sec * sr))
    hop_samples = int(round(hop_sec * sr))
    if frame_samples <= 0 or hop_samples <= 0:
        raise ValueError("frame_sec and hop_sec must produce at least one sample")
    frames, start_indices = _frame_audio(waveform, frame_samples, hop_samples)
    if frames.shape[0] == 0:
        LOGGER.warning(
            "Audio shorter than a single frame (%.2fs); emitting no sound events", frame_sec
        )
        return []

    model_file, label_file = _resolve_model_assets(model_path)
    label_names = _load_label_names(label_file)
    coarse_mapping = _build_coarse_mapping(label_names)

    import onnxruntime as ort  # Imported lazily for light module imports

    session_options = ort.SessionOptions()
    if threads is not None and threads > 0:
        session_options.intra_op_num_threads = int(threads)
        session_options.inter_op_num_threads = max(1, int(threads // 2) or 1)
    session = ort.InferenceSession(
        str(model_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    frame_posteriors = np.zeros((frames.shape[0], len(label_names)), dtype=np.float32)
    inference_start = time.perf_counter()
    for idx, frame in enumerate(frames):
        frame_input = np.expand_dims(frame.astype(np.float32, copy=False), axis=0)
        outputs = session.run(None, {input_name: frame_input})
        clipwise = outputs[0]
        frame_posteriors[idx] = clipwise[0]
    inference_elapsed = time.perf_counter() - inference_start

    coarse_posteriors, coarse_labels, frame_topk = _collapse_labels(
        frame_posteriors, coarse_mapping, topk
    )

    if median_size > 1:
        if median_size % 2 == 0:
            median_size += 1
        coarse_posteriors = medfilt(coarse_posteriors, kernel_size=(median_size, 1)).astype(
            np.float32, copy=False
        )

    events: list[dict[str, float | str]] = []
    total_duration = waveform.shape[0] / sr
    for label_index, label in enumerate(coarse_labels):
        probs = coarse_posteriors[:, label_index]
        runs = _hysteresis_runs(probs, enter_thresh, exit_thresh, min_dur, merge_gap, hop_sec)
        for start_frame, end_frame, peak in runs:
            start_time_sec = start_frame * hop_sec
            end_time_sec = min(total_duration, end_frame * hop_sec + frame_sec)
            events.append(
                {
                    "start": float(start_time_sec),
                    "end": float(end_time_sec),
                    "label": label,
                    "score": float(max(min(peak, 1.0), 0.0)),
                }
            )

    events.sort(key=lambda item: (item["start"], -(item["score"])))

    global _LAST_DEBUG
    frame_times = start_indices.astype(np.float32) / float(sr)
    _LAST_DEBUG = SedDebugInfo(
        frame_times=frame_times,
        coarse_labels=coarse_labels,
        coarse_posteriors=coarse_posteriors,
        coarse_topk=frame_topk,
        fine_labels=label_names,
        fine_posteriors=frame_posteriors,
    )

    elapsed = time.perf_counter() - start_time
    LOGGER.info(
        "Detected %d events from %s in %.2fs (inference %.2fs)",
        len(events),
        path.name,
        elapsed,
        inference_elapsed,
    )
    return events


def get_last_debug() -> SedDebugInfo | None:
    """Return the most recent :class:`SedDebugInfo` produced by :func:`run_sed`."""

    return _LAST_DEBUG


def _write_events_csv(
    output_path: Path, file_id: str, events: Sequence[dict[str, float | str]]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_id", "start", "end", "label", "score"])
        for event in events:
            writer.writerow(
                [
                    file_id,
                    f"{event['start']:.3f}",
                    f"{event['end']:.3f}",
                    event["label"],
                    f"{event['score']:.3f}",
                ]
            )


def _write_debug_jsonl(output_path: Path, debug: SedDebugInfo) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for frame_idx, (time_sec, topk, probs) in enumerate(
            zip(debug.frame_times, debug.coarse_topk, debug.coarse_posteriors, strict=False)
        ):
            payload = {
                "frame_index": frame_idx,
                "time": float(time_sec),
                "coarse_topk": topk,
                "coarse_posteriors": {
                    label: float(prob)
                    for label, prob in zip(debug.coarse_labels, probs, strict=False)
                },
            }
            handle.write(json.dumps(payload) + "\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sound event detection with PANNs CNN14 (ONNX)")
    parser.add_argument("input", help="Path to input WAV file (16 kHz mono recommended)")
    parser.add_argument("output", help="Path to events_timeline.csv to write")
    parser.add_argument("--model", dest="model", help="Optional model file or directory")
    parser.add_argument(
        "--frame", type=float, default=1.0, help="Frame length in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--hop", type=float, default=0.5, help="Frame hop in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--enter", dest="enter", type=float, default=0.50, help="Hysteresis enter threshold"
    )
    parser.add_argument(
        "--exit", dest="exit", type=float, default=0.35, help="Hysteresis exit threshold"
    )
    parser.add_argument(
        "--min-dur",
        dest="min_dur",
        type=float,
        default=0.30,
        help="Minimum event duration in seconds",
    )
    parser.add_argument(
        "--merge-gap",
        dest="merge_gap",
        type=float,
        default=0.20,
        help="Merge gap threshold in seconds",
    )
    parser.add_argument(
        "--median", dest="median", type=int, default=5, help="Median filter size (odd integer)"
    )
    parser.add_argument(
        "--mel", dest="mel", type=int, default=64, help="Number of mel bins (informational)"
    )
    parser.add_argument(
        "--topk", dest="topk", type=int, default=3, help="Top-K labels cached per frame"
    )
    parser.add_argument(
        "--threads", dest="threads", type=int, help="Override ONNX Runtime thread count"
    )
    parser.add_argument(
        "--debug-jsonl",
        dest="debug_jsonl",
        help="Optional path to write frame-level posterior debug JSONL",
    )
    parser.add_argument(
        "--backend",
        choices=("panns", "yamnet"),
        default="panns",
        help="Select detection backend (default: panns)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")

    backend = args.backend.lower()
    if backend != "panns":
        if backend == "yamnet":
            from . import sed_yamnet_tf

            events = sed_yamnet_tf.run_sed(
                args.input,
                frame_sec=args.frame,
                hop_sec=args.hop,
                enter_thresh=args.enter,
                exit_thresh=args.exit,
                min_dur=args.min_dur,
                merge_gap=args.merge_gap,
                topk=args.topk,
            )
        else:  # pragma: no cover - defensive
            parser.error(f"Unsupported backend: {args.backend}")
            return 2
    else:
        events = run_sed(
            args.input,
            frame_sec=args.frame,
            hop_sec=args.hop,
            mel_bins=args.mel,
            median_size=args.median,
            enter_thresh=args.enter,
            exit_thresh=args.exit,
            min_dur=args.min_dur,
            merge_gap=args.merge_gap,
            topk=args.topk,
            threads=args.threads,
            model_path=args.model,
        )

    output_path = Path(args.output)
    file_id = Path(args.input).stem
    _write_events_csv(output_path, file_id, events)
    LOGGER.info("Wrote %d events to %s", len(events), output_path)

    if args.debug_jsonl:
        debug = get_last_debug()
        if debug is None:
            LOGGER.warning("No debug data available for %s", args.input)
        else:
            _write_debug_jsonl(Path(args.debug_jsonl), debug)
            LOGGER.info("Wrote frame debug JSONL to %s", args.debug_jsonl)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
