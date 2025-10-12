from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _set_env_cpu_only() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Ensure transformers does not try to import torch in this process
    os.environ.setdefault("TRANSFORMERS_NO_PYTORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _load_wav(path: Path, *, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return np.asarray(y, dtype=np.float32), int(target_sr)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load audio {path}: {exc}")


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _iter_segments_from_csv(csv_path: Path, limit: int | None) -> Iterable[Segment]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            try:
                start = float(row.get("start", 0.0) or 0.0)
                end = float(row.get("end", start) or start)
            except Exception:
                continue
            text = (row.get("text") or "").strip()
            yield Segment(start, end, text)
            count += 1
            if limit is not None and count >= limit:
                break


def _iter_segments_fixed(duration: float, window: float = 2.0, hop: float = 2.0) -> Iterable[Segment]:
    t = 0.0
    while t < duration:
        yield Segment(t, min(t + window, duration), "")
        t += hop


def main() -> None:
    _set_env_cpu_only()

    parser = argparse.ArgumentParser(description="Run affect analysis only (ONNX-first).")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input audio file")
    parser.add_argument("--outdir", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--segments-csv", type=Path, default=None, help="Optional segments CSV to drive analysis (uses start/end/text)")
    parser.add_argument("--limit", type=int, default=20, help="Limit segments processed (for quick checks)")
    parser.add_argument("--models", type=Path, default=Path("D:/models"), help="Models root directory (defaults to D:/models)")
    args = parser.parse_args()

    models_root = args.models.expanduser().resolve()
    os.environ.setdefault("DIAREMOT_MODEL_DIR", os.fspath(models_root))

    # Resolve model subdirs
    text_dir = models_root / "text_emotions"
    ser_dir = models_root / "Affect" / "ser8-onnx-int8"
    if not ser_dir.exists():
        ser_dir = models_root / "Affect" / "ser8"
    vad_dir = models_root / "Affect" / "VAD_dim"
    intent_dir = models_root / "intent"

    # Lazy import after env guards
    from diaremot.affect.emotion_analyzer import EmotionIntentAnalyzer

    analyzer = EmotionIntentAnalyzer(
        affect_backend="onnx",
        affect_text_model_dir=os.fspath(text_dir),
        affect_ser_model_dir=os.fspath(ser_dir),
        affect_vad_model_dir=os.fspath(vad_dir),
        affect_intent_model_dir=os.fspath(intent_dir),
        disable_downloads=True,
    )

    y, sr = _load_wav(args.input)
    duration = len(y) / float(sr) if len(y) else 0.0

    # Build segments
    segments: list[Segment] = []
    if args.segments_csv and args.segments_csv.exists():
        segments.extend(_iter_segments_from_csv(args.segments_csv, args.limit))
    else:
        segments.extend(_iter_segments_fixed(duration, 2.0, 2.0))

    # Run
    results: list[dict[str, Any]] = []
    for seg in segments:
        i0 = max(0, int(seg.start * sr))
        i1 = max(i0, int(seg.end * sr))
        clip = y[i0:i1] if i1 > i0 else np.asarray([], dtype=np.float32)
        payload = analyzer.analyze(wav=clip, sr=sr, text=seg.text)
        results.append(
            {
                "start": seg.start,
                "end": seg.end,
                "emotion_top": payload.get("speech_emotion", {}).get("top"),
                "intent_top": payload.get("intent", {}).get("top"),
                "affect_hint": payload.get("affect_hint"),
                "valence": payload.get("vad", {}).get("valence"),
                "arousal": payload.get("vad", {}).get("arousal"),
                "dominance": payload.get("vad", {}).get("dominance"),
            }
        )

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "affect_only_summary.json").write_text(
        json.dumps({"count": len(results), "segments": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "count": len(results), "out": os.fspath(outdir)}, indent=2))


if __name__ == "__main__":
    main()

