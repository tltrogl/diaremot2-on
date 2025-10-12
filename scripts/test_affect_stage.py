from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import librosa  # type: ignore

        y, sr = librosa.load(str(path), sr=16000, mono=True)
        return np.asarray(y, dtype=np.float32), int(sr)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to load audio {path}: {exc}")


def _load_norm_tx(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:
                    break
                try:
                    start = float(row.get("start", 0.0) or 0.0)
                    end = float(row.get("end", start) or start)
                except Exception:
                    start, end = 0.0, 0.0
                rows.append(
                    {
                        "start": start,
                        "end": end,
                        "text": row.get("text", ""),
                        "speaker_id": row.get("speaker_id", "Speaker_1"),
                        "speaker_name": row.get("speaker_name", "Speaker_1"),
                        "asr_logprob_avg": row.get("asr_logprob_avg"),
                        "snr_db": row.get("snr_db"),
                    }
                )
    except FileNotFoundError:
        pass
    return rows


def main() -> None:
    from diaremot.pipeline.orchestrator import AudioAnalysisPipelineV2
    from diaremot.pipeline.stages.base import PipelineState
    from diaremot.pipeline.logging_utils import StageGuard
    from diaremot.pipeline.stages import affect as stage_affect

    root = Path.cwd()
    in_path = root / "data" / "sample1.mp3"
    out_dir = root / "outputs" / "_affect_stage_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = _load_audio(in_path)

    # Build a tiny norm_tx from prior run if present; otherwise synthesize one
    prior_csv = root / "outputs" / "sample1_run" / "diarized_transcript_with_emotion.csv"
    norm_tx = _load_norm_tx(prior_csv)
    if not norm_tx:
        # Fallback segments (2s windows)
        norm_tx = [
            {"start": 0.0, "end": 2.0, "text": "hello there", "speaker_id": "Speaker_1", "speaker_name": "Speaker_1"},
            {"start": 2.0, "end": 4.0, "text": "testing affect stage", "speaker_id": "Speaker_1", "speaker_name": "Speaker_1"},
        ]

    # Configure pipeline with affect enabled; allow neutral defaults if models missing
    cfg: dict[str, Any] = {
        "disable_affect": False,
        "affect_backend": "auto",
        "disable_downloads": True,
        "enable_sed": False,
        "quiet": True,
    }
    pipe = AudioAnalysisPipelineV2(cfg)
    pipe.stats.file_id = in_path.name

    state = PipelineState(input_audio_path=str(in_path), out_dir=out_dir)
    state.y = y
    state.sr = sr
    state.norm_tx = norm_tx

    guard = StageGuard(pipe.corelog, pipe.stats, "affect_and_assemble")
    with guard:
        stage_affect.run(pipe, state, guard)

    # Summarize first few rows
    summary = []
    for row in state.segments_final[:5]:
        summary.append(
            {
                "start": row.get("start"),
                "end": row.get("end"),
                "emotion_top": row.get("emotion_top"),
                "intent_top": row.get("intent_top"),
                "affect_hint": row.get("affect_hint"),
            }
        )

    (out_dir / "affect_stage_summary.json").write_text(
        json.dumps({"segments": summary, "count": len(state.segments_final)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "count": len(state.segments_final), "out": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

