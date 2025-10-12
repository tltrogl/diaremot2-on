"""Thin wrapper around TensorFlow YAMNet for optional SED fallback."""

from __future__ import annotations

from typing import Sequence

LOGGER_NAME = "diaremot.sed.sed_yamnet_tf"


def run_sed(
    wav_path: str,
    *,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5,
    enter_thresh: float = 0.5,
    exit_thresh: float = 0.35,
    min_dur: float = 0.3,
    merge_gap: float = 0.2,
    topk: int = 3,
) -> list[dict[str, float | str]]:
    """Placeholder backend until the TensorFlow port is re-integrated."""

    raise RuntimeError(
        "The YAMNet backend requires TensorFlow and is not bundled by default. "
        "Install the optional dependencies and re-run with --backend yamnet."
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - defensive
    raise SystemExit(
        "This module is only imported indirectly via diaremot.sed.sed_panns_onnx --backend yamnet"
    )
