"""Utility helpers shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_audio_sha16(y: np.ndarray) -> str:
    try:
        arr = np.array(y, dtype=np.float32)
    except Exception:
        return hashlib.blake2s(b"", digest_size=16).hexdigest()

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr, dtype=np.float32, copy=True)
        except Exception:
            return hashlib.blake2s(b"", digest_size=16).hexdigest()
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    return hashlib.blake2s(arr.tobytes(), digest_size=16).hexdigest()


def compute_audio_sha16_from_file(path: str) -> str:
    """Compute hash from raw file bytes (fast, no audio decode)."""
    try:
        hasher = hashlib.blake2s(digest_size=16)
        with open(path, "rb") as f:
            # Read in 1MB chunks
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def compute_pp_signature(pp_conf: Any) -> str:
    """Compute a deterministic string signature of preprocessing config."""
    keys = ["target_sr", "denoise", "loudness_mode"]
    sig: dict[str, Any] = {}
    for key in keys:
        try:
            sig[key] = getattr(pp_conf, key)
        except Exception:
            sig[key] = None
    return json.dumps(sig, sort_keys=True)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json_safe(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


__all__ = [
    "atomic_write_json",
    "compute_audio_sha16",
    "compute_audio_sha16_from_file",
    "compute_pp_signature",
    "read_json_safe",
]
