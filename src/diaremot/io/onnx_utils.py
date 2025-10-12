"""Utilities for ensuring and loading ONNX Runtime models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.hash import hash_file
from .onnx_runtime_guard import (
    OnnxRuntimeUnavailable,
    ensure_onnxruntime,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onnxruntime import InferenceSession as OrtInferenceSession
else:  # pragma: no cover - runtime safe fallback
    OrtInferenceSession = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _check_sha256(path: Path, sha256: str) -> None:
    """Validate file integrity via SHA256."""
    digest = hash_file(path, algo="sha256")
    if digest.lower() != sha256.lower():
        raise RuntimeError(f"SHA256 mismatch for {path}: {digest} != {sha256}")


def ensure_onnx_model(
    path_or_hf_id: str | Path,
    *,
    sha256: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local path for a model, downloading from Hugging Face if needed.

    Parameters
    ----------
    path_or_hf_id:
        Either a local filesystem path or a Hugging Face identifier of the form
        ``hf://<repo>/<filename>`` or ``<repo>/<filename>``.
    sha256:
        Optional hex digest to verify file integrity.

    Returns
    -------
    Path
        Local filesystem path to the requested file. Raises on failure.
    """
    path = Path(path_or_hf_id)
    if path.exists():
        if sha256:
            _check_sha256(path, sha256)
        return path

    ident = str(path_or_hf_id)
    try:
        from huggingface_hub import hf_hub_download

        if ident.startswith("hf://"):
            repo_id, filename = ident[5:].rsplit("/", 1)
        elif "/" in ident and not ident.startswith("./") and not ident.startswith("../"):
            repo_id, filename = ident.rsplit("/", 1)
        else:
            raise FileNotFoundError(ident)

        path = Path(hf_hub_download(repo_id, filename, local_files_only=local_files_only))
        if sha256:
            _check_sha256(path, sha256)
        return path
    except Exception as exc:  # pragma: no cover - network/environment issues
        logger.info("Could not retrieve %s: %s", ident, exc)
        raise


def create_onnx_session(
    model_path: str | Path, *, cpu_only: bool = True, threads: int = 1
) -> OrtInferenceSession:
    """Create an ONNX Runtime session with consistent CPU behaviour."""
    ort = ensure_onnxruntime()
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
    # Enable full graph optimizations for better CPU performance
    try:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        # Older ORT versions may not expose the enum; safe to ignore
        pass
    providers = ["CPUExecutionProvider"] if cpu_only else ort.get_available_providers()
    try:
        return ort.InferenceSession(str(model_path), providers=providers, sess_options=opts)
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OnnxRuntimeUnavailable(
            f"Failed to initialize ONNX Runtime session for {model_path}: {exc}",
            cause=exc,
        ) from exc


__all__ = ["ensure_onnx_model", "create_onnx_session"]
