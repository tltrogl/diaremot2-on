"""Helpers for importing ONNXRuntime with graceful degradation.

This module centralizes the import of :mod:`onnxruntime` so that the rest of
the codebase can share consistent fallback behaviour when the native extension
fails to load (commonly observed on Windows when the Microsoft Visual C++
runtime is missing).

The helper keeps track of the last import failure and exposes a descriptive
message that downstream callers can surface before switching to the PyTorch
fallback implementations.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from types import ModuleType

logger = logging.getLogger(__name__)

_CACHED_ORT: ModuleType | None = None
_CACHED_ERROR: Exception | None = None
_ATTEMPTED = False


class OnnxRuntimeUnavailable(RuntimeError):
    """Raised when the onnxruntime Python extension cannot be imported."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause


def _format_windows_hint() -> str:
    if sys.platform.startswith("win"):
        return (
            "Install the Microsoft Visual C++ 2015-2022 Redistributable and "
            "ensure the onnxruntime wheel matches your CPU architecture."
        )
    return ""


def _format_generic_hint() -> str:
    return (
        "ONNX features will be disabled automatically; PyTorch fallbacks will "
        "be used when available."
    )


def _import_onnxruntime(force_reload: bool = False) -> ModuleType | None:
    global _CACHED_ORT, _CACHED_ERROR, _ATTEMPTED

    if not force_reload and _ATTEMPTED:
        return _CACHED_ORT

    try:
        ort = importlib.import_module("onnxruntime")
    except Exception as exc:  # pragma: no cover - platform/runtime dependent
        _CACHED_ORT = None
        _CACHED_ERROR = exc
        _ATTEMPTED = True
        logger.warning("ONNXRuntime import failed: %s", exc)
        return None

    _CACHED_ORT = ort
    _CACHED_ERROR = None
    _ATTEMPTED = True
    return ort


def get_onnxruntime(force_reload: bool = False) -> ModuleType | None:
    """Return the onnxruntime module or ``None`` if it failed to import."""

    return _import_onnxruntime(force_reload=force_reload)


def onnxruntime_available() -> bool:
    """Return ``True`` when :mod:`onnxruntime` imported successfully."""

    return get_onnxruntime() is not None


def last_onnxruntime_error() -> Exception | None:
    """Return the last import error, if any."""

    return _CACHED_ERROR


def format_unavailable_message() -> str:
    """Construct a human readable explanation for an unavailable import."""

    base = str(_CACHED_ERROR) if _CACHED_ERROR else "onnxruntime import failed"
    hints = [_format_generic_hint()]
    win_hint = _format_windows_hint()
    if win_hint:
        hints.append(win_hint)
    env_hint = os.environ.get("DIAREMOT_MODEL_DIR")
    if env_hint:
        hints.append(f"Checked DIAREMOT_MODEL_DIR={env_hint}")
    return f"{base}. {' '.join(hints)}".strip()


def ensure_onnxruntime(force_reload: bool = False) -> ModuleType:
    """Return the imported module or raise :class:`OnnxRuntimeUnavailable`."""

    ort = get_onnxruntime(force_reload=force_reload)
    if ort is None:
        message = format_unavailable_message()
        raise OnnxRuntimeUnavailable(message, cause=_CACHED_ERROR)
    return ort


__all__ = [
    "OnnxRuntimeUnavailable",
    "ensure_onnxruntime",
    "format_unavailable_message",
    "get_onnxruntime",
    "last_onnxruntime_error",
    "onnxruntime_available",
]
