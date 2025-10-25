"""Optional dependency management for paralinguistics features."""

from __future__ import annotations

from importlib import import_module
from importlib import util as importlib_util
import warnings
from types import ModuleType
from typing import Final


def _load_optional_module(module_name: str) -> ModuleType | None:
    """Return the imported module when available, otherwise ``None``."""

    try:
        spec = importlib_util.find_spec(module_name)
    except (ImportError, ValueError):  # ValueError when extension modules expose no spec
        return None
    if spec is None:
        return None
    try:
        return import_module(module_name)
    except Exception:  # pragma: no cover - defensive best-effort import
        warnings.warn(
            f"failed to import optional module '{module_name}'", RuntimeWarning, stacklevel=2
        )
        return None


librosa = _load_optional_module("librosa")  # type: ignore[assignment]
if librosa is None:
    warnings.warn(
        "librosa not available - paralinguistic features will be limited",
        RuntimeWarning,
        stacklevel=2,
    )
LIBROSA_AVAILABLE: Final[bool] = librosa is not None

scipy_signal = _load_optional_module("scipy.signal")  # type: ignore[assignment]
if scipy_signal is None:
    warnings.warn(
        "scipy not available - some voice quality features disabled",
        RuntimeWarning,
        stacklevel=2,
    )
SCIPY_AVAILABLE: Final[bool] = scipy_signal is not None

parselmouth = _load_optional_module("parselmouth")  # type: ignore[assignment]
if parselmouth is None:
    call = None  # type: ignore[assignment]
else:
    praat = _load_optional_module("parselmouth.praat")
    call = getattr(praat, "call", None) if praat is not None else None
if parselmouth is None or call is None:
    warnings.warn(
        "parselmouth not available - Praat-based voice metrics disabled",
        RuntimeWarning,
        stacklevel=2,
    )
PARSELMOUTH_AVAILABLE: Final[bool] = parselmouth is not None and call is not None

__all__ = [
    "librosa",
    "scipy_signal",
    "parselmouth",
    "call",
    "LIBROSA_AVAILABLE",
    "SCIPY_AVAILABLE",
    "PARSELMOUTH_AVAILABLE",
]
