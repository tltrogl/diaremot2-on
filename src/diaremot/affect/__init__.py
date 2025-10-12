"""Affect analysis modules (emotion, intent, paralinguistics)."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_SUBMODULES: dict[str, str] = {
    "emotion_analysis": "diaremot.affect.emotion_analysis",
    "emotion_analyzer": "diaremot.affect.emotion_analyzer",
    "intent_defaults": "diaremot.affect.intent_defaults",
    "paralinguistics": "diaremot.affect.paralinguistics",
    "sed_panns": "diaremot.affect.sed_panns",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
    """Lazily import affect submodules when accessed via the package."""

    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

+


def __dir__() -> list[str]:
    """Expose lazy submodules through :func:`dir`."""

    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()))
