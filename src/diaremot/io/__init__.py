"""Persistence and model I/O helpers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_SUBMODULES: dict[str, str] = {
    "download_utils": "diaremot.io.download_utils",
    "onnx_utils": "diaremot.io.onnx_utils",
    "speaker_registry_manager": "diaremot.io.speaker_registry_manager",
}

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUBMODULES.keys()))
