"""Summary generation utilities and conversation analytics."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_SUBMODULES: dict[str, str] = {
    "conversation_analysis": "diaremot.summaries.conversation_analysis",
    "html_summary_generator": "diaremot.summaries.html_summary_generator",
    "pdf_summary_generator": "diaremot.summaries.pdf_summary_generator",
    "speakers_summary_builder": "diaremot.summaries.speakers_summary_builder",
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
