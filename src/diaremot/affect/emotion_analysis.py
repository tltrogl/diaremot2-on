"""
Compatibility shim for legacy imports.

This module re-exports the ONNX-first EmotionAnalyzer implemented in
`emotion_analyzer.py`. Prefer importing from `diaremot.affect.emotion_analyzer`.

Keeping this file ensures older code that imports
`diaremot.affect.emotion_analysis` does not break.
"""

from .emotion_analyzer import (
    GOEMOTIONS_LABELS,
    SER8_LABELS,
    EmotionAnalyzer,
    EmotionOutputs,
)

__all__ = [
    "EmotionAnalyzer",
    "EmotionOutputs",
    "GOEMOTIONS_LABELS",
    "SER8_LABELS",
]
