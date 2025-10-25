"""Affect helpers shared across pipeline executors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..outputs import default_affect


class AffectMixin:
    def _affect_hint(self, v, a, d, intent):
        try:
            if a is None or v is None:
                return "neutral-status"
            if a > 0.5 and v < 0:
                return "agitated-negative"
            if a < 0.3 and v > 0.2:
                return "calm-positive"
            return f"neutral-{intent}"
        except Exception:  # pragma: no cover - defensive
            return "neutral-status"

    def _affect_unified(self, wav: np.ndarray, sr: int, text: str) -> dict[str, Any]:
        try:
            if hasattr(self.affect, "analyze"):
                res = self.affect.analyze(wav=wav, sr=sr, text=text)
                if getattr(self.affect, "issues", None):
                    for issue in self.affect.issues:
                        if issue not in self.stats.issues:
                            self.stats.issues.append(issue)
                return res or default_affect()
            return default_affect()
        except Exception as exc:  # pragma: no cover - best effort fallback
            self.corelog.warn(f"Affect analysis failed: {exc}")
            return default_affect()
