"""Thread-safe convenience wrapper around the diarization speaker registry.

This module provides a lightweight manager facade that keeps access to the
underlying :class:`diaremot.pipeline.speaker_diarization.SpeakerRegistry`
serialised across threads and offers a small set of convenience helpers
(auto-flush, reload).

The public factory :func:`diaremot.get_registry_manager` wires to this
manager so callers consistently receive a functional registry even though the
project does not ship a standalone ``speaker_registry_manager`` module.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


class SpeakerRegistryManager:
    """Manage persistence and concurrency for the speaker registry."""

    def __init__(self, registry_path: str, auto_flush: bool = True):
        self.path = Path(registry_path)
        self.auto_flush = auto_flush
        self._lock = threading.RLock()
        self._registry = self._create_registry()

    def _create_registry(self):
        from ..pipeline.speaker_diarization import SpeakerRegistry

        return SpeakerRegistry(str(self.path))

    def reload(self) -> None:
        """Reload registry state from disk."""

        with self._lock:
            self._registry = self._create_registry()

    def save(self) -> None:
        """Persist current registry state to disk."""

        with self._lock:
            self._registry.save()

    def has(self, name: str) -> bool:
        with self._lock:
            return self._registry.has(name)

    def enroll(self, name: str, centroid: Any) -> None:
        with self._lock:
            self._registry.enroll(name, centroid)
            if self.auto_flush:
                self._registry.save()

    def update_centroid(self, name: str, centroid: Any, alpha: float = 0.30) -> None:
        with self._lock:
            self._registry.update_centroid(name, centroid, alpha=alpha)
            if self.auto_flush:
                self._registry.save()

    def match(self, centroid: Any) -> tuple[str | None, float]:
        with self._lock:
            return self._registry.match(centroid)

    def get_speakers(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the stored speakers."""

        with self._lock:
            # ``SpeakerRegistry`` keeps the payload in ``_speakers``; copy it to
            # avoid leaking a mutable reference.
            return {k: dict(v) for k, v in getattr(self._registry, "_speakers", {}).items()}

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return dict(getattr(self._registry, "_metadata", {}))


__all__ = ["SpeakerRegistryManager"]
