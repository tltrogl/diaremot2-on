"""Helpers for discovering DiaRemot model roots in a cross-platform way."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["iter_model_roots", "iter_model_subpaths", "collect_model_roots"]


def _dedupe_paths(paths: Iterable[Path]) -> Iterator[Path]:
    seen: set[str] = set()
    for path in paths:
        # Expand user but avoid resolving non-existent drives (e.g., D:/ on POSIX).
        expanded = path.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        yield expanded


def iter_model_roots(extra: Iterable[str | Path] | None = None) -> Iterator[Path]:
    """Yield candidate model roots in priority order.

    Priority:
    1. Explicit entries in extra (already expanded by callers)
    2. DIAREMOT_MODEL_DIR environment variable
    3. Current working directory ./models
    4. User home ~/models
    5. OS-specific defaults (Windows: D:/models and D:/diaremot/diaremot2-1/models;
       POSIX: /models and /opt/diaremot/models)
    """

    extra_paths = [] if extra is None else [Path(p) for p in extra]
    env_root = os.environ.get("DIAREMOT_MODEL_DIR")
    roots: list[Path] = []
    roots.extend(extra_paths)
    if env_root:
        roots.append(Path(env_root))
    roots.append(Path.cwd() / "models")
    roots.append(Path.home() / "models")
    if os.name == "nt":
        roots.append(Path("D:/models"))
        roots.append(Path("D:/diaremot/diaremot2-1/models"))
    else:
        roots.append(Path("/models"))
        roots.append(Path("/opt/diaremot/models"))

    yield from _dedupe_paths(roots)


def iter_model_subpaths(
    relative_path: str | Path, *, extra_roots: Iterable[str | Path] | None = None
) -> Iterator[Path]:
    """Yield candidate concrete paths under each known model root."""

    rel = Path(relative_path)
    for root in iter_model_roots(extra_roots):
        yield root / rel


def collect_model_roots(extra: Iterable[str | Path] | None = None) -> list[Path]:
    """Return a materialized list of candidate model roots."""

    return list(iter_model_roots(extra))
