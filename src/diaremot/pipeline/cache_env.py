"""Shared helpers for configuring cache-related environment variables."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["configure_local_cache_env"]


def _should_skip(existing: str, target_path: Path, cache_root: Path) -> bool:
    """Return True if existing path already satisfies the cache requirement."""
    try:
        existing_path = Path(existing).resolve()
    except (OSError, RuntimeError, ValueError):
        return False

    if existing_path == target_path:
        return True
    try:
        # Python <3.9 doesn't have is_relative_to; emulate to preserve behaviour.
        existing_path.relative_to(cache_root)
        return True
    except ValueError:
        return False


def configure_local_cache_env() -> None:
    """Ensure HuggingFace/Torch caches resolve inside the repo-local `.cache` dir."""
    cache_root = (Path(__file__).resolve().parents[3] / ".cache").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    targets = {
        "HF_HOME": cache_root / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_root / "hf",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }

    for env_name, target in targets.items():
        target_path = target.resolve()
        existing = os.environ.get(env_name)
        if existing and _should_skip(existing, target_path, cache_root):
            continue
        target_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(target_path)


_configured = False


def _configure_once() -> None:
    global _configured
    if not _configured:
        configure_local_cache_env()
        _configured = True


_configure_once()
