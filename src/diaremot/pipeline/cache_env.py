"""Compatibility shim for legacy cache configuration imports."""

from __future__ import annotations

from .runtime_env import configure_local_cache_env

__all__ = ["configure_local_cache_env"]
