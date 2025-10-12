"""Lightweight provider registry used for dependency injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Factory = Callable[..., Any]
_PROVIDERS: dict[str, Factory] = {}


def register_provider(name: str, factory: Factory) -> None:
    """Register a factory for pipeline components."""

    _PROVIDERS[name] = factory


def get_provider(name: str, default: Factory | None = None) -> Factory:
    """Fetch a provider, optionally falling back to a default factory."""

    if name in _PROVIDERS:
        return _PROVIDERS[name]
    if default is None:
        raise KeyError(f"No provider registered for '{name}'")
    _PROVIDERS[name] = default
    return default


def provide(name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve a provider and invoke it with the supplied arguments."""

    factory = get_provider(name)
    return factory(*args, **kwargs)


__all__ = ["register_provider", "get_provider", "provide"]
