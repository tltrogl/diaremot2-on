"""Lightweight runtime dependency validation for the audio pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from typing import Any

try:  # pragma: no cover - packaging may not be installed in some runtimes
    from packaging.version import Version
except Exception:  # pragma: no cover - fallback if packaging missing
    Version = None  # type: ignore

from .pipeline_config import CORE_DEPENDENCY_REQUIREMENTS


def _iter_dependency_status() -> Iterator[
    tuple[str, str, Any, str | None, Exception | None, Exception | None]
]:
    for mod, min_ver in CORE_DEPENDENCY_REQUIREMENTS.items():
        import_error: Exception | None = None
        metadata_error: Exception | None = None
        module = None
        try:
            module = __import__(mod.replace("-", "_"))
        except Exception as exc:
            import_error = exc

        version: str | None = None
        if module is not None:
            try:
                version = importlib_metadata.version(mod)
            except importlib_metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            except Exception as exc:
                metadata_error = exc
        yield mod, min_ver, module, version, import_error, metadata_error


def _verify_core_dependencies(require_versions: bool = False) -> tuple[bool, list[str]]:
    """Verify core runtime dependencies are importable (and optionally versioned)."""

    issues: list[str] = []

    for (
        mod,
        min_ver,
        module,
        version,
        import_error,
        metadata_error,
    ) in _iter_dependency_status():
        if import_error is not None or module is None:
            issues.append(f"Missing or failed to import: {mod} ({import_error})")
            continue

        if not require_versions:
            continue

        if version is None:
            reason = metadata_error or "version metadata unavailable"
            issues.append(f"Version unknown for {mod}; require >= {min_ver} ({reason})")
            continue

        if Version is None:
            continue

        try:
            if Version(version) < Version(min_ver):
                issues.append(f"{mod} version {version} < required {min_ver}")
        except Exception as exc:
            issues.append(f"Version check failed for {mod}: {exc}")

    return (len(issues) == 0), issues


def _dependency_health_summary() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for (
        mod,
        min_ver,
        module,
        version,
        import_error,
        metadata_error,
    ) in _iter_dependency_status():
        entry: dict[str, Any] = {"required_min": min_ver}

        if import_error is not None or module is None:
            entry["status"] = "error"
            entry["issue"] = str(import_error)
            summary[mod] = entry
            continue

        entry["status"] = "ok"

        if metadata_error is not None:
            entry["status"] = "warn"
            entry["issue"] = f"version lookup failed: {metadata_error}"

        if version is not None:
            entry["version"] = str(version)
            if Version is not None:
                try:
                    if Version(version) < Version(min_ver):
                        entry["status"] = "warn"
                        entry["issue"] = f"version {version} < required {min_ver}"
                except Exception as exc:
                    entry["status"] = "warn"
                    entry["issue"] = f"version comparison failed: {exc}"
        else:
            entry.setdefault("issue", "version metadata unavailable")

        summary[mod] = entry

    return summary


def verify_dependencies(strict: bool = False) -> tuple[bool, list[str]]:
    """Expose lightweight dependency verification for external callers."""

    return _verify_core_dependencies(require_versions=strict)


def diagnostics(require_versions: bool = False) -> dict[str, Any]:
    """Return diagnostic information about optional runtime dependencies."""

    ok, issues = _verify_core_dependencies(require_versions=require_versions)
    return {
        "ok": ok,
        "issues": issues,
        "summary": _dependency_health_summary(),
        "strict_versions": require_versions,
    }


def dependency_summary() -> dict[str, dict[str, Any]]:
    """Expose the structured dependency summary for diagnostics consumers."""

    return _dependency_health_summary()


__all__ = [
    "dependency_summary",
    "diagnostics",
    "verify_dependencies",
]
