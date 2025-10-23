"""Runtime environment helpers for pipeline execution."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "WINDOWS_MODELS_ROOT",
    "DEFAULT_MODELS_ROOT",
    "MODEL_ROOTS",
    "DEFAULT_WHISPER_MODEL",
    "configure_local_cache_env",
    "resolve_default_whisper_model",
    "iter_model_roots",
    "set_primary_model_root",
]


WINDOWS_CANONICAL_MODEL_ROOT = Path("D:/models")
LINUX_CANONICAL_MODEL_ROOT = Path("/srv/models")
# Canonical cache base used across platforms for deterministic tests
WINDOWS_CACHE_BASE = Path("D:/srv/.cache")
LINUX_CACHE_BASE = Path("/srv/.cache")


PROJECT_ROOT_MARKERS = ("pyproject.toml", ".git")


_MODEL_SENTINELS: tuple[tuple[str, ...], ...] = (
    ("Diarization", "ecapa-onnx", "ecapa_tdnn.onnx"),
    ("Diarization", "silaro_vad", "silero_vad.onnx"),
    ("Affect", "ser8", "model.int8.onnx"),
    ("Affect", "ser8", "model.onnx"),
    ("Affect", "VAD_dim", "model.onnx"),
    ("Affect", "sed_panns", "model.onnx"),
    ("text_emotions", "model.int8.onnx"),
    ("text_emotions", "model.onnx"),
    ("intent", "model_int8.onnx"),
    ("intent", "model.onnx"),
    ("tiny.en", "model.bin"),
    ("faster-whisper", "tiny.en", "model.bin"),
)


def _find_project_root(start: Path) -> Path | None:
    """Walk upward from ``start`` until a project marker is found."""

    current = start
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        for marker in PROJECT_ROOT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return None


def _is_cache_dir(path: Path) -> bool:
    """Return True when ``path`` appears to represent a cache location."""

    parts = {part.lower() for part in path.parts}
    return any(part.startswith(".cache") or part.endswith(".cache") for part in parts)


def _has_model_artifacts(root: Path) -> bool:
    """Heuristically detect whether ``root`` contains installed model files."""

    try:
        if not root.exists():
            return False
    except OSError:
        return False

    try:
        for child in root.iterdir():
            if child.is_file() and child.suffix.lower() == ".onnx":
                return True
    except OSError:
        pass

    for relative in _MODEL_SENTINELS:
        try:
            if (root.joinpath(*relative)).exists():
                return True
        except OSError:
            continue

    return False


def _candidate_cache_roots(script_path: Path) -> Iterable[Path]:
    project_root = _find_project_root(script_path)
    candidates: list[Path] = []
    # Always start from the canonical Linux-style cache root to keep behavior
    # consistent across platforms and tests.
    candidates.append(LINUX_CACHE_BASE if os.name != "nt" else WINDOWS_CACHE_BASE)
    if project_root is not None:
        candidates.append(project_root / ".cache")

    candidates.extend(
        [
            Path.cwd() / ".cache",
            Path.home() / ".cache" / "diaremot",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _ensure_writable_directory(path: Path) -> bool:
    """Return ``True`` when ``path`` can be created and written to."""

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    probe = path / ".cache_write_test"
    try:
        probe.touch(exist_ok=True)
    except OSError:
        return False
    else:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            return False
    return os.access(path, os.W_OK | os.X_OK)


def configure_local_cache_env() -> None:
    """Ensure all cache directories resolve to a writable, local cache root."""

    cache_root = None
    script_path = Path(__file__).resolve()
    for candidate in _candidate_cache_roots(script_path):
        resolved = candidate.resolve()
        if _ensure_writable_directory(resolved):
            cache_root = resolved
            break
    if cache_root is None:
        raise PermissionError("Unable to locate a writable cache directory for DiaRemot")

    # Use a stable subdirectory layout across platforms
    hf_home = cache_root / "hf"
    targets = {
        "HF_HOME": hf_home,
        "HUGGINGFACE_HUB_CACHE": hf_home,
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
    }
    for env_name, target in targets.items():
        target_path = target.resolve()
        existing = os.environ.get(env_name)
        if existing:
            try:
                existing_path = Path(existing).expanduser().resolve()
            except (OSError, RuntimeError, ValueError):
                existing_path = None
            if existing_path is not None:
                try:
                    if existing_path == target_path or existing_path.is_relative_to(cache_root):
                        continue
                except AttributeError:
                    if str(existing_path).startswith(str(cache_root)):
                        continue
        target_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(target_path)

    # Disable HF accelerated transfer by default to avoid requiring the
    # optional 'hf_transfer' package in minimal/air‑gapped installs.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


configure_local_cache_env()


def _default_model_root() -> Path:
    canonical = WINDOWS_CANONICAL_MODEL_ROOT if os.name == "nt" else LINUX_CANONICAL_MODEL_ROOT
    project_root = _find_project_root(Path(__file__).resolve())

    candidates: list[Path] = [canonical]
    if project_root is not None:
        candidates.append(project_root / "models")
    candidates.extend(
        [
            Path.cwd() / "models",
            Path.home() / "models",
        ]
    )

    for candidate in candidates:
        if _has_model_artifacts(candidate) or _ensure_writable_directory(candidate):
            return candidate

    fallback = Path.cwd() / ".cache" / "models"
    _ensure_writable_directory(fallback)
    return fallback


def _resolve_model_root_from_env() -> Path:
    env_root = os.environ.get("DIAREMOT_MODEL_DIR")
    if env_root:
        return Path(env_root).expanduser()

    default_root = _default_model_root()
    os.environ.setdefault("DIAREMOT_MODEL_DIR", str(default_root))
    return default_root


_PRIMARY_MODEL_ROOT = _resolve_model_root_from_env()


def _discover_model_roots() -> list[Path]:
    roots: list[Path] = []
    roots.append(_PRIMARY_MODEL_ROOT)

    if os.name == "nt":
        roots.append(WINDOWS_CANONICAL_MODEL_ROOT)
    else:
        roots.append(LINUX_CANONICAL_MODEL_ROOT)

    project_root = _find_project_root(Path(__file__).resolve())
    if project_root:
        roots.append(project_root / "models")

    roots.append(Path.cwd() / "models")
    roots.append(Path.home() / "models")
    roots.append(Path.cwd() / ".cache" / "models")

    seen: set[str] = set()
    deduped: list[Path] = []
    for candidate in roots:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    local_roots = [root for root in deduped if not _is_cache_dir(root) and _has_model_artifacts(root)]
    if local_roots:
        primary_local = local_roots[0]
        ordered: list[Path] = [primary_local]
        for root in deduped:
            if root == primary_local:
                continue
            if _is_cache_dir(root):
                continue
            ordered.append(root)
        deduped = ordered
    return deduped


MODEL_ROOTS: tuple[Path, ...] = tuple(_discover_model_roots()) or (Path("/models"),)
DEFAULT_MODELS_ROOT: Path = MODEL_ROOTS[0]
WINDOWS_MODELS_ROOT = DEFAULT_MODELS_ROOT if os.name == "nt" else None


def iter_model_roots() -> tuple[Path, ...]:
    """Return an ordered tuple of candidate model roots."""

    return MODEL_ROOTS


def set_primary_model_root(path: Path) -> None:
    """Override the primary model root and refresh cached search paths."""

    global _PRIMARY_MODEL_ROOT, MODEL_ROOTS, DEFAULT_MODELS_ROOT, WINDOWS_MODELS_ROOT

    resolved = Path(path).expanduser().resolve()
    os.environ["DIAREMOT_MODEL_DIR"] = str(resolved)
    _PRIMARY_MODEL_ROOT = resolved
    MODEL_ROOTS = tuple(_discover_model_roots()) or (resolved,)
    DEFAULT_MODELS_ROOT = MODEL_ROOTS[0]
    if os.name == "nt":
        WINDOWS_MODELS_ROOT = DEFAULT_MODELS_ROOT


def _iter_whisper_candidates() -> list[Path]:
    candidates: list[Path] = []
    for root in MODEL_ROOTS:
        candidates.extend(
            [
                root / "tiny.en",
                root / "faster-whisper" / "tiny.en",
                root / "tiny.en",
                root / "ct2" / "tiny.en",
            ]
        )
    candidates.append(Path.home() / "whisper_models" / "tiny.en")
    return candidates


def resolve_default_whisper_model() -> Path:
    env_override = os.environ.get("WHISPER_MODEL_PATH")
    if env_override:
        return Path(env_override)

    for candidate in _iter_whisper_candidates():
        if candidate.exists():
            return candidate

    return _iter_whisper_candidates()[0]


DEFAULT_WHISPER_MODEL = resolve_default_whisper_model()
