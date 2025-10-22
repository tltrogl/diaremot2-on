"""Configuration defaults and dependency helpers for the DiaRemot pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from dataclasses import fields as dataclass_fields
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:  # pragma: no cover - packaging optional during tests
    from packaging.version import Version
except Exception:  # pragma: no cover - defensive fallback
    Version = None  # type: ignore

from .speaker_diarization import DiarizationConfig


def _ensure_numeric_range(
    name: str,
    value: float,
    *,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
) -> None:
    """Validate numeric range constraints for configuration fields."""

    if ge is not None and value < ge:
        raise ValueError(f"{name} must be >= {ge}")
    if gt is not None and value <= gt:
        raise ValueError(f"{name} must be > {gt}")
    if le is not None and value > le:
        raise ValueError(f"{name} must be <= {le}")
    if lt is not None and value >= lt:
        raise ValueError(f"{name} must be < {lt}")


def _coerce_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


@dataclass(slots=True)
class PipelineConfig:
    """Validated configuration for the end-to-end pipeline."""

    registry_path: Path = Path("speaker_registry.json")
    ahc_distance_threshold: float = DiarizationConfig.ahc_distance_threshold
    speaker_limit: int | None = None
    clustering_backend: str = "ahc"
    min_speakers: int | None = None
    max_speakers: int | None = None
    whisper_model: str = "tiny.en"
    asr_backend: str = "faster"
    compute_type: str = "int8"
    cpu_threads: int = 1
    language: str | None = None
    language_mode: str = "auto"
    ignore_tx_cache: bool = False
    quiet: bool = False
    disable_affect: bool = False
    affect_backend: str = "onnx"
    affect_text_model_dir: Path | None = None
    affect_intent_model_dir: Path | None = None
    affect_ser_model_dir: Path | None = None
    affect_vad_model_dir: Path | None = None
    affect_analyzer_threads: int | None = None
    beam_size: int = 1
    temperature: float = 0.0
    no_speech_threshold: float = 0.20
    noise_reduction: bool = False
    enable_sed: bool = True
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = 60.0
    chunk_size_minutes: float = 20.0
    chunk_overlap_seconds: float = 30.0
    vad_threshold: float = 0.22
    vad_min_speech_sec: float = 0.25
    vad_min_silence_sec: float = 0.25
    vad_speech_pad_sec: float = 0.2
    vad_backend: str = "auto"
    disable_energy_vad_fallback: bool = False
    energy_gate_db: float = -33.0
    energy_hop_sec: float = 0.01
    max_asr_window_sec: int = 480
    segment_timeout_sec: float = 300.0
    batch_timeout_sec: float = 1200.0
    cpu_diarizer: bool = False
    # Prefer local model assets before any remote download/caching.
    local_first: bool = True
    validate_dependencies: bool = False
    strict_dependency_versions: bool = False
    cache_root: Path = Path(".cache")
    cache_roots: list[Path] = dataclass_field(default_factory=list)
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    target_sr: int = 16000
    loudness_mode: str = "asr"
    run_id: str | None = None
    text_emotion_model: str | None = None
    intent_labels: list[str] | None = None
    sed_mode: str = "auto"
    sed_window_sec: float = 1.0
    sed_hop_sec: float = 0.5
    sed_enter: float = 0.50
    sed_exit: float = 0.35
    sed_min_dur: dict[str, float] = dataclass_field(default_factory=dict)
    sed_merge_gap: float = 0.20
    sed_classmap_csv: Path | None = None
    sed_median_k: int = 5
    sed_timeline_jsonl: bool = False
    sed_batch_size: int = 256
    sed_default_min_dur: float = 0.30

    def __post_init__(self) -> None:  # noqa: D401 - dataclass validation helper
        """Validate and normalise configuration fields."""

        self.registry_path = Path(self.registry_path)
        self.cache_root = Path(self.cache_root)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.affect_text_model_dir = _coerce_optional_path(self.affect_text_model_dir)
        self.affect_intent_model_dir = _coerce_optional_path(self.affect_intent_model_dir)
        self.affect_ser_model_dir = _coerce_optional_path(
            getattr(self, "affect_ser_model_dir", None)
        )
        self.affect_vad_model_dir = _coerce_optional_path(
            getattr(self, "affect_vad_model_dir", None)
        )
        self.sed_classmap_csv = _coerce_optional_path(self.sed_classmap_csv)

        if isinstance(self.cache_roots, (str, Path)):
            self.cache_roots = [Path(self.cache_roots)]
        else:
            self.cache_roots = [Path(path) for path in (self.cache_roots or [])]

        if self.intent_labels is not None:
            if not isinstance(self.intent_labels, Iterable) or isinstance(
                self.intent_labels, (str, bytes)
            ):
                raise ValueError("intent_labels must be an iterable of strings")
            self.intent_labels = [str(label) for label in self.intent_labels]

        if self.affect_analyzer_threads is not None:
            try:
                threads = int(self.affect_analyzer_threads)
            except (TypeError, ValueError) as exc:
                raise ValueError("affect_analyzer_threads must be an integer > 0") from exc
            self._validate_positive_int("affect_analyzer_threads", threads)
            self.affect_analyzer_threads = threads

        self.affect_backend = self._lower_choice(
            "affect_backend", self.affect_backend, {"auto", "onnx", "torch"}
        )
        self.asr_backend = self._lower_choice("asr_backend", self.asr_backend, None)
        self.vad_backend = self._lower_choice("vad_backend", self.vad_backend, {"auto", "onnx"})
        self.loudness_mode = self._lower_choice(
            "loudness_mode", self.loudness_mode, {"asr", "broadcast"}
        )
        self.language_mode = self._lower_choice("language_mode", self.language_mode, None)

        if self.speaker_limit is not None and self.speaker_limit < 1:
            raise ValueError("speaker_limit must be >= 1 or None")

        self._validate_positive_int("cpu_threads", self.cpu_threads)
        self._validate_positive_int("beam_size", self.beam_size)
        self._validate_positive_int("max_asr_window_sec", self.max_asr_window_sec)
        self._validate_positive_float("chunk_threshold_minutes", self.chunk_threshold_minutes)
        self._validate_positive_float("chunk_size_minutes", self.chunk_size_minutes)
        _ensure_numeric_range("chunk_overlap_seconds", self.chunk_overlap_seconds, ge=0.0)
        if isinstance(self.enable_sed, bool):
            pass
        elif isinstance(self.enable_sed, (int, float)):
            self.enable_sed = bool(self.enable_sed)
        else:
            raise ValueError("enable_sed must be a boolean value")
        self.enable_sed = bool(self.enable_sed)
        _ensure_numeric_range("vad_threshold", self.vad_threshold, ge=0.0, le=1.0)
        _ensure_numeric_range("temperature", self.temperature, ge=0.0, le=1.0)
        _ensure_numeric_range("no_speech_threshold", self.no_speech_threshold, ge=0.0, le=1.0)
        _ensure_numeric_range("vad_min_speech_sec", self.vad_min_speech_sec, ge=0.0)
        _ensure_numeric_range("vad_min_silence_sec", self.vad_min_silence_sec, ge=0.0)
        _ensure_numeric_range("vad_speech_pad_sec", self.vad_speech_pad_sec, ge=0.0)
        _ensure_numeric_range("energy_hop_sec", self.energy_hop_sec, gt=0.0)
        _ensure_numeric_range("segment_timeout_sec", self.segment_timeout_sec, gt=0.0)
        _ensure_numeric_range("batch_timeout_sec", self.batch_timeout_sec, gt=0.0)
        self._validate_positive_int("target_sr", self.target_sr)
        _ensure_numeric_range("ahc_distance_threshold", self.ahc_distance_threshold, ge=0.0)

        # Diarization clustering backend validation
        self.clustering_backend = self._lower_choice(
            "clustering_backend", self.clustering_backend, {"ahc", "spectral"}
        )
        if self.min_speakers is not None:
            self._validate_positive_int("min_speakers", int(self.min_speakers))
        if self.max_speakers is not None:
            self._validate_positive_int("max_speakers", int(self.max_speakers))
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and int(self.min_speakers) > int(self.max_speakers)
        ):
            raise ValueError("min_speakers must be <= max_speakers")

        if self.chunk_size_minutes * 60.0 <= self.chunk_overlap_seconds:
            raise ValueError("chunk_overlap_seconds must be smaller than chunk_size_minutes * 60")
        if self.chunk_threshold_minutes < self.chunk_size_minutes:
            raise ValueError("chunk_threshold_minutes must be >= chunk_size_minutes")

        self.sed_mode = self._lower_choice("sed_mode", self.sed_mode, {"auto", "global", "timeline"})
        _ensure_numeric_range("sed_window_sec", self.sed_window_sec, gt=0.0)
        _ensure_numeric_range("sed_hop_sec", self.sed_hop_sec, gt=0.0)
        _ensure_numeric_range("sed_enter", self.sed_enter, ge=0.0, le=1.0)
        _ensure_numeric_range("sed_exit", self.sed_exit, ge=0.0, le=1.0)
        if self.sed_enter < self.sed_exit:
            raise ValueError("sed_enter must be >= sed_exit")
        _ensure_numeric_range("sed_merge_gap", self.sed_merge_gap, ge=0.0)
        _ensure_numeric_range("sed_default_min_dur", self.sed_default_min_dur, ge=0.0)
        self.sed_median_k = int(self.sed_median_k or 1)
        self._validate_positive_int("sed_median_k", self.sed_median_k)
        if self.sed_median_k % 2 == 0:
            self.sed_median_k += 1
        self.sed_batch_size = int(self.sed_batch_size or 1)
        self._validate_positive_int("sed_batch_size", self.sed_batch_size)
        if not isinstance(self.sed_min_dur, dict):
            raise ValueError("sed_min_dur must be a mapping of label to duration")
        cleaned_min_dur: dict[str, float] = {}
        for key, value in self.sed_min_dur.items():
            try:
                duration = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - validation path
                raise ValueError(f"Invalid sed_min_dur value for {key!r}: {value}") from exc
            if duration < 0:
                raise ValueError(f"sed_min_dur for {key!r} must be >= 0")
            cleaned_min_dur[str(key).lower()] = duration
        self.sed_min_dur = cleaned_min_dur

    @staticmethod
    def _lower_choice(name: str, value: Any, allowed: set[str] | None) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        lowered = value.lower()
        if allowed is not None and lowered not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}")
        return lowered

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be an integer > 0")

    @staticmethod
    def _validate_positive_float(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a number > 0")

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:  # noqa: D401 - compatibility shim
        """Return the configuration as a dictionary."""

        if mode not in {"python", "json"}:  # pragma: no cover - defensive guard
            raise ValueError("mode must be 'python' or 'json'")
        data: dict[str, Any] = {}
        for field in dataclass_fields(self):
            data[field.name] = getattr(self, field.name)
        return data

    @classmethod
    def model_validate(cls, data: Mapping[str, Any] | PipelineConfig) -> PipelineConfig:
        """Validate a mapping and construct a configuration instance."""

        if isinstance(data, PipelineConfig):
            return data
        if not isinstance(data, Mapping):
            raise ValueError("PipelineConfig.model_validate expects a mapping")
        return cls(**dict(data))


DEFAULT_PIPELINE_CONFIG: dict[str, Any] = PipelineConfig().model_dump(mode="python")

CORE_DEPENDENCY_REQUIREMENTS: dict[str, str] = {
    "numpy": "1.24",
    "scipy": "1.10",
    "librosa": "0.10",
    "soundfile": "0.12",
    "ctranslate2": "4.0",
    "faster_whisper": "1.0",
    "pandas": "2.0",
    "onnxruntime": "1.16",
    "transformers": "4.30",
}

__all__ = [
    "DEFAULT_PIPELINE_CONFIG",
    "CORE_DEPENDENCY_REQUIREMENTS",
    "PipelineConfig",
    "build_pipeline_config",
    "verify_dependencies",
    "diagnostics",
    "dependency_health_summary",
]


def build_pipeline_config(
    overrides: dict[str, Any] | PipelineConfig | None = None,
) -> dict[str, Any]:
    """Return a validated pipeline configuration merged with overrides."""

    if isinstance(overrides, PipelineConfig):
        return overrides.model_dump(mode="python")

    base = PipelineConfig()
    if not overrides:
        return base.model_dump(mode="python")

    merged: dict[str, Any] = base.model_dump(mode="python")
    for key, value in overrides.items():
        if key not in merged:
            raise ValueError(f"Unknown configuration key: {key}")
        if value is None:
            continue
        merged[key] = value

    try:
        validated = PipelineConfig.model_validate(merged)
    except (
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - surface readable error upstream
        raise ValueError(str(exc)) from exc
    return validated.model_dump(mode="python")


def _iter_dependency_status() -> Iterator[
    tuple[str, str, Any, str | None, Exception | None, Exception | None]
]:
    for mod, min_ver in CORE_DEPENDENCY_REQUIREMENTS.items():
        import_error: Exception | None = None
        metadata_error: Exception | None = None
        module = None
        try:
            module = __import__(mod.replace("-", "_"))
        except Exception as exc:  # pragma: no cover - defensive import guard
            import_error = exc

        version: str | None = None
        if module is not None:
            try:
                version = importlib_metadata.version(mod)
            except importlib_metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            except Exception as exc:  # pragma: no cover - metadata failure
                metadata_error = exc
        yield mod, min_ver, module, version, import_error, metadata_error


def _verify_core_dependencies(require_versions: bool = False) -> tuple[bool, list[str]]:
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
        except Exception as exc:  # pragma: no cover - comparison safety
            issues.append(f"Version check failed for {mod}: {exc}")

    return (len(issues) == 0), issues


def dependency_health_summary() -> dict[str, dict[str, Any]]:
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
                except Exception as exc:  # pragma: no cover - comparison safety
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
        "summary": dependency_health_summary(),
        "strict_versions": require_versions,
    }
