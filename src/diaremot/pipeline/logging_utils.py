from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert values into JSON-serialisable types."""
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {key: _make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        converted = [_make_json_safe(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(converted)
        if isinstance(obj, set):
            return converted
        return converted
    return obj


class JSONLWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_make_json_safe(record), ensure_ascii=False) + "\n")

        except PermissionError as exc:
            print(f"Warning: Could not write to log file {self.path}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: Error writing to log file {self.path}: {exc}")


@dataclass
class RunStats:
    run_id: str
    file_id: str
    schema_version: str = "2.0.0"
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def mark(self, stage: str, elapsed_ms: float, counts: dict[str, int] | None = None) -> None:
        self.stage_timings_ms[stage] = self.stage_timings_ms.get(stage, 0.0) + float(elapsed_ms)
        if counts:
            slot = self.stage_counts.setdefault(stage, {})
            for key, value in counts.items():
                slot[key] = slot.get(key, 0) + int(value)


class CoreLogger:
    def __init__(self, run_id: str, jsonl_path: Path, console_level: int = logging.INFO):
        self.run_id = run_id
        self.jsonl = JSONLWriter(jsonl_path)
        self.log = logging.getLogger(f"pipeline.{run_id}")
        self.log.setLevel(console_level)
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(console_level)
            fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M")
            handler.setFormatter(fmt)
            self.log.addHandler(handler)

    def event(self, stage: str, event: str, **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "run_id": self.run_id,
            "stage": stage,
            "event": event,
        }
        record.update(fields)
        self.jsonl.emit(record)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an informational message forwarding formatting arguments."""

        self.log.info(message, *args, **kwargs)

    def warn(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message forwarding formatting arguments."""

        self.log.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message forwarding formatting arguments."""

        self.log.error(message, *args, **kwargs)


def _fmt_hms(seconds: float) -> str:
    """Return a human readable H:MM:SS style string for ``seconds``."""

    safe_seconds = max(0.0, float(seconds))
    total_seconds = int(safe_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _fmt_hms_ms(milliseconds: float) -> str:
    """Return a human readable string with millisecond precision."""

    safe_ms = max(0.0, float(milliseconds))
    seconds = safe_ms / 1000.0
    base_seconds = int(seconds)
    fractional_ms = int(round((seconds - base_seconds) * 1000))

    if fractional_ms == 1000:
        base_seconds += 1
        fractional_ms = 0

    hours, remainder = divmod(base_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{fractional_ms:03d}"
    if minutes:
        return f"{minutes:02d}:{secs:02d}.{fractional_ms:03d}"
    return f"00:{secs:02d}.{fractional_ms:03d}"


class StageGuard(AbstractContextManager["StageGuard"]):
    _OPTIONAL_STAGE_EXCEPTION_MAP = {
        "paralinguistics": (
            ImportError,
            ModuleNotFoundError,
        ),
        "affect_and_assemble": (
            ImportError,
            ModuleNotFoundError,
        ),
        "overlap_interruptions": (
            AttributeError,
            ImportError,
            ModuleNotFoundError,
        ),
        "conversation_analysis": (ValueError,),
        "speaker_rollups": (
            ValueError,
            TypeError,
        ),
    }
    _CRITICAL_STAGES = {"preprocess", "outputs"}
    _TIMEOUT_STAGES = {"diarize", "transcribe"}

    def __init__(self, corelog: CoreLogger, stats: RunStats, stage: str):
        self.corelog = corelog
        self.stats = stats
        self.stage = stage
        self.start: float | None = None
        self.corelog.info(f"[{self.stage}] start")

    def __enter__(self) -> StageGuard:
        self.start = time.time()
        self.corelog.event(self.stage, "start")
        return self

    def done(self, **counts: int) -> None:
        if counts:
            self.stats.mark(self.stage, 0.0, counts)

    def _is_known_nonfatal(self, exc: BaseException) -> bool:
        if isinstance(exc, TimeoutError | subprocess.TimeoutExpired) and (
            self.stage in self._TIMEOUT_STAGES
        ):
            return True

        optional = self._OPTIONAL_STAGE_EXCEPTION_MAP.get(self.stage, ())
        return isinstance(exc, optional)

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        if self.start is None:
            self.start = time.time()
        elapsed_ms = max(0.0, (time.time() - self.start) * 1000.0)

        if exc:
            known_nonfatal = self._is_known_nonfatal(exc)
            trace_hash = hashlib.blake2s(
                f"{self.stage}:{type(exc).__name__}".encode(), digest_size=8
            ).hexdigest()
            self.corelog.event(
                self.stage,
                "error",
                elapsed_ms=elapsed_ms,
                error=f"{type(exc).__name__}: {exc}",
                trace_hash=trace_hash,
                handled=known_nonfatal,
            )

            dur_txt = _fmt_hms_ms(elapsed_ms)
            log_fn = self.corelog.warn if known_nonfatal else self.corelog.error
            log_fn(
                f"[{self.stage}] {'handled ' if known_nonfatal else ''}"
                f"{type(exc).__name__}: {exc} ({dur_txt})"
            )
            self.stats.mark(self.stage, elapsed_ms)
            try:
                message = f"{self.stage}: {type(exc).__name__}: {exc}"
                self.stats.warnings.append(message)
                self.stats.errors.append(message)
                if known_nonfatal:
                    self.stats.issues.append(message)

                def _suggest_fix(stage: str, err: BaseException) -> str:
                    text = str(err).lower()
                    if stage == "preprocess":
                        if "libsndfile" in text or "soundfile" in text:
                            return "Install libsndfile: apt-get install libsndfile1 (Linux) or brew install libsndfile (macOS)."

                        if "ffmpeg" in text or "audioread" in text:
                            return "Install ffmpeg and ensure it is on PATH."
                        if "file not found" in text or "no such file" in text:
                            return "Check input path and permissions."
                        return "Verify audio codec support (try converting to WAV 16kHz mono)."
                    if stage == "transcribe":
                        if isinstance(err, TimeoutError | subprocess.TimeoutExpired):
                            return (
                                "Increase --asr-segment-timeout or choose a smaller Whisper model."
                            )
                        if "faster_whisper" in text or "ctranslate2" in text:
                            return "Install faster-whisper and ctranslate2; confirm CPU wheels are compatible."
                        if "whisper" in text and "tiny" in text:
                            return "OpenAI whisper fallback failed; try reinstalling whisper or using a local model."
                        if "model" in text and ("not found" in text or "download" in text):
                            return "Model not found; provide a valid local model path or enable network access."
                        return (
                            "Reduce model size, set compute_type=float32, and verify dependencies."
                        )
                    if stage == "paralinguistics":
                        return "Install librosa/scipy extras or run with --disable_paralinguistics."
                    if stage == "affect_and_assemble":
                        return "Install emotion/intent model dependencies or run with --disable_affect."
                    if stage == "background_sed":
                        return "Install and configure SED dependencies; this stage is required."
                    if stage == "overlap_interruptions":
                        return (
                            "Install paralinguistics extras for overlap metrics or skip this stage."
                        )
                    if stage == "conversation_analysis":
                        return "Ensure numpy/pandas are available for analytics or review conversation inputs."
                    if stage == "speaker_rollups":
                        return "Inspect segment data integrity before computing speaker rollups."
                    if stage == "outputs":
                        return "Ensure outdir is writable and disk has space."
                    return "Check logs for details; ensure dependencies and file permissions."

                self.stats.failures.append(
                    {
                        "stage": self.stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_ms": elapsed_ms,
                        "suggestion": _suggest_fix(self.stage, exc),
                    }
                )
                if self.stage == "preprocess":
                    self.stats.config_snapshot["preprocess_failed"] = True
                if self.stage == "transcribe":
                    self.stats.config_snapshot["transcribe_failed"] = True

            except Exception:  # pragma: no cover - stats updates are best effort
                pass
            swallow = known_nonfatal and self.stage not in self._CRITICAL_STAGES
            return swallow
        else:
            self.corelog.event(self.stage, "stop", elapsed_ms=elapsed_ms)
            dur_txt = _fmt_hms_ms(elapsed_ms)
            self.corelog.info(f"[{self.stage}] ok in {dur_txt}")
            self.stats.mark(self.stage, elapsed_ms)
            return False


__all__ = [
    "RunStats",
    "CoreLogger",
    "StageGuard",
    "_fmt_hms",
    "_fmt_hms_ms",
]
