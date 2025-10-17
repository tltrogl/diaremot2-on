"""Command line interface for the DiaRemot audio intelligence pipeline."""

from __future__ import annotations

import array
import json
import math
import shutil
import subprocess
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import typer

from .pipeline.logging_utils import _make_json_safe
from .pipeline.runtime_env import DEFAULT_WHISPER_MODEL

app = typer.Typer(help="High level CLI wrapper for the DiaRemot audio pipeline.")

# Ensure Optional is available when annotations are evaluated by inspect on Python 3.11.
globals()["Optional"] = Optional


@lru_cache
def _core():
    try:
        return import_module("diaremot.pipeline.audio_pipeline_core")
    except ModuleNotFoundError:
        return import_module("audio_pipeline_core")


def core_build_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return _core().build_pipeline_config(overrides)


def core_diagnostics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _core().diagnostics(*args, **kwargs)


def core_resume(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _core().resume(*args, **kwargs)


def core_run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _core().run_pipeline(*args, **kwargs)


BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "fast": {
        "whisper_model": str(DEFAULT_WHISPER_MODEL),
        "beam_size": 1,
        "temperature": 0.0,
        "affect_backend": "torch",
        "enable_sed": False,
    },
    "accurate": {
        "whisper_model": str(DEFAULT_WHISPER_MODEL),
        "beam_size": 4,
        "temperature": 0.0,
        "no_speech_threshold": 0.2,
    },
    "offline": {
        "affect_backend": "onnx",
        "disable_affect": False,
        "ignore_tx_cache": False,
    },
}


def _load_profile(profile: Optional[str]) -> dict[str, Any]:
    if not profile:
        return {}

    if profile in BUILTIN_PROFILES:
        return dict(BUILTIN_PROFILES[profile])

    profile_path = Path(profile)
    if not profile_path.exists():
        raise typer.BadParameter(f"Profile '{profile}' not found.")

    try:
        data = json.loads(profile_path.read_text())
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise typer.BadParameter(f"Profile file '{profile}' is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Profile file '{profile}' must contain a JSON object of overrides."
        )
    return data


def _normalise_path(value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    return str(value.expanduser().resolve())


def _parse_min_dur_map(value: Optional[str]) -> Optional[dict[str, float]]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        mapping: dict[str, float] = {}
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            return {}
        for part in parts:
            if "=" not in part:
                raise ValueError(
                    "sed_min_dur must be JSON or comma-separated key=value entries"
                )
            key, raw_val = part.split("=", 1)
            mapping[key.strip()] = float(raw_val)
        return mapping
    if not isinstance(data, dict):
        raise ValueError("sed_min_dur JSON must be an object mapping labels to seconds")
    result: dict[str, float] = {}
    for key, val in data.items():
        result[str(key)] = float(val)
    return result


def _validate_assets(input_path: Path, output_dir: Path, config: dict[str, Any]) -> None:
    errors = []

    if not input_path.exists():
        errors.append(f"Input file '{input_path}' does not exist.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to create output directory '{output_dir}': {exc}")

    registry_path = Path(config.get("registry_path", "speaker_registry.json"))
    try:
        registry_path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to prepare speaker registry directory: {exc}")

    if str(config.get("affect_backend", "onnx")).lower() == "onnx":
        for key in ("affect_text_model_dir", "affect_intent_model_dir"):
            path_value = config.get(key)
            if path_value:
                resolved = Path(str(path_value)).expanduser()
                if not resolved.exists():
                    errors.append(f"Configured {key}='{path_value}' but the path is missing.")

    cache_root = Path(config.get("cache_root", ".cache"))
    try:
        cache_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem failure
        errors.append(f"Unable to prepare cache directory '{cache_root}': {exc}")

    classmap = config.get("sed_classmap_csv")
    if classmap:
        try:
            classmap_path = Path(classmap).expanduser()
        except TypeError:
            classmap_path = None
        if classmap_path is not None and not classmap_path.exists():
            errors.append(f"SED class map '{classmap_path}' does not exist.")

    if errors:
        raise typer.BadParameter("\n".join(errors))


def _merge_configs(
    profile_overrides: dict[str, Any], cli_overrides: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(profile_overrides)
    defaults = _default_config()
    for key, value in cli_overrides.items():
        if value is None:
            continue
        if key in profile_overrides and key in defaults and defaults.get(key) == value:
            # Preserve profile-provided overrides when the CLI value merely reflects
            # the default generated configuration.
            continue
        merged[key] = value
    return merged


@lru_cache
def _default_config() -> dict[str, Any]:
    return core_build_config({})


def _common_options(**kwargs: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "registry_path": kwargs.get("registry_path"),
        "ahc_distance_threshold": kwargs.get("ahc_distance_threshold"),
        "speaker_limit": kwargs.get("speaker_limit"),
        "whisper_model": kwargs.get("whisper_model"),
        "compute_type": kwargs.get("asr_compute_type"),
        "cpu_threads": kwargs.get("asr_cpu_threads"),
        "language": kwargs.get("language"),
        "language_mode": kwargs.get("language_mode"),
        "ignore_tx_cache": kwargs.get("ignore_tx_cache"),
        "quiet": kwargs.get("quiet"),
        "disable_affect": kwargs.get("disable_affect"),
        "affect_text_model_dir": kwargs.get("affect_text_model_dir"),
        "affect_intent_model_dir": kwargs.get("affect_intent_model_dir"),
        "affect_ser_model_dir": kwargs.get("affect_ser_model_dir"),
        "affect_vad_model_dir": kwargs.get("affect_vad_model_dir"),
        "beam_size": kwargs.get("beam_size"),
        "temperature": kwargs.get("temperature"),
        "no_speech_threshold": kwargs.get("no_speech_threshold"),
        "noise_reduction": kwargs.get("noise_reduction"),
        "enable_sed": kwargs.get("enable_sed"),
        "auto_chunk_enabled": kwargs.get("chunk_enabled"),
        "chunk_threshold_minutes": kwargs.get("chunk_threshold_minutes"),
        "chunk_size_minutes": kwargs.get("chunk_size_minutes"),
        "chunk_overlap_seconds": kwargs.get("chunk_overlap_seconds"),
        "vad_threshold": kwargs.get("vad_threshold"),
        "vad_min_speech_sec": kwargs.get("vad_min_speech_sec"),
        "vad_min_silence_sec": kwargs.get("vad_min_silence_sec"),
        "vad_speech_pad_sec": kwargs.get("vad_speech_pad_sec"),
        "disable_energy_vad_fallback": kwargs.get("disable_energy_vad_fallback"),
        "energy_gate_db": kwargs.get("energy_gate_db"),
        "energy_hop_sec": kwargs.get("energy_hop_sec"),
        "max_asr_window_sec": kwargs.get("asr_window_sec"),
        "segment_timeout_sec": kwargs.get("asr_segment_timeout"),
        "batch_timeout_sec": kwargs.get("asr_batch_timeout"),
        "cpu_diarizer": kwargs.get("cpu_diarizer"),
        "sed_window_sec": kwargs.get("sed_window_sec"),
        "sed_hop_sec": kwargs.get("sed_hop_sec"),
        "sed_enter": kwargs.get("sed_enter"),
        "sed_exit": kwargs.get("sed_exit"),
        "sed_min_dur": kwargs.get("sed_min_dur"),
        "sed_merge_gap": kwargs.get("sed_merge_gap"),
        "sed_classmap_csv": kwargs.get("sed_classmap_csv"),
        "sed_timeline_jsonl": kwargs.get("sed_timeline_jsonl"),
        "sed_batch_size": kwargs.get("sed_batch_size"),
        "sed_median_k": kwargs.get("sed_median_k"),
        "sed_mode": kwargs.get("sed_mode"),
        "sed_default_min_dur": kwargs.get("sed_default_min_dur"),
    }

    backend = kwargs.get("affect_backend")
    if backend is not None:
        overrides["affect_backend"] = str(backend).lower()

    asr_backend = kwargs.get("asr_backend")
    if asr_backend is not None:
        overrides["asr_backend"] = str(asr_backend).lower()

    vad_backend = kwargs.get("vad_backend")
    if vad_backend is not None:
        overrides["vad_backend"] = str(vad_backend).lower()

    sed_mode = kwargs.get("sed_mode")
    if sed_mode is not None:
        overrides["sed_mode"] = str(sed_mode).lower()

    return overrides


def _assemble_config(profile: Optional[str], cli_overrides: dict[str, Any]) -> dict[str, Any]:
    profile_overrides = _load_profile(profile)
    merged = _merge_configs(profile_overrides, _common_options(**cli_overrides))
    return core_build_config(merged)


def _generate_sample_audio(
    target: Path,
    duration: float,
    sample_rate: int,
    ffmpeg_bin: Optional[str] = None,
) -> str:
    """Generate a sine-wave sample clip for smoke testing."""

    if duration <= 0:
        raise typer.BadParameter("duration must be positive")

    if sample_rate <= 0:
        raise typer.BadParameter("sample rate must be positive")

    target.parent.mkdir(parents=True, exist_ok=True)

    resolved_ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if resolved_ffmpeg:
        command = [
            resolved_ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
            "-ac",
            "1",
            str(target),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
            )
            return "ffmpeg"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    total_samples = int(duration * sample_rate)
    if total_samples <= 0:
        raise typer.BadParameter("duration/sample-rate combination produced no audio")

    sine = array.array("h")
    amplitude = 32767
    angular = 2 * math.pi * 440
    for index in range(total_samples):
        value = int(amplitude * math.sin(angular * (index / sample_rate)))
        sine.append(value)

    import wave

    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(sine.tobytes())

    return "python"


@app.command()
def run(
    input: Path = typer.Option(..., "--input", "-i", help="Path to input audio file."),
    outdir: Path = typer.Option(..., "--outdir", "-o", help="Directory to write outputs."),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=f"Configuration profile to load ({', '.join(BUILTIN_PROFILES)} or path to JSON).",
    ),
    registry_path: Path = typer.Option(
        Path("speaker_registry.json"), help="Speaker registry path."
    ),
    ahc_distance_threshold: float = typer.Option(
        0.15, help="Agglomerative clustering distance threshold."
    ),
    speaker_limit: Optional[int] = typer.Option(None, help="Maximum number of speakers to keep."),
    clustering_backend: str = typer.Option(
        "ahc", help="Clustering backend: 'ahc' or 'spectral'."
    ),
    min_speakers: Optional[int] = typer.Option(
        None, help="Minimum speakers (for spectral clustering)."
    ),
    max_speakers: Optional[int] = typer.Option(
        None, help="Maximum speakers (for spectral clustering)."
    ),
    whisper_model: str = typer.Option(
        str(DEFAULT_WHISPER_MODEL), help="Whisper/Faster-Whisper model identifier."
    ),
    asr_backend: str = typer.Option("faster", help="ASR backend", show_default=True),
    asr_compute_type: str = typer.Option("float32", help="CT2 compute type for faster-whisper."),
    asr_cpu_threads: int = typer.Option(1, help="CPU threads for ASR backend."),
    language: Optional[str] = typer.Option(None, help="Override ASR language"),
    language_mode: str = typer.Option("auto", help="Language detection mode"),
    ignore_tx_cache: bool = typer.Option(
        False,
        "--ignore-tx-cache",
        help="Ignore cached diarization/transcription results.",
        is_flag=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Reduce console verbosity.",
        is_flag=True,
    ),
    disable_affect: bool = typer.Option(
        False,
        "--disable-affect",
        help="Skip affect analysis.",
        is_flag=True,
    ),
    affect_backend: str = typer.Option("onnx", help="Affect backend (auto/torch/onnx)."),
    affect_text_model_dir: Optional[Path] = typer.Option(
        None, help="Path to GoEmotions model directory."
    ),
    affect_intent_model_dir: Optional[Path] = typer.Option(
        None, help="Path to intent model directory."
    ),
    affect_ser_model_dir: Optional[Path] = typer.Option(
        None, help="Path to speech emotion model directory."
    ),
    affect_vad_model_dir: Optional[Path] = typer.Option(
        None, help="Path to valence/arousal/dominance model directory."
    ),
    beam_size: int = typer.Option(1, help="Beam size for ASR decoding."),
    temperature: float = typer.Option(0.0, help="Sampling temperature for ASR."),
    no_speech_threshold: float = typer.Option(0.20, help="No-speech threshold for Whisper."),
    noise_reduction: bool = typer.Option(
        False,
        "--noise-reduction",
        help="Enable gentle noise reduction.",
        is_flag=True,
    ),
    enable_sed: bool = typer.Option(
        True,
        "--enable-sed/--disable-sed",
        help="Toggle background sound event detection stage.",
    ),
    sed_mode: str = typer.Option("auto", help="SED mode: global, timeline, or auto."),
    sed_window_sec: float = typer.Option(1.0, help="Timeline SED window length (seconds)."),
    sed_hop_sec: float = typer.Option(0.5, help="Timeline SED hop length (seconds)."),
    sed_enter: float = typer.Option(0.50, help="Timeline SED hysteresis enter threshold."),
    sed_exit: float = typer.Option(0.35, help="Timeline SED hysteresis exit threshold."),
    sed_min_dur: Optional[str] = typer.Option(
        None,
        help="JSON or comma list mapping collapsed labels to minimum event duration seconds.",
    ),
    sed_merge_gap: float = typer.Option(0.20, help="Merge SED events separated by <= gap seconds."),
    sed_classmap: Optional[Path] = typer.Option(
        None,
        help="Optional CSV mapping AudioSet labels to collapsed groups for timeline SED.",
    ),
    sed_timeline_jsonl: bool = typer.Option(
        False,
        "--sed-write-jsonl/--sed-no-jsonl",
        help="Write per-frame SED debug JSONL alongside events timeline.",
    ),
    chunk_enabled: Optional[bool] = typer.Option(
        None,
        "--chunk-enabled",
        help="Set automatic chunking of long files (true/false).",
    ),
    chunk_threshold_minutes: float = typer.Option(30.0, help="Chunking activation threshold."),
    chunk_size_minutes: float = typer.Option(20.0, help="Chunk size in minutes."),
    chunk_overlap_seconds: float = typer.Option(30.0, help="Overlap between chunks in seconds."),
    vad_threshold: float = typer.Option(0.35, help="Silero VAD probability threshold."),
    vad_min_speech_sec: float = typer.Option(0.8, help="Minimum detected speech duration."),
    vad_min_silence_sec: float = typer.Option(0.8, help="Minimum detected silence duration."),
    vad_speech_pad_sec: float = typer.Option(0.1, help="Padding added around VAD speech regions."),
    vad_backend: str = typer.Option("auto", help="Silero VAD backend (auto/torch/onnx)."),
    disable_energy_vad_fallback: bool = typer.Option(
        False,
        "--disable-energy-fallback",
        help="Disable energy VAD fallback when Silero VAD fails.",
        is_flag=True,
    ),
    energy_gate_db: float = typer.Option(-33.0, help="Energy VAD gating threshold."),
    energy_hop_sec: float = typer.Option(0.01, help="Energy VAD hop length."),
    asr_window_sec: int = typer.Option(480, help="Maximum audio length per ASR window."),
    asr_segment_timeout: float = typer.Option(300.0, help="Timeout per ASR segment."),
    asr_batch_timeout: float = typer.Option(1200.0, help="Timeout for a batch of ASR segments."),
    cpu_diarizer: bool = typer.Option(
        False,
        "--cpu-diarizer",
        help="Enable CPU optimised diarizer wrapper.",
        is_flag=True,
    ),
    clear_cache: bool = typer.Option(
        False,
        "--clear-cache",
        help="Clear cached diarization/transcription data before running.",
        is_flag=True,
    ),
):
    run_overrides: dict[str, Any] = {
        "registry_path": _normalise_path(registry_path),
        "ahc_distance_threshold": ahc_distance_threshold,
        "speaker_limit": speaker_limit,
        "clustering_backend": clustering_backend,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "whisper_model": whisper_model,
        "asr_backend": asr_backend,
        "asr_compute_type": asr_compute_type,
        "asr_cpu_threads": asr_cpu_threads,
        "language": language,
        "language_mode": language_mode,
        "ignore_tx_cache": ignore_tx_cache,
        "quiet": quiet,
        "disable_affect": disable_affect,
        "affect_backend": affect_backend,
        "affect_text_model_dir": _normalise_path(affect_text_model_dir),
        "affect_intent_model_dir": _normalise_path(affect_intent_model_dir),
        "affect_ser_model_dir": _normalise_path(affect_ser_model_dir),
        "affect_vad_model_dir": _normalise_path(affect_vad_model_dir),
        "beam_size": beam_size,
        "temperature": temperature,
        "no_speech_threshold": no_speech_threshold,
        "noise_reduction": noise_reduction,
        "chunk_enabled": chunk_enabled,
        "chunk_threshold_minutes": chunk_threshold_minutes,
        "chunk_size_minutes": chunk_size_minutes,
        "chunk_overlap_seconds": chunk_overlap_seconds,
        "vad_threshold": vad_threshold,
        "vad_min_speech_sec": vad_min_speech_sec,
        "vad_min_silence_sec": vad_min_silence_sec,
        "vad_speech_pad_sec": vad_speech_pad_sec,
        "vad_backend": vad_backend,
        "enable_sed": enable_sed,
        "sed_mode": sed_mode,
        "sed_window_sec": sed_window_sec,
        "sed_hop_sec": sed_hop_sec,
        "sed_enter": sed_enter,
        "sed_exit": sed_exit,
        "sed_merge_gap": sed_merge_gap,
        "sed_classmap_csv": _normalise_path(sed_classmap),
        "sed_timeline_jsonl": sed_timeline_jsonl,
        "disable_energy_vad_fallback": disable_energy_vad_fallback,
        "energy_gate_db": energy_gate_db,
        "energy_hop_sec": energy_hop_sec,
        "asr_window_sec": asr_window_sec,
        "asr_segment_timeout": asr_segment_timeout,
        "asr_batch_timeout": asr_batch_timeout,
        "cpu_diarizer": cpu_diarizer,
    }

    if sed_min_dur is not None:
        try:
            min_dur_map = _parse_min_dur_map(sed_min_dur)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if min_dur_map is not None:
            run_overrides["sed_min_dur"] = min_dur_map

    config = _assemble_config(profile, run_overrides)

    _validate_assets(input, outdir, config)

    try:
        manifest = core_run_pipeline(
            str(input), str(outdir), config=config, clear_cache=clear_cache
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.secho(f"Pipeline execution failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(_make_json_safe(manifest), indent=2))


@app.command()
def smoke(
    outdir: Path = typer.Option(
        ..., "--outdir", "-o", help="Directory to write smoke test outputs."
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=f"Optional configuration profile ({', '.join(BUILTIN_PROFILES)} or path).",
    ),
    duration: float = typer.Option(3.0, help="Duration of generated sample audio in seconds."),
    sample_rate: int = typer.Option(16000, help="Sample rate for generated audio."),
    disable_affect: bool = typer.Option(
        True,
        "--disable-affect/--enable-affect",
        help="Disable affect stages for a faster smoke test run.",
    ),
    ffmpeg_bin: Optional[Path] = typer.Option(
        None,
        "--ffmpeg-bin",
        help="Explicit ffmpeg binary to synthesise the audio (defaults to PATH lookup).",
    ),
    keep_audio: bool = typer.Option(
        False,
        "--keep-audio",
        help="Retain the generated sample WAV after the run completes.",
        is_flag=True,
    ),
):
    """Generate a demo audio file and execute the pipeline against it."""

    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sample_path = outdir / "diaremot_smoke_input.wav"
    ffmpeg_override = _normalise_path(ffmpeg_bin)

    try:
        _generate_sample_audio(
            sample_path,
            duration=duration,
            sample_rate=sample_rate,
            ffmpeg_bin=ffmpeg_override,
        )
    except typer.BadParameter:
        if not keep_audio and sample_path.exists():
            sample_path.unlink()
        raise

    smoke_overrides: dict[str, Any] = {
        "registry_path": _normalise_path(Path("speaker_registry.json")),
        "disable_affect": disable_affect,
        "affect_backend": "onnx",
        "enable_sed": True,
        "noise_reduction": False,
        "chunk_enabled": None,
        "chunk_threshold_minutes": None,
        "chunk_size_minutes": None,
        "chunk_overlap_seconds": None,
        "vad_backend": "auto",
    }

    config = _assemble_config(profile, smoke_overrides)

    _validate_assets(sample_path, outdir, config)

    try:
        manifest = core_run_pipeline(str(sample_path), str(outdir), config=config, clear_cache=True)
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.secho(f"Smoke test failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    finally:
        if not keep_audio:
            try:
                sample_path.unlink()
            except FileNotFoundError:  # pragma: no cover - defensive cleanup
                pass

    typer.echo(json.dumps(_make_json_safe(manifest), indent=2))


@app.command()
def resume(
    input: Path = typer.Option(..., "--input", "-i", help="Original input audio file."),
    outdir: Path = typer.Option(
        ..., "--outdir", "-o", help="Output directory used in the previous run."
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help=f"Configuration profile to load ({', '.join(BUILTIN_PROFILES)} or path to JSON).",
    ),
    registry_path: Path = typer.Option(
        Path("speaker_registry.json"), help="Speaker registry path."
    ),
    ahc_distance_threshold: float = typer.Option(
        0.15, help="Agglomerative clustering distance threshold."
    ),
    speaker_limit: Optional[int] = typer.Option(None, help="Maximum number of speakers to keep."),
    whisper_model: str = typer.Option(
        str(DEFAULT_WHISPER_MODEL), help="Whisper/Faster-Whisper model identifier."
    ),
    asr_backend: str = typer.Option("faster", help="ASR backend", show_default=True),
    asr_compute_type: str = typer.Option("float32", help="CT2 compute type for faster-whisper."),
    asr_cpu_threads: int = typer.Option(1, help="CPU threads for ASR backend."),
    language: Optional[str] = typer.Option(None, help="Override ASR language"),
    language_mode: str = typer.Option("auto", help="Language detection mode"),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Reduce console verbosity.",
        is_flag=True,
    ),
    disable_affect: bool = typer.Option(
        False,
        "--disable-affect",
        help="Skip affect analysis.",
        is_flag=True,
    ),
    affect_backend: str = typer.Option("onnx", help="Affect backend (auto/torch/onnx)."),
    affect_text_model_dir: Optional[Path] = typer.Option(
        None, help="Path to GoEmotions model directory."
    ),
    affect_intent_model_dir: Optional[Path] = typer.Option(
        None, help="Path to intent model directory."
    ),
    affect_ser_model_dir: Optional[Path] = typer.Option(
        None, help="Path to speech emotion model directory."
    ),
    affect_vad_model_dir: Optional[Path] = typer.Option(
        None, help="Path to valence/arousal/dominance model directory."
    ),
    beam_size: int = typer.Option(1, help="Beam size for ASR decoding."),
    temperature: float = typer.Option(0.0, help="Sampling temperature for ASR."),
    no_speech_threshold: float = typer.Option(0.20, help="No-speech threshold for Whisper."),
    noise_reduction: bool = typer.Option(
        False,
        "--noise-reduction",
        help="Enable gentle noise reduction.",
        is_flag=True,
    ),
    enable_sed: bool = typer.Option(
        True,
        "--enable-sed/--disable-sed",
        help="Toggle background sound event detection stage.",
    ),
    chunk_enabled: Optional[bool] = typer.Option(
        None,
        "--chunk-enabled",
        help="Set automatic chunking of long files (true/false).",
    ),
    chunk_threshold_minutes: float = typer.Option(30.0, help="Chunking activation threshold."),
    chunk_size_minutes: float = typer.Option(20.0, help="Chunk size in minutes."),
    chunk_overlap_seconds: float = typer.Option(30.0, help="Overlap between chunks in seconds."),
    vad_threshold: float = typer.Option(0.35, help="Silero VAD probability threshold."),
    vad_min_speech_sec: float = typer.Option(0.8, help="Minimum detected speech duration."),
    vad_min_silence_sec: float = typer.Option(0.8, help="Minimum detected silence duration."),
    vad_speech_pad_sec: float = typer.Option(0.1, help="Padding added around VAD speech regions."),
    vad_backend: str = typer.Option("auto", help="Silero VAD backend (auto/torch/onnx)."),
    disable_energy_vad_fallback: bool = typer.Option(
        False,
        "--disable-energy-fallback",
        help="Disable energy VAD fallback when Silero VAD fails.",
        is_flag=True,
    ),
    energy_gate_db: float = typer.Option(-33.0, help="Energy VAD gating threshold."),
    energy_hop_sec: float = typer.Option(0.01, help="Energy VAD hop length."),
    asr_window_sec: int = typer.Option(480, help="Maximum audio length per ASR window."),
    asr_segment_timeout: float = typer.Option(300.0, help="Timeout per ASR segment."),
    asr_batch_timeout: float = typer.Option(1200.0, help="Timeout for a batch of ASR segments."),
    cpu_diarizer: bool = typer.Option(
        False,
        "--cpu-diarizer",
        help="Enable CPU optimised diarizer wrapper.",
        is_flag=True,
    ),
):
    resume_overrides: dict[str, Any] = {
        "registry_path": _normalise_path(registry_path),
        "ahc_distance_threshold": ahc_distance_threshold,
        "speaker_limit": speaker_limit,
        "whisper_model": whisper_model,
        "asr_backend": asr_backend,
        "asr_compute_type": asr_compute_type,
        "asr_cpu_threads": asr_cpu_threads,
        "language": language,
        "language_mode": language_mode,
        "ignore_tx_cache": False,
        "quiet": quiet,
        "disable_affect": disable_affect,
        "affect_backend": affect_backend,
        "affect_text_model_dir": _normalise_path(affect_text_model_dir),
        "affect_intent_model_dir": _normalise_path(affect_intent_model_dir),
        "affect_ser_model_dir": _normalise_path(affect_ser_model_dir),
        "affect_vad_model_dir": _normalise_path(affect_vad_model_dir),
        "beam_size": beam_size,
        "temperature": temperature,
        "no_speech_threshold": no_speech_threshold,
        "noise_reduction": noise_reduction,
        "chunk_enabled": chunk_enabled,
        "chunk_threshold_minutes": chunk_threshold_minutes,
        "chunk_size_minutes": chunk_size_minutes,
        "chunk_overlap_seconds": chunk_overlap_seconds,
        "vad_threshold": vad_threshold,
        "vad_min_speech_sec": vad_min_speech_sec,
        "vad_min_silence_sec": vad_min_silence_sec,
        "vad_speech_pad_sec": vad_speech_pad_sec,
        "vad_backend": vad_backend,
        "enable_sed": enable_sed,
        "disable_energy_vad_fallback": disable_energy_vad_fallback,
        "energy_gate_db": energy_gate_db,
        "energy_hop_sec": energy_hop_sec,
        "asr_window_sec": asr_window_sec,
        "asr_segment_timeout": asr_segment_timeout,
        "asr_batch_timeout": asr_batch_timeout,
        "cpu_diarizer": cpu_diarizer,
    }

    config = _assemble_config(profile, resume_overrides)

    _validate_assets(input, outdir, config)

    try:
        manifest = core_resume(str(input), str(outdir), config=config)
    except Exception as exc:  # pragma: no cover - runtime failure
        typer.secho(f"Pipeline resume failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(_make_json_safe(manifest), indent=2))


@app.command()
def diagnostics(
    strict: bool = typer.Option(False, help="Require minimum dependency versions."),
):
    """Run dependency diagnostics and emit a JSON summary."""

    result = core_diagnostics(require_versions=strict)
    typer.echo(json.dumps(result, indent=2))


def main_diagnostics() -> None:
    """Console script entry point for :func:`diagnostics`."""

    typer.run(diagnostics)


def main() -> None:
    """Console script entry point for the DiaRemot CLI (Typer app)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    app()
