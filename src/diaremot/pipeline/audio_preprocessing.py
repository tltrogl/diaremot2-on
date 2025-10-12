# audio_preprocessing.py with integrated auto-chunking for long files
# Extended version that automatically splits very long audio files
# and processes them in manageable chunks to prevent memory issues

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# Quieter logs from third-parties
warnings.filterwarnings("ignore", category=UserWarning, module="librosa.core.audio")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# ---------------------------
# Dataclasses / Config
# ---------------------------


@dataclass
class PreprocessConfig:
    target_sr: int = 16000
    mono: bool = True

    # Auto-chunking for long audio files
    auto_chunk_enabled: bool = True
    chunk_threshold_minutes: float = 60.0  # Split audio longer than this
    chunk_size_minutes: float = 20.0  # Each chunk duration
    chunk_overlap_seconds: float = 30.0  # Overlap between chunks
    chunk_temp_dir: str | None = None  # Use system temp if None

    # High-pass
    hpf_hz: float = 80.0
    hpf_order: int = 2

    # Denoise (soft spectral subtraction with temporal smoothing + backoff)
    denoise: str = "spectral_sub_soft"  # "spectral_sub_soft" | "none"
    denoise_alpha_db: float = 3.0  # over-subtraction in dB
    denoise_beta: float = 0.06  # spectral floor as fraction of noise (0..1)
    mask_exponent: float = 1.0  # 1 ~ Wiener-ish
    smooth_t: int = 3  # median smoothing width (frames) for mask
    high_clip_backoff: float = 0.12  # backoff if floor_clipping_ratio exceeds this

    # Noise tracking
    noise_update_alpha: float = 0.10  # EMA for noise profile updates (lower = smoother)
    min_noise_frames: int = 30  # min non-speech frames to trust VAD noise estimate

    # VAD (RMS-gated; CPU-friendly)
    use_vad: bool = True
    frame_ms: int = 20
    hop_ms: int = 10
    vad_rel_db: float = 12.0  # speech if rms_db > noise_floor_db + vad_rel_db
    vad_floor_percentile: float = 20.0

    # Gated upward gain
    gate_db: float = -45.0  # below this, do not boost
    target_db: float = -23.0  # aim per-frame towards this
    max_boost_db: float = 18.0  # cap upward gain
    gain_smooth_ms: int = 250
    gain_smooth_method: str = "hann"  # "hann" | "exp"
    exp_smooth_alpha: float = 0.15

    # Compression (transparent)
    comp_ratio: float = 2.0
    comp_thresh_db: float = -26.0
    comp_knee_db: float = 6.0

    # Loudness norm (approximate)
    loudness_mode: str = "asr"  # "asr" -> hotter (-20 LUFS equiv), "broadcast" -> -23
    lufs_target_asr: float = -20.0
    lufs_target_broadcast: float = -23.0

    # QC / metrics
    oversample_factor: int = 4  # for intersample peak check
    silence_db: float = -60.0  # below counts as silence


@dataclass
class AudioHealth:
    snr_db: float
    clipping_detected: bool
    silence_ratio: float
    rms_db: float
    est_lufs: float
    dynamic_range_db: float
    floor_clipping_ratio: float
    is_chunked: bool = False
    chunk_info: dict | None = None


@dataclass
class PreprocessResult:
    """Structured result emitted by :class:`AudioPreprocessor`."""

    audio: np.ndarray
    sample_rate: int
    health: AudioHealth | None
    duration_s: float
    is_chunked: bool = False
    chunk_details: dict | None = None

    def to_tuple(self) -> tuple[np.ndarray, int, AudioHealth | None]:
        """Return the legacy tuple representation (audio, sr, health)."""

        return self.audio, self.sample_rate, self.health

    def __iter__(self):
        """Allow unpacking ``PreprocessResult`` like the historical tuple."""

        yield from (self.audio, self.sample_rate, self.health)


@dataclass
class ChunkInfo:
    chunk_id: int
    start_time: float
    end_time: float
    duration: float
    overlap_start: float
    overlap_end: float
    temp_path: str


# ---------------------------
# Utility helpers
# ---------------------------


PCM_FORMATS = {"WAV", "WAVEX", "AIFF", "AIFFC"}
PCM_FALLBACK_SUBTYPES = {"PCM", "FLOAT", "DOUBLE"}


def _is_uncompressed_pcm(info: sf.Info) -> bool:
    """Return True when the file is a WAV/AIFF PCM variant we can read directly."""

    try:
        fmt = (info.format or "").upper()
        subtype = (info.subtype or "").upper()
    except AttributeError:
        return False

    if fmt not in PCM_FORMATS:
        return False

    return any(token in subtype for token in PCM_FALLBACK_SUBTYPES)


def _load_uncompressed_with_soundfile(
    source: Path, target_sr: int, mono: bool
) -> tuple[np.ndarray, int]:
    """Read PCM WAV/AIFF directly via libsndfile."""

    y, sr = sf.read(source, always_2d=False, dtype="float32")
    if mono and y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        import soxr
        y = soxr.resample(y, sr, target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def _decode_with_ffmpeg(source: Path, target_sr: int, mono: bool) -> tuple[np.ndarray, int]:
    """Decode arbitrary containers via ffmpeg → temp wav → soundfile."""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(target_sr),
        "-f",
        "wav",
        "-loglevel",
        "quiet",  # Suppress ffmpeg output
        tmp_wav,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=300,  # 5 minute timeout
        )
        y, sr = sf.read(tmp_wav, always_2d=False, dtype="float32")
        if mono and y.ndim > 1:
            y = np.mean(y, axis=1)
        logger.debug(f"Decoded {source} via primary ffmpeg path")
        return y.astype(np.float32), sr
    except subprocess.TimeoutExpired as exc:
        logger.error(f"FFmpeg timeout for {source}")
        raise RuntimeError(f"Audio decoding timeout for {source}") from exc
    except subprocess.CalledProcessError as exc:
        stderr_output = exc.stderr.decode() if exc.stderr else "No error details"
        logger.debug(f"FFmpeg returned non-zero exit for {source}: {stderr_output}")
        raise RuntimeError(
            f"Cannot decode audio file {source}. FFmpeg error: {stderr_output}"
        ) from exc
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass


def _safe_load_audio(path: str, target_sr: int, mono: bool = True) -> tuple[np.ndarray, int]:
    p = Path(path)

    info: sf.Info | None = None
    try:
        info = sf.info(p)
    except Exception as exc_info:
        logger.debug(f"soundfile could not inspect {p}: {exc_info}")

    if info and _is_uncompressed_pcm(info):
        logger.debug(f"Loading PCM audio directly via soundfile: {p}")
        return _load_uncompressed_with_soundfile(p, target_sr=target_sr, mono=mono)

    try:
        logger.debug(f"Decoding {p} via ffmpeg → WAV")
        return _decode_with_ffmpeg(p, target_sr=target_sr, mono=mono)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to decode compressed audio containers; install ffmpeg"
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot decode audio file {p} via ffmpeg: {exc}") from exc


def _get_audio_duration(path: str, info: sf.Info | None = None) -> float:
    """Get audio duration with a robust fallback chain."""

    if info is not None:
        try:
            duration = getattr(info, "duration", None)
        except Exception:
            duration = None
        if duration and duration > 0:
            logger.debug(f"Got duration via soundfile metadata: {duration}s")
            return float(duration)

    try:
        info = sf.info(path)
        if info and info.duration and info.duration > 0:
            logger.debug(f"Got duration via soundfile metadata: {info.duration}s")
            return float(info.duration)
    except Exception as exc:
        logger.debug(f"soundfile failed for duration: {exc}")

    try:
        import av  # type: ignore

        with av.open(path) as container:
            dur = None
            for s in container.streams:
                if s.type == "audio" and s.duration and s.time_base:
                    dur = float(s.duration * s.time_base)
                    break
            if dur is None and container.duration is not None:
                dur = float(container.duration) / 1e6
            if dur is not None and dur > 0:
                logger.debug(f"Got duration via PyAV: {dur}s")
                return dur
    except Exception as exc_av:
        logger.debug(f"PyAV failed for duration: {exc_av}")

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            logger.debug(f"Got duration via ffprobe: {duration}s")
            return duration
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        ValueError,
    ) as exc_probe:
        logger.debug(f"ffprobe failed for duration: {exc_probe}")

    logger.warning(f"Could not determine duration for {path}, using 0.0")
    return 0.0


def _probe_audio_metadata(path: str) -> tuple[float, sf.Info | None]:
    """Return the duration and cached soundfile.Info for ``path``."""

    info: sf.Info | None = None
    try:
        info = sf.info(path)
    except Exception as exc_info:
        logger.debug(f"soundfile could not inspect {path}: {exc_info}")

    duration = _get_audio_duration(path, info=info)
    return duration, info


def _create_audio_chunks(
    audio_path: str,
    config: PreprocessConfig,
    *,
    duration: float | None = None,
    info: sf.Info | None = None,
) -> list[ChunkInfo]:
    logger.info(f"[chunks] Creating audio chunks for long file: {audio_path}")

    # Load audio file info (robust to compressed containers)
    if duration is None or duration <= 0:
        duration = _get_audio_duration(audio_path, info=info)

    if info is None:
        try:
            info = sf.info(audio_path)
        except Exception as exc_info:
            logger.debug(f"soundfile could not inspect {audio_path}: {exc_info}")

    if info and info.samplerate:
        sr = int(info.samplerate)
    else:
        sr = int(config.target_sr)

    logger.info(
        f"[chunks] Audio duration: {duration / 60:.1f} minutes; threshold={config.chunk_threshold_minutes} min; size={config.chunk_size_minutes} min; overlap={config.chunk_overlap_seconds}s"
    )

    chunk_duration = config.chunk_size_minutes * 60.0
    overlap_duration = config.chunk_overlap_seconds

    # Create temp directory for chunks
    if config.chunk_temp_dir:
        temp_dir = Path(config.chunk_temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))

    chunks = []
    chunk_id = 0
    start_time = 0.0

    while start_time < duration:
        end_time = min(start_time + chunk_duration, duration)

        # Calculate actual start/end with overlap
        if chunk_id > 0:
            actual_start = max(0.0, start_time - overlap_duration)
        else:
            actual_start = start_time

        if end_time < duration:
            actual_end = min(duration, end_time + overlap_duration)
        else:
            actual_end = end_time

        # Extract chunk - FIXED VERSION
        temp_chunk_raw = None
        try:
            t0 = time.time()
            # Try direct chunk extraction with ffmpeg (faster for M4A)
            temp_chunk_raw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_chunk_raw.close()

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                audio_path,
                "-ss",
                str(actual_start),
                "-t",
                str(actual_end - actual_start),
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-ac",
                "1",
                "-loglevel",
                "quiet",
                temp_chunk_raw.name,
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                chunk_audio, _ = sf.read(temp_chunk_raw.name, dtype="float32")
                logger.info(
                    f"[chunks] Extracted chunk {chunk_id} via ffmpeg in {time.time() - t0:.2f}s ({actual_start:.1f}s→{actual_end:.1f}s)"
                )
            else:
                raise subprocess.CalledProcessError(result.returncode, cmd)

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ):
            if info and _is_uncompressed_pcm(info):
                logger.debug(
                    f"ffmpeg chunk extraction failed; reading PCM chunk via soundfile for chunk {chunk_id}"
                )
                frames_start = int(round(actual_start * sr))
                frames_end = int(round(actual_end * sr))
                frames = max(frames_end - frames_start, 0)
                with sf.SoundFile(audio_path) as snd:
                    snd.seek(frames_start)
                    chunk_audio = snd.read(frames, dtype="float32", always_2d=False)
                if chunk_audio.ndim > 1:
                    chunk_audio = np.mean(chunk_audio, axis=1)
                chunk_audio = chunk_audio.astype(np.float32)
                logger.info(
                    f"[chunks] Extracted chunk {chunk_id} via soundfile in {time.time() - t0:.2f}s ({actual_start:.1f}s→{actual_end:.1f}s)"
                )
            else:
                raise RuntimeError(
                    "ffmpeg chunk extraction failed and only PCM WAV/AIFF fallback is supported"
                )
        finally:
            # Clean up temp file
            if temp_chunk_raw is not None:
                try:
                    os.unlink(temp_chunk_raw.name)
                except Exception:
                    pass

        # Save chunk to temp file
        chunk_filename = f"chunk_{chunk_id:03d}_{int(start_time):04d}s-{int(end_time):04d}s.wav"
        chunk_path = temp_dir / chunk_filename
        sf.write(chunk_path, chunk_audio, sr)

        chunk_info = ChunkInfo(
            chunk_id=chunk_id,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            overlap_start=start_time - actual_start,
            overlap_end=actual_end - end_time,
            temp_path=str(chunk_path),
        )
        chunks.append(chunk_info)

        logger.info(
            f"[chunks] Saved chunk {chunk_id}: {chunk_filename} ({chunk_info.duration:.1f}s)"
        )

        # Move to next chunk
        start_time = end_time
        chunk_id += 1

    logger.info(f"[chunks] Created {len(chunks)} chunks in {temp_dir}")
    return chunks


def _merge_chunked_audio(chunks: list[tuple[np.ndarray, ChunkInfo]], target_sr: int) -> np.ndarray:
    logger.info(f"Merging {len(chunks)} processed chunks")

    if not chunks:
        return np.array([], dtype=np.float32)

    if len(chunks) == 1:
        return chunks[0][0]

    # Sort chunks by start time
    chunks.sort(key=lambda x: x[1].start_time)

    merged_parts = []

    for i, (chunk_audio, chunk_info) in enumerate(chunks):
        if i == 0:
            # First chunk: use everything
            merged_parts.append(chunk_audio)
        else:
            # Subsequent chunks: skip overlap from beginning
            overlap_samples = int(chunk_info.overlap_start * target_sr)
            if overlap_samples < len(chunk_audio):
                chunk_audio = chunk_audio[overlap_samples:]
            merged_parts.append(chunk_audio)

    # Concatenate all parts
    merged = np.concatenate(merged_parts, axis=0)
    logger.info(f"Merged audio: {len(merged) / target_sr:.1f}s total")

    return merged.astype(np.float32)


def _cleanup_chunks(chunks: list[ChunkInfo]) -> None:
    """Robust cleanup of temporary chunk files with retry logic."""
    logger.info(f"Cleaning up {len(chunks)} temporary chunk files")
    temp_dirs = set()
    failed_cleanups = []

    for chunk in chunks:
        chunk_path = Path(chunk.temp_path)
        if chunk_path.exists():
            temp_dirs.add(chunk_path.parent)
            try:
                chunk_path.unlink()
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to remove {chunk_path}: {e}")
                failed_cleanups.append(chunk_path)

    # Retry failed cleanups once after brief delay
    if failed_cleanups:
        import time

        time.sleep(0.1)
        for chunk_path in failed_cleanups:
            try:
                if chunk_path.exists():
                    chunk_path.unlink()
            except Exception:
                logger.warning(f"Permanent cleanup failure: {chunk_path}")

    # Remove empty temp directories
    for temp_dir in temp_dirs:
        try:
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
                logger.debug(f"Removed temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory {temp_dir}: {e}")


def _butter_highpass(y: np.ndarray, sr: int, freq: float, order: int = 2) -> np.ndarray:
    if freq <= 0:
        return y
    nyq = 0.5 * sr
    Wn = min(0.999, max(1e-6, freq / nyq))
    b, a = butter(order, Wn, btype="high", analog=False)
    return filtfilt(b, a, y)


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(1e-12, x)))


def _rms_db(y: np.ndarray) -> float:
    return _db(float(np.sqrt(np.mean(np.square(y)) + 1e-12)))


def _frame_params(sr: int, frame_ms: int, hop_ms: int) -> tuple[int, int]:
    n_fft = int(round(frame_ms * 0.001 * sr))
    n_fft = max(256, 1 << int(np.ceil(np.log2(max(8, n_fft)))))
    hop = int(round(hop_ms * 0.001 * sr))
    hop = max(1, min(hop, n_fft // 2))
    return n_fft, hop


def _hann_smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    w = np.hanning(win)
    w = w / (w.sum() + 1e-12)
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, w, mode="valid")


def _exp_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def _interp_per_sample(env: np.ndarray, hop: int, length: int) -> np.ndarray:
    # Map per-frame env → per-sample linearly; env length equals num frames
    t_env = np.arange(len(env)) * hop
    t = np.arange(length)
    return np.interp(t, t_env, env, left=env[0], right=env[-1])


def _oversampled_clip_detect(y: np.ndarray, factor: int = 4, thresh: float = 0.999) -> bool:
    if factor <= 1:
        return bool(np.any(np.abs(y) >= thresh))
    # Linear oversample (cheap; good enough for QC)
    idx = np.arange(len(y), dtype=np.float64)
    fine = np.linspace(0, len(y) - 1, num=(len(y) - 1) * factor + 1)
    y2 = np.interp(fine, idx, y.astype(np.float64))
    return bool(np.any(np.abs(y2) >= thresh))


def _percentile_db(y_abs: np.ndarray, p: float) -> float:
    return _db(float(np.percentile(y_abs, p)))


def _dynamic_range_db(y: np.ndarray) -> float:
    y_abs = np.abs(y) + 1e-12
    hi = _percentile_db(y_abs, 95.0)
    lo = _percentile_db(y_abs, 5.0)
    return max(0.0, hi - lo)


def _estimate_loudness_lufs_approx(y: np.ndarray, sr: int) -> float:
    # Approx integrated loudness: 400ms RMS with -10 dB relative gate, no K-weight
    win = int(0.400 * sr)
    hop = int(0.100 * sr)
    if win <= 0 or len(y) < win:
        return _rms_db(y)
    frames = librosa.util.frame(y, frame_length=win, hop_length=hop).T
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms + 1e-12)
    ungated_mean = np.mean(loud)
    gated = loud[loud > (ungated_mean - 10.0)]
    lufs = float(np.mean(gated) if len(gated) else ungated_mean)
    return lufs


def _simple_vad(
    y: np.ndarray, sr: int, frame_ms: int, hop_ms: int, floor_pct: float, rel_db: float
) -> np.ndarray:
    # Framewise RMS; speech if above noise floor + margin
    n_fft, hop = _frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(S)
    rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
    rms_db = 20 * np.log10(rms + 1e-12)
    floor = np.percentile(rms_db, floor_pct)
    speech = rms_db > (floor + rel_db)
    return speech.astype(np.bool_)


# ---------------------------
# Denoise: soft spectral subtraction (VAD-aware)
# ---------------------------


def _spectral_subtract_soft_vad(
    y: np.ndarray,
    sr: int,
    speech_mask: np.ndarray | None,
    alpha_db: float,
    beta: float,
    p: float,
    smooth_t: int,
    noise_ema_alpha: float,
    min_noise_frames: int,
    frame_ms: int,
    hop_ms: int,
    backoff_thresh: float,
) -> tuple[np.ndarray, float]:
    n_fft, hop = _frame_params(sr, frame_ms, hop_ms)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag, phase = np.abs(S), np.angle(S)

    # Noise estimate using VAD if available, otherwise percentile
    if speech_mask is not None and np.sum(~speech_mask) >= max(min_noise_frames, 5):
        noise_mag = np.median(mag[:, ~speech_mask], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)

    # Over-subtraction factor (linear)
    alpha = 10.0 ** (alpha_db / 20.0)

    residual = mag - alpha * noise_mag
    floor = beta * noise_mag
    clean_mag = np.maximum(residual, floor)

    # Soft gain mask 0..1
    M = (clean_mag / (clean_mag + alpha * noise_mag + 1e-12)) ** p

    # Temporal median smoothing to reduce musical noise
    if smooth_t > 1:
        M = median_filter(M, size=(1, smooth_t))

    # Compute floor-clipping ratio (how many bins at floor)
    floor_hits = (residual <= floor).sum()
    total_bins = residual.size
    floor_ratio = float(floor_hits) / float(total_bins + 1e-12)

    # Backoff if too high
    if floor_ratio > backoff_thresh:
        alpha *= 0.75
        beta2 = min(0.08, beta * 1.25)
        residual2 = mag - alpha * noise_mag
        floor2 = beta2 * noise_mag
        clean_mag2 = np.maximum(residual2, floor2)
        M = (clean_mag2 / (clean_mag2 + alpha * noise_mag + 1e-12)) ** p
        if smooth_t > 1:
            M = median_filter(M, size=(1, smooth_t))
        floor_hits = (residual2 <= floor2).sum()
        total_bins = residual2.size
        floor_ratio = float(floor_hits) / float(total_bins + 1e-12)
        logger.warning("[denoise] High floor clipping; applied backoff (ratio=%.3f)", floor_ratio)

    S_hat = M * mag * np.exp(1j * phase)
    y_hat = librosa.istft(S_hat, hop_length=hop, window="hann", length=len(y))
    return y_hat.astype(np.float32), float(floor_ratio)


# ---------------------------
# Core processor with auto-chunking
# ---------------------------


class AudioPreprocessor:
    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()

    def process_file(self, path: str) -> PreprocessResult:
        """Load ``path`` and run the preprocessing chain."""

        duration, info = _probe_audio_metadata(path)
        threshold_seconds = self.config.chunk_threshold_minutes * 60.0

        if self.config.auto_chunk_enabled and duration >= threshold_seconds:
            logger.info(
                "Long audio detected (%.1fmin), auto-chunking into ~%d min windows",
                duration / 60.0,
                int(self.config.chunk_size_minutes),
            )
            return self._process_file_chunked(path, duration, info)

        logger.info(f"Processing audio normally ({duration / 60:.1f}min)")
        y, sr = _safe_load_audio(path, target_sr=self.config.target_sr, mono=self.config.mono)
        return self.process_array(y, sr)

    def _process_file_chunked(
        self,
        path: str,
        duration: float,
        info: sf.Info | None,
    ) -> PreprocessResult:
        chunks_info = _create_audio_chunks(
            path,
            self.config,
            duration=duration,
            info=info,
        )

        if not chunks_info:
            logger.warning("No chunks created, falling back to normal processing")
            y, sr = _safe_load_audio(path, target_sr=self.config.target_sr, mono=self.config.mono)
            return self.process_array(y, sr)

        processed_chunks: list[tuple[np.ndarray, ChunkInfo]] = []
        chunk_healths: list[AudioHealth] = []

        try:
            for chunk_info in chunks_info:
                logger.info(f"Processing chunk {chunk_info.chunk_id}/{len(chunks_info) - 1}")
                y_chunk, sr = _safe_load_audio(
                    chunk_info.temp_path,
                    target_sr=self.config.target_sr,
                    mono=self.config.mono,
                )
                chunk_result = self.process_array(y_chunk, sr)
                processed_chunks.append((chunk_result.audio, chunk_info))
                if chunk_result.health:
                    chunk_healths.append(chunk_result.health)

            merged_audio = _merge_chunked_audio(processed_chunks, self.config.target_sr)

            chunk_meta = {
                "num_chunks": len(chunks_info),
                "chunk_duration_minutes": self.config.chunk_size_minutes,
                "total_duration_minutes": len(merged_audio) / self.config.target_sr / 60.0,
                "overlap_seconds": self.config.chunk_overlap_seconds,
            }

            combined_health = self._combine_chunk_health(chunk_healths, len(chunks_info))
            if combined_health:
                combined_health.is_chunked = True
                combined_health.chunk_info = chunk_meta

            duration_s = len(merged_audio) / self.config.target_sr if self.config.target_sr else 0.0

            logger.info(
                f"Chunked processing complete: {len(merged_audio) / self.config.target_sr:.1f}s total"
            )

            return PreprocessResult(
                audio=merged_audio.astype(np.float32),
                sample_rate=self.config.target_sr,
                health=combined_health,
                duration_s=float(duration_s),
                is_chunked=True,
                chunk_details=chunk_meta,
            )
        finally:
            _cleanup_chunks(chunks_info)

    def _combine_chunk_health(
        self, chunk_healths: list[AudioHealth], num_chunks: int
    ) -> AudioHealth | None:
        if not chunk_healths:
            if num_chunks <= 0:
                return None
            return AudioHealth(
                snr_db=0.0,
                clipping_detected=False,
                silence_ratio=1.0,
                rms_db=-60.0,
                est_lufs=-60.0,
                dynamic_range_db=0.0,
                floor_clipping_ratio=0.0,
                is_chunked=True,
            )

        avg_snr = float(np.mean([h.snr_db for h in chunk_healths]))
        any_clipping = any(h.clipping_detected for h in chunk_healths)
        avg_silence = float(np.mean([h.silence_ratio for h in chunk_healths]))
        avg_rms = float(np.mean([h.rms_db for h in chunk_healths]))
        avg_lufs = float(np.mean([h.est_lufs for h in chunk_healths]))
        avg_dynamic_range = float(np.mean([h.dynamic_range_db for h in chunk_healths]))
        max_floor_clipping = float(np.max([h.floor_clipping_ratio for h in chunk_healths]))

        return AudioHealth(
            snr_db=avg_snr,
            clipping_detected=any_clipping,
            silence_ratio=avg_silence,
            rms_db=avg_rms,
            est_lufs=avg_lufs,
            dynamic_range_db=avg_dynamic_range,
            floor_clipping_ratio=max_floor_clipping,
            is_chunked=True,
        )

    def process_array(self, y: np.ndarray, sr: int) -> PreprocessResult:
        if y is None or len(y) == 0:
            empty = np.zeros(1, dtype=np.float32)
            return PreprocessResult(
                audio=empty,
                sample_rate=sr,
                health=None,
                duration_s=0.0,
                is_chunked=False,
            )

        y = y.astype(np.float32)

        y_hp = self._apply_highpass(y, sr)
        speech_mask = self._run_vad(y_hp, sr)
        y_denoised, floor_ratio = self._apply_denoise(y_hp, sr, speech_mask)

        n_fft, hop = _frame_params(sr, self.config.frame_ms, self.config.hop_ms)
        y_boosted = self._apply_upward_gain(y_denoised, sr, n_fft, hop)
        y_compressed = self._apply_compression(y_boosted, sr, n_fft, hop)
        y_loud = self._apply_loudness(y_compressed, sr)
        y_final = self._apply_safety_limit(y_loud)

        health = self._build_health(y_final, sr, floor_ratio)
        duration_s = len(y_final) / sr if sr else 0.0

        return PreprocessResult(
            audio=y_final,
            sample_rate=sr,
            health=health,
            duration_s=float(duration_s),
            is_chunked=False,
        )

    def _apply_highpass(self, y: np.ndarray, sr: int) -> np.ndarray:
        return _butter_highpass(y, sr, self.config.hpf_hz, self.config.hpf_order)

    def _run_vad(self, y: np.ndarray, sr: int) -> np.ndarray | None:
        if not self.config.use_vad:
            return None
        return _simple_vad(
            y,
            sr,
            self.config.frame_ms,
            self.config.hop_ms,
            floor_pct=self.config.vad_floor_percentile,
            rel_db=self.config.vad_rel_db,
        )

    def _apply_denoise(
        self, y: np.ndarray, sr: int, speech_mask: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        if self.config.denoise == "spectral_sub_soft":
            return _spectral_subtract_soft_vad(
                y,
                sr,
                speech_mask,
                alpha_db=self.config.denoise_alpha_db,
                beta=self.config.denoise_beta,
                p=self.config.mask_exponent,
                smooth_t=self.config.smooth_t,
                noise_ema_alpha=self.config.noise_update_alpha,
                min_noise_frames=self.config.min_noise_frames,
                frame_ms=self.config.frame_ms,
                hop_ms=self.config.hop_ms,
                backoff_thresh=self.config.high_clip_backoff,
            )
        return y, 0.0

    def _apply_upward_gain(self, y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
        mag = np.abs(S)
        frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-12)
        frame_db = 20 * np.log10(frame_rms + 1e-12)

        gain_db = np.zeros_like(frame_db)
        gate_db = float(self.config.gate_db)
        target_db = float(self.config.target_db)
        max_boost = float(self.config.max_boost_db)
        needs_boost = (frame_db > gate_db) & (frame_db < target_db)
        gain_db[needs_boost] = np.minimum(target_db - frame_db[needs_boost], max_boost)

        smooth_len = max(1, int(round(self.config.gain_smooth_ms / self.config.hop_ms)))
        if self.config.gain_smooth_method == "hann":
            gain_db_sm = _hann_smooth(gain_db, smooth_len)
        else:
            gain_db_sm = _exp_smooth(gain_db, alpha=float(self.config.exp_smooth_alpha))

        gain_lin = np.power(10.0, gain_db_sm / 20.0)
        env = _interp_per_sample(gain_lin, hop, len(y))
        return y * env.astype(np.float32)

    def _apply_compression(self, y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
        mag = np.abs(S)
        lvl_db = 20 * np.log10(np.sqrt(np.mean(mag**2, axis=0)) + 1e-12)

        thr = float(self.config.comp_thresh_db)
        ratio = float(self.config.comp_ratio)
        knee = float(self.config.comp_knee_db)

        over = lvl_db - thr
        comp_gain_db = np.zeros_like(over)
        lower = -knee / 2.0
        upper = knee / 2.0
        for i, o in enumerate(over):
            if o <= lower:
                comp_gain_db[i] = 0.0
            elif o < upper:
                t = (o - lower) / (knee + 1e-12)
                desired = thr + o / ratio
                comp_gain_db[i] = (desired - (thr + o)) * t
            else:
                comp_gain_db[i] = (thr + o / ratio) - (thr + o)

        comp_gain_lin = np.power(10.0, comp_gain_db / 20.0)
        comp_env = _interp_per_sample(comp_gain_lin, hop, len(y))
        return y * comp_env.astype(np.float32)

    def _apply_loudness(self, y: np.ndarray, sr: int) -> np.ndarray:
        current_lufs = _estimate_loudness_lufs_approx(y, sr)
        target_lufs = (
            self.config.lufs_target_asr
            if self.config.loudness_mode == "asr"
            else self.config.lufs_target_broadcast
        )
        loudness_gain_db = np.clip(target_lufs - current_lufs, -12.0, 12.0)
        loudness_gain_lin = 10.0 ** (loudness_gain_db / 20.0)
        return y * float(loudness_gain_lin)

    def _apply_safety_limit(self, y: np.ndarray) -> np.ndarray:
        if y.size == 0:
            return y.astype(np.float32)
        peak = float(np.max(np.abs(y)))
        if peak > 0.95:
            safety_gain = 0.95 / peak
            y = y * safety_gain
            logger.warning(f"Applied safety limiting: {20 * np.log10(safety_gain):.1f} dB")
        return y.astype(np.float32)

    def _build_health(self, y: np.ndarray, sr: int, floor_ratio: float) -> AudioHealth:
        signal_power = float(np.mean(y**2)) if y.size else 0.0
        noise_estimate = float(np.percentile(y**2, 10)) if y.size else 0.0
        snr_db = 10 * np.log10(signal_power / max(noise_estimate, 1e-12)) if signal_power > 0 else 0.0

        silence_thresh = 10.0 ** (self.config.silence_db / 20.0)
        silence_frames = float(np.sum(np.abs(y) < silence_thresh)) if y.size else 0.0
        silence_ratio = silence_frames / float(len(y)) if len(y) > 0 else 1.0

        clipping_detected = _oversampled_clip_detect(y, self.config.oversample_factor)
        dynamic_range_db = _dynamic_range_db(y)
        rms_db = _rms_db(y)
        est_lufs = _estimate_loudness_lufs_approx(y, sr)

        return AudioHealth(
            snr_db=float(snr_db),
            clipping_detected=bool(clipping_detected),
            silence_ratio=float(silence_ratio),
            rms_db=float(rms_db),
            est_lufs=float(est_lufs),
            dynamic_range_db=float(dynamic_range_db),
            floor_clipping_ratio=float(floor_ratio),
        )


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio preprocessing with auto-chunking")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output audio file")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument(
        "--denoise", choices=["none", "spectral_sub_soft"], default="spectral_sub_soft"
    )
    parser.add_argument("--loudness-mode", choices=["asr", "broadcast"], default="asr")
    parser.add_argument(
        "--chunk-threshold",
        type=float,
        default=30.0,
        help="Auto-chunk threshold (minutes)",
    )
    parser.add_argument("--chunk-size", type=float, default=20.0, help="Chunk size (minutes)")
    parser.add_argument("--no-chunking", action="store_true", help="Disable auto-chunking")

    args = parser.parse_args()

    # Create config
    config = PreprocessConfig(
        target_sr=args.target_sr,
        denoise=args.denoise,
        loudness_mode=args.loudness_mode,
        auto_chunk_enabled=not args.no_chunking,
        chunk_threshold_minutes=args.chunk_threshold,
        chunk_size_minutes=args.chunk_size,
    )

    # Create preprocessor
    preprocessor = AudioPreprocessor(config)

    print(f"Processing {args.input}...")
    start_time = time.time()

    try:
        result = preprocessor.process_file(args.input)

        # Save output
        sf.write(args.output, result.audio, result.sample_rate)

        elapsed = time.time() - start_time
        duration = result.duration_s

        print(f"✓ Processing complete in {elapsed:.1f}s")
        print(f"  Output: {args.output}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Sample rate: {result.sample_rate} Hz")

        if result.health:
            print("  Audio Health:")
            print(f"    SNR: {result.health.snr_db:.1f} dB")
            print(f"    RMS: {result.health.rms_db:.1f} dB")
            print(f"    Est. LUFS: {result.health.est_lufs:.1f}")
            print(f"    Dynamic range: {result.health.dynamic_range_db:.1f} dB")
            print(f"    Silence ratio: {result.health.silence_ratio:.1%}")
            print(f"    Clipping detected: {result.health.clipping_detected}")
            if result.health.is_chunked and result.health.chunk_info:
                print(f"    Processed in chunks: {result.health.chunk_info['num_chunks']}")

    except Exception as e:
        print(f"✗ Processing failed: {e}")
        import traceback

        traceback.print_exc()
