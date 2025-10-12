"""Chunked CPU diarization utilities.

This module wraps an existing diarizer instance and runs it on long audio
recordings by processing small chunks.  Low-energy regions can be skipped to
avoid unnecessary work.  The resulting speaker segments are stitched back
into a single timeline and overlapping segments from the same speaker are
merged.  The wrapper also exposes utilities to retrieve embeddings from the
most recent diarization run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Reuse clustering helper from the baseline diarizer if available
    from .speaker_diarization import _agglo
except Exception:  # pragma: no cover - fallback if import fails
    _agglo = None


@dataclass
class CPUOptimizationConfig:
    """Configuration for chunked CPU diarization."""

    chunk_size_sec: float = 30.0
    overlap_sec: float = 2.0
    max_speakers: int | None = None
    enable_vad: bool = True
    energy_threshold_db: float = -60.0


class CPUOptimizedSpeakerDiarizer:
    """Chunked, CPU-friendly diarization wrapper."""

    def __init__(self, base_diarizer, config: CPUOptimizationConfig):
        self.base_diarizer = base_diarizer
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._last_segments: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _chunks(self, audio: np.ndarray, sr: int) -> Iterable[tuple[int, int]]:
        """Yield start/end sample indices for each processing chunk."""
        step = int((self.config.chunk_size_sec - self.config.overlap_sec) * sr)
        size = int(self.config.chunk_size_sec * sr)
        if step <= 0:
            step = size
        cursor = 0
        while cursor < len(audio):
            end = min(cursor + size, len(audio))
            yield cursor, end
            if cursor + size >= len(audio):
                break
            cursor += step

    def _merge_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge overlapping segments from the same speaker."""
        if not segments:
            return []
        segments.sort(key=lambda s: s["start"])
        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            if seg["start"] <= last["end"] and seg["speaker"] == last["speaker"]:
                last["end"] = max(last["end"], seg["end"])
                if seg.get("embedding") is not None and last.get("embedding") is not None:
                    last_emb = np.asarray(last["embedding"], dtype=np.float32)
                    seg_emb = np.asarray(seg["embedding"], dtype=np.float32)
                    last["embedding"] = np.mean([last_emb, seg_emb], axis=0).tolist()
                elif seg.get("embedding") is not None:
                    last["embedding"] = seg["embedding"]
            else:
                merged.append(seg)
        return merged

    def _rms_db(self, audio: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(audio))) + 1e-10
        return 20 * np.log10(rms)

    # ------------------------------------------------------------------
    def diarize_audio(self, audio: np.ndarray, sr: int) -> list[dict[str, Any]]:
        """Diarize ``audio`` using chunked processing and simple energy gating."""
        try:
            if audio is None or len(audio) == 0:
                return []

            audio = audio.astype(np.float32, copy=False)
            segments: list[dict[str, Any]] = []

            for start, end in self._chunks(audio, sr):
                chunk = audio[start:end]

                if self.config.enable_vad and self._rms_db(chunk) < self.config.energy_threshold_db:
                    continue

                offset = start / sr
                result = self.base_diarizer.diarize_audio(chunk, sr)
                # Robustly coerce segments to dictionaries and skip invalid items
                if isinstance(result, Mapping):
                    result_iter = [result]
                elif isinstance(result, (list, tuple)):
                    result_iter = result
                else:
                    self.logger.debug(f"Skipping unexpected diarize result type: {type(result)}")
                    result_iter = []

                for seg in result_iter:
                    try:
                        if isinstance(seg, Mapping):
                            s = dict(seg)
                        elif hasattr(seg, "__dict__"):
                            s = dict(vars(seg))
                        else:
                            # Unsupported type (e.g., str); skip gracefully
                            self.logger.debug(f"Skipping unexpected segment type: {type(seg)}")
                            continue

                        # Normalize required fields
                        s_start = float(s.get("start", 0.0)) + offset
                        s_end = float(s.get("end", 0.0)) + offset
                        if s_end < s_start:
                            s_start, s_end = s_end, s_start
                        s["start"], s["end"] = s_start, s_end
                        s.setdefault("speaker", "Speaker_1")

                        segments.append(s)
                    except Exception:
                        # Skip malformed items without failing the whole run
                        continue

            # Ensure all items are mappings
            segments = [s for s in segments if isinstance(s, dict)]

            # Re-cluster embeddings across chunks for consistent speaker labels
            try:
                emb_list = []
                for s in segments:
                    if isinstance(s, dict) and s.get("embedding") is not None:
                        emb_list.append(np.asarray(s["embedding"], dtype=np.float32))
                if emb_list and _agglo is not None:
                    X = np.vstack(emb_list)
                    cfg = getattr(self.base_diarizer, "config", None)
                    if cfg and getattr(cfg, "speaker_limit", None):
                        clusterer = _agglo(
                            distance_threshold=None,
                            n_clusters=cfg.speaker_limit,
                            linkage=getattr(cfg, "ahc_linkage", "average"),
                            metric="cosine",
                        )
                    else:
                        clusterer = _agglo(
                            distance_threshold=getattr(cfg, "ahc_distance_threshold", 0.15),
                            linkage=getattr(cfg, "ahc_linkage", "average"),
                            metric="cosine",
                        )
                    labels = clusterer.fit_predict(X)
                    idx = 0
                    for seg in segments:
                        if seg.get("embedding") is not None:
                            seg["speaker"] = f"Speaker_{labels[idx] + 1}"
                            idx += 1
            except Exception as e:  # pragma: no cover - clustering is optional
                self.logger.warning(f"Global clustering failed: {e}")

            segments = self._merge_segments(segments)

            if self.config.max_speakers:
                durations: dict[str, float] = {}
                for seg in segments:
                    durations.setdefault(seg["speaker"], 0.0)
                    durations[seg["speaker"]] += seg["end"] - seg["start"]
                top = sorted(durations, key=durations.get, reverse=True)[: self.config.max_speakers]
                segments = [s for s in segments if s["speaker"] in top]

            self._last_segments = segments
            return segments
        except Exception as e:  # pragma: no cover - fallback behaviour
            self.logger.warning(f"CPU-optimized diarization failed: {e}")
            duration = len(audio) / sr if audio is not None else 0.0
            self._last_segments = [{"start": 0.0, "end": duration, "speaker": "Speaker_1"}]
            return self._last_segments

    # ------------------------------------------------------------------
    @property
    def registry(self):
        """Expose the underlying diarizer's speaker registry if present."""
        return getattr(self.base_diarizer, "registry", None)

    def get_segment_embeddings(self) -> list[dict[str, Any]]:
        """Return embeddings from the most recent diarization run."""
        return [
            {"speaker": s["speaker"], "embedding": s["embedding"]}
            for s in self._last_segments
            if s.get("embedding") is not None
        ]
