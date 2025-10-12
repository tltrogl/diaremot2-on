"""Speaker diarization utilities.

This module contains a light-weight clustering wrapper that assigns
consistent speaker identifiers to contiguous segments.  A regression was
reported where a recording containing a single person was labelled as 26
unique speakers.  The root cause turned out to be that the code was
enumerating the segments when formatting labels instead of reusing the
cluster assignments returned by the clustering model.  As a result every
segment was forced to become a new speaker.

The :class:`DiarizationClustering` class below fixes the issue by keeping a
stable mapping between cluster ids and the speaker labels we expose to the
rest of the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass
class Segment:
    """Light weight representation of an audio segment.

    Attributes
    ----------
    start:
        Start time of the segment in seconds.
    end:
        End time of the segment in seconds.
    embedding:
        A sequence of floats containing the speaker embedding for the
        segment.  We deliberately avoid depending on :mod:`numpy` so this
        package can operate in minimal environments.
    speaker:
        Optional speaker label assigned after clustering.
    """

    start: float
    end: float
    embedding: Sequence[float]
    speaker: str | None = None


class DiarizationClustering:
    """Wrap a clustering model used for speaker diarization.

    Parameters
    ----------
    clusterer:
        Any object exposing a ``fit_predict`` method compatible with the
        scikit-learn API.  The estimator is expected to return integer
        cluster labels where the same integer is used for segments belonging
        to the same speaker.
    label_prefix:
        Prefix used when generating human readable speaker labels.
    """

    def __init__(self, clusterer, *, label_prefix: str = "SPK") -> None:
        self._clusterer = clusterer
        self._label_prefix = label_prefix

    def assign_speakers(self, segments: Sequence[Segment]) -> list[Segment]:
        """Assign speaker ids to the provided segments."""

        if not segments:
            return []

        embeddings = [tuple(segment.embedding) for segment in segments]
        labels = self._fit_predict(embeddings)
        speaker_labels = self._labels_to_speakers(labels)

        labelled_segments: list[Segment] = []
        for segment, speaker in zip(segments, speaker_labels):
            labelled_segments.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    embedding=segment.embedding,
                    speaker=speaker,
                )
            )
        return labelled_segments

    # --- internal helpers -------------------------------------------------
    def _fit_predict(self, embeddings: Sequence[Sequence[float]]) -> list[int]:
        """Execute the underlying clustering model."""

        if len(embeddings) == 1:
            return [0]

        labels = self._clusterer.fit_predict(embeddings)
        return [int(label) for label in labels]

    def _labels_to_speakers(self, labels: Iterable[int]) -> list[str]:
        """Convert raw cluster ids into stable speaker labels."""

        speaker_map: dict[int, str] = {}
        result: list[str] = []
        for label in labels:
            if label not in speaker_map:
                speaker_map[label] = f"{self._label_prefix}_{len(speaker_map) + 1}"
            result.append(speaker_map[label])
        return result


__all__ = ["DiarizationClustering", "Segment"]
