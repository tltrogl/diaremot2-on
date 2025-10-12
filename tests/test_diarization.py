"""Tests for the diarization clustering helper."""

from __future__ import annotations

import numpy as np
import pytest

from diaremot.pipeline.speaker_diarization import DiarizedTurn, collapse_single_speaker_turns
from diaremot2_on.diarization import DiarizationClustering, Segment


class _MockClusterer:
    """Simple mock returning preconfigured cluster labels."""

    def __init__(self, labels):
        self._labels = list(labels)
        self.calls = 0

    def fit_predict(self, embeddings):  # pragma: no cover - trivial wiring
        self.calls += 1
        assert len(embeddings) == len(self._labels)
        return self._labels


@pytest.fixture
def embeddings():
    """Create deterministic embeddings for three segments."""

    base = [round(i * 0.25, 2) for i in range(4)]
    return [tuple(value + offset for value in base) for offset in (0.0, 0.1, -0.1)]


def test_single_speaker_is_not_split(embeddings):
    """All segments mapped to the same cluster must share a label."""

    clusterer = _MockClusterer([0, 0, 0])
    diarizer = DiarizationClustering(clusterer)

    segments = [Segment(i * 1.5, (i + 1) * 1.5, emb) for i, emb in enumerate(embeddings)]
    labelled = diarizer.assign_speakers(segments)

    speakers = {segment.speaker for segment in labelled}
    assert speakers == {"SPK_1"}
    # Ensure we did not mutate the original start/end times.
    assert labelled[0].start == pytest.approx(0.0)
    assert labelled[-1].end == pytest.approx(4.5)


def test_clusters_map_to_distinct_labels(embeddings):
    """Different cluster ids should produce distinct speaker labels."""

    clusterer = _MockClusterer([1, 2, 1])
    diarizer = DiarizationClustering(clusterer, label_prefix="SPEAKER")

    segments = [Segment(i, i + 0.5, emb) for i, emb in enumerate(embeddings)]
    labelled = diarizer.assign_speakers(segments)

    assert [segment.speaker for segment in labelled] == ["SPEAKER_1", "SPEAKER_2", "SPEAKER_1"]


def test_singleton_input_shortcuts_clusterer():
    """A single segment should not invoke the estimator."""

    clusterer = _MockClusterer([0, 0])
    diarizer = DiarizationClustering(clusterer)

    segment = Segment(0.0, 1.0, (0.1, 0.2, 0.3))
    labelled = diarizer.assign_speakers([segment])

    assert [segment.speaker for segment in labelled] == ["SPK_1"]
    assert clusterer.calls == 0


def _turn(speaker: str, start: float, end: float, *, embedding: np.ndarray | None = None) -> DiarizedTurn:
    emb = embedding if embedding is not None else np.ones(3, dtype=np.float32)
    return DiarizedTurn(
        start=start,
        end=end,
        speaker=speaker,
        speaker_name=speaker,
        embedding=emb,
    )


def test_collapse_single_speaker_by_dominance():
    turns = [
        _turn("Speaker_1", 0.0, 5.0, embedding=np.ones(4, dtype=np.float32)),
        _turn("Speaker_1", 5.0, 10.0, embedding=np.ones(4, dtype=np.float32)),
        _turn("Speaker_2", 10.0, 10.4, embedding=np.ones(4, dtype=np.float32) * 0.99),
        _turn("Speaker_3", 10.4, 10.8, embedding=np.ones(4, dtype=np.float32) * 1.01),
    ]

    collapsed, label, reason = collapse_single_speaker_turns(
        turns, dominance_threshold=0.8, centroid_threshold=0.2, min_turns=3
    )

    assert collapsed
    assert label == "Speaker_1"
    assert reason is not None
    assert {turn.speaker for turn in turns} == {"Speaker_1"}


def test_collapse_respects_true_multi_speaker():
    turns = [
        _turn("Speaker_A", 0.0, 5.0, embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        _turn("Speaker_B", 5.0, 10.0, embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]

    collapsed, label, reason = collapse_single_speaker_turns(
        turns, dominance_threshold=0.8, centroid_threshold=0.05, min_turns=2
    )

    assert not collapsed
    assert label is None
    assert reason is None
    assert {turn.speaker for turn in turns} == {"Speaker_A", "Speaker_B"}
