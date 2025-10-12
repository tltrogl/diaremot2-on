"""Tests for the diarization clustering helper."""

from __future__ import annotations

import pytest

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
