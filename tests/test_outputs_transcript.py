from __future__ import annotations

import json

from diaremot.pipeline.outputs import write_human_transcript


def test_write_human_transcript(tmp_path):
    segments = [
        {
            "start": 5.2,
            "end": 12.9,
            "speaker_name": "Speaker A",
            "text": "Hello world",
            "valence": 0.1,
            "arousal": -0.2,
            "dominance": 0.3,
            "emotion_top": "happy",
            "affect_hint": "positive-greeting",
            "intent_top": "greeting",
            "intent_top3_json": json.dumps(
                [
                    {"label": "greeting", "score": 0.9},
                    {"label": "status_update", "score": 0.5},
                    {"label": "question", "score": 0.2},
                ]
            ),
            "events_top3_json": json.dumps(
                [
                    {"label": "Speech", "score": 0.8},
                    {"label": "Alarm", "score": 0.1},
                ]
            ),
            "noise_tag": "Speech",
            "vad_unstable": False,
            "duration_s": 7.7,
            "wpm": 120.5,
        }
    ]

    out_file = tmp_path / "readable.txt"
    write_human_transcript(out_file, segments)

    rendered = out_file.read_text(encoding="utf-8")
    assert "[00:00:05 - 00:00:13] Speaker A" in rendered
    assert "Text: Hello world" in rendered
    assert "Affect: emotion happy; valence +0.10, arousal -0.20, dominance +0.30; hint positive-greeting" in rendered
    assert "Intent: greeting (top3: greeting 0.90, status_update 0.50, question 0.20)" in rendered
    assert "VAD: stable | SED: Speech, Alarm | Noise tag: Speech" in rendered
    assert "Duration: 7.70s | Speech rate: 120.50 wpm" in rendered
