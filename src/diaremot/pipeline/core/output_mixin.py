"""Output helpers used by the pipeline executor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..outputs import (
    ensure_segment_keys,
    write_human_transcript,
    write_qc_report,
    write_segments_csv,
    write_segments_jsonl,
    write_speakers_summary,
    write_timeline_csv,
)
from ..pipeline_checkpoint_system import ProcessingStage
from ...summaries.conversation_analysis import ConversationMetrics


class OutputMixin:
    def _write_outputs(
        self,
        input_audio_path: str,
        outp: Path,
        segments_final: list[dict[str, Any]],
        speakers_summary: list[dict[str, Any]],
        health: Any,
        turns: list[dict[str, Any]],
        overlap_stats: dict[str, Any],
        per_speaker_interrupts: dict[str, Any],
        conv_metrics: ConversationMetrics | None,
        duration_s: float,
        sed_info: dict[str, Any] | None,
    ) -> None:
        write_segments_csv(outp / "diarized_transcript_with_emotion.csv", segments_final)
        write_segments_jsonl(outp / "segments.jsonl", segments_final)
        write_timeline_csv(outp / "timeline.csv", segments_final)
        write_human_transcript(outp / "diarized_transcript_readable.txt", segments_final)
        write_qc_report(
            outp / "qc_report.json",
            self.stats,
            health,
            n_turns=len(turns),
            n_segments=len(segments_final),
            segments=segments_final,
        )
        write_speakers_summary(outp / "speakers_summary.csv", speakers_summary)

        try:
            html_path = self.html.render_to_html(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            html_path = None
            self.corelog.warn(
                f"HTML summary skipped: {exc}. Verify HTML template assets or install report dependencies."
            )

        try:
            pdf_path = self.pdf.render_to_pdf(
                out_dir=str(outp),
                file_id=self.stats.file_id,
                segments=segments_final,
                speakers_summary=speakers_summary,
                overlap_stats=overlap_stats,
            )
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            pdf_path = None
            self.corelog.warn(
                f"PDF summary skipped: {exc}. Ensure wkhtmltopdf/LaTeX prerequisites are installed."
            )

        self.checkpoints.create_checkpoint(
            input_audio_path,
            ProcessingStage.SUMMARY_GENERATION,
            {"html": html_path, "pdf": pdf_path},
            progress=90.0,
        )

    def _summarize_speakers(
        self,
        segments: list[dict[str, Any]],
        per_speaker_interrupts: dict[str, dict[str, Any]],
        overlap_stats: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        prof: dict[str, dict[str, Any]] = {}
        for s in segments:
            sid = str(s.get("speaker_id", "Unknown"))
            start = float(s.get("start", 0.0) or 0.0)
            end = float(s.get("end", 0.0) or 0.0)
            dur = max(0.0, end - start)
            p = prof.setdefault(
                sid,
                {
                    "speaker_name": s.get("speaker_name"),
                    "total_duration": 0.0,
                    "word_count": 0,
                    "avg_wpm": 0.0,
                    "avg_valence": 0.0,
                    "avg_arousal": 0.0,
                    "avg_dominance": 0.0,
                    "interruptions_made": 0,
                    "interruptions_received": 0,
                    "overlap_ratio": 0.0,
                },
            )
            p["total_duration"] += dur
            words = len((s.get("text") or "").split())
            p["word_count"] += words

            for k_src, k_dst in (
                ("valence", "avg_valence"),
                ("arousal", "avg_arousal"),
                ("dominance", "avg_dominance"),
            ):
                val = s.get(k_src, None)
                if val is not None:
                    prev = p[k_dst]
                    cnt = p.get("_n_" + k_dst, 0) + 1
                    p[k_dst] = (prev * (cnt - 1) + float(val)) / float(cnt)
                    p["_n_" + k_dst] = cnt

        for sid, vals in (per_speaker_interrupts or {}).items():
            p = prof.setdefault(str(sid), {})
            p["interruptions_made"] = int(vals.get("made", 0) or 0)
            p["interruptions_received"] = int(vals.get("received", 0) or 0)

        for sid, p in prof.items():
            if not p.get("speaker_name"):
                p["speaker_name"] = sid
            for k in [k for k in p.keys() if k.startswith("_n_")]:
                del p[k]

        return prof

    def _quick_take(self, speakers: dict[str, dict[str, Any]], duration_s: float) -> str:
        if not speakers:
            return "No speakers identified."
        most = max(speakers.items(), key=lambda kv: float(kv[1].get("total_duration", 0.0)))[1]
        tone = "neutral"
        v = float(most.get("avg_valence", 0.0))
        if v > 0.2:
            tone = "positive"
        elif v < -0.2:
            tone = "negative"
        return f"{len(speakers)} speakers over {int(duration_s // 60)} min; most-active tone {tone}."

    def _moments_to_check(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not segments:
            return []
        arr = [(i, float(s.get("arousal", 0.0) or 0.0)) for i, s in enumerate(segments)]
        arr.sort(key=lambda kv: kv[1], reverse=True)
        picks = arr[:10]
        out: list[dict[str, Any]] = []
        for i, _ in picks:
            s = segments[i]
            out.append(
                {
                    "timestamp": float(s.get("start", 0.0) or 0.0),
                    "speaker": str(s.get("speaker_id", "Unknown")),
                    "description": (s.get("text") or "")[:180],
                    "type": "peak",
                }
            )
        out.sort(key=lambda m: m["timestamp"])
        return out

    def _action_items(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in segments:
            text = (s.get("text") or "").lower()
            intent = str(s.get("intent_top") or s.get("intent") or "")
            if (
                intent in {"command", "instruction", "request", "suggestion"}
                or "let's " in text
                or "we will" in text
            ):
                out.append(
                    {
                        "type": "action",
                        "text": s.get("text") or "",
                        "speaker": str(s.get("speaker_id", "Unknown")),
                        "timestamp": float(s.get("start", 0.0) or 0.0),
                        "confidence": 0.8,
                        "intent": intent or "unknown",
                    }
                )
        return out

    @staticmethod
    def ensure_segment(segment: dict[str, Any], file_id: str) -> dict[str, Any]:
        return ensure_segment_keys(
            {
                "file_id": file_id,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "speaker_id": segment.get("speaker_id", "Unknown"),
                "speaker_name": segment.get("speaker_name", "Unknown"),
                "text": segment.get("text", ""),
            }
        )
