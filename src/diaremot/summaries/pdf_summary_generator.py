"""
pdf_summary_generator.py — Human-friendly PDF mirroring HTML features.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _fmt_hms(seconds: float) -> str:
    try:
        s = int(round(max(0.0, float(seconds))))
    except Exception:
        s = 0
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _safe_json(s):
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, str) and s.strip():
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _build_speaker_rows_from_profiles(
    spk_profiles: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for sid, prof in (spk_profiles or {}).items():
        rows.append(
            {
                "speaker_id": sid,
                "speaker_name": prof.get("speaker_name", ""),
                "total_duration": prof.get("total_duration", 0.0),
                "word_count": prof.get("word_count", 0),
                "avg_wpm": prof.get("avg_wpm", prof.get("avg_speech_rate", 0.0)),
                "avg_valence": prof.get("avg_valence", 0.0),
                "avg_arousal": prof.get("avg_arousal", 0.0),
                "avg_dominance": prof.get("avg_dominance", 0.0),
                "text_emotions_top5_json": (
                    json.dumps(prof.get("text_emotions_top5", []), ensure_ascii=False)
                    if isinstance(prof.get("text_emotions_top5"), list)
                    else prof.get("text_emotions_top5_json", "")
                ),
            }
        )
    return rows


class PDFSummaryGenerator:
    def render_to_pdf(
        self,
        out_dir: str,
        file_id: str,
        segments: list[dict[str, Any]],
        speakers_summary: list[dict[str, Any]],
        overlap_stats: dict[str, Any],
    ) -> str:
        """
        Core interface: generate PDF from pipeline data
        """
        out_path = Path(out_dir) / "summary.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        duration = max((s.get("end", 0) or 0) for s in segments) if segments else 0
        title = f"Summary - {file_id}"

        return self._generate_pdf(
            out_path,
            title,
            duration,
            len(speakers_summary),
            len(segments),
            speakers_summary,
            segments,
        )

    def _generate_pdf(
        self,
        out_path: Path,
        title: str,
        duration: float,
        num_speakers: int,
        num_segments: int,
        speakers: list,
        segments: list,
    ) -> str:
        gen_ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            leftMargin=0.7 * inch,
            rightMargin=0.7 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
        )
        story = []

        # Title
        title_style = ParagraphStyle("title", fontSize=18, leading=22, spaceAfter=12)
        meta_style = ParagraphStyle("meta", fontSize=10, textColor=colors.grey, spaceAfter=6)
        story.append(Paragraph(f"<b>{title}</b>", title_style))
        story.append(
            Paragraph(
                f"Duration: {_fmt_hms(duration)} • Speakers: {num_speakers} • Segments: {num_segments} • Generated: {gen_ts}",
                meta_style,
            )
        )
        story.append(Spacer(1, 8))

        # Executive summary
        story.append(
            Paragraph(
                "<b>Executive Summary</b>",
                ParagraphStyle("h2", fontSize=14, spaceBefore=6, spaceAfter=6),
            )
        )
        if speakers:
            lead = max(speakers, key=lambda r: _float(r.get("total_duration", 0)))
            lead_name = lead.get("speaker_name") or lead.get("speaker_id")
            share = 100.0 * _float(lead.get("total_duration", 0)) / max(1.0, duration)
            summary = f"{lead_name} spoke most (~{share:.0f}%). Audio: {_fmt_hms(duration)}, {num_speakers} speakers."
        else:
            summary = f"Audio length {_fmt_hms(duration)}"
        story.append(Paragraph(summary, ParagraphStyle("p", fontSize=11, spaceAfter=8)))

        # Speaker table
        if speakers:
            story.append(
                Paragraph(
                    "<b>Speaker Analysis</b>",
                    ParagraphStyle("h2", fontSize=14, spaceBefore=6, spaceAfter=6),
                )
            )
            data = [
                [
                    "Speaker",
                    "Time",
                    "Words",
                    "WPM",
                    "Val",
                    "Aro",
                    "Dom",
                    "VQ Hint",
                    "VQ (jit/shm/HNR/CPPS)",
                    "Top Emotions",
                ]
            ]

            for spk in speakers:
                name = spk.get("speaker_name") or spk.get("speaker_id") or "Unknown"
                spk_dur = _float(spk.get("total_duration", 0))
                words = int(_float(spk.get("word_count", 0)))
                wpm = _float(spk.get("avg_wpm", 0))
                val = _float(spk.get("avg_valence", 0))
                aro = _float(spk.get("avg_arousal", 0))
                dom = _float(spk.get("avg_dominance", 0))

                vq_hint = spk.get("voice_quality_hint", "") or "—"
                vq_summary = (
                    f"{_float(spk.get('vq_jitter_pct_med', 0)):.2f}%/"
                    f"{_float(spk.get('vq_shimmer_db_med', 0)):.2f}dB/"
                    f"{_float(spk.get('vq_hnr_db_med', 0)):.1f}dB/"
                    f"{_float(spk.get('vq_cpps_db_med', 0)):.1f}dB"
                )

                emotions = _safe_json(spk.get("text_emotions_top5_json", "")) or []
                top3 = ", ".join(
                    [
                        f"{e.get('label', '?')} {100 * _float(e.get('score', 0)):.0f}%"
                        for e in emotions[:3]
                    ]
                )

                data.append(
                    [
                        name,
                        _fmt_hms(spk_dur),
                        str(words),
                        f"{wpm:.0f}",
                        f"{val:.2f}",
                        f"{aro:.2f}",
                        f"{dom:.2f}",
                        vq_hint,
                        vq_summary,
                        top3 or "—",
                    ]
                )

            tbl = Table(data, repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ]
                )
            )
            story.append(tbl)

        # Key moments
        moments = []
        if segments:
            sorted_segs = sorted(segments, key=lambda s: _float(s.get("arousal", 0)), reverse=True)[
                :5
            ]
            for seg in sorted_segs:
                if _float(seg.get("arousal", 0)) > 0.3:
                    text = seg.get("text", "") or ""
                    preview = text[:100] + "..." if len(text) > 100 else text
                    moments.append(
                        {
                            "time": _fmt_hms(_float(seg.get("start", 0))),
                            "speaker": seg.get("speaker_name")
                            or seg.get("speaker_id")
                            or "Unknown",
                            "preview": preview,
                        }
                    )

        if moments:
            story.append(
                Paragraph(
                    "<b>Key Moments</b>",
                    ParagraphStyle("h2", fontSize=14, spaceBefore=6, spaceAfter=6),
                )
            )
            for m in moments:
                story.append(
                    Paragraph(
                        f"<b>{m['time']} - {m['speaker']}:</b> {m['preview']}",
                        ParagraphStyle("p", fontSize=10, spaceAfter=4),
                    )
                )

        doc.build(story)
        return str(out_path)

    def generate_pdf_legacy(self, analysis_results: dict[str, Any], output_path: str) -> str:
        """Legacy interface for backward compatibility"""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = analysis_results.get("metadata", {}) or {}
        title = meta.get("title") or "Conversation Summary"
        duration = _float(meta.get("duration_seconds", 0.0))
        nspeakers = int(_float(meta.get("num_speakers", 0)))
        nsegments = int(_float(meta.get("num_segments", 0)))

        # Get speaker data
        spk_rows = _build_speaker_rows_from_profiles(analysis_results.get("speaker_profiles", {}))
        segments = analysis_results.get("segments", [])

        return self._generate_pdf(
            out_path, title, duration, nspeakers, nsegments, spk_rows, segments
        )
