"""Enhanced HTML summary generator with core interface compatibility."""

import json
from pathlib import Path
from typing import Any


def fmt_hms(seconds: float) -> str:
    s = max(0, int(round(float(seconds or 0))))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _coerce_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _safe_json(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if s and isinstance(s, str) else None
    except Exception:
        return None


def _get_speaker_name(spk_id, name=None):
    if name and name.strip():
        return name.strip()
    return f"Speaker {spk_id}" if spk_id != "Unknown" else "Unknown"


class HTMLSummaryGenerator:
    def render_to_html(
        self,
        out_dir: str,
        file_id: str,
        segments: list[dict[str, Any]],
        speakers_summary: list[dict[str, Any]],
        overlap_stats: dict[str, Any],
    ) -> str:
        """Core interface: generate HTML from pipeline data"""
        out_path = Path(out_dir) / "summary.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build metadata from inputs
        duration = max((seg.get("end", 0) or 0) for seg in segments) if segments else 0
        metadata = {
            "title": f"Summary - {file_id}",
            "file_name": file_id,
            "duration_seconds": duration,
            "num_segments": len(segments),
            "num_speakers": len(speakers_summary),
        }

        html = self._build_html(metadata, segments, speakers_summary, overlap_stats)

        with out_path.open("w", encoding="utf-8") as f:
            f.write(html)

        return str(out_path)

    def _build_html(
        self, metadata: dict, segments: list, speakers: list, overlap_stats: dict
    ) -> str:
        duration = _coerce_float(metadata.get("duration_seconds", 0))
        title = metadata.get("title", "Audio Analysis")

        # Executive summary
        lead_speaker = (
            max(speakers, key=lambda s: _coerce_float(s.get("total_duration", 0)))
            if speakers
            else None
        )
        if lead_speaker:
            lead_name = _get_speaker_name(
                lead_speaker.get("speaker_id"), lead_speaker.get("speaker_name")
            )
            lead_pct = 100 * _coerce_float(lead_speaker.get("total_duration", 0)) / max(1, duration)
            exec_summary = f"{lead_name} spoke most ({lead_pct:.0f}%). Total: {fmt_hms(duration)}, {len(speakers)} speakers."
        else:
            exec_summary = f"Audio length: {fmt_hms(duration)}"

        # Key moments (top arousal segments)
        moments = []
        if segments:
            sorted_segs = sorted(
                segments, key=lambda s: _coerce_float(s.get("arousal", 0)), reverse=True
            )[:5]
            for seg in sorted_segs:
                if _coerce_float(seg.get("arousal", 0)) > 0.3:
                    moments.append(
                        {
                            "time": fmt_hms(_coerce_float(seg.get("start", 0))),
                            "speaker": _get_speaker_name(
                                seg.get("speaker_id"), seg.get("speaker_name")
                            ),
                            "text": (seg.get("text", "") or "")[:100]
                            + ("..." if len(seg.get("text", "") or "") > 100 else ""),
                            "arousal": _coerce_float(seg.get("arousal", 0)),
                        }
                    )

        # Aggregate simple voice-quality metrics
        def _avg(key: str):
            vals = []
            for s in segments or []:
                v = s.get(key)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            return (sum(vals) / len(vals)) if vals else None

        vq_jitter = _avg("vq_jitter_pct")
        vq_shimmer = _avg("vq_shimmer_db")
        vq_hnr = _avg("vq_hnr_db")
        vq_cpps = _avg("vq_cpps_db")

        return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
body {{ font: 14px system-ui; margin: 0; background: #f8f9fa; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px; }}
.header h1 {{ margin: 0 0 10px; font-size: 2.5rem; }}
.header .meta {{ opacity: 0.9; font-size: 1.1rem; }}
.section {{ background: white; border-radius: 8px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.section h2 {{ margin: 0 0 20px; color: #333; }}
.speakers-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
.speaker-card {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; }}
.speaker-name {{ font-weight: bold; font-size: 1.2rem; margin-bottom: 10px; }}
.duration-bar {{ height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin: 10px 0; }}
.duration-fill {{ height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-top: 15px; }}
.metric {{ display: flex; justify-content: space-between; padding: 8px; background: #f8f9fa; border-radius: 4px; }}
.moments {{ display: flex; flex-direction: column; gap: 15px; }}
.moment {{ display: flex; gap: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #667eea; }}
.moment-time {{ font-weight: bold; color: #667eea; min-width: 60px; }}
.moment-content {{ flex: 1; }}
.transcript {{ height: clamp(240px, 50vh, 600px); overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; }}
.transcript-row {{ display: flex; gap: 15px; padding: 10px 15px; border-bottom: 1px solid #f0f0f0; }}
.transcript-time {{ min-width: 60px; color: #666; font-family: monospace; }}
.transcript-speaker {{ min-width: 100px; font-weight: bold; }}
.transcript-text {{ flex: 1; overflow-wrap: anywhere; word-break: break-word; }}
.transcript-row:hover {{ background: #f8f9fa; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Audio Analysis Report</h1>
    <div class="meta">{metadata.get("file_name", "Unknown")} &bull; {fmt_hms(duration)} &bull; {len(speakers)} speakers</div>
  </div>

  <div class="section">
    <h2>Executive Summary</h2>
    <p>{exec_summary}</p>
  </div>

  <div class="section">
    <h2>Speaker Analysis</h2>
    <div class="speakers-grid">
      {self._build_speaker_cards(speakers, duration)}
    </div>
  </div>

  <div class="section">
    <h2>Voice Quality</h2>
    <div class="metrics">
      <div class="metric"><span>Jitter (pct):</span><span>{(vq_jitter if vq_jitter is not None else "—")}</span></div>
      <div class="metric"><span>Shimmer (dB):</span><span>{(vq_shimmer if vq_shimmer is not None else "—")}</span></div>
      <div class="metric"><span>HNR (dB):</span><span>{(vq_hnr if vq_hnr is not None else "—")}</span></div>
      <div class="metric"><span>CPPS (dB):</span><span>{(vq_cpps if vq_cpps is not None else "—")}</span></div>
    </div>
  </div>

  {self._build_key_moments_section(moments)}

  <div class="section">
    <h2>Transcript</h2>
    <div class="transcript">
      {self._build_transcript(segments)}
    </div>
  </div>
</div>
</body>
</html>"""

    def _build_speaker_cards(self, speakers: list, total_duration: float) -> str:
        if not speakers:
            return "<p>No speakers identified</p>"

        cards = []
        for spk in speakers:
            name = _get_speaker_name(spk.get("speaker_id"), spk.get("speaker_name"))
            duration = _coerce_float(spk.get("total_duration", 0))
            pct = 100 * duration / max(1, total_duration)

            # Get emotion data
            emotions = _safe_json(spk.get("text_emotions_top5_json", "")) or []
            top_emotions = ", ".join(
                [
                    f"{e.get('label', '?')} {100 * _coerce_float(e.get('score', 0)):.0f}%"
                    for e in emotions[:3]
                ]
            )

            cards.append(
                f"""
            <div class="speaker-card">
              <div class="speaker-name">{name}</div>
              <div class="duration-bar">
                <div class="duration-fill" style="width:{pct:.1f}%"></div>
              </div>
              <div>{pct:.1f}% ({fmt_hms(duration)})</div>
              <div class="metrics">
                <div class="metric"><span>Words:</span><span>{int(_coerce_float(spk.get("word_count", 0)))}</span></div>
                <div class="metric"><span>WPM:</span><span>{_coerce_float(spk.get("avg_wpm", 0)):.0f}</span></div>
                <div class="metric"><span>Valence:</span><span>{_coerce_float(spk.get("avg_valence", 0)):+.2f}</span></div>
                <div class="metric"><span>Arousal:</span><span>{_coerce_float(spk.get("avg_arousal", 0)):+.2f}</span></div>
                <div class="metric"><span>Dominance:</span><span>{_coerce_float(spk.get("avg_dominance", 0)):+.2f}</span></div>
                <div class="metric"><span>VQ Hint:</span><span>{spk.get("voice_quality_hint", "") or "—"}</span></div>
                <div class="metric"><span>VQ Jitter (%):</span><span>{_coerce_float(spk.get("vq_jitter_pct_med", 0)):.2f}</span></div>
                <div class="metric"><span>VQ Shimmer (dB):</span><span>{_coerce_float(spk.get("vq_shimmer_db_med", 0)):.2f}</span></div>
                <div class="metric"><span>HNR (dB):</span><span>{_coerce_float(spk.get("vq_hnr_db_med", 0)):.1f}</span></div>
                <div class="metric"><span>CPPS (dB):</span><span>{_coerce_float(spk.get("vq_cpps_db_med", 0)):.1f}</span></div>
                <div class="metric"><span>F0 mean (Hz):</span><span>{_coerce_float(spk.get("f0_mean_hz", 0)):.1f}</span></div>
                <div class="metric"><span>F0 spread (Hz):</span><span>{_coerce_float(spk.get("f0_std_hz", 0)):.1f}</span></div>
                <div class="metric"><span>Loudness med:</span><span>{_coerce_float(spk.get("loudness_rms_med", 0)):.2f}</span></div>
                <div class="metric"><span>Loudness DR (dB):</span><span>{_coerce_float(spk.get("loudness_dr_db", 0)):.1f}</span></div>
                <div class="metric"><span>Pauses (sec):</span><span>{_coerce_float(spk.get("pause_total_sec", 0)):.1f}</span></div>
                <div class="metric"><span>Pause count:</span><span>{int(_coerce_float(spk.get("pause_count", 0)))}</span></div>
                <div class="metric"><span>Pause ratio:</span><span>{_coerce_float(spk.get("pause_ratio", 0)):.2f}</span></div>
                <div class="metric"><span>Interruptions (made/recv):</span><span>{int(_coerce_float(spk.get("interruptions_made", 0)))}/{int(_coerce_float(spk.get("interruptions_received", 0)))}</span></div>
                <div class="metric"><span>Overlap ratio:</span><span>{spk.get("overlap_ratio", "") if spk.get("overlap_ratio", "") != "" else "—"}</span></div>
                <div class="metric"><span>Dominant speech:</span><span>{spk.get("dominant_speech_emotion", "neutral")}</span></div>
                <div class="metric"><span>Dominant text:</span><span>{spk.get("dominant_text_emotion", "neutral")}</span></div>
              </div>
              <div style="margin-top:10px; font-size:0.9em; color:#666;">
                {top_emotions or "No emotions detected"}
              </div>
              </div>
            """
            )

        return "".join(cards)

    def _build_key_moments_section(self, moments: list) -> str:
        if not moments:
            return ""

        moment_html = []
        for m in moments:
            moment_html.append(
                f"""
            <div class="moment">
              <div class="moment-time">{m["time"]}</div>
              <div class="moment-content">
                <div><strong>{m["speaker"]}</strong> (arousal: {m["arousal"]:.2f})</div>
                <div>{m["text"]}</div>
              </div>
            </div>
            """
            )

        return f"""
        <div class="section">
          <h2>Key Moments</h2>
          <div class="moments">
            {"".join(moment_html)}
          </div>
        </div>
        """

    def _build_transcript(self, segments: list) -> str:
        if not segments:
            return "<p>No transcript available</p>"

        rows = []
        for seg in segments[:100]:  # Limit for performance
            time = fmt_hms(_coerce_float(seg.get("start", 0)))
            speaker = _get_speaker_name(seg.get("speaker_id"), seg.get("speaker_name"))
            text = seg.get("text", "") or ""

            rows.append(
                f"""
            <div class="transcript-row">
              <div class="transcript-time">{time}</div>
              <div class="transcript-speaker">{speaker}</div>
              <div class="transcript-text">{text}</div>
            </div>
            """
            )

        return "".join(rows)


def render_summary_html(
    *,
    file_id: str,
    segments: list[dict[str, Any]],
    speakers_summary: list[dict[str, Any]],
    quick_take: str,
    key_moments: list[dict[str, Any]],
    action_items: list[dict[str, Any]],
    overlap_stats: dict[str, Any],
    outdir: str,
    conv_metrics: Any = None,
) -> str:
    """Compatibility wrapper expected by the pipeline.

    Builds the full HTML string and also writes it to outdir/summary.html.
    """
    gen = HTMLSummaryGenerator()
    # Reuse generator HTML building; write file and return HTML.
    # The class render_to_html writes and returns path, but we also need the HTML string
    # for the pipeline's assignment. So call the internal builder directly.
    duration = max((seg.get("end", 0) or 0) for seg in segments) if segments else 0
    metadata = {
        "title": f"Summary - {file_id}",
        "file_name": file_id,
        "duration_seconds": duration,
        "num_segments": len(segments),
        "num_speakers": len(speakers_summary),
    }
    html = gen._build_html(metadata, segments, speakers_summary, overlap_stats)
    # Also write a file for convenience (mirrors prior behavior)
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "summary.html").write_text(html, encoding="utf-8")
    return html


__all__ = ["HTMLSummaryGenerator", "render_summary_html"]
