"""Conversation flow analysis and metrics calculation with robust error handling."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConversationMetrics:
    turn_taking_balance: float  # entropy of speaker participation
    interruption_rate_per_min: float
    avg_turn_duration_sec: float
    conversation_pace_turns_per_min: float
    silence_ratio: float
    speaker_dominance: dict[str, float]  # speaker_id -> % of total time
    response_latency_stats: dict[str, float]  # avg, median response times
    topic_coherence_score: float  # 0-1, semantic consistency
    energy_flow: list[dict[str, Any]]  # time-series of engagement
    interruptions_per_speaker: dict[str, float] = field(default_factory=dict)


def analyze_conversation_flow(
    segments: list[dict[str, Any]], total_duration_sec: float
) -> ConversationMetrics:
    """Analyze conversation patterns and dynamics with robust error handling."""
    try:
        if not segments or total_duration_sec <= 0:
            return _empty_metrics()

        # Sort segments by time safely
        try:
            segs = sorted(segments, key=lambda x: float(x.get("start", 0) or 0))
        except (TypeError, ValueError):
            segs = segments  # Use original order if sorting fails

        # 1. Speaker participation with safe division
        speaker_times = {}
        total_speech_time = 0.0

        for seg in segs:
            try:
                spk = seg.get("speaker_id") or seg.get("speaker") or "Unknown"
                start = float(seg.get("start", 0) or 0)
                end = float(seg.get("end", 0) or 0)
                duration = max(0.0, end - start)  # Ensure non-negative

                speaker_times[spk] = speaker_times.get(spk, 0.0) + duration
                total_speech_time += duration
            except (TypeError, ValueError):
                # Skip malformed segments
                continue

        # Ensure we have valid data
        if not speaker_times or total_speech_time <= 0:
            return _empty_metrics()

        # Dominance percentages with safe division
        dominance = {}
        for spk, time in speaker_times.items():
            try:
                dominance[spk] = (time / total_speech_time) * 100.0
            except ZeroDivisionError:
                dominance[spk] = 0.0

        # Turn-taking balance (entropy) with safety checks
        balance = 0.0
        if len(speaker_times) > 1 and total_speech_time > 0:
            try:
                probs = [t / total_speech_time for t in speaker_times.values()]
                # Filter out zero probabilities
                probs = [p for p in probs if p > 0]
                if probs:
                    entropy = -sum(p * np.log2(p) for p in probs)
                    max_entropy = np.log2(len(probs))
                    balance = entropy / max_entropy if max_entropy > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                balance = 0.0

        # 2. Turn-taking dynamics with interruption tracking
        turns = []
        prev_speaker = None
        prev_end = 0.0
        interrupts: dict[str, int] = {}

        for seg in segs:
            try:
                current_speaker = seg.get("speaker_id") or seg.get("speaker") or "Unknown"
                start = float(seg.get("start", 0) or 0)
                end = float(seg.get("end", 0) or 0)
                duration = max(0.0, end - start)

                if current_speaker != prev_speaker and duration > 0:
                    turns.append(
                        {
                            "speaker": current_speaker,
                            "start": start,
                            "end": end,
                            "duration": duration,
                        }
                    )
                if prev_speaker is not None and start < prev_end:
                    interrupts[current_speaker] = interrupts.get(current_speaker, 0) + 1
                prev_speaker = current_speaker
                prev_end = end
            except (TypeError, ValueError):
                continue

        # Turn statistics with safety checks
        turn_durations = [t["duration"] for t in turns if t.get("duration", 0) > 0]
        avg_turn_duration = np.mean(turn_durations) if turn_durations else 0.0

        # Safe division for turns per minute
        try:
            duration_minutes = total_duration_sec / 60.0
            turns_per_min = len(turns) / duration_minutes if duration_minutes > 0 else 0.0
        except ZeroDivisionError:
            turns_per_min = 0.0

        # Interruption rate per minute
        try:
            interrupt_rate = (
                sum(interrupts.values()) / duration_minutes if duration_minutes > 0 else 0.0
            )
            interrupts_per_speaker = {
                spk: (cnt / duration_minutes if duration_minutes > 0 else 0.0)
                for spk, cnt in interrupts.items()
            }
        except Exception:
            interrupt_rate = 0.0
            interrupts_per_speaker = {}

        # 3. Response latencies with bounds checking
        response_times = []
        for i in range(1, len(turns)):
            try:
                prev_end = turns[i - 1]["end"]
                curr_start = turns[i]["start"]
                latency = curr_start - prev_end
                # Only include reasonable response times (0-5 seconds)
                if 0 <= latency <= 5.0:
                    response_times.append(latency)
            except (KeyError, TypeError, ValueError):
                continue

        response_stats = {
            "avg_sec": float(np.mean(response_times)) if response_times else 0.0,
            "median_sec": float(np.median(response_times)) if response_times else 0.0,
            "count": len(response_times),
        }

        # 4. Topic coherence with error handling
        coherence = _calculate_topic_coherence(segs)

        # 5. Energy flow over time with bounds checking
        energy_flow = _calculate_energy_flow(segs, total_duration_sec)

        # 6. Silence ratio with safe division
        silence_ratio = 0.0
        try:
            silence_ratio = max(
                0.0,
                min(1.0, (total_duration_sec - total_speech_time) / total_duration_sec),
            )
        except ZeroDivisionError:
            silence_ratio = 1.0

        return ConversationMetrics(
            turn_taking_balance=float(balance),
            interruption_rate_per_min=float(interrupt_rate),
            avg_turn_duration_sec=float(avg_turn_duration),
            conversation_pace_turns_per_min=float(turns_per_min),
            silence_ratio=float(silence_ratio),
            speaker_dominance=dominance,
            response_latency_stats=response_stats,
            topic_coherence_score=float(coherence),
            energy_flow=energy_flow,
            interruptions_per_speaker=interrupts_per_speaker,
        )

    except Exception as e:
        # Fallback to empty metrics on any error
        print(f"Warning: conversation analysis failed: {e}")
        return _empty_metrics()


def _calculate_topic_coherence(segments: list[dict[str, Any]]) -> float:
    """Calculate topic coherence with error handling."""
    try:
        if len(segments) < 2:
            return 1.0

        coherence_scores = []

        for i in range(1, len(segments)):
            try:
                prev_text = str(segments[i - 1].get("text", "")).lower().split()
                curr_text = str(segments[i].get("text", "")).lower().split()

                if not prev_text or not curr_text:
                    continue

                # Simple keyword overlap - only content words
                prev_words = set(w for w in prev_text if len(w) > 3)
                curr_words = set(w for w in curr_text if len(w) > 3)

                if prev_words and curr_words:
                    overlap = len(prev_words & curr_words)
                    union = len(prev_words | curr_words)
                    if union > 0:
                        jaccard = overlap / union
                        coherence_scores.append(jaccard)
            except (TypeError, AttributeError):
                continue

        return np.mean(coherence_scores) if coherence_scores else 0.0

    except Exception:
        return 0.0


def _calculate_energy_flow(
    segments: list[dict[str, Any]], total_duration: float
) -> list[dict[str, Any]]:
    """Calculate engagement/energy over time windows with error handling."""
    try:
        if total_duration <= 0:
            return []

        window_sec = 30.0  # 30-second windows
        num_windows = max(1, int(total_duration / window_sec))

        flow = []
        for i in range(num_windows):
            try:
                window_start = i * window_sec
                window_end = min((i + 1) * window_sec, total_duration)

                # Find segments in this window with bounds checking
                window_segs = []
                for s in segments:
                    try:
                        seg_start = float(s.get("start", 0) or 0)
                        seg_end = float(s.get("end", 0) or 0)
                        if seg_start < window_end and seg_end > window_start:
                            window_segs.append(s)
                    except (TypeError, ValueError):
                        continue

                if not window_segs:
                    energy = 0.0
                    speakers = 0
                    speech_density = 0.0
                else:
                    # Energy = average arousal * speech density
                    arousals = []
                    for s in window_segs:
                        try:
                            arousal = float(s.get("arousal", 0) or 0)
                            arousals.append(arousal)
                        except (TypeError, ValueError):
                            arousals.append(0.0)

                    avg_arousal = np.mean(arousals) if arousals else 0.0

                    # Calculate speech time in window
                    speech_time = 0.0
                    for s in window_segs:
                        try:
                            seg_start = float(s.get("start", 0) or 0)
                            seg_end = float(s.get("end", 0) or 0)
                            # Clip to window boundaries
                            clipped_start = max(seg_start, window_start)
                            clipped_end = min(seg_end, window_end)
                            if clipped_end > clipped_start:
                                speech_time += clipped_end - clipped_start
                        except (TypeError, ValueError):
                            continue

                    speech_density = speech_time / window_sec if window_sec > 0 else 0.0
                    energy = avg_arousal * speech_density

                    # Count unique speakers
                    speakers = len(
                        set(
                            s.get("speaker_id") or s.get("speaker") or "Unknown"
                            for s in window_segs
                        )
                    )

                flow.append(
                    {
                        "window_start_sec": window_start,
                        "window_end_sec": window_end,
                        "energy_score": float(energy),
                        "active_speakers": int(speakers),
                        "speech_density": float(speech_density),
                    }
                )

            except Exception:
                # Skip this window on error
                continue

        return flow

    except Exception:
        return []


def _empty_metrics() -> ConversationMetrics:
    """Return empty/default metrics for edge cases."""
    return ConversationMetrics(
        turn_taking_balance=0.0,
        interruption_rate_per_min=0.0,
        avg_turn_duration_sec=0.0,
        conversation_pace_turns_per_min=0.0,
        silence_ratio=1.0,
        speaker_dominance={},
        response_latency_stats={"avg_sec": 0.0, "median_sec": 0.0, "count": 0},
        topic_coherence_score=0.0,
        energy_flow=[],
        interruptions_per_speaker={},
    )


def build_conversation_analysis(
    segments: list[dict[str, Any]],
    total_duration_sec: float,
    overlap_stats: dict[str, Any],
) -> dict[str, Any]:
    """Core interface: build conversation analysis from pipeline data"""
    metrics = analyze_conversation_flow(segments, total_duration_sec)

    return {
        "turn_taking_balance": metrics.turn_taking_balance,
        "interruption_rate_per_min": metrics.interruption_rate_per_min,
        "avg_turn_duration_sec": metrics.avg_turn_duration_sec,
        "conversation_pace_turns_per_min": metrics.conversation_pace_turns_per_min,
        "silence_ratio": metrics.silence_ratio,
        "speaker_dominance": metrics.speaker_dominance,
        "response_latency_stats": metrics.response_latency_stats,
        "topic_coherence_score": metrics.topic_coherence_score,
        "energy_flow": metrics.energy_flow,
        "interruptions_per_speaker": metrics.interruptions_per_speaker,
        "overlap_stats": overlap_stats,
    }
