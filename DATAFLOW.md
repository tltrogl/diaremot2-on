# DiaRemot2 Data Flow Map

**Version:** 2.2.0  
**Updated:** 2025-01-15

## Pipeline Overview

```
Input Audio File
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: dependency_check                                    │
│ Validates runtime dependencies and model availability        │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: preprocess                                          │
│ • Load audio file                                            │
│ • Resample to 16kHz mono                                     │
│ • Normalize to -20 LUFS                                      │
│ • Apply spectral subtraction denoising                       │
│ • Auto-chunk if >60 min                                      │
│ • Compute audio health metrics                               │
│ Output: y (audio array), sr, duration_s, health, audio_sha16 │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: background_sed                                      │
│ • Run PANNs CNN14 on full audio                              │
│ • Generate global background classification                  │
│ • Optionally create detailed timeline (if noise ≥0.30)      │
│ Output: sed_info {top, dominant_label, noise_score,         │
│         timeline_csv?, timeline_jsonl?}                      │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: diarize                                             │
│ • Run Silero VAD to detect speech regions                    │
│ • Extract ECAPA-TDNN embeddings (192-dim)                    │
│ • Perform Agglomerative Hierarchical Clustering (AHC)        │
│ • Apply merge rules (collar=0.25s, min_turn=1.5s)           │
│ • Compute VAD stability metrics                              │
│ Output: turns [{start, end, speaker, speaker_name}]         │
│         vad_unstable (bool)                                  │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: transcribe                                          │
│ • Batch turns into optimal ASR windows                       │
│ • Run Faster-Whisper on each batch                           │
│ • Extract text, timestamps, confidence scores                │
│ Output: norm_tx [{start, end, speaker_id, speaker_name,     │
│         text, asr_logprob_avg, snr_db}]                      │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: paralinguistics                                     │
│ • Extract Praat voice quality metrics per segment            │
│   - Jitter, Shimmer, HNR, CPPS                               │
│ • Compute prosody features                                   │
│   - WPM, pause metrics, F0, RMS loudness                     │
│ • Detect disfluencies                                        │
│ Output: para_map {seg_idx: {wpm, f0_mean_hz, vq_jitter_pct, │
│         pause_count, ...}}                                   │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: affect_and_assemble                                 │
│ • For each segment:                                          │
│   - Audio affect: VAD model → valence/arousal/dominance      │
│   - Audio emotion: SER8 → 8-class emotion                    │
│   - Text emotion: GoEmotions → 28-class distribution         │
│   - Intent: BART-MNLI → zero-shot classification             │
│   - Attach top-3 background sounds from SED                  │
│ • Merge all features into segments_final                     │
│ Output: segments_final (39 columns per SEGMENT_COLUMNS)     │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 8: overlap_interruptions                               │
│ • Detect overlapping speech between speakers                 │
│ • Classify interruptions (successful vs unsuccessful)        │
│ • Compute per-speaker interrupt statistics                   │
│ Output: overlap_stats, per_speaker_interrupts                │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 9: conversation_analysis                               │
│ • Compute turn-taking patterns                               │
│ • Calculate speaker dominance scores                         │
│ • Analyze conversation flow                                  │
│ Output: conv_metrics (ConversationMetrics object)            │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 10: speaker_rollups                                    │
│ • Aggregate per-speaker statistics:                          │
│   - Total duration, word count, avg WPM                      │
│   - Avg affect (valence, arousal, dominance)                 │
│   - Emotion distributions                                    │
│   - Voice quality averages                                   │
│   - Interruption counts                                      │
│ Output: speakers_summary                                     │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 11: outputs                                            │
│ • Write CSV: diarized_transcript_with_emotion.csv            │
│ • Write JSONL: segments.jsonl                                │
│ • Write timeline: timeline.csv                               │
│ • Write readable: diarized_transcript_readable.txt           │
│ • Generate HTML: summary.html                                │
│ • Generate PDF: summary.pdf (if wkhtmltopdf available)       │
│ • Write QC: qc_report.json                                   │
│ • Write speakers: speakers_summary.csv                       │
│ Output: manifest with all output file paths                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Stage Data Flow

### Stage 1: dependency_check
**Input:** None  
**Process:**
- Check FFmpeg availability
- Verify Python dependencies (librosa, torch, etc.)
- Test model file accessibility
- Report dependency health

**Output:**
```python
PipelineState:
  # No state changes, only validates environment
```

---

### Stage 2: preprocess
**Input:** `input_audio_path` (string)  
**Process:**
1. Load audio file using librosa
2. Resample to 16kHz mono
3. Apply loudness normalization to -20 LUFS
4. Run spectral subtraction denoising
5. Auto-chunk if duration > 60 minutes
6. Compute audio health metrics (SNR, clipping, silence ratio)
7. Save to cache (preprocessed.npz)

**Output:**
```python
PipelineState:
  y: np.ndarray            # Audio samples (float32)
  sr: int                  # Sample rate (16000)
  duration_s: float        # Duration in seconds
  health: AudioHealth      # Metrics (snr_db, clipping_detected, etc.)
  audio_sha16: str         # SHA-1 hash of audio (first 16 chars)
  pp_sig: str              # Preprocessing config signature
  cache_dir: Path          # Cache directory for this audio
```

**Cache Format:**
```python
# preprocessed.npz
{
  "audio": np.ndarray,       # Preprocessed audio
  "sample_rate": int,        # 16000
  "duration_s": float,       # Duration
  "pp_signature": str,       # Config hash for cache validation
  "health": dict             # Audio health metrics
}
```

---

### Stage 3: background_sed
**Input:** `state.y`, `state.sr`  
**Process:**
1. Run PANNs CNN14 on entire audio
2. Generate global background classification
3. If noise_score ≥ 0.30 (or mode="timeline"):
   - Create detailed event timeline
   - Save to events_timeline.csv, events.jsonl

**Output:**
```python
PipelineState:
  sed_info: dict
    {
      "top": [{"label": str, "score": float}, ...],  # Top 10 labels
      "dominant_label": str,                         # Most common class
      "noise_score": float,                          # 0.0-1.0
      "enabled": bool,                               # True if ran
      "timeline_csv": str?,                          # Path if generated
      "timeline_jsonl": str?,                        # Path if generated
      "timeline_events": list?,                      # Event list
    }
```

---

### Stage 4: diarize
**Input:** `state.y`, `state.sr`, `state.duration_s`  
**Process:**
1. Run Silero VAD to detect speech segments
2. For each speech segment:
   - Extract audio chunk
   - Compute log-mel spectrogram
   - Extract ECAPA-TDNN embedding (192-dim)
3. Perform Agglomerative Hierarchical Clustering on embeddings
4. Apply post-processing:
   - Merge turns with gap ≤ 1.0s
   - Apply collar (0.25s) to prevent overlap
   - Enforce minimum turn duration (1.5s)
5. Compute VAD stability (detect excessive toggle rate)
6. Save to cache (diar.json)

**Output:**
```python
PipelineState:
  turns: list[dict]
    [
      {
        "start": float,           # Seconds
        "end": float,             # Seconds
        "speaker": str,           # Speaker ID (e.g., "Speaker_1")
        "speaker_name": str,      # Human-readable name
        "embedding": list[float]? # 192-dim ECAPA embedding (optional)
      },
      ...
    ]
  vad_unstable: bool              # True if VAD toggled >60 times/minute
```

**Cache Format:**
```python
# diar.json
{
  "version": str,          # Cache version (e.g., "v3")
  "audio_sha16": str,      # Audio hash
  "pp_signature": str,     # Preprocessing signature
  "turns": list[dict],     # Diarization turns
  "saved_at": float        # Unix timestamp
}
```

---

### Stage 5: transcribe
**Input:** `state.turns`, `state.y`, `state.sr`  
**Process:**
1. Convert turns to ASR segments with speaker info
2. Batch short segments (<8s) into groups (target: 60s, max: 300s)
3. For each batch/segment:
   - Extract audio chunk
   - Run Faster-Whisper ASR
   - Extract text, word timestamps, confidence (log prob)
4. Normalize output format
5. Save to cache (tx.json)

**Output:**
```python
PipelineState:
  norm_tx: list[dict]
    [
      {
        "start": float,          # Seconds
        "end": float,            # Seconds
        "speaker_id": str,       # From diarization
        "speaker_name": str,     # From diarization
        "text": str,             # Transcribed text
        "asr_logprob_avg": float # ASR confidence (avg log prob)
        "snr_db": float?,        # SNR estimate
        "error_flags": str       # Error markers (empty if OK)
      },
      ...
    ]
```

**Cache Format:**
```python
# tx.json
{
  "version": str,          # Cache version
  "audio_sha16": str,      # Audio hash
  "pp_signature": str,     # Preprocessing signature
  "segments": list[dict],  # Transcribed segments (norm_tx format)
  "saved_at": float        # Unix timestamp
}
```

---

### Stage 6: paralinguistics
**Input:** `state.norm_tx`, `state.y`, `state.sr`  
**Process:**
1. For each transcribed segment:
   - Extract audio chunk
   - Run Praat-Parselmouth analysis:
     * Voice quality: jitter, shimmer, HNR, CPPS
     * Prosody: F0 (pitch), RMS loudness
     * Speech rate: WPM, pause detection
     * Disfluencies: filler word count
2. Handle failures gracefully (fallback to zeros)

**Output:**
```python
PipelineState:
  para_map: dict[int, dict]
    {
      segment_index: {
        "wpm": float,                # Words per minute
        "duration_s": float,         # Segment duration
        "words": int,                # Word count
        "pause_count": int,          # Number of pauses
        "pause_time_s": float,       # Total pause duration
        "pause_ratio": float,        # pause_time / duration
        "f0_mean_hz": float,         # Mean F0 (pitch)
        "f0_std_hz": float,          # F0 standard deviation
        "loudness_rms": float,       # RMS loudness
        "disfluency_count": int,     # Filler words
        "vq_jitter_pct": float,      # Jitter %
        "vq_shimmer_db": float,      # Shimmer dB
        "vq_hnr_db": float,          # Harmonics-to-Noise Ratio
        "vq_cpps_db": float,         # Cepstral Peak Prominence
      },
      ...
    }
```

---

### Stage 7: affect_and_assemble
**Input:** `state.norm_tx`, `state.para_map`, `state.sed_info`, `state.y`, `state.sr`  
**Process:**
1. For each segment:
   - Extract audio chunk
   - **Audio Affect:**
     * VAD model → valence, arousal, dominance (-1 to +1)
     * SER8 model → 8-class emotion + scores
   - **Text Affect:**
     * GoEmotions model → 28-class emotion distribution
     * BART-MNLI → intent classification (zero-shot)
   - **Background Context:**
     * Attach top-3 background sounds from SED
     * Compute noise tag
2. Assemble final segment structure with all 39 columns
3. Compute derived fields:
   - affect_hint (e.g., "calm-positive", "agitated-negative")
   - voice_quality_hint (interpretation of Praat metrics)
   - low_confidence_ser (flag if SER score < 0.5)

**Output:**
```python
PipelineState:
  segments_final: list[dict]
    [
      {
        # Temporal
        "file_id": str,
        "start": float,
        "end": float,
        "duration_s": float,
        
        # Speaker
        "speaker_id": str,
        "speaker_name": str,
        
        # Content
        "text": str,
        "words": int,
        "language": str?,
        
        # ASR
        "asr_logprob_avg": float,
        "low_confidence_ser": bool,
        
        # Audio Emotion
        "valence": float,           # -1 to +1
        "arousal": float,           # -1 to +1
        "dominance": float,         # -1 to +1
        "emotion_top": str,         # Top audio emotion
        "emotion_scores_json": str, # JSON: 8-class scores
        
        # Text Emotions
        "text_emotions_top5_json": str,  # JSON: Top 5
        "text_emotions_full_json": str,  # JSON: All 28
        
        # Intent
        "intent_top": str,          # Top intent label
        "intent_top3_json": str,    # JSON: Top 3 + scores
        
        # Background
        "events_top3_json": str,    # JSON: Top 3 sounds
        "noise_tag": str,           # Dominant background class
        
        # Signal Quality
        "snr_db": float,
        "snr_db_sed": float,
        "vad_unstable": bool,
        
        # Prosody
        "wpm": float,
        "pause_count": int,
        "pause_time_s": float,
        "pause_ratio": float,
        "f0_mean_hz": float,
        "f0_std_hz": float,
        "loudness_rms": float,
        "disfluency_count": int,
        
        # Voice Quality
        "vq_jitter_pct": float,
        "vq_shimmer_db": float,
        "vq_hnr_db": float,
        "vq_cpps_db": float,
        "voice_quality_hint": str,
        
        # Hints/Flags
        "affect_hint": str,
        "error_flags": str,
      },
      ...
    ]
```

---

### Stage 8: overlap_interruptions
**Input:** `state.segments_final`  
**Process:**
1. Detect overlapping speech regions (start_A < start_B < end_A)
2. Classify interruptions:
   - Successful: interrupter continues speaking
   - Unsuccessful: interrupted speaker resumes
3. Aggregate per-speaker:
   - Interruptions made
   - Interruptions received

**Output:**
```python
PipelineState:
  overlap_stats: dict
    {
      "total_overlaps": int,
      "overlap_duration_s": float,
      "overlap_ratio": float,      # overlap_duration / total_duration
      "interruptions": int,
    }
  
  per_speaker_interrupts: dict[str, dict]
    {
      speaker_id: {
        "made": int,               # Interruptions initiated
        "received": int,           # Times interrupted
      },
      ...
    }
```

---

### Stage 9: conversation_analysis
**Input:** `state.segments_final`, `state.overlap_stats`  
**Process:**
1. Compute turn-taking metrics:
   - Average turn duration
   - Turn distribution across speakers
2. Calculate speaker dominance scores
3. Analyze conversation flow patterns

**Output:**
```python
PipelineState:
  conv_metrics: ConversationMetrics
    {
      "total_turns": int,
      "avg_turn_duration_s": float,
      "turn_distribution": dict[str, float],  # speaker_id → % of turns
      "dominance_scores": dict[str, float],   # speaker_id → score
      "flow_quality": float,                  # 0-1 metric
    }
```

---

### Stage 10: speaker_rollups
**Input:** `state.segments_final`, `state.per_speaker_interrupts`, `state.overlap_stats`  
**Process:**
1. For each speaker:
   - Sum total speaking time
   - Count words
   - Compute average WPM, affect (V/A/D)
   - Aggregate emotion distributions
   - Average voice quality metrics
   - Include interruption stats

**Output:**
```python
PipelineState:
  speakers_summary: list[dict]
    [
      {
        "speaker_id": str,
        "speaker_name": str,
        "total_duration": float,    # Total speaking time (s)
        "word_count": int,
        "avg_wpm": float,
        "avg_valence": float,
        "avg_arousal": float,
        "avg_dominance": float,
        "top_emotion": str,
        "emotion_distribution": dict[str, float],
        "interruptions_made": int,
        "interruptions_received": int,
        "overlap_ratio": float,
        "avg_vq_jitter_pct": float,
        "avg_vq_shimmer_db": float,
        "avg_vq_hnr_db": float,
        "avg_vq_cpps_db": float,
      },
      ...
    ]
```

---

### Stage 11: outputs
**Input:** All state data  
**Process:**
1. Write primary CSV (39 columns, fixed order)
2. Write JSONL segments
3. Write timeline CSV
4. Write human-readable transcript
5. Generate HTML summary with interactivity
6. Generate PDF summary (if wkhtmltopdf available)
7. Write QC report with processing stats
8. Write speaker summary CSV

**Output:**
```python
manifest: dict
  {
    "run_id": str,
    "file_id": str,
    "out_dir": str,
    "outputs": {
      "csv": str,                  # diarized_transcript_with_emotion.csv
      "jsonl": str,                # segments.jsonl
      "timeline": str,             # timeline.csv
      "summary_html": str,         # summary.html
      "summary_pdf": str,          # summary.pdf
      "qc_report": str,            # qc_report.json
      "speakers_summary": str,     # speakers_summary.csv
      "events_timeline": str?,     # events_timeline.csv (if SED ran)
      "events_jsonl": str?,        # events.jsonl (if SED ran)
      "speaker_registry": str,     # speaker_registry.json
    },
    "issues": list[str]?,          # Any warnings/issues
    "dependency_ok": bool,
    "dependency_unhealthy": list[str],
  }
```

---

## Key Data Structures

### SEGMENT_COLUMNS (39 columns, fixed order)
```python
SEGMENT_COLUMNS = [
    "file_id",              # 1
    "start",                # 2
    "end",                  # 3
    "speaker_id",           # 4
    "speaker_name",         # 5
    "text",                 # 6
    "valence",              # 7
    "arousal",              # 8
    "dominance",            # 9
    "emotion_top",          # 10
    "emotion_scores_json",  # 11
    "text_emotions_top5_json",   # 12
    "text_emotions_full_json",   # 13
    "intent_top",           # 14
    "intent_top3_json",     # 15
    "events_top3_json",     # 16
    "noise_tag",            # 17
    "asr_logprob_avg",      # 18
    "snr_db",               # 19
    "snr_db_sed",           # 20
    "wpm",                  # 21
    "duration_s",           # 22
    "words",                # 23
    "pause_ratio",          # 24
    "low_confidence_ser",   # 25
    "vad_unstable",         # 26
    "affect_hint",          # 27
    "pause_count",          # 28
    "pause_time_s",         # 29
    "f0_mean_hz",           # 30
    "f0_std_hz",            # 31
    "loudness_rms",         # 32
    "disfluency_count",     # 33
    "error_flags",          # 34
    "vq_jitter_pct",        # 35
    "vq_shimmer_db",        # 36
    "vq_hnr_db",            # 37
    "vq_cpps_db",           # 38
    "voice_quality_hint",   # 39
]
```

---

## Cache Strategy

### Cache Invalidation Rules
Cache is invalidated when:
1. Audio file changes (audio_sha16 mismatch)
2. Preprocessing config changes (pp_signature mismatch)
3. Cache version changes (version mismatch)

### Cache Files
```
.cache/
  {audio_sha16}/
    preprocessed.npz  # Preprocessed audio + metadata
    diar.json         # Diarization results
    tx.json           # Transcription results
```

### Resume Behavior
- **preprocessed.npz hit:** Skip preprocessing, load audio
- **diar.json + tx.json hit:** Skip diarization + ASR, load segments
- **tx.json hit only:** Skip ASR, reconstruct turns from segments
- **diar.json hit only:** Skip diarization, run ASR
- **No cache:** Run full pipeline

---

## Model Integration Points

### Preprocessing (Stage 2)
- **librosa:** Audio loading, resampling
- **scipy:** Signal processing, denoising
- **pyloudnorm:** Loudness normalization

### Diarization (Stage 4)
- **Silero VAD (ONNX):** Voice activity detection
- **ECAPA-TDNN (ONNX):** Speaker embeddings
- **scikit-learn:** Agglomerative clustering

### Transcription (Stage 5)
- **CTranslate2 + Faster-Whisper:** ASR inference

### Paralinguistics (Stage 6)
- **Praat-Parselmouth:** Voice quality metrics

### Affect (Stage 7)
- **SER8 (ONNX):** 8-class audio emotion
- **VAD (ONNX):** Valence/Arousal/Dominance
- **GoEmotions (ONNX):** 28-class text emotion
- **BART-MNLI (ONNX):** Intent classification

### Background SED (Stage 3)
- **PANNs CNN14 (ONNX):** 527-class sound events

---

## Error Handling

### Graceful Degradation
- **Missing models:** Fall back to PyTorch (slower) or skip stage
- **Stage failure:** Continue pipeline, mark error_flags
- **Paralinguistics failure:** Use lightweight fallback (zeros)
- **ASR timeout:** Generate placeholder segments
- **SED unavailable:** Emit empty background tags

### Error Propagation
Errors are logged but don't halt the pipeline. Each stage:
1. Attempts processing
2. On failure, uses fallback or placeholder
3. Logs warning + suggestion
4. Sets error_flags in segment data
5. Continues to next stage

---

## Performance Optimizations

### Intelligent Batching (ASR)
- Groups short segments (<8s) into batches
- Target batch size: 60 seconds
- Max batch duration: 300 seconds
- Reduces overhead by 2-5x

### Auto-Chunking (Long Audio)
- Activates for files >60 minutes
- Splits into 20-minute chunks with 30s overlap
- Processes chunks independently
- Merges results with overlap handling

### Caching
- Avoids redundant preprocessing
- Enables resume from any stage
- Multi-root cache search (primary + fallbacks)

### INT8 Quantization
- 30-40% faster than float32
- Available for: ASR, SER8, GoEmotions, BART-MNLI
- Minimal accuracy loss (<2%)

---

**End of Data Flow Map**
