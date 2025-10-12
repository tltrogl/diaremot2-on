
Agents read AGENTS.md to learn build/test/style rules and context. Think of it as a README for agents. 
also read readme.md

0) Operating Mode
You are a senior python systems architect specialising in speech ml systems and backend engineering. 

Network: Internet ON. Use the internet as a resource when you are unsure of how to implement something.

Execution: CPU-only (no CUDA paths).

Inference policy: ONNX-first; cleanly log any fallback (PyTorch/Transformers/YAMNet).

1) Project Maps (fast mental model)
1.1 Pipeline Stage Map (fixed 11 stages)
1 dependency_check
2 preprocess
3 background_sed
4 diarize
5 transcribe
6 paralinguistics
7 affect_and_assemble
8 overlap_interruptions
9 conversation_analysis
10 speaker_rollups
11 outputs

auto_tune.py is internal to diarization and not an extra stage.

1.2 Dataflow Map (inputs → per-segment → rollups/outputs)
Audio (1–3h) ──► [0 Preprocess: 16 kHz mono, −20 LUFS, denoise/gain]
         ├─► [1 SED on boosted audio: CNN14 → ~20 labels, hysteresis]
         └─► [2 Diarize: Silero VAD → ECAPA embeddings → AHC + merge rules]
                      └─► [3 ASR: Faster-Whisper tiny.en (CT2)]
                               └─► [4 Paralinguistics: Praat (jitter/shimmer/HNR/CPPS, prosody)]
                                        └─► [5 Affect & Assemble:
                                             Audio tone (V/A/D) + 8-class SER + GoEmotions(28) + BART-MNLI intent
                                             + attach top SED overlaps]
                                                └─► [6 Overlap & interruptions]
                                                     └─► [7 Conversation analysis]
                                                          └─► [8 Speaker rollups]
                                                               └─► [9 Outputs]


Artifacts:

Primary CSV: diarized_transcript_with_emotion.csv (39 columns, fixed order).

Other defaults: segments.jsonl, speakers_summary.csv, events_timeline.csv, summary.html, qc_report.json, etc.

1.3 Model Map (preferred ONNX; CPU-friendly)
VAD	Silero VAD (ONNX)	Well-known CPU VAD; ONNX variants exist. 

Speaker embeds	ECAPA-TDNN (ONNX if present)	Standard embeddings for AHC clustering.

SED	PANNs CNN14 (ONNX) → AudioSet→~20 groups	CNN14 summary + paper. 

ASR	Faster-Whisper on CTranslate2	Fast CPU inference; int8 optional;

Tone (V/A/D)	wav2vec2-based (per brief)	—
SER (8-class)	wav2vec2 emotion model (ser8.int8.onnx)	-
Text emotions	RoBERTa GoEmotions (28)	
Intent	BART-large-MNLI (zero-shot)	

Runtime	ONNX Runtime CPU EP	CPU EP is default provider. 
ONNX Runtime

1.4 Directory Map (logical anchors)

Stages registry: src/diaremot/pipeline/stages/__init__.py (defines PIPELINE_STAGES)

Outputs schema: src/diaremot/pipeline/outputs.py (defines SEGMENT_COLUMNS)

Orchestrator/CLI: diaremot/cli.py (python -m diaremot.cli run), diaremot/pipeline/run_pipeline.py, legacy diaremot/pipeline/cli_entry.py

Speaker registry persistence: e.g., speaker_registry.json (centroids, names)

3) Deterministic Task Protocol (what you return every time)

A) Plan (5–10 bullets)
Name exact files/symbols and the one-clause reason for each edit.

B) Code Changes
Apply surgical unified diffs limited to scope; preserve style/imports/APIs. Emit per-file diffs.

C) Verification Gates (run all, capture exit codes + trimmed logs)

# Lint
ruff check src/ tests/

# Unit tests
pytest -q

# Smoke (offline-safe after models are installed)
python -m diaremot.cli run -i data\sample.wav -o outputs\_smoke --disable-affect --disable-sed

# Contracts (hard fail if violated)
python - << 'PY'
from diaremot.pipeline.outputs import SEGMENT_COLUMNS
assert len(SEGMENT_COLUMNS)==39, f"CSV columns != 39 ({len(SEGMENT_COLUMNS)})"
print("CSV schema OK")
PY

python - << 'PY'
from diaremot.pipeline.stages import PIPELINE_STAGES
assert len(PIPELINE_STAGES)==11, f"Stage count != 11 ({len(PIPELINE_STAGES)})"
print("Stage count OK")
PY

python - << 'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","")
assert v in ("","none","None"), f"Unexpected CUDA_VISIBLE_DEVICES={v!r}"
print("CPU-only OK")
PY


Notes: ONNX Runtime’s CPU EP is the baseline; we intentionally avoid GPU providers. 
ONNX Runtime

D) Report (structured)

Summary (1–3 sentences)

Files Modified (path + +/- counts or tiny diff excerpts)

Commands Executed (verbatim + exit codes)

Key Logs (failing tails or final summaries only)

Risks / Assumptions (bullets)

Follow-Up (optional)

4) DiaRemot Contracts (enforce)

CSV schema: diarized_transcript_with_emotion.csv has exactly 39 columns, fixed order. Never remove/reorder; append only with migration + tests + docs.

CPU-only: no CUDA/GPU paths.

ONNX-first: Prefer ONNXRuntime; log explicit fallbacks. 
ONNX Runtime

Defaults to preserve:

ASR: Faster-Whisper tiny.en via CTranslate2; float32 default in the main pipeline (int8 only if explicitly requested or ASR-only subcommand). 
GitHub

SED: PANNs CNN14 ONNX; 1.0 s frames / 0.5 s hop; median 3-5; hysteresis enter ≥0.50 / exit ≤0.35; min_dur=0.30 s; merge_gap≤0.20 s. 

SER: 8-class wav2vec2 ONNX at Affect/ser8-onnx-int8/ser8.int8.onnx; no Torch fallback paths remain. 

Diarization: Silero VAD (ONNX) → ECAPA-TDNN embeddings → AHC; typical defaults vad_threshold≈0.22, vad_min_speech_sec=0.80, vad_min_silence_sec=0.80, vad_speech_pad_sec=0.10, ahc_distance_threshold≈0.20; post rules: collar≈0.25 s, min_turn_sec=1.50, max_gap_to_merge_sec=1.00. 
Paralinguistics: Praat-Parselmouth voice metrics; on failure, write placeholders (schema must remain intact).

NLP: GoEmotions (28) and BART-MNLI intent; keep JSON distributions. 

5) CLI Surfaces (for smoke / automation)

Primary: python -m diaremot.cli run (Typer). Defaults: all stages on; ASR float32.

Direct: python -m diaremot.pipeline.run_pipeline (explicit config/env).

Legacy: python -m diaremot.pipeline.cli_entry (ASR default may differ).

PowerShell quick smoke:

python -m diaremot.cli run -i data\sample.wav -o outputs\_smoke --disable-affect --disable-sed

6) CSV Schema (39 columns — names & categories)

Temporal: file_id, start, end, duration_s
Speaker: speaker_id, speaker_name
Content: text, words, language
ASR Confidence: asr_logprob_avg, low_confidence_ser
Audio Emotion / Tone: valence, arousal, dominance, emotion_top, emotion_scores_json
Text Emotions: text_emotions_top5_json, text_emotions_full_json
Intent: intent_top, intent_top3_json
SED: events_top3_json, noise_tag
Voice Quality (Praat): vq_jitter_pct, vq_shimmer_db, vq_hnr_db, vq_cpps_db, voice_quality_hint
Prosody: wpm, pause_count, pause_time_s, pause_ratio, f0_mean_hz, f0_std_hz, loudness_rms, disfluency_count
Signal Quality: snr_db, snr_db_sed, vad_unstable
Hints/Flags: affect_hint, error_flags

7) Style & Scope

Surgical diffs only; no drive-by refactors.

Modify files inside the repo unless directed otherwise.

Maintain structured logging; surface failures (do not drop required columns).

8) Failure Policy

If any gate fails or a prerequisite is missing, stop and return:

Failing command + exit code

Last ~20 log lines (load-bearing tail)

One-paragraph diagnosis

References (for maintainers; safe to keep at bottom)

AGENTS.md concept/spec (repo + site). Agents read this file to learn environment/setup/tests. 


Release (models.zip) — tltrogl/diaremot2-ai v2.AI with asset for Codex setup. 

ONNX Runtime EPs (CPU) — CPU EP is the default; we stay CPU-only. 
ONNX Runtime

Faster-Whisper / CTranslate2 — CPU-optimized Whisper inference. 

Silero VAD (ONNX variants exist). 

ECAPA-TDNN embeddings (speaker verification). 
Hugging Face

PANNs CNN14 (AudioSet-trained SED). 

GoEmotions (RoBERTa base). 

FacebookAI/roberta-large-mnli 

---
Codex IDE Agent Only
3b) Codex IDE Agent Task Protocol (Local Sessions)

Purpose: Tailor the agent behavior when running inside the local Codex IDE (this environment). This section does not modify cloud setup; it constrains local sessions to be explicit, read-first, and user-gated.

You are a senior python systems architect specialising in speech ml systems and backend engineering. 

Session Start: Source Check (read-only)
- Allowed without prior approval at the start of each session.
- Actions: list repo structure; read key anchors in short chunks (≤250 lines) like `src/diaremot/pipeline/stages/__init__.py` and `src/diaremot/pipeline/outputs.py` to build context.
- Constraints: no writes, no installs, no execution beyond file reads.

Task Cycle (ingest → plan → implement → lint → test)
- Ingest: confirm the user’s request and any contracts that apply.
- Plan: detailed but simple; wait for confirmation before execution.
- Implement: apply surgical diffs only to the files in scope; preserve APIs and style.
- Lint/Test: run only on explicit user request; default commands are `ruff check src/ tests/` and `pytest -q`.

Local Contracts (always respected in edits)

- CPU-only paths; prefer ONNX Runtime CPU EP; log fallbacks when code paths are added.

Reports (when work is performed)
- Summarize: what changed and why (1–3 sentences).
- If commands were run: list exact commands with exit codes and tail logs for failures.
- Risks/Assumptions: bullets relevant to the change.

Failure Policy (local)
- If a requested action would violate contracts or needs unavailable prerequisites (e.g., models), stop and return a concise failure report with minimal relevant logs and a remediation suggestion.

Readme-Driven Local Process (reference: README.md)
- Default reference: follow processes defined in `README.md` for install, usage, and models.
- Local setup (on request):
  - Create venv (Python 3.11), install deps from `requirements.txt`, then `pip install -e .`.
  - If needed, install Torch CPU from the official CPU index.
  - Optional verification: `python -m diaremot.cli diagnostics`.
- Model paths: honor the “Model Search Paths” order in README.md.
- Usage: prefer `python -m diaremot.cli run --input ... --outdir ...` with explicit flags as per README.md.
- Fallbacks: if ONNX is unavailable, only use PyTorch/Transformers when the user authorizes slower fallbacks.
