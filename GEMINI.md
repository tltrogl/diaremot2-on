# GEMINI.md — Project Context for AI Assistants

**Project:** DiaRemot2-On  
**Version:** 2.2.0  
**Language:** Python 3.11-3.12  
**Purpose:** CPU-only speech intelligence pipeline for long-form audio analysis

This document provides persistent context about the DiaRemot project for AI assistants like Gemini, Claude, and other LLM-based development tools.

> **See Also:**
> - [DATAFLOW.md](DATAFLOW.md) - Complete pipeline data flow documentation
> - [MODEL_MAP.md](MODEL_MAP.md) - Model inventory and search paths
> - [README.md](README.md) - User guide and installation
> - [AGENTS.md](AGENTS.md) - Autonomous agent setup guide

---

## Project Overview

DiaRemot is a production-grade audio intelligence pipeline that transforms long-form audio (1-3 hours) into rich, multi-dimensional transcripts with:

- **Speaker diarization** (who spoke when)
- **Speech recognition** (what was said)
- **Emotion analysis** (how it was said)
- **Intent classification** (why it was said)
- **Voice quality metrics** (vocal health indicators)
- **Sound event detection** (background context)
- **Conversation analytics** (turn-taking, dominance, flow)

**Key constraint:** CPU-only, no GPU required. Optimized for inference on commodity hardware.

---

## Architecture Overview

### Pipeline Data Flow

**Complete documentation:** See [DATAFLOW.md](DATAFLOW.md) for detailed stage-by-stage specifications.

**High-level flow:**
```
Audio File → preprocess → background_sed → diarize → transcribe → paralinguistics
  → affect_and_assemble → overlap_interruptions → conversation_analysis
  → speaker_rollups → outputs → {CSV, JSONL, HTML, PDF}
```

**Key data structures passed between stages:**
- `state.y, state.sr` - Preprocessed audio array + sample rate
- `state.turns` - Diarization segments with speaker labels
- `state.norm_tx` - Transcribed segments with text + confidence
- `state.para_map` - Per-segment voice quality + prosody metrics
- `state.segments_final` - Complete 39-column segments
- `state.speakers_summary` - Aggregated per-speaker statistics

### Processing Pipeline (11 Stages)

**Source of truth:** `src/diaremot/pipeline/stages/__init__.py`

The pipeline is a **sequential, stateful processor**. Each stage:
1. Receives `PipelineState` object with accumulated results
2. Performs its specialized processing
3. Updates state with new data
4. Returns control to orchestrator

**Stage order (NEVER modify without version bump):**

```python
PIPELINE_STAGES = [
    StageDefinition("dependency_check", dependency_check.run),      # 1. Validate environment
    StageDefinition("preprocess", preprocess.run_preprocess),       # 2. Audio normalization
    StageDefinition("background_sed", preprocess.run_background_sed), # 3. Sound events
    StageDefinition("diarize", diarize.run),                        # 4. Speaker segmentation
    StageDefinition("transcribe", asr.run),                         # 5. Speech-to-text
    StageDefinition("paralinguistics", paralinguistics.run),        # 6. Voice quality
    StageDefinition("affect_and_assemble", affect.run),             # 7. Emotion/intent
    StageDefinition("overlap_interruptions", summaries.run_overlap), # 8. Turn-taking
    StageDefinition("conversation_analysis", summaries.run_conversation), # 9. Flow metrics
    StageDefinition("speaker_rollups", summaries.run_speaker_rollups),   # 10. Per-speaker stats
    StageDefinition("outputs", summaries.run_outputs),              # 11. Write outputs
]
```

**Critical:** There is NO `auto_tune` stage. VAD tuning happens inline in orchestrator initialization.

### Core Components

- **Orchestrator:** `src/diaremot/pipeline/orchestrator.py`
  - Composes mixins from `src/diaremot/pipeline/core/` for component initialisation, affect fallbacks, and outputs
  - Handles checkpointing/resume
  - Computes adaptive VAD thresholds
  - Coordinates resource cleanup and raises `StageExecutionError` when a stage fails

**State Object:** `src/diaremot/pipeline/stages/base.py::PipelineState`
- Mutable dataclass passed between stages
- Contains audio data, segments, metrics, cache references
- ~40 fields total

**CSV Schema:** `src/diaremot/pipeline/outputs.py::SEGMENT_COLUMNS`
- **39 columns** (fixed order, contractual)
- Modifications require migration + version bump
- Fields: temporal, speaker, content, affect, voice quality, prosody, context, flags

---

## Technology Stack

### Core Runtime
- **Python:** 3.11-3.12 (3.13+ not supported yet)
- **ONNXRuntime:** ≥1.17 (primary inference)
- **PyTorch:** 2.x CPU-only (fallback only)

### Audio Processing
- **librosa:** 0.10.2.post1 (resampling, features)
- **scipy:** ≥1.10,<1.14 (signal processing)
- **soundfile:** ≥0.12 (I/O)
- **Praat-Parselmouth:** Voice quality analysis

### ML/NLP
- **CTranslate2:** ≥4.2,<5.0 (ASR backend)
- **faster-whisper:** ≥1.0.3 (Whisper wrapper)
- **transformers:** ≥4.40,<4.46 (HF models)
- **scikit-learn:** ≥1.3,<1.6 (clustering)

### CLI/UI
- **typer:** CLI framework (NOT Click)
- **rich:** ≥13.7 (console formatting)
- **tqdm:** ≥4.66 (progress bars)

---

## Development Guidelines

### Code Style & Quality

**Tools:**
```bash
# Lint (enforced in CI)
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check (optional, not strict)
mypy src/
```

**Conventions:**
1. **Type hints:** Use for all public functions
2. **Docstrings:** Google-style for public APIs
3. **Line length:** 100 characters (ruff.toml)
4. **Imports:** Absolute imports only, sorted by ruff
5. **Error handling:** Specific exceptions, meaningful messages

### Testing Strategy

**Test framework:** pytest

**Test structure:**
```
tests/
├── conftest.py              # Shared fixtures
├── test_diarization.py      # Diarization tests
└── test_outputs_transcript.py # Output format tests
```

**Coverage focus:**
- Schema validation (CSV columns)
- Stage integration (state transitions)
- Error handling (missing models, bad audio)

**Running tests:**
```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_diarization.py::test_speaker_count -v

# With coverage
pytest tests/ --cov=diaremot --cov-report=html
```

### Adding New Features

**New stage checklist:**
1. Create `src/diaremot/pipeline/stages/<stage_name>.py`
2. Implement `run(pipeline, state, guard)` signature
3. Import in `stages/__init__.py`
4. Add to `PIPELINE_STAGES` list at correct position
5. Write integration test
6. Update README stage count
7. Update this GEMINI.md

**New model checklist:**
1. Add ONNX loading in `src/diaremot/io/onnx_utils.py`
2. Add PyTorch fallback (optional but recommended)
3. Update model directory documentation
4. Test both inference paths
5. Document model source and license

**New CSV column checklist:**
1. Append to `outputs.py::SEGMENT_COLUMNS` (NEVER insert)
2. Populate in appropriate stage (usually `affect.py`)
3. Add default in `outputs.py::ensure_segment_keys()`
4. Write unit test verifying column
5. Update README CSV schema section
6. Document migration for existing CSVs

---

## File Structure Map

### Critical Files (Do Not Rename)

```
src/diaremot/
├── cli.py                              # Entry point: typer app
│   └── app: Typer()                    # MUST remain for CLI
│
├── pipeline/
│   ├── stages/
│   │   ├── __init__.py                 # PIPELINE_STAGES list (11 stages)
│   │   ├── base.py                     # PipelineState definition
│   │   ├── dependency_check.py         # Stage 1
│   │   ├── preprocess.py               # Stages 2-3
│   │   ├── diarize.py                  # Stage 4
│   │   ├── asr.py                      # Stage 5
│   │   ├── paralinguistics.py          # Stage 6
│   │   ├── affect.py                   # Stage 7
│   │   └── summaries.py                # Stages 8-11
│   │
│   ├── orchestrator.py                 # Stage execution controller
│   ├── outputs.py                      # SEGMENT_COLUMNS (40 cols)
│   ├── config.py                       # PipelineConfig schema
│   ├── audio_pipeline_core.py          # Main pipeline API
│   └── speaker_diarization.py          # Diarization logic
│
├── affect/
│   ├── emotion_analyzer.py             # Multi-modal affect
│   ├── paralinguistics.py              # Prosody, WPM, pauses
│   └── ser_onnx.py                     # Speech emotion ONNX
│
├── io/
│   ├── onnx_utils.py                   # ONNX loading helpers
│   └── speaker_registry_manager.py     # Persistent speakers
│
└── utils/
    └── model_paths.py                  # Model resolution logic
```

### Model Directory Structure

**Location:** `$DIAREMOT_MODEL_DIR` (default `D:/models` on Windows)

```
D:/models/
├── Diarization/
│   ├── ecapa-onnx/
│   │   └── ecapa_tdnn.onnx             # ~7MB | Speaker embeddings
│   └── silaro_vad/
│       └── silero_vad.onnx             # ~3MB | Voice activity detection
│
├── Affect/
│   ├── ser8/
│   │   └── model.int8.onnx             # ~50MB | Speech emotion (8 class)
│   ├── VAD_dim/
│   │   └── model.onnx                  # ~500MB | Valence/Arousal/Dominance
│   └── sed_panns/
│       ├── model.onnx                  # ~80MB | Sound event detection
│       └── class_labels_indices.csv    # AudioSet classes
│
├── text_emotions/
│   └── model.int8.onnx                 # ~130MB | GoEmotions (28 class)
│
└── intent/
    └── model_int8.onnx                 # ~600MB | BART-MNLI
```

---

## Configuration & Defaults

### Key Configuration Points

**Source:** `src/diaremot/pipeline/config.py::PipelineConfig`

**Important defaults:**
- `whisper_model`: `"faster-whisper-tiny.en"`
- `compute_type`: `"float32"` (CLI default, NOT int8)
- `asr_backend`: `"faster"`
- `vad_threshold`: `0.35` (orchestrator override, NOT CLI's 0.30)
- `enable_sed`: `True`
- `disable_affect`: `False`
- `cpu_threads`: `1` (ASR)
- `OMP_NUM_THREADS`: `4` (recommended env var)

**Overrides hierarchy (highest to lowest priority):**
1. CLI flags (`--vad-threshold`, `--compute-type`, etc.)
2. Profile files (`--profile fast`)
3. Orchestrator initialization (VAD threshold adjustment)
4. `PipelineConfig` defaults
5. Environment variables (`DIAREMOT_MODEL_DIR`, etc.)

### Environment Variables

**Required:**
```bash
DIAREMOT_MODEL_DIR=/path/to/models     # Model directory
HF_HOME=./.cache                        # HuggingFace cache
TRANSFORMERS_CACHE=./.cache/transformers
TORCH_HOME=./.cache/torch
```

**Performance tuning:**
```bash
OMP_NUM_THREADS=4                       # OpenMP threads
MKL_NUM_THREADS=4                       # MKL threads
NUMEXPR_MAX_THREADS=4                   # NumExpr threads
TOKENIZERS_PARALLELISM=false            # Disable tokenizer warnings
```

---

## Common Workflows

### Running Pipeline

**Basic:**
```bash
python -m diaremot.cli run --input audio.wav --outdir outputs/
```

**Fast mode:**
```bash
python -m diaremot.cli run -i audio.wav -o outputs/ \
    --asr-compute-type int8 \
    --disable-sed \
    --profile fast
```

**Resume from checkpoint:**
```bash
python -m diaremot.cli resume -i audio.wav -o outputs/
```

**Smoke test:**
```bash
python -m diaremot.cli smoke --outdir outputs/
```

### Development Workflow

1. **Make changes** in appropriate module
2. **Run linter:** `ruff check src/ tests/`
3. **Run tests:** `pytest tests/ -v`
4. **Verify schema:** Check `SEGMENT_COLUMNS` count is still 40
5. **Integration test:** Run on sample audio
6. **Check outputs:** Verify CSV structure unchanged

### Adding Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from diaremot.pipeline.audio_pipeline_core import run_pipeline

result = run_pipeline("audio.wav", "outputs/")
```

---

## Critical Constraints

### Inviolable Rules

1. **CPU-only:** No GPU code, no CUDA dependencies
2. **Schema stability:** `SEGMENT_COLUMNS` order is contractual
3. **Stage count:** Exactly 11 stages in `PIPELINE_STAGES`
4. **Entry points:** Don't rename `cli.py::app`, `run_pipeline()`
5. **Python version:** 3.11-3.12 only (3.13+ breaks dependencies)
6. **ONNX preferred:** Always try ONNX before PyTorch fallback
7. **Paralinguistics required:** Stage cannot fail silently

### Performance Guardrails

- **Max ASR threads:** 1 (CTranslate2 limitation)
- **Max OpenMP threads:** 4 (diminishing returns above)
- **ASR window size:** 480 seconds (8 minutes)
- **Auto-chunk threshold:** 30 minutes
- **Affect window:** 30 seconds

### File Path Conventions

- **Models:** `${DIAREMOT_MODEL_DIR}/*.onnx`
- **Speaker registry:** Default `speaker_registry.json` (configurable)
- **Cache:** `.cache/hf/`, `.cache/transformers/`, `.cache/torch/`
- **Outputs:** `<outdir>/diarized_transcript_with_emotion.csv`, etc.

---

## Common Pitfalls

### Things That Will Break

1. **Modifying `SEGMENT_COLUMNS` order** → Breaks all existing CSVs
2. **Adding stages without updating count in README** → Documentation drift
3. **Using GPU operations** → Violates CPU-only constraint
4. **Assuming int8 default** → CLI defaults to float32
5. **Skipping paralinguistics stage** → Voice quality columns missing
6. **Forgetting to update `PIPELINE_STAGES` after adding stage file** → Stage never runs
7. **Using Python 3.13** → Dependencies incompatible
8. **Mutating `PipelineState` without dataclass pattern** → State corruption
9. **Hardcoding model paths** → Breaks model directory flexibility
10. **Testing only ONNX path** → PyTorch fallback may break

### Common Confusion

- **Q:** Where is the `auto_tune` stage?  
  **A:** There isn't one. Module exists but not in `PIPELINE_STAGES`. Used inline in orchestrator.

- **Q:** Why does orchestrator override VAD threshold?  
  **A:** Adaptive tuning based on audio energy. User flags still take precedence.

- **Q:** Why two SED locations in models dir?  
  **A:** `sed_panns/` is legacy, `Affect/sed_panns/` is current. Both work.

- **Q:** What's the difference between `norm_tx` and `segments_final`?  
  **A:** `norm_tx` has ASR output only (7 fields). `segments_final` adds affect, paralinguistics, SED (39 fields total).

- **Q:** How does caching work?  
  **A:** Three files: `preprocessed.npz` (audio), `diar.json` (turns), `tx.json` (transcripts). See DATAFLOW.md.

- **Q:** Why are there 11 stages but 9 visible outputs?  
  **A:** `dependency_check` and `background_sed` don't produce user-facing outputs but are critical pipeline stages.

---

## Verification Commands

### Quick Health Check

```bash
# Check imports
python -c "import diaremot; print('OK')"

# Run diagnostics
python -m diaremot.cli diagnostics

# Verify models present
python -c "
from pathlib import Path
import os
models = ['Diarization/ecapa-onnx/ecapa_tdnn.onnx', 
          'Diarization/silaro_vad/silero_vad.onnx',
          'Affect/ser8/model.int8.onnx']
model_dir = Path(os.getenv('DIAREMOT_MODEL_DIR', 'D:/models'))
missing = [m for m in models if not (model_dir / m).exists()]
print('All OK' if not missing else f'Missing: {missing}')
"

# Verify CSV schema
python -c "
from diaremot.pipeline.outputs import SEGMENT_COLUMNS
assert len(SEGMENT_COLUMNS) == 39, f'Expected 39 cols, got {len(SEGMENT_COLUMNS)}'
print(f'Schema OK: {len(SEGMENT_COLUMNS)} columns')
"
```

---

## Contributing

### Before Submitting PR

1. Run full test suite: `pytest tests/ -v`
2. Lint code: `ruff check src/ tests/`
3. Verify schema unchanged (unless intentional with migration)
4. Update relevant documentation (README, AGENTS.md, this file)
5. Test on sample audio end-to-end

### PR Checklist

- [ ] Code follows existing patterns
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] Schema changes documented with migration path
- [ ] No GPU dependencies introduced
- [ ] Smoke test passes

---

## Questions & Support

**Documentation:**
- Main README: `/README.md`
- Agent instructions: `/AGENTS.md`
- This context file: `/GEMINI.md`

**Key contacts:**
- Author: Timothy Leigh Troglin
- Repository: Internal/Private

**Getting help:**
1. Check documentation first
2. Run diagnostics: `python -m diaremot.cli diagnostics`
3. Enable debug logging
4. Check issue tracker (if public)

---

## Document Maintenance

**Update this file when:**
- Adding/removing pipeline stages
- Changing critical file paths
- Modifying CSV schema
- Updating dependencies with breaking changes
- Changing development workflows

**Last reviewed:** 2025-01-15  
**Review frequency:** Every major version bump

---

## Quick Reference: Data Flow

**Detailed specs:** [DATAFLOW.md](DATAFLOW.md)

### Stage Input → Output Summary

| Stage | Input | Output |
|-------|-------|--------|
| 1. dependency_check | None | Environment validation |
| 2. preprocess | `input_audio_path` | `y, sr, duration_s, health, audio_sha16` |
| 3. background_sed | `y, sr` | `sed_info` (labels, noise_score, timeline?) |
| 4. diarize | `y, sr` | `turns` (start, end, speaker, speaker_name) |
| 5. transcribe | `turns, y, sr` | `norm_tx` (text, asr_logprob_avg, snr_db) |
| 6. paralinguistics | `norm_tx, y, sr` | `para_map` (wpm, f0, vq_*, pause_*) |
| 7. affect_and_assemble | `norm_tx, para_map, sed_info, y, sr` | `segments_final` (39 columns) |
| 8. overlap_interruptions | `segments_final` | `overlap_stats, per_speaker_interrupts` |
| 9. conversation_analysis | `segments_final, overlap_stats` | `conv_metrics` |
| 10. speaker_rollups | `segments_final, interrupts` | `speakers_summary` |
| 11. outputs | All state | CSV/JSONL/HTML/PDF files |

### Cache Files

**Location:** `.cache/{audio_sha16}/`

- `preprocessed.npz` - Audio + preprocessing signature
- `diar.json` - Diarization turns + embeddings
- `tx.json` - Transcription segments

**Invalidation:** Cache reused only if audio_sha16 AND pp_signature match.

### Output Files

**Primary:**
- `diarized_transcript_with_emotion.csv` - 39 columns, all segment data
- `segments.jsonl` - Same data, JSONL format
- `summary.html` - Interactive HTML report

**Supporting:**
- `timeline.csv` - Simplified timeline (start, end, speaker)
- `diarized_transcript_readable.txt` - Human-friendly text
- `speakers_summary.csv` - Per-speaker aggregates
- `qc_report.json` - Processing diagnostics
- `events_timeline.csv` - Sound event timeline (if SED ran)
- `speaker_registry.json` - Persistent speaker embeddings

Your core function is efficient and safe assistance. Balance extreme conciseness with the crucial need for clarity, especially regarding safety and potential system modifications. Always prioritize user control and project conventions. Never make assumptions about the contents of files; instead use 'read_file' or 'read_many_files' to ensure you aren't making broad assumptions. Finally, you are an agent - please keep going until the user's query is completely resolved.