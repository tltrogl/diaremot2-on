# DiaRemot - CPU-Only Speech Intelligence Pipeline

**Version 2.2.0**

DiaRemot is a production-ready, CPU-only speech intelligence system that processes long-form audio (1-3 hours) into comprehensive diarized transcripts with deep affect, paralinguistic, and acoustic analysis. Built for research and production environments requiring detailed speaker analytics without GPU dependencies.

## Core Capabilities

- **Speaker Diarization** – Silero VAD + ECAPA-TDNN embeddings with Agglomerative Hierarchical Clustering
- **Automatic Speech Recognition** – Faster-Whisper (CTranslate2) with intelligent batching
- **Emotion Analysis** – Multi-modal (audio + text) with 8 speech emotions + 28 text emotions (GoEmotions)
- **Intent Classification** – Zero-shot intent detection via BART-MNLI
- **Sound Event Detection** – PANNs CNN14 for ambient sound classification (527 AudioSet classes)
- **Voice Quality Analysis** – Praat-Parselmouth metrics (jitter, shimmer, HNR, CPPS)
- **Paralinguistics** – Prosody, speech rate (WPM), pause patterns, disfluency detection
- **Persistent Speaker Registry** – Cross-file speaker tracking via embedding centroids

---

## Documentation

- **README.md** (this file) - User guide and reference
- **DATAFLOW.md** - Detailed pipeline data flow documentation
- **MODEL_MAP.md** - Complete model inventory and search paths
- **GEMINI.md** - Project context for AI assistants (Gemini, Claude, etc.)
- **AGENTS.md** - Setup guide for autonomous agents
- **CLOUD_BUILD_GUIDE.md** - Cloud deployment instructions

---

## 11-Stage Processing Pipeline

1. **dependency_check** – Validate runtime dependencies and model availability
2. **preprocess** – Audio normalization, denoising, auto-chunking for long files
3. **background_sed** – Sound event detection (music, keyboard, ambient noise)
4. **diarize** – Speaker segmentation with adaptive VAD tuning
5. **transcribe** – Speech-to-text with intelligent batching
6. **paralinguistics** – Voice quality and prosody extraction
7. **affect_and_assemble** – Emotion/intent analysis and segment assembly
8. **overlap_interruptions** – Turn-taking and interruption pattern analysis
9. **conversation_analysis** – Flow metrics and speaker dominance
10. **speaker_rollups** – Per-speaker statistical summaries
11. **outputs** – Generate CSV, JSON, HTML, PDF reports

### Modular orchestration

- Core mixins live in `src/diaremot/pipeline/core/` and provide targeted responsibilities for
  component initialisation, affect handling, paralinguistics fallbacks, and output generation.
- The public orchestrator (`src/diaremot/pipeline/orchestrator.py`) now composes these mixins,
  surfaces structured `StageExecutionError` instances from `src/diaremot/pipeline/errors.py`, and
  improves failure diagnostics without altering the 11-stage contract.
- Component bootstrap logic in `ComponentFactoryMixin` raises these structured errors as soon as a
  dependency is missing, keeping cache and checkpoint handling consistent with pre-refactor runs.
- The paralinguistics stack is now a proper package under `src/diaremot/affect/paralinguistics/`,
  splitting configuration, audio/voice quality primitives, aggregate analytics, benchmarking, and
  the CLI into focused modules while preserving the legacy `extract` API for pipeline callers.

### Data Flow Diagram

```
Audio File (WAV/MP3/M4A)
    ↓
[1] dependency_check → Validate environment
    ↓
[2] preprocess → 16kHz mono, -20 LUFS, denoise, auto-chunk
    ↓ {y, sr, duration_s, health, audio_sha16}
    ↓
[3] background_sed → PANNs CNN14 (global + timeline if noisy)
    ↓ {sed_info: top labels, dominant_label, noise_score, timeline?}
    ↓
[4] diarize → Silero VAD + ECAPA embeddings + AHC clustering
    ↓ {turns: [{start, end, speaker, speaker_name}], vad_unstable}
    ↓
[5] transcribe → Faster-Whisper with intelligent batching
    ↓ {norm_tx: [{start, end, speaker, text, asr_logprob_avg}]}
    ↓
[6] paralinguistics → Praat (jitter/shimmer/HNR/CPPS) + prosody (WPM/F0/pauses)
    ↓ {para_map: {seg_idx: {wpm, f0_mean_hz, vq_jitter_pct, ...}}}
    ↓
[7] affect_and_assemble → Audio (VAD+SER8) + Text (GoEmotions+BART-MNLI) + SED context
    ↓ {segments_final: 39 columns per segment}
    ↓
[8] overlap_interruptions → Detect overlaps + classify interruptions
    ↓ {overlap_stats, per_speaker_interrupts}
    ↓
[9] conversation_analysis → Turn-taking + dominance + flow metrics
    ↓ {conv_metrics: ConversationMetrics}
    ↓
[10] speaker_rollups → Aggregate per-speaker stats
    ↓ {speakers_summary: [{speaker, duration, affect, voice_quality, ...}]}
    ↓
[11] outputs → Write CSV/JSONL/HTML/PDF/QC reports
    ↓
Output Files:
  • diarized_transcript_with_emotion.csv (39 columns)
  • segments.jsonl
  • timeline.csv
  • diarized_transcript_readable.txt
  • summary.html
  • summary.pdf
  • qc_report.json
  • speakers_summary.csv
  • events_timeline.csv (if SED ran)
```

> **Detailed Documentation:** See [DATAFLOW.md](DATAFLOW.md) for complete stage-by-stage data flow, data structures, cache strategy, and error handling

---

## Output Files

### Primary Outputs

**`diarized_transcript_with_emotion.csv`** - 40-column master transcript
- **Temporal**: start, end, duration_s
- **Speaker**: speaker_id, speaker_name
- **Content**: text, asr_logprob_avg
- **Affect**: valence, arousal, dominance, emotion scores
- **Voice Quality**: jitter, shimmer, HNR, CPPS
- **Prosody**: WPM, pause metrics, pitch statistics
- **Context**: sound events, SNR estimates
- **Quality Flags**: low confidence, VAD instability, error flags

**`diarized_transcript_readable.txt`** - Plain-text transcript with diarized turns, human-friendly timestamps, VAD stability, top sound events, dominant intent, and affect snapshot (valence/arousal/dominance + emotion hint).

**`summary.html`** - Interactive HTML report
- Quick Take overview
- Speaker snapshots with analytics
- Timeline with clickable timestamps
- Sound event log
- Action items and key moments

**`speakers_summary.csv`** – Per-speaker statistics
- Average affect (V/A/D)
- Emotion distribution
- Voice quality metrics
- Turn-taking patterns
- Dominance scores

### Supporting Files

- **`segments.jsonl`** – Full segment payloads with audio features
- **`speaker_registry.json`** – Persistent speaker embeddings for cross-file tracking
- **`events_timeline.csv`** – Sound event timeline with confidence scores
- **`timeline.csv`** – Simplified timeline for quick review
- **`qc_report.json`** – Quality control metrics and processing diagnostics
- **`summary.pdf`** – PDF version of HTML report (requires wkhtmltopdf)

---

## Installation

### Prerequisites

**Required:**
- Python 3.11 or 3.12 (3.13+ not yet supported)
- FFmpeg on PATH (`ffmpeg -version` must work)
- 4+ GB RAM
- 4+ CPU cores (recommended)

**Optional:**
- `wkhtmltopdf` – For PDF report generation

### Quick Start (Windows)

**Option 1: Automated Setup (Recommended)**
```powershell
# Clone repository
git clone <repository-url>
cd diaremot2-on

# Run setup script
.\setup.ps1
```

**Option 2: Manual Setup**
```powershell
# 1. Clone repository
git clone <repository-url>
cd diaremot2-on

# 2. Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -U pip wheel setuptools
pip install -e .

# 5. Verify installation
python -m diaremot.cli diagnostics
```

### Quick Start (Linux/macOS)

**Option 1: Automated Setup (Recommended)**
```bash
# Clone repository
git clone <repository-url>
cd diaremot2-on

# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

**Option 2: Manual Setup**
```bash
# 1. Clone repository
git clone <repository-url>
cd diaremot2-on

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -U pip wheel setuptools
pip install -e .

# 4. Verify installation
python -m diaremot.cli diagnostics
```

### Development Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint code
ruff check src/ tests/

# Type check
mypy src/
```

---

## Configuration

### Environment Variables

**Required:**
```bash
export DIAREMOT_MODEL_DIR=D:/models         # Model directory (Windows)
# export DIAREMOT_MODEL_DIR=/srv/models     # Model directory (Linux)

export HF_HOME=./.cache                     # HuggingFace cache
export HUGGINGFACE_HUB_CACHE=./.cache/hub
export TRANSFORMERS_CACHE=./.cache/transformers
export TORCH_HOME=./.cache/torch

# CPU Threading (optimize for your system)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4

# Disable tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false
```

### Local-first model resolution

- DiaRemot now prefers models that already exist on disk. Every stage checks local paths and caches before attempting a network download.
 - The CLI defaults to this behaviour. Pass `--remote-first` to `run`, `resume`, or `smoke` if you intentionally want to prefer fresh downloads.
 - Need to target a different model directory for a particular invocation? Supply `--model-root D:/alt-models` (or set `DIAREMOT_MODEL_DIR`) and the pipeline will rebuild its search paths dynamically.
- Programmatic integrations can control this via `build_pipeline_config({... 'local_first': False ...})` when a remote-first run is desired.

### Model Search Paths

DiaRemot uses a priority-based model discovery system. For each model, the system searches these locations in order:

**Priority Order:**
1. Explicit environment variable (if set) - e.g., `SILERO_VAD_ONNX_PATH`
2. `$DIAREMOT_MODEL_DIR` (if set)
3. Current working directory: `./models`
4. User home directory: `~/models` or `%USERPROFILE%\models`
5. OS-specific defaults:
   - **Windows**: `D:/models`, `D:/diaremot/diaremot2-1/models`
   - **Linux/Mac**: `/models`, `/opt/diaremot/models`

**How It Works:**
- Pipeline searches relative paths under each root (e.g., `Diarization/ecapa-onnx/ecapa_tdnn.onnx`)
- First existing file wins
- If no ONNX found, falls back to PyTorch (slower, auto-downloads)

---

## Model Assets

### Downloading the official bundle (v2.AI)

- Download the curated model pack published at
  [tltrogl/diaremot2-ai · v2.AI](https://github.com/tltrogl/diaremot2-ai/releases/tag/v2.AI).
- Codex Cloud workers automatically pull `models.zip` from this release; mirror
  the same archive locally for parity with production.
- Verify the checksum before extracting so you know the asset matches CI:

```bash
curl -L https://github.com/tltrogl/diaremot2-ai/releases/download/v2.AI/models.zip -o models.zip
sha256sum models.zip  # Expect 3cc2115f4ef7cd4f9e43cfcec376bf56ea2a8213cb760ab17b27edbc2cac206c
unzip -q models.zip -d ./models
```

`models.zip.sha256` in the repository mirrors the expected hash for quick
automation-friendly validation.

### Required ONNX Models (~2.8GB total)

**ACTUAL MODEL DIRECTORY STRUCTURE** (v2.2.0):

The structure below shows what the code actually searches for and expects. **All models must be in subdirectories** - there are no root-level ONNX files.

```
D:/models/                                      # Windows default
/srv/models/                                    # Linux default
│
├── Diarization/                                # Speaker diarization models
│   ├── ecapa-onnx/
│   │   └── ecapa_tdnn.onnx                    # ~7MB   | ECAPA speaker embeddings
│   │                                          # Search: Diarization/ecapa-onnx/ecapa_tdnn.onnx
│   │                                          #         OR ecapa_onnx/ecapa_tdnn.onnx
│   │                                          #         OR ecapa_tdnn.onnx (root fallback)
│   │                                          # Env:    ECAPA_ONNX_PATH
│   │
│   └── silaro_vad/
│       └── silero_vad.onnx                    # ~3MB   | Silero VAD
│                                              # Search: silero_vad.onnx (multiple locations)
│                                              #         OR silero/vad.onnx
│                                              # Env:    SILERO_VAD_ONNX_PATH
│
├── Affect/                                     # Affect analysis models
│   ├── ser8/                                  # Speech Emotion Recognition (8-class)
│   │   ├── model.int8.onnx                   # ~50MB  | Quantized SER (PREFERRED)
│   │   ├── model.onnx                         # ~200MB | Float32 SER (if available)
│   │   ├── config.json                        # Required for tokenizer
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   │                                          # Search: ser8.int8.onnx
│   │                                          #         OR model.onnx  
│   │                                          #         OR ser_8class.onnx
│   │                                          # Default: Affect/ser8/
│   │                                          # Env:    DIAREMOT_SER_ONNX
│   │
│   ├── VAD_dim/                               # Valence/Arousal/Dominance
│   │   ├── model.onnx                         # ~500MB | V/A/D regression model
│   │   ├── config.json                        # Required for tokenizer
│   │   ├── added_tokens.json
│   │   ├── preprocessor_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   │                                          # Search: model.onnx
│   │                                          #         OR vad_model.onnx
│   │                                          # Default: Affect/VAD_dim/
│   │                                          # Env:    AFFECT_VAD_DIM_MODEL_DIR
│   │
│   └── sed_panns/                             # Sound Event Detection (PANNs)
│       ├── model.onnx                         # ~80MB  | PANNs CNN14
│       └── class_labels_indices.csv           # ~12KB  | 527 AudioSet class labels
│                                              # Search: cnn14.onnx + labels.csv
│                                              #         OR panns_cnn14.onnx + audioset_labels.csv
│                                              #         OR model.onnx + class_labels_indices.csv
│                                              # Default: sed_panns/ OR panns/ OR panns_cnn14/
│                                              # Env:    DIAREMOT_PANNS_DIR
│
├── text_emotions/                              # Text emotion classification
│   ├── model.int8.onnx                        # ~130MB | GoEmotions quantized (PREFERRED)
│   ├── model.onnx                             # ~500MB | GoEmotions float32 (if available)
│   ├── config.json                            # Required for tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
│                                              # Search: model.onnx
│                                              #         OR roberta-base-go_emotions.onnx
│                                              # Default: text_emotions/
│                                              # Env:    DIAREMOT_TEXT_EMO_MODEL_DIR
│
├── intent/                                     # Intent classification (zero-shot NLI)
│   ├── model_int8.onnx                        # ~600MB | BART-MNLI quantized (PREFERRED)
│   ├── model_uint8.onnx                       # ~600MB | Alternative quantization (if available)
│   ├── model.onnx                             # ~1.5GB | Float32 (if available)
│   ├── config.json                            # Required - must contain model_type
│   ├── merges.txt                             # Required for tokenizer
│   ├── special_tokens_map.json
│   ├── tokenizer.json                         # Required OR vocab.json+merges.txt
│   ├── tokenizer_config.json
│   └── vocab.json
│                                              # Search: model_uint8.onnx
│                                              #         OR model.onnx
│                                              #         OR any .onnx file
│                                              # Default: intent/ OR bart/ OR bart-large-mnli/
│                                              #         OR facebook/bart-large-mnli/
│                                              # Env:    DIAREMOT_INTENT_MODEL_DIR
│
└── faster-whisper/                             # ASR models (auto-downloaded)
    └── models--Systran--faster-whisper-tiny.en/
        └── tiny.en/                           # CTranslate2 format
            ├── config.json
            ├── model.bin                      # ~40MB  | Whisper tiny.en quantized
            ├── tokenizer.json
            └── vocabulary.txt
                                               # Auto-download location: HF_HOME/hub/
                                               # Models: tiny.en, base.en, small.en, medium.en, large-v2
```

### Critical Notes on Model Files

**What's Actually Present:**
- ✅ **Quantized models only** for SER8, text emotions, and intent (no float32 versions)
- ✅ **model.int8.onnx** is used for SER8 and text emotions (NOT `model.onnx`)
- ✅ **model_int8.onnx** is used for intent (NOT `model_uint8.onnx`)
- ✅ All models are in **subdirectories** - no root-level ONNX files
- ✅ Tokenizer files (config.json, vocab.json, etc.) must be colocated with ONNX files

**What's Missing from Common Docs:**
- ❌ No `D:/models/silero_vad.onnx` (it's in `Diarization/silaro_vad/`)
- ❌ No `D:/models/ecapa_tdnn.onnx` (it's in `Diarization/ecapa-onnx/`)
- ❌ No float32 SER8 (`Affect/ser8/model.onnx` doesn't exist, only `.int8`)
- ❌ No float32 text emotions (`text_emotions/model.onnx` doesn't exist, only `.int8`)
- ❌ No uint8 intent model (`intent/model_uint8.onnx` doesn't exist, only `_int8`)

### Model Search Behavior

**Example: ECAPA Model Discovery**

When loading ECAPA embeddings, the system searches:
```python
# Priority 1: Environment variable (if set)
$ECAPA_ONNX_PATH

# Priority 2-6: Search under each model root with these relative paths:
1. ecapa_onnx/ecapa_tdnn.onnx
2. Diarization/ecapa-onnx/ecapa_tdnn.onnx  # <-- DOCUMENTED PATH
3. ecapa_tdnn.onnx

# Model roots searched (in order):
- $DIAREMOT_MODEL_DIR
- ./models
- ~/models
- D:/models (Windows) or /models (Linux)
```

**Example: Silero VAD Discovery**

```python
# Priority 1: Environment variable
$SILERO_VAD_ONNX_PATH

# Priority 2-6: Search paths
1. silero_vad.onnx
2. silero/vad.onnx

# Roots: (same priority as ECAPA)
```

### Environment Variable Overrides

Override specific model paths to skip search:

```bash
# Diarization models
export SILERO_VAD_ONNX_PATH="D:/models/Diarization/silaro_vad/silero_vad.onnx"
export ECAPA_ONNX_PATH="D:/models/Diarization/ecapa-onnx/ecapa_tdnn.onnx"

# Affect models (full directory paths, not specific files)
export DIAREMOT_SER_ONNX="D:/models/Affect/ser8/model.int8.onnx"
export DIAREMOT_TEXT_EMO_MODEL_DIR="D:/models/text_emotions"
export AFFECT_VAD_DIM_MODEL_DIR="D:/models/Affect/VAD_dim"
export DIAREMOT_INTENT_MODEL_DIR="D:/models/intent"
export DIAREMOT_PANNS_DIR="D:/models/Affect/sed_panns"

# Model root (affects all relative paths)
export DIAREMOT_MODEL_DIR="D:/models"
```

### CTranslate2 Models (Auto-downloaded)

Faster-Whisper models auto-download on first use:
- **Default**: `faster-whisper-tiny.en` (39 MB)
- **Location**: `$HF_HOME/hub/models--Systran--faster-whisper-{MODEL}/`
- **Supported compute types**: `float32`, `int8`, `int8_float16`
- **Available models**: tiny.en, base.en, small.en, medium.en, large-v2

### PyTorch Fallback Models

When ONNX models are unavailable, system auto-downloads from:
- **Silero VAD** → TorchHub (`snakers4/silero-vad`)
- **PANNs SED** → `panns_inference` library
- **Emotion/Intent** → HuggingFace Hub via `transformers`

⚠️ **Warning:** PyTorch fallbacks are 2-3x slower than ONNX and consume more memory.

---

## Usage

### Basic Commands

```bash
# Standard processing (float32 ASR)
python -m diaremot.cli run --input audio.wav --outdir outputs/

# Fast mode (int8 quantization)
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
    --asr-compute-type int8

# Override VAD tuning
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
    --vad-threshold 0.30 \
    --vad-min-speech-sec 0.80 \
    --ahc-distance-threshold 0.12

# Use preset profile
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
    --profile fast

# Disable optional stages
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
    --disable-sed \
    --disable-affect

# Resume from checkpoint
python -m diaremot.cli resume --input audio.wav --outdir outputs/

# Clear cache before run
python -m diaremot.cli run --input audio.wav --outdir outputs/ \
    --clear-cache

# Run smoke test
python -m diaremot.cli smoke --outdir outputs/
```

### Key CLI Flags

**Input/Output:**
- `--input, -i` – Audio file path (WAV, MP3, M4A, FLAC)
- `--outdir, -o` – Output directory

**Performance:**
- `--asr-compute-type` – `float32` (default) | `int8` | `int8_float16`
- `--asr-cpu-threads` – Thread count for CPU operations (default: 1)

**VAD/Diarization:**
- `--vad-threshold` – Override adaptive VAD threshold (0.0-1.0)
- `--vad-min-speech-sec` – Minimum speech segment duration
- `--vad-speech-pad-sec` – Padding around speech segments
- `--ahc-distance-threshold` – Speaker clustering threshold
- `--clustering-backend` – Clustering method (ahc or spectral)

**Features:**
- `--disable-sed` / `--enable-sed` – Toggle sound event detection
- `--disable-affect` – Skip emotion/intent analysis
- `--profile` – Preset configuration (default|fast|accurate|offline)

**Diagnostics:**
- `--quiet` – Reduce console output
- `--strict` – Enforce strict dependency versions (diagnostics command)

### Profile Presets

```bash
# Default: Balanced speed/quality
python -m diaremot.cli run -i audio.wav -o out/ --profile default

# Fast: Optimized for speed (int8, minimal features)
python -m diaremot.cli run -i audio.wav -o out/ --profile fast

# Accurate: Maximum quality (float32, all features)
python -m diaremot.cli run -i audio.wav -o out/ --profile accurate

# Offline: No model downloads, use local only
python -m diaremot.cli run -i audio.wav -o out/ --profile offline
```

### Programmatic API

```python
from diaremot.pipeline.audio_pipeline_core import run_pipeline, build_pipeline_config

# Configure pipeline
config = build_pipeline_config({
    "whisper_model": "faster-whisper-tiny.en",
    "asr_backend": "faster",
    "compute_type": "int8",
    "vad_threshold": 0.35,
    "enable_sed": True,
    "disable_affect": False,
})

# Run pipeline
result = run_pipeline("audio.wav", "outputs/", config=config)

# Access results
print(f"Processed {result['num_segments']} segments")
print(f"Speakers: {result['num_speakers']}")
print(f"Output directory: {result['out_dir']}")
```

---

## Architecture Details

### Technology Stack

**Core Runtime:**
- Python 3.11-3.12
- ONNXRuntime ≥1.17 (primary inference engine)
- PyTorch 2.x (minimal fallback use)

**Audio Processing:**
- librosa 0.10.2.post1 (resampling, feature extraction)
- scipy ≥1.10,<1.14 (signal processing)
- soundfile ≥0.12 (I/O)
- resampy ≥0.4.3 (high-quality resampling)
- pydub ≥0.25 (audio utilities)

> Optional signal libraries are discovered lazily via `importlib.util.find_spec` so
> the paralinguistics stage can warn and fall back gracefully if librosa, scipy, or
> Parselmouth are missing at runtime.
> The pipeline explicitly imports `importlib.util` to keep this probe available even
> in embedded Python builds where attribute access on `importlib` may be limited.

**ML/NLP:**
- CTranslate2 ≥4.2,<5.0 (ASR backend)
- faster-whisper ≥1.0.3 (ASR wrapper)
- transformers ≥4.40,<4.46 (HuggingFace models)
- scikit-learn ≥1.3,<1.6 (clustering)

**Data/Reporting:**
- pandas ≥2.0,<2.3 (data handling)
- jinja2 ≥3.1 (HTML templating)
- markdown-it-py ≥3.0 (markdown processing)

**CLI/UI:**
- typer (CLI framework)
- rich ≥13.7 (console output)
- tqdm ≥4.66 (progress bars)

### Processing Flow

```
Audio Input
    ↓
[Preprocessing] → Normalize, denoise, chunk if >30min
    ↓
[Sound Events] → PANNs CNN14 → Ambient classification
    ↓
[Diarization] → Silero VAD → ECAPA embeddings → AHC clustering
    ↓
[Transcription] → Faster-Whisper → Intelligent batching
    ↓
[Paralinguistics] → Praat analysis → WPM, pauses, voice quality
    ↓
[Affect Analysis] → Audio + text emotion → Intent → Assembly
    ↓
[Analysis] → Overlaps, flow metrics, speaker summaries
    ↓
[Output Generation] → CSV, JSON, HTML, PDF
```

### Adaptive VAD Tuning

Pipeline automatically adjusts VAD threshold based on audio characteristics:
- Analyzes median energy in dB
- Computes adaptive threshold (0.25-0.45 range)
- User overrides via `--vad-threshold` take precedence

### Intelligent Batching

Transcription module employs sophisticated batching:
- Groups short segments (<8s) into batches
- Target batch size: 60 seconds
- Maximum batch duration: 300 seconds
- Reduces ASR overhead by 2-5x for conversational audio

---

## CSV Schema Reference

The primary output `diarized_transcript_with_emotion.csv` contains **40 columns** (in this exact order):

### Column Order (CRITICAL - DO NOT MODIFY)
```python
SEGMENT_COLUMNS = [
    "file_id",                      # 1.  File identifier
    "start",                        # 2.  Segment start time (seconds)
    "end",                          # 3.  Segment end time (seconds)
    "speaker_id",                   # 4.  Internal speaker ID
    "speaker_name",                 # 5.  Human-readable speaker name
    "text",                         # 6.  Transcribed text
    "valence",                      # 7.  Valence (-1 to +1)
    "arousal",                      # 8.  Arousal (-1 to +1)
    "dominance",                    # 9.  Dominance (-1 to +1)
    "emotion_top",                  # 10. Top speech emotion label
    "emotion_scores_json",          # 11. All 8 emotion scores (JSON)
    "text_emotions_top5_json",      # 12. Top 5 text emotions (JSON)
    "text_emotions_full_json",      # 13. All 28 text emotions (JSON)
    "intent_top",                   # 14. Top intent label
    "intent_top3_json",             # 15. Top 3 intents with confidence (JSON)
    "events_top3_json",             # 16. Top 3 background sounds (JSON)
    "noise_tag",                    # 17. Dominant background class
    "asr_logprob_avg",              # 18. ASR confidence (avg log prob)
    "snr_db",                       # 19. Signal-to-noise ratio estimate
    "snr_db_sed",                   # 20. SNR from SED noise score
    "wpm",                          # 21. Words per minute
    "duration_s",                   # 22. Segment duration
    "words",                        # 23. Word count
    "pause_ratio",                  # 24. Pause time / total duration
    "low_confidence_ser",           # 25. Low speech emotion confidence flag
    "vad_unstable",                 # 26. VAD instability flag
    "affect_hint",                  # 27. Human-readable affect state
    "pause_count",                  # 28. Number of pauses
    "pause_time_s",                 # 29. Total pause duration
    "f0_mean_hz",                   # 30. Mean fundamental frequency
    "f0_std_hz",                    # 31. F0 standard deviation
    "loudness_rms",                 # 32. RMS loudness
    "disfluency_count",             # 33. Filler word count
    "error_flags",                  # 34. Processing error indicators
    "vq_jitter_pct",                # 35. Jitter percentage
    "vq_shimmer_db",                # 36. Shimmer in dB
    "vq_hnr_db",                    # 37. Harmonics-to-Noise Ratio
    "vq_cpps_db",                   # 38. Cepstral Peak Prominence Smoothed
    "voice_quality_hint",           # 39. Human-readable quality interpretation
]
```

**CRITICAL:** This schema is contractual. Modifications require version bumps and migration plans.

---

## Testing

### Smoke Test

```bash
# Quick validation with generated audio
python -m diaremot.cli smoke --outdir /tmp/smoke_test

# Verify outputs
ls /tmp/smoke_test/diarized_transcript_with_emotion.csv
ls /tmp/smoke_test/qc_report.json
```

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_diarization.py -v

# Run with coverage
pytest tests/ --cov=diaremot --cov-report=html
```

### Dependency Validation

```bash
# Basic check
python -m diaremot.cli diagnostics

# Strict version check
python -m diaremot.cli diagnostics --strict
```

---

## Troubleshooting

### Common Issues

**"Module not found" errors**
```bash
# Ensure package is installed
pip install -e .

# Verify imports
python -c "import diaremot; print(diaremot.__file__)"
```

**"Model not found" errors**
```bash
# Check model directory
echo $DIAREMOT_MODEL_DIR
ls -lh $DIAREMOT_MODEL_DIR/

# Verify critical models exist (ACTUAL PATHS)
python -c "
from pathlib import Path
import os
models = [
    'Diarization/silaro_vad/silero_vad.onnx',
    'Diarization/ecapa-onnx/ecapa_tdnn.onnx',
    'Affect/ser8/model.int8.onnx',
    'Affect/VAD_dim/model.onnx',
    'Affect/sed_panns/model.onnx',
    'text_emotions/model.int8.onnx',
    'intent/model_int8.onnx'
]
model_dir = Path(os.getenv('DIAREMOT_MODEL_DIR', 'D:/models'))
missing = [m for m in models if not (model_dir / m).exists()]
print('✓ All models present' if not missing else f'Missing models: {missing}')
"
```

**Poor diarization results**
```bash
# Try adjusting VAD threshold
python -m diaremot.cli run -i audio.wav -o out/ --vad-threshold 0.25

# Increase AHC distance threshold for fewer speakers
python -m diaremot.cli run -i audio.wav -o out/ --ahc-distance-threshold 0.20

# Add more speech padding
python -m diaremot.cli run -i audio.wav -o out/ --vad-speech-pad-sec 0.30
```

**Slow processing**
```bash
# Use int8 quantization
python -m diaremot.cli run -i audio.wav -o out/ --asr-compute-type int8

# Disable optional stages
python -m diaremot.cli run -i audio.wav -o out/ \
    --disable-sed --disable-affect

# Use fast profile
python -m diaremot.cli run -i audio.wav -o out/ --profile fast
```

**Memory issues with long files**
```bash
# Auto-chunking activates at 30 minutes
# Force smaller chunks:
python -m diaremot.cli run -i long_audio.wav -o out/ \
    --chunk-threshold-minutes 15.0 \
    --chunk-size-minutes 10.0 \
    --chunk-overlap-seconds 20.0
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from diaremot.pipeline.audio_pipeline_core import run_pipeline

# Now run pipeline with verbose output
result = run_pipeline("audio.wav", "outputs/")
```

### Clear Cache

```bash
# Clear all caches
rm -rf .cache/

# Clear specific cache
rm -rf .cache/hf/
rm -rf .cache/torch/

# Use CLI flag
python -m diaremot.cli run -i audio.wav -o out/ --clear-cache
```

---

## Performance Benchmarks

Typical processing times on Intel i7 (4 cores, 3.6GHz):

| Audio Length | Configuration | Processing Time | Real-time Factor |
|--------------|---------------|-----------------|------------------|
| 5 min | float32, all stages | ~8 min | 1.6x |
| 5 min | int8, all stages | ~5 min | 1.0x |
| 5 min | int8, no SED/affect | ~3 min | 0.6x |
| 30 min | int8, all stages | ~28 min | 0.93x |
| 60 min | int8, all stages | ~54 min | 0.90x |
| 120 min | int8 (auto-chunked) | ~105 min | 0.88x |

**Key Optimization Factors:**
- `int8` quantization: 30-40% faster than `float32`
- Intelligent batching: 2-5x speedup on conversational audio
- Auto-chunking: Maintains performance on long files (>30 min)
- SED disabled: ~15% faster
- Affect disabled: ~20% faster

---

## Contributing

### Development Guidelines

1. **Follow existing patterns** – Match code style in similar modules
2. **Preserve module boundaries** – Don't merge unrelated logic
3. **Minimal diffs** – Touch only necessary code
4. **Complete implementations** – No placeholder TODOs
5. **Test before committing** – Run linter and tests

### Code Quality

```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/ -v
```

### Adding New Stages

1. Create stage module in `src/diaremot/pipeline/stages/`
2. Add to `PIPELINE_STAGES` list in `stages/__init__.py`
3. Add tests to `tests/`
4. Document in `README.md` and `GEMINI.md`

### Adding New Models

1. Add ONNX export logic to `src/diaremot/io/onnx_utils.py`
2. Update model loading in appropriate module
3. Document in `README.md` models section
4. Test both ONNX and fallback paths

---

## Project Structure

```
diaremot2-on/
├── src/
│   ├── audio_pipeline_core.py       # Legacy location (transitional)
│   └── diaremot/                    # Main package
│       ├── cli.py                   # Typer-based CLI
│       ├── api.py                   # Public API
│       ├── pipeline/
│       │   ├── stages/              # 11 pipeline stages
│       │   │   ├── base.py          # Stage definition
│       │   │   ├── dependency_check.py
│       │   │   ├── preprocess.py
│       │   │   ├── diarize.py
│       │   │   ├── asr.py
│       │   │   ├── paralinguistics.py
│       │   │   ├── affect.py
│       │   │   ├── summaries.py
│       │   │   └── __init__.py      # PIPELINE_STAGES registry
│       │   ├── audio_pipeline_core.py
│       │   ├── orchestrator.py
│       │   ├── speaker_diarization.py
│       │   ├── transcription_module.py
│       │   ├── outputs.py           # CSV schema (40 columns)
│       │   ├── config.py
│       │   ├── runtime_env.py
│       │   ├── pipeline_checkpoint_system.py
│       │   └── ...
│       ├── affect/
│       │   ├── emotion_analyzer.py
│       │   ├── emotion_analysis.py
│       │   ├── paralinguistics.py
│       │   ├── ser_onnx.py
│       │   ├── sed_panns.py
│       │   └── intent_defaults.py
│       ├── io/
│       │   ├── onnx_utils.py
│       │   ├── download_utils.py
│       │   └── speaker_registry_manager.py
│       ├── sed/
│       │   ├── sed_panns_onnx.py
│       │   └── sed_yamnet_tf.py
│       ├── summaries/
│       │   ├── conversation_analysis.py
│       │   ├── html_summary_generator.py
│       │   ├── pdf_summary_generator.py
│       │   └── speakers_summary_builder.py
│       ├── tools/
│       │   └── deps_check.py
│       └── utils/
│           ├── model_paths.py
│           └── hash.py
├── tests/
│   ├── conftest.py
│   ├── test_diarization.py
│   └── test_outputs_transcript.py
├── scripts/
│   ├── affect_only.py
│   ├── audit_models_all.py
│   └── test_affect_stage.py
├── data/                            # Data files
├── checkpoints/                     # Pipeline checkpoints
├── outputs/                         # Default output directory
├── logs/                            # Log files
├── dist/                            # Build artifacts
├── js/                              # JavaScript assets (if any)
├── .cache/                          # HuggingFace/model cache
├── pyproject.toml                   # Package configuration
├── requirements.txt                 # Python dependencies
├── requirements.in                  # Requirements source
├── setup.sh                         # Automated setup script (Linux/macOS)
├── setup.ps1                        # Automated setup script (Windows)
├── Dockerfile                       # Container definition
├── Dockerfile.cloudrun              # Cloud Run specific
├── deploy-cloudrun.sh/.ps1          # Deployment scripts
├── README.md                        # This file
├── GEMINI.md                        # AI assistant context/instructions
├── AGENTS.md                        # Agent setup guide
├── CLOUD_BUILD_GUIDE.md            # Cloud build instructions
└── test_*.py                        # Root-level test files
```

---
"Your core function is efficient and safe assistance. Balance extreme conciseness with the crucial need for clarity, especially regarding safety and potential system modifications. Always prioritize user control and project conventions. Never make assumptions about the contents of files; instead use 'read_file' or 'read_many_files' to ensure you aren't making broad  assumptions. Finally, you are an agent - please keep going until the user's query is completely resolved."

## Citation

If you use DiaRemot in your research, please cite:

```bibtex
@software{diaremot2024,
  title = {DiaRemot: CPU-Only Speech Intelligence Pipeline},
  author = {Timothy Leigh Troglin},
  year = {2024},
  version = {2.2.0},
  url = {https://github.com/tltrogl/diaremot2-on}
}
```

---

## License

MIT License. See LICENSE file for details.

---

## Acknowledgments

Built on:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein
- [Silero VAD](https://github.com/snakers4/silero-vad) by Silero Team
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) by Qiuqiang Kong
- [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth) by Yannick Jadoul
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

Special thanks to the open-source ML community.

---

**Last Updated:** 2025-01-15  
**Version:** 2.2.0  
**Python:** 3.11-3.12  
**License:** MIT

[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/tltrogl/diaremot2-on&cloudshell_git_branch=main&cloudshell_workspace=.&cloudshell_open_in_editor=README.md&show=ide%2Cterminal)
