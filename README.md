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

---

## Output Files

### Primary Outputs

**`diarized_transcript_with_emotion.csv`** – 39-column master transcript
- **Temporal**: start, end, duration_s
- **Speaker**: speaker_id, speaker_name
- **Content**: text, asr_logprob_avg
- **Affect**: valence, arousal, dominance, emotion scores
- **Voice Quality**: jitter, shimmer, HNR, CPPS
- **Prosody**: WPM, pause metrics, pitch statistics
- **Context**: sound events, SNR estimates
- **Quality Flags**: low confidence, VAD instability, error flags

**`summary.html`** – Interactive HTML report
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
- Python 3.11+
- FFmpeg on PATH (`ffmpeg -version` must work)
- 4+ GB RAM
- 4+ CPU cores (recommended)

**Optional:**
- `wkhtmltopdf` – For PDF report generation

### Quick Start (Windows)

```powershell
# 1. Clone repository
git clone https://github.com/your-org/diaremot2-ai.git
cd diaremot2-ai

# 2. Create virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt

# 4. Install PyTorch CPU (requires special index)
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu


# 5. Install package
pip install -e .

# 6. Verify installation
python -m diaremot.cli diagnostics
```

**Alternatively, use the provided setup script:**
```powershell
.\setup.ps1
```

### Quick Start (Linux/macOS)

```bash
# 1. Clone repository
git clone https://github.com/your-org/diaremot2-ai.git
cd diaremot2-ai

# 2. Run setup script (handles everything)
./setup.sh

# 3. Verify installation
python -m diaremot.cli diagnostics
```

### Development Setup

```bash
# Install dev tools
pip install ruff pytest mypy

# Run tests
pytest tests/ -v

# Lint code
ruff check src/ tests/

# Type check (if configured)
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

### Model Search Paths (in priority order)

1. `$DIAREMOT_MODEL_DIR` (if set)
2. `D:/models` (Windows) or `/srv/models` (Linux)
3. `./models` (project directory)
4. `$HOME/models`

---

## Model Assets

### Required ONNX Models (~2.8GB total)

Models must be placed in your model directory (default: `D:/models` on Windows, `/srv/models` on Linux):

```
D:/models/                              # Windows
├── silero_vad.onnx                    # ~3MB   | Silero VAD
├── ecapa_tdnn.onnx                    # ~7MB   | ECAPA speaker embeddings (legacy location)
│
├── Diarization/
│   └── ecapa-onnx/
│       └── ecapa_tdnn.onnx            # ~7MB   | Speaker embeddings (preferred)
│
├── Affect/
│   ├── ser8/
│   │   ├── model.onnx                 # ~100MB | Speech emotion recognition (8 classes)
│   │   ├── model.int8.onnx           # ~50MB  | Quantized version (optional)
│   │   ├── config.json
│   │   ├── vocab.json
│   │   └── tokenizer configs...
│   │
│   ├── VAD_dim/
│   │   ├── model.onnx                 # ~500MB | Valence/Arousal/Dominance
│   │   ├── config.json
│   │   └── tokenizer configs...
│   │
│   └── sed_panns/
│       ├── model.onnx                 # ~80MB  | PANNs CNN14 sound event detection
│       └── class_labels_indices.csv   # ~12KB  | AudioSet 527-class labels
│
├── text_emotions/
│   ├── model.onnx                     # ~500MB | RoBERTa GoEmotions (28 emotions)
│   ├── model.int8.onnx               # ~130MB | Quantized version (optional)
│   ├── config.json
│   ├── vocab.json
│   ├── merges.txt
│   └── tokenizer configs...
│
└── intent/
    ├── model_int8.onnx                # ~600MB | BART-MNLI intent classification (preferred)
    ├── model_uint8.onnx              # ~600MB | Alternative quantization (optional)
    ├── config.json
    ├── vocab.json
    ├── merges.txt
    └── tokenizer configs...
```

> Note: Release v2.AI installs the quantized SER bundle at
> `D:/models/Affect/ser8-onnx-int8/ser8.int8.onnx` (Windows) or
> `/srv/models/Affect/ser8-onnx-int8/ser8.int8.onnx` (Linux). Legacy
> `Affect/ser8/model.onnx` remains available for compatibility but is no longer the default.

### Environment Variable Overrides

Override specific model paths:
```bash
# Individual model overrides
export SILERO_VAD_ONNX_PATH=/path/to/silero_vad.onnx
export ECAPA_ONNX_PATH=/path/to/ecapa_tdnn.onnx
export DIAREMOT_SER_ONNX="D:/models/Affect/ser8-onnx-int8/ser8.int8.onnx"

# Affect model directory overrides
export DIAREMOT_INTENT_MODEL_DIR=/path/to/intent
```

### Download Models

**Method 1: Automated (Recommended)**
```bash
cd $DIAREMOT_MODEL_DIR
wget https://github.com/tltrogl/diaremot2-ai/releases/download/v2.1/models.zip
sha256sum models.zip  # Verify checksum from release page
unzip -q models.zip
```

**Method 2: Python Download Utility**
```python
from pathlib import Path
from diaremot.io.download_utils import download_file

models_dir = Path("D:/models")  # or Path("/srv/models") on Linux
models_dir.mkdir(parents=True, exist_ok=True)

download_file(
    url="https://github.com/tltrogl/diaremot2-ai/releases/download/v2.1/models.zip",
    destination=models_dir / "models.zip",
    timeout=300
)
```

### CTranslate2 Models (Auto-downloaded)

Faster-Whisper models auto-download to HuggingFace cache:
- `faster-whisper-tiny.en` (39 MB) – Default model
- Supported compute types: `float32`, `int8`, `int8_float16`

### PyTorch Fallback Models

When ONNX models unavailable, system auto-downloads from:
- Silero VAD → TorchHub (`snakers4/silero-vad`)
- PANNs SED → `panns_inference` library
- Emotion/Intent → HuggingFace Hub via `transformers`

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
```

### Key CLI Flags

**Input/Output:**
- `--input, -i` – Audio file path (WAV, MP3, M4A, FLAC)
- `--outdir, -o` – Output directory

**Performance:**
- `--asr-compute-type` – `float32` (default) | `int8` | `int8_float16`
- `--cpu-threads` – Thread count for CPU operations (default: 1)

**VAD/Diarization:**
- `--vad-threshold` – Override adaptive VAD threshold (0.0-1.0)
- `--vad-min-speech-sec` – Minimum speech segment duration
- `--speech-pad-sec` – Padding around speech segments
- `--ahc-distance-threshold` – Speaker clustering threshold

**Features:**
- `--disable-sed` – Skip sound event detection
- `--disable-affect` – Skip emotion/intent analysis
- `--profile` – Preset configuration (default|fast|accurate|offline)

**Diagnostics:**
- `--quiet` – Reduce console output
- `--validate-dependencies` – Check all dependencies before processing
- `--strict-dependency-versions` – Enforce exact version requirements

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
from diaremot.pipeline.audio_pipeline_core import AudioAnalysisPipelineV2

# Configure pipeline
config = {
    "whisper_model": "faster-whisper-tiny.en",
    "asr_backend": "faster",
    "compute_type": "int8",
    "vad_threshold": 0.35,
    "disable_sed": False,
    "disable_affect": False,
}

# Initialize and run
pipeline = AudioAnalysisPipelineV2(config)
result = pipeline.process_audio_file("audio.wav", "outputs/")

# Access results
print(f"Processed {result['num_segments']} segments")
print(f"Speakers: {result['num_speakers']}")
print(f"Output directory: {result['out_dir']}")
```

---

## Architecture Details

### Technology Stack

**Core Runtime:**
- Python 3.11
- ONNXRuntime 1.17.1 (primary inference engine)
- PyTorch 2.4.1+cpu (minimal fallback use)

**Audio Processing:**
- librosa 0.10.2 (resampling, feature extraction)
- scipy 1.10.1 (signal processing)
- soundfile 0.12.1 (I/O)
- Praat-Parselmouth 0.4.3 (voice quality)

**ML/NLP:**
- CTranslate2 4.6.0 (ASR backend)
- faster-whisper 1.1.0 (ASR wrapper)
- transformers 4.38.2 (HuggingFace models)

**Data/Reporting:**
- pandas 2.0.3 (data handling)
- reportlab 4.1.0 (PDF generation)
- jinja2 3.1.6 (HTML templating)

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

The primary output `diarized_transcript_with_emotion.csv` contains 39 columns:

### Temporal Fields
- `start` – Segment start time (seconds)
- `end` – Segment end time (seconds)
- `duration_s` – Segment duration

### Speaker Fields
- `speaker_id` – Internal speaker ID
- `speaker_name` – Human-readable speaker name

### Content Fields
- `text` – Transcribed text
- `asr_logprob_avg` – ASR confidence (average log probability)
  - Negative values closer to 0 indicate higher confidence
  - Typical range: -0.1 to -2.0 for good quality

### Emotion Fields
- `valence` – Valence (-1 to +1, negative to positive)
- `arousal` – Arousal (-1 to +1, calm to excited)
- `dominance` – Dominance (-1 to +1, submissive to dominant)
- `emotion_top` – Top speech emotion label
- `emotion_scores_json` – All 8 emotion scores (JSON)
- `text_emotions_top5_json` – Top 5 text emotions (JSON)
- `text_emotions_full_json` – All 28 text emotions (JSON)
- `affect_hint` – Human-readable affect state

### Intent Fields
- `intent_top` – Top intent label
- `intent_top3_json` – Top 3 intents with confidence (JSON)

### Sound Event Fields
- `events_top3_json` – Top 3 background sounds (JSON)
- `noise_tag` – Dominant background class

### Quality Metrics
- `snr_db` – Signal-to-noise ratio estimate
- `snr_db_sed` – SNR from SED noise score
- `low_confidence_ser` – Low speech emotion confidence flag
- `vad_unstable` – VAD instability flag
- `error_flags` – Processing error indicators

### Prosody & Paralinguistics
- `wpm` – Words per minute
- `words` – Word count
- `pause_count` – Number of pauses
- `pause_time_s` – Total pause duration
- `pause_ratio` – Pause time / total duration
- `disfluency_count` – Filler word count
- `f0_mean_hz` – Mean fundamental frequency
- `f0_std_hz` – F0 standard deviation
- `loudness_rms` – RMS loudness

### Voice Quality (Praat)
- `vq_jitter_pct` – Jitter percentage
- `vq_shimmer_db` – Shimmer in dB
- `vq_hnr_db` – Harmonics-to-Noise Ratio
- `vq_cpps_db` – Cepstral Peak Prominence Smoothed
- `voice_quality_hint` – Human-readable quality interpretation

**CRITICAL:** This schema is contractual. Modifications require version bumps and migration plans.

---

## Testing

### Smoke Test

```bash
# Quick validation
python -m diaremot.cli run \
    --input tests/fixtures/sample.wav \
    --outdir /tmp/smoke_test

# Verify outputs
ls /tmp/smoke_test/diarized_transcript_with_emotion.csv
ls /tmp/smoke_test/qc_report.json
```

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_paralinguistics.py -v

# Run with coverage
pytest tests/ --cov=diaremot --cov-report=html
```

### Dependency Validation

```bash
# Basic check
python -m diaremot.cli diagnostics

# Strict version check
python -m diaremot.cli diagnostics --strict

# Programmatic check
python -c "from diaremot.pipeline.config import diagnostics; print(diagnostics())"
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

# Verify all required models
python -c "
from pathlib import Path
import os
models = [
    'silero_vad.onnx',
    'ecapa_tdnn.onnx',
    'Affect/ser8-onnx-int8/ser8.int8.onnx',
    'Affect/ser8/model.onnx',
    'Affect/VAD_dim/model.onnx',
    'Affect/sed_panns/model.onnx',
    'text_emotions/model.onnx',
    'intent/model_int8.onnx'
]
model_dir = Path(os.getenv('DIAREMOT_MODEL_DIR', 'D:/models'))
missing = [m for m in models if not (model_dir / m).exists()]
print('Missing:' if missing else 'All models present:', missing or '✓')
"
```

**Poor diarization results**
```bash
# Try adjusting VAD threshold
python -m diaremot.cli run -i audio.wav -o out/ --vad-threshold 0.25

# Increase AHC distance threshold for fewer speakers
python -m diaremot.cli run -i audio.wav -o out/ --ahc-distance-threshold 0.20

# Add more speech padding
python -m diaremot.cli run -i audio.wav -o out/ --speech-pad-sec 0.30
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
python -c "
from diaremot.pipeline.audio_pipeline_core import AudioAnalysisPipelineV2
config = {
    'chunk_threshold_minutes': 15.0,
    'chunk_size_minutes': 10.0,
    'chunk_overlap_seconds': 20.0,
}
pipeline = AudioAnalysisPipelineV2(config)
pipeline.process_audio_file('long_audio.wav', 'outputs/')
"
```

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from diaremot.pipeline.audio_pipeline_core import AudioAnalysisPipelineV2

# Now run pipeline with verbose output
pipeline = AudioAnalysisPipelineV2({})
result = pipeline.process_audio_file("audio.wav", "outputs/")
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

# Type check (if configured)
mypy src/

# Run tests
pytest tests/ -v
```

### Adding New Stages

1. Create stage module in `src/diaremot/pipeline/stages/`
2. Add to `PIPELINE_STAGES` list in `stages/__init__.py`
3. Update `AI_INDEX.yaml`
4. Add tests to `tests/`
5. Document in `README.md` and `CLAUDE.md`

### Adding New Models

1. Add ONNX export logic to `src/diaremot/io/onnx_utils.py`
2. Update model loading in appropriate module
3. Document in `README.md` models section
4. Add to `models.zip` release asset
5. Test both ONNX and fallback paths

---

## Project Structure

```
diaremot2-ai/
├── src/diaremot/
│   ├── pipeline/
│   │   ├── stages/              # 11 pipeline stages
│   │   ├── audio_pipeline_core.py
│   │   ├── orchestrator.py
│   │   ├── speaker_diarization.py
│   │   ├── transcription_module.py
│   │   ├── outputs.py           # CSV schema
│   │   └── config.py
│   ├── affect/
│   │   ├── emotion_analyzer.py
│   │   └── paralinguistics.py
│   ├── io/
│   │   ├── onnx_utils.py
│   │   └── download_utils.py
│   ├── utils/
│   └── cli.py                   # Main CLI entry point
├── tests/
│   ├── fixtures/
│   └── test_*.py
├── requirements.txt
├── pyproject.toml
├── README.md                    # This file
├── CLAUDE.md                    # AI assistant instructions
├── AGENTS.md                    # Agent setup guide
└── AI_INDEX.yaml                # Architecture reference
```

---

## Citation

If you use DiaRemot in your research, please cite:

```bibtex
@software{diaremot2024,
  title = {DiaRemot: CPU-Only Speech Intelligence Pipeline},
  author = {DiaRemot Authors},
  year = {2024},
  version = {2.1.0},
  url = {https://github.com/your-org/diaremot2-ai}
}
```

---

## License

MIT License. See LICENSE file for details.

---

## Support

**Issues:** https://github.com/your-org/diaremot2-ai/issues  
**Documentation:** https://diaremot.readthedocs.io  
**Email:** support@diaremot.com

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

**Last Updated:** 2025-10-11  
**Version:** 2.2.0  
**Python:** 3.11+  
**License:** MIT
