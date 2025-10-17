# DiaRemot Model Map - Complete Reference

**Date:** 2025-01-15  
**Version:** 2.2.0

> **See Also:** [DATAFLOW.md](DATAFLOW.md) - Detailed pipeline stage-by-stage data flow documentation

---

## Complete Model Inventory

| # | Component | HuggingFace Model | Actual File Path | Size | Quantization | Task |
|---|-----------|-------------------|------------------|------|--------------|------|
| 1 | Silero VAD | snakers4/silero-vad (TorchHub) | `Diarization/silaro_vad/silero_vad.onnx` | 3MB | N/A | Voice Activity Detection |
| 2 | ECAPA Speaker Embeddings | speechbrain/spkrec-ecapa-voxceleb | `Diarization/ecapa-onnx/ecapa_tdnn.onnx` | 7MB | N/A | Speaker embeddings (192-dim) |
| 3 | Speech Emotion Recognition | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition | `Affect/ser8/model.int8.onnx` | 50MB | INT8 | 8-class emotion (audio) |
| 4 | Valence/Arousal/Dominance | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | `Affect/VAD_dim/model.onnx` | 500MB | Float32 | Dimensional emotion |
| 5 | Text Emotions | **SamLowe/roberta-base-go_emotions** | `text_emotions/model.int8.onnx` | 130MB | INT8 | 28-class GoEmotions |
| 6 | Intent Classification | **facebook/bart-large-mnli** | `intent/model_int8.onnx` | 600MB | INT8 | Zero-shot NLI |
| 7 | Sound Event Detection | qiuqiangkong/panns_cnn14 | `Affect/sed_panns/model.onnx` | 80MB | Float32 | 527 AudioSet classes |
| 8 | ASR (Whisper) | Systran/faster-whisper-tiny.en | `(auto-download)` | 40MB | CTranslate2 | Speech-to-text |

---

## Model Details by Component

### 1. Silero VAD (Voice Activity Detection)

**Purpose:** Detects speech vs non-speech regions in audio

| Property | Value |
|----------|-------|
| **HuggingFace/Source** | `snakers4/silero-vad` (TorchHub) |
| **File Path** | `D:/models/Diarization/silaro_vad/silero_vad.onnx` |
| **Search Paths** | `silero_vad.onnx`, `silero/vad.onnx` |
| **Size** | ~3MB |
| **Input** | 16kHz mono audio (512 sample chunks) |
| **Output** | Speech probability per chunk [0.0-1.0] |
| **Env Override** | `SILERO_VAD_ONNX_PATH` |
| **PyTorch Fallback** | TorchHub: `snakers4/silero-vad` |

**Code Reference:**
```python
# speaker_diarization.py line ~105
candidate_paths = list(iter_model_subpaths("silero_vad.onnx"))
candidate_paths.extend(list(iter_model_subpaths(Path("silero") / "vad.onnx")))
```

---

### 2. ECAPA-TDNN Speaker Embeddings

**Purpose:** Extract speaker embeddings for diarization clustering

| Property | Value |
|----------|-------|
| **HuggingFace/Source** | speechbrain/spkrec-ecapa-voxceleb |
| **File Path** | `D:/models/Diarization/ecapa-onnx/ecapa_tdnn.onnx` |
| **Search Paths** | `ecapa_onnx/ecapa_tdnn.onnx`<br>`Diarization/ecapa-onnx/ecapa_tdnn.onnx`<br>`ecapa_tdnn.onnx` |
| **Size** | ~7MB |
| **Input** | Log-mel spectrograms (80 mels) |
| **Output** | 192-dimensional speaker embedding |
| **Env Override** | `ECAPA_ONNX_PATH` |
| **PyTorch Fallback** | None (ONNX required) |

**Code Reference:**
```python
# speaker_diarization.py line ~325
candidate_paths = list(iter_model_subpaths(Path("ecapa_onnx") / "ecapa_tdnn.onnx"))
candidate_paths.extend(
    list(iter_model_subpaths(Path("Diarization") / "ecapa-onnx" / "ecapa_tdnn.onnx"))
)
```

---

### 3. Speech Emotion Recognition (SER8)

**Purpose:** Classify audio into 8 emotion categories

| Property | Value |
|----------|-------|
| **HuggingFace Model** | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition |
| **File Path** | `D:/models/Affect/ser8/model.int8.onnx` |
| **Search Order** | 1. `ser8.int8.onnx`<br>2. `model.onnx`<br>3. `ser_8class.onnx` |
| **Size** | ~50MB (quantized) |
| **Quantization** | INT8 ONLY (no float32 version exists) |
| **Classes** | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| **Input** | 16kHz mono audio (waveform or mel-spectrogram) |
| **Output** | 8-class probability distribution |
| **Default Dir** | `Affect/ser8/` |
| **Env Override** | `DIAREMOT_SER_ONNX` |
| **PyTorch Fallback** | None |

**Required Files:**
- `model.int8.onnx` (ONNX model)
- `config.json` (model config)
- `preprocessor_config.json` (audio preprocessing)
- `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`

**Code Reference:**
```python
# emotion_analyzer.py line ~198
self.path_ser8_onnx = str(
    _select_first_existing(
        self.ser_model_dir,
        ("ser8.int8.onnx", "model.onnx", "ser_8class.onnx"),
    )
)
```

---

### 4. Valence/Arousal/Dominance (VAD)

**Purpose:** Predict continuous emotion dimensions

| Property | Value |
|----------|-------|
| **HuggingFace Model** | audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim |
| **File Path** | `D:/models/Affect/VAD_dim/model.onnx` |
| **Search Order** | 1. `model.onnx`<br>2. `vad_model.onnx` |
| **Size** | ~500MB |
| **Quantization** | Float32 |
| **Output** | Valence [-1,+1], Arousal [-1,+1], Dominance [-1,+1] |
| **Input** | 16kHz mono audio |
| **Default Dir** | `Affect/VAD_dim/` |
| **Env Override** | `AFFECT_VAD_DIM_MODEL_DIR` |

**Required Files:**
- `model.onnx`
- `config.json`, `added_tokens.json`
- `preprocessor_config.json`
- `tokenizer_config.json`, `vocab.json`, `special_tokens_map.json`

**Code Reference:**
```python
# emotion_analyzer.py line ~202
self.path_vad_onnx = str(
    _select_first_existing(
        self.vad_model_dir,
        ("model.onnx", "vad_model.onnx"),
    )
)
```

---

### 5. Text Emotions (GoEmotions)

**Purpose:** Classify text into 28 emotion categories

| Property | Value |
|----------|-------|
| **HuggingFace Model** | **SamLowe/roberta-base-go_emotions** |
| **Base Architecture** | RoBERTa-base |
| **File Path** | `D:/models/text_emotions/model.int8.onnx` |
| **Search Order** | 1. `model.onnx`<br>2. `roberta-base-go_emotions.onnx` |
| **Size** | ~130MB (quantized) |
| **Quantization** | INT8 ONLY (no float32 version exists) |
| **Classes** | 28 emotions (admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral) |
| **Input** | Text (max 128 tokens) |
| **Output** | 28-class probability distribution |
| **Default Dir** | `text_emotions/` |
| **Env Override** | `DIAREMOT_TEXT_EMO_MODEL_DIR` |
| **PyTorch Fallback** | `SamLowe/roberta-base-go_emotions` (via transformers) |

**Required Files:**
- `model.int8.onnx`
- `config.json`, `merges.txt`
- `tokenizer.json`, `tokenizer_config.json`, `vocab.json`
- `special_tokens_map.json`

**Code Reference:**
```python
# emotion_analyzer.py line ~148
# GoEmotions 28 labels (SamLowe/roberta-base-go_emotions)

# emotion_analyzer.py line ~393
candidates.append(("SamLowe/roberta-base-go_emotions", {}))

# emotion_analyzer.py line ~494
self.pipe = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    truncation=True,
)
```

---

### 6. Intent Classification (Zero-Shot NLI)

**Purpose:** Classify utterance intent using zero-shot classification

| Property | Value |
|----------|-------|
| **HuggingFace Model** | **facebook/bart-large-mnli** |
| **Base Architecture** | BART-large fine-tuned on MNLI |
| **File Path** | `D:/models/intent/model_int8.onnx` |
| **Search Order** | 1. `model_uint8.onnx` (DOCUMENTED but doesn't exist)<br>2. `model_int8.onnx` (ACTUAL)<br>3. `model.onnx`<br>4. Any `.onnx` file |
| **Size** | ~600MB |
| **Quantization** | INT8 (named `_int8` not `_uint8`) |
| **Method** | Natural Language Inference (entailment detection) |
| **Default Classes** | status_update, question, request, command, small_talk, opinion, complaint, agreement, disagreement, appreciation |
| **Input** | Text pairs (premise, hypothesis) |
| **Output** | Entailment/contradiction/neutral logits |
| **Default Dirs** | `intent/`, `bart/`, `bart-large-mnli/`, `facebook/bart-large-mnli/` |
| **Env Override** | `DIAREMOT_INTENT_MODEL_DIR` |
| **PyTorch Fallback** | `facebook/bart-large-mnli` (via transformers zero-shot pipeline) |

**Required Files:**
- `model_int8.onnx` (NOT `model_uint8.onnx`!)
- `config.json` (must contain `model_type`)
- `tokenizer.json` OR (`vocab.json` + `merges.txt`)
- `tokenizer_config.json`, `special_tokens_map.json`

**CRITICAL BUG:** Code searches for `model_uint8.onnx` first but actual file is `model_int8.onnx`

**Code Reference:**
```python
# emotion_analyzer.py line ~148
DEFAULT_INTENT_MODEL = "facebook/bart-large-mnli"

# emotion_analyzer.py line ~719
def _select_onnx_model(self, model_dir: Path) -> Path | None:
    for name in ("model_uint8.onnx", "model.onnx"):  # <-- BUG: searches uint8 first
        candidate = model_dir / name
        if candidate.exists():
            return candidate
```

---

### 7. Sound Event Detection (PANNs CNN14)

**Purpose:** Detect ambient sounds and background events

| Property | Value |
|----------|-------|
| **HuggingFace Model** | qiuqiangkong/panns_cnn14 |
| **File Path** | `D:/models/Affect/sed_panns/model.onnx` (primary)<br>`D:/models/sed_panns/model.onnx` (legacy) |
| **Search Dirs** | `sed_panns/`, `panns/`, `panns_cnn14/` |
| **Search Pairs** | 1. `cnn14.onnx` + `labels.csv`<br>2. `panns_cnn14.onnx` + `audioset_labels.csv`<br>3. `model.onnx` + `class_labels_indices.csv` |
| **Size** | ~80MB |
| **Quantization** | Float32 |
| **Classes** | 527 AudioSet classes |
| **Input** | 32kHz mono audio |
| **Output** | 527-class probability distribution |
| **Env Override** | `DIAREMOT_PANNS_DIR` |
| **PyTorch Fallback** | `panns_inference` library |

**Required Files:**
- `model.onnx` (CNN14 ONNX export)
- `class_labels_indices.csv` (527 AudioSet labels)

**Code Reference:**
```python
# sed_panns.py line ~43
_PANNS_SUBDIR_CANDIDATES = ("sed_panns", "panns", "panns_cnn14")

# sed_panns.py line ~77
_ONNX_FILENAME_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("cnn14.onnx", "labels.csv"),
    ("panns_cnn14.onnx", "audioset_labels.csv"),
    ("model.onnx", "class_labels_indices.csv"),
)
```

---

### 8. ASR - Faster-Whisper (CTranslate2)

**Purpose:** Speech-to-text transcription

| Property | Value |
|----------|-------|
| **HuggingFace Model** | Systran/faster-whisper-{MODEL} |
| **Default Model** | `faster-whisper-tiny.en` |
| **Location** | Auto-download to `$HF_HOME/hub/models--Systran--faster-whisper-*/` |
| **Size** | ~40MB (tiny.en, int8) |
| **Format** | CTranslate2 (optimized Whisper) |
| **Compute Types** | `float32`, `int8`, `int8_float16` |
| **Available Models** | tiny.en, base.en, small.en, medium.en, large-v2 |
| **Input** | 16kHz mono audio |
| **Output** | Timestamped transcription with confidence scores |

**Model Files (per model):**
- `config.json` (CTranslate2 config)
- `model.bin` (CTranslate2 model weights)
- `tokenizer.json` (Whisper tokenizer)
- `vocabulary.txt` (token vocabulary)

---

## Environment Variables Summary

| Variable | Purpose | Default Search |
|----------|---------|----------------|
| `DIAREMOT_MODEL_DIR` | Root model directory | `D:/models` (Win), `/srv/models` (Linux) |
| `SILERO_VAD_ONNX_PATH` | Override Silero VAD location | Multiple subdirs |
| `ECAPA_ONNX_PATH` | Override ECAPA location | `Diarization/ecapa-onnx/` |
| `DIAREMOT_SER_ONNX` | Override SER8 location | `Affect/ser8/` |
| `DIAREMOT_TEXT_EMO_MODEL_DIR` | Text emotions directory | `text_emotions/` |
| `AFFECT_VAD_DIM_MODEL_DIR` | VAD directory | `Affect/VAD_dim/` |
| `DIAREMOT_INTENT_MODEL_DIR` | Intent model directory | `intent/`, `bart/`, etc. |
| `DIAREMOT_PANNS_DIR` | PANNs directory | `sed_panns/`, `Affect/sed_panns/` |

---

## Search Priority System

For **each model**, DiaRemot searches in this order:

```
1. Explicit environment variable (if set)
   ↓
2. $DIAREMOT_MODEL_DIR + relative paths
   ↓
3. ./models + relative paths (current directory)
   ↓
4. ~/models + relative paths (user home)
   ↓
5. OS-specific defaults:
   Windows:  D:/models, D:/diaremot/diaremot2-1/models
   Linux:    /models, /opt/diaremot/models

First existing file WINS.
```

---

## Quick Verification Script

```python
from pathlib import Path
import os

models_to_check = {
    "Silero VAD": "Diarization/silaro_vad/silero_vad.onnx",
    "ECAPA Embeddings": "Diarization/ecapa-onnx/ecapa_tdnn.onnx",
    "SER8 (Speech Emotion)": "Affect/ser8/model.int8.onnx",
    "VAD Emotion": "Affect/VAD_dim/model.onnx",
    "Text Emotions (GoEmotions)": "text_emotions/model.int8.onnx",
    "Intent (BART-MNLI)": "intent/model_int8.onnx",
    "PANNs SED": "Affect/sed_panns/model.onnx",
}

model_dir = Path(os.getenv('DIAREMOT_MODEL_DIR', 'D:/models'))

print(f"Checking models in: {model_dir}\n")

for name, rel_path in models_to_check.items():
    full_path = model_dir / rel_path
    status = "✓ FOUND" if full_path.exists() else "✗ MISSING"
    size = f"{full_path.stat().st_size / 1024 / 1024:.1f}MB" if full_path.exists() else "N/A"
    print(f"{status:10} {name:30} {size:>8}  {rel_path}")
```

---

## Key Takeaways

### ✅ What's Correct

1. **All models in subdirectories** - No root-level ONNX files
2. **Quantized models preferred** - INT8 used for SER8, text emotions, intent
3. **Priority search system** - Environment vars > model dir > local > defaults
4. **Fallback system** - PyTorch auto-download when ONNX unavailable

### ❌ Common Mistakes in Documentation

1. **Intent model name**: It's `model_int8.onnx` NOT `model_uint8.onnx`
2. **Text emotions model**: SamLowe/roberta-base-go_emotions NOT facebook/bart-large-mnli
3. **Intent model**: facebook/bart-large-mnli (not used for text emotions)
4. **Root-level models**: Don't document `D:/models/silero_vad.onnx` - it's in a subdirectory
5. **Float32 versions**: SER8 and text emotions only have INT8 versions

### 🐛 Code Bugs

**Intent model search bug** (emotion_analyzer.py:719):
```python
# BUG: Searches for model_uint8.onnx but actual file is model_int8.onnx
for name in ("model_uint8.onnx", "model.onnx"):
```

**Should be:**
```python
for name in ("model_int8.onnx", "model.onnx"):
```

---

**Last Updated:** 2025-01-15  
**Author:** Based on code analysis of diaremot2-on v2.2.0
