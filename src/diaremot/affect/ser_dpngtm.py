"""Torch/HuggingFace implementation of the dpngtm SER model."""

from __future__ import annotations

import logging
import os

import numpy as np

try:  # pragma: no cover - optional heavy dependencies
    import torch
    from transformers import (  # type: ignore
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Processor,
    )
except Exception:  # pragma: no cover - handled lazily
    torch = None  # type: ignore
    Wav2Vec2ForSequenceClassification = None  # type: ignore
    Wav2Vec2Processor = None  # type: ignore


LOGGER = logging.getLogger(__name__)

# Default label mapping shipped with the original dpngtm release
ID2LABEL = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

# Public model repo that mirrors the local snapshot used in production builds.
DEFAULT_MODEL_ID = "dpngtm/wav2vec2-large-xlsr-en-speech-emotion-recognition"


class SERDpngtm:
    """Speech emotion recogniser based on dpngtm's Wav2Vec2 finetune.

    Parameters
    ----------
    model_dir:
        Local directory or HuggingFace repo id containing the model weights. If
        ``None`` the loader checks ``DIAREMOT_SER_MODEL_DIR`` and finally falls
        back to :data:`DEFAULT_MODEL_ID`.
    allow_downloads:
        When ``False`` the loader only uses local files and raises if the model
        snapshot is missing. This is tied to ``EmotionAnalyzer.disable_downloads``.
    device:
        Torch device to run inference on (defaults to CPU).
    """

    def __init__(
        self,
        model_dir: str | None = None,
        *,
        allow_downloads: bool = True,
        device: str | None = None,
    ) -> None:
        if torch is None or Wav2Vec2Processor is None or Wav2Vec2ForSequenceClassification is None:
            raise RuntimeError(
                "PyTorch/transformers are required for SERDpngtm but were not found."
            )

        resolved_dir = model_dir or os.getenv("DIAREMOT_SER_MODEL_DIR") or DEFAULT_MODEL_ID

        local_files_only = not allow_downloads
        # If the resolved path is a real directory we can keep local_files_only
        # strict; for repo ids we rely on allow_downloads to decide behaviour.
        if os.path.isdir(resolved_dir):
            model_source = resolved_dir
        else:
            model_source = resolved_dir
            if local_files_only:
                LOGGER.warning(
                    "SERDpngtm configured with allow_downloads=False but repo '%s'"
                    " is remote; attempting local cache only.",
                    resolved_dir,
                )

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            )
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # pragma: no cover - external dependency error paths
            raise RuntimeError(f"Unable to load SER model from '{model_source}': {exc}") from exc

        self.device = torch.device(device or "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Normalise label mapping so downstream callers can rely on consistent
        # keys regardless of the checkpoint metadata.
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and id2label:
            self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        else:
            self.id2label = {idx: label for idx, label in enumerate(ID2LABEL)}

    def predict_16k_f32(self, wav_16k_f32: np.ndarray) -> tuple[str, dict[str, float]]:
        """Predict emotion scores for a 16 kHz mono waveform."""

        if torch is None:
            raise RuntimeError("PyTorch is required for SER inference")

        if getattr(wav_16k_f32, "ndim", 1) > 1:
            wav_16k_f32 = wav_16k_f32.mean(axis=-1)

        inputs = self.processor(
            wav_16k_f32,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        scores = {
            self.id2label.get(i, ID2LABEL[i] if i < len(ID2LABEL) else str(i)): float(p)
            for i, p in enumerate(probs)
        }
        top_label = max(scores.items(), key=lambda item: item[1])[0]
        return top_label, scores


__all__ = ["SERDpngtm", "ID2LABEL", "DEFAULT_MODEL_ID"]
