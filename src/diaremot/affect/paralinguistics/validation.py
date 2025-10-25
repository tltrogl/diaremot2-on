"""Self-diagnostics for the paralinguistics package."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ParalinguisticsConfig, get_config_preset
from .environment import LIBROSA_AVAILABLE, PARSELMOUTH_AVAILABLE, SCIPY_AVAILABLE
from .features import compute_segment_features_v2, process_segments_batch_v2


def validate_module() -> dict[str, Any]:
    """Run a quick functional and dependency validation."""

    validation = {"status": "unknown", "dependencies": {}, "features": {}, "errors": []}

    validation["dependencies"]["librosa"] = LIBROSA_AVAILABLE
    validation["dependencies"]["scipy"] = SCIPY_AVAILABLE
    validation["dependencies"]["parselmouth"] = PARSELMOUTH_AVAILABLE
    validation["dependencies"]["numpy"] = True

    try:
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = 0.1 * np.sin(2 * np.pi * 150 * t)
        test_text = "Hello world test"

        cfg = ParalinguisticsConfig()
        validation["features"]["config_creation"] = True

        features = compute_segment_features_v2(test_audio, sr, 0.0, duration, test_text, cfg)
        validation["features"]["basic_extraction"] = True
        validation["features"]["feature_count"] = len(
            [
                key
                for key, value in features.items()
                if value is not None and not (isinstance(value, float) and np.isnan(value))
            ]
        )

        for preset in ["fast", "balanced", "quality", "research"]:
            try:
                _ = get_config_preset(preset)
                validation["features"][f"preset_{preset}"] = True
            except Exception as exc:
                validation["errors"].append(f"Preset {preset} failed: {exc}")
                validation["features"][f"preset_{preset}"] = False

        segments = [
            (test_audio, sr, 0.0, 1.0, "test one"),
            (test_audio, sr, 1.0, 2.0, "test two"),
        ]
        batch_results = process_segments_batch_v2(segments, cfg)
        validation["features"]["batch_processing"] = len(batch_results) == 2

        validation["status"] = "success"
    except Exception as exc:
        validation["status"] = "failed"
        validation["errors"].append(f"Validation failed: {exc}")

    return validation


__all__ = ["validate_module"]
