from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _reload_runtime_env() -> object:
    sys.modules.pop("diaremot.pipeline.runtime_env", None)
    return importlib.import_module("diaremot.pipeline.runtime_env")


def _contains_cache_segment(path: Path) -> bool:
    return any(part.lower().startswith(".cache") or part.lower().endswith(".cache") for part in path.parts)


def test_runtime_env_prefers_local_models(monkeypatch, tmp_path):
    local_models = tmp_path / "local_models"
    (local_models / "Diarization" / "ecapa-onnx").mkdir(parents=True)
    (local_models / "Diarization" / "ecapa-onnx" / "ecapa_tdnn.onnx").touch()

    monkeypatch.setenv("DIAREMOT_MODEL_DIR", str(local_models))

    try:
        runtime_env = _reload_runtime_env()
        assert runtime_env.DEFAULT_MODELS_ROOT.resolve() == local_models.resolve()
        assert runtime_env.MODEL_ROOTS
        assert all(
            not _contains_cache_segment(Path(root))
            for root in runtime_env.MODEL_ROOTS
        )
    finally:
        monkeypatch.delenv("DIAREMOT_MODEL_DIR", raising=False)
        _reload_runtime_env()
