import importlib
import sys
import types
from typing import Any


def _install_reportlab_stub() -> None:
    if "reportlab" in sys.modules:
        return

    reportlab = types.ModuleType("reportlab")

    lib = types.ModuleType("reportlab.lib")
    colors = types.SimpleNamespace(grey="#888888", lightgrey="#cccccc")
    lib.colors = colors

    pagesizes = types.ModuleType("reportlab.lib.pagesizes")
    pagesizes.letter = (612, 792)
    styles = types.ModuleType("reportlab.lib.styles")

    class ParagraphStyle:  # pragma: no cover - stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    styles.ParagraphStyle = ParagraphStyle

    units = types.ModuleType("reportlab.lib.units")
    units.inch = 72

    platypus = types.ModuleType("reportlab.platypus")

    class _SimpleStub:  # pragma: no cover - stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build(self, *_args, **_kwargs) -> None:
            return None

    class TableStyle(_SimpleStub):
        pass

    platypus.Paragraph = _SimpleStub
    platypus.SimpleDocTemplate = _SimpleStub
    platypus.Spacer = _SimpleStub
    platypus.Table = _SimpleStub
    platypus.TableStyle = TableStyle

    sys.modules["reportlab"] = reportlab
    sys.modules["reportlab.lib"] = lib
    sys.modules["reportlab.lib.colors"] = lib.colors
    sys.modules["reportlab.lib.pagesizes"] = pagesizes
    sys.modules["reportlab.lib.styles"] = styles
    sys.modules["reportlab.lib.units"] = units
    sys.modules["reportlab.platypus"] = platypus
    sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    scipy = types.ModuleType("scipy")
    scipy.__path__ = []  # type: ignore[attr-defined]
    scipy.signal = types.ModuleType("scipy.signal")
    scipy.signal.butter = lambda *_args, **_kwargs: ([], [])
    scipy.signal.filtfilt = lambda *_args, **_kwargs: []
    sys.modules.setdefault("scipy", scipy)
    sys.modules.setdefault("scipy.signal", scipy.signal)
    ndimage = types.ModuleType("scipy.ndimage")

    def _median_filter_stub(array, size=3):  # pragma: no cover - stub
        return array

    ndimage.median_filter = _median_filter_stub
    sys.modules.setdefault("scipy.ndimage", ndimage)

    soundfile = types.ModuleType("soundfile")

    class _SoundFileStub:  # pragma: no cover - stub
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, *_args, **_kwargs):
            return [], 16000

    class _Info:  # pragma: no cover - stub
        duration = 0.0

        def __init__(self, *_args, **_kwargs) -> None:
            pass

    soundfile.SoundFile = _SoundFileStub
    soundfile.Info = _Info
    soundfile.read = lambda *_args, **_kwargs: ([], 16000)
    soundfile.info = lambda *_args, **_kwargs: _Info()
    soundfile.write = lambda *_args, **_kwargs: None
    sys.modules.setdefault("soundfile", soundfile)


_install_reportlab_stub()

component_factory = importlib.import_module("src.diaremot.pipeline.component_factory")


class DummyLogger:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def info(self, *args, **kwargs):
        return None

    def warn(self, message, *args, **kwargs):  # pragma: no cover - logging is incidental
        self.warnings.append(message % args if args else message)

    def error(self, message, *args, **kwargs):
        rendered = message % args if args else message
        self.errors.append(rendered)


def _install_component_stubs(monkeypatch):
    class StubPreprocessor:
        def __init__(self, conf):
            self.conf = conf

    class StubDiarizationConfig:
        vad_threshold = 0.35
        vad_min_speech_sec = 0.80
        vad_min_silence_sec = 0.80
        speech_pad_sec = 0.10
        energy_gate_db = 30.0
        energy_hop_sec = 0.02

        def __init__(self, target_sr, registry_path=None, ahc_distance_threshold=0.15, **kwargs):
            self.target_sr = target_sr
            self.registry_path = registry_path
            self.ahc_distance_threshold = ahc_distance_threshold
            self.speaker_limit = kwargs.get("speaker_limit")

    class StubDiarizer:
        def __init__(self, conf):
            self.conf = conf

    class StubTranscriber:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubAutoTuner:
        pass

    class StubAffect:
        def __init__(self, *_args, **_kwargs) -> None:
            self.issues: list[str] = []

    class StubSed:
        available = True

        def __init__(self, *_args, **_kwargs):
            pass

    class StubHTML:
        pass

    class StubPDF:
        pass

    monkeypatch.setattr(component_factory, "AudioPreprocessor", StubPreprocessor)
    monkeypatch.setattr(
        component_factory._speaker_diarization,
        "DiarizationConfig",
        StubDiarizationConfig,
    )
    monkeypatch.setattr(
        component_factory._speaker_diarization,
        "SpeakerDiarizer",
        StubDiarizer,
    )
    monkeypatch.setattr(component_factory, "AutoTuner", StubAutoTuner)
    monkeypatch.setattr(component_factory, "HTMLSummaryGenerator", StubHTML)
    monkeypatch.setattr(component_factory, "PDFSummaryGenerator", StubPDF)
    monkeypatch.setattr(component_factory, "EmotionIntentAnalyzer", StubAffect)
    monkeypatch.setattr(component_factory, "PANNSEventTagger", StubSed)
    monkeypatch.setattr(component_factory, "SEDConfig", lambda: types.SimpleNamespace())

    stub_tx_module = types.ModuleType("transcription_module")
    stub_tx_module.AudioTranscriber = StubTranscriber
    monkeypatch.setitem(
        sys.modules,
        "src.diaremot.pipeline.transcription_module",
        stub_tx_module,
    )


def test_build_component_bundle_happy_path(monkeypatch):
    _install_component_stubs(monkeypatch)

    logger = DummyLogger()
    bundle = component_factory.build_component_bundle({}, logger)

    assert bundle.pre is not None
    assert bundle.diar is not None
    assert not logger.errors, f"unexpected errors: {logger.errors}"
    assert bundle.tx is not None
    assert bundle.pp_conf is not None
    assert bundle.config_snapshot["target_sr"] == bundle.pp_conf.target_sr
    assert bundle.models["transcriber"] == "StubTranscriber"
    assert not logger.errors


def test_build_component_bundle_handles_preprocessor_failure(monkeypatch):
    _install_component_stubs(monkeypatch)

    def failing_preprocessor(_conf):
        raise RuntimeError("boom")

    monkeypatch.setattr(component_factory, "AudioPreprocessor", failing_preprocessor)

    logger = DummyLogger()
    bundle = component_factory.build_component_bundle({}, logger)

    assert bundle.pre is None
    assert bundle.models["preprocessor"] == "NoneType"
    assert logger.errors[0].startswith("[preprocessor] initialization failed")
    assert "preprocessor initialization failed" in bundle.issues[0]
