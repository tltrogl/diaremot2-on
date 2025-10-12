"""
Dependency reality check utility for DiaRemot.

Features
- Builds a project import map from `src/` using AST (top-level modules only).
- Maps imports to installed distributions and to pinned specs
  (requirements.txt + pyproject.project.dependencies).
- Validates ABI-sensitive stacks:
  * torch ↔ torchaudio ↔ torchvision ↔ numpy
  * onnxruntime CPU only (no GPU provider)
  * faster-whisper ↔ ctranslate2
  * soundfile/libsndfile presence, pydub ↔ ffmpeg binary
- Fails fast when a third-party import is not pinned (repro hazard).

Usage
  python -m diaremot.tools.deps_check [--graph] [--json] [--strict]

Exit codes
  0: OK
  1: Any hard error (missing pins, ABI mismatch, import failures)
  2: Only warnings encountered (when not in --strict)
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    import tomllib  # py311+
except Exception:  # pragma: no cover - fallback for older Pythons
    tomllib = None  # type: ignore

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"


# Map import module → PyPI distribution name
IMPORT_TO_DIST: dict[str, str] = {
    # Core scientific
    "numpy": "numpy",
    "scipy": "scipy",
    "librosa": "librosa",
    "numba": "numba",
    "llvmlite": "llvmlite",
    "resampy": "resampy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    # PyTorch stack
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    # ASR / NLP
    "ctranslate2": "ctranslate2",
    "faster_whisper": "faster-whisper",
    "whisper": "openai-whisper",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "huggingface_hub": "huggingface_hub",
    # ONNX
    "onnxruntime": "onnxruntime",
    # Audio I/O and utils
    "soundfile": "soundfile",
    "audioread": "audioread",
    "av": "av",
    "pydub": "pydub",
    "ffmpeg": "ffmpeg-python",
    # Visualization / reporting
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "reportlab": "reportlab",
    # Web/HTML
    "jinja2": "jinja2",
    "bs4": "beautifulsoup4",
    # Utilities
    "tqdm": "tqdm",
    "joblib": "joblib",
    "psutil": "psutil",
    "packaging": "packaging",
    # Affect / SER
    "parselmouth": "praat-parselmouth",
    "panns_inference": "panns-inference",
}


STD_LIB_SENTINELS = set(sys.builtin_module_names)


def _is_stdlib(mod: str) -> bool:
    if mod in STD_LIB_SENTINELS:
        return True
    try:
        spec = find_spec(mod)
    except Exception:
        return False
    if spec is None or spec.origin is None:
        return False
    origin = str(spec.origin)
    # Heuristic: stdlib lives under base prefix's lib folder and not site-packages
    base = str(Path(sys.base_prefix))
    return ("site-packages" not in origin) and origin.startswith(base)


def _iter_py_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        # Skip virtual envs or packaging folders if accidentally under src
        if any(part in {".venv", "build", "dist", "packaging"} for part in path.parts):
            continue
        yield path


def build_import_map(root: Path = SRC_ROOT) -> dict[str, set[Path]]:
    """Parse Python files and collect top-level imported module names → file set."""
    imports: dict[str, set[Path]] = {}
    for file in _iter_py_files(root):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except Exception:
            # Skip unreadable/invalid files
            continue
        for node in ast.walk(tree):
            mod_name: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name.split(".")[0]
                    if mod_name:
                        _record_import(imports, mod_name, file)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue  # relative import
                if node.module:
                    mod_name = node.module.split(".")[0]
                    _record_import(imports, mod_name, file)
    # Remove local package imports
    imports.pop("diaremot", None)
    return imports


def _record_import(imports: dict[str, set[Path]], mod_name: str, file: Path) -> None:
    if not mod_name:
        return
    # Skip common stdlib prefixes early
    if mod_name in {
        "__future__",
        "typing",
        "pathlib",
        "dataclasses",
        "json",
        "os",
        "sys",
        "re",
        "math",
    }:
        return
    imports.setdefault(mod_name, set()).add(file)


def _load_pins_from_requirements(req_path: Path) -> dict[str, SpecifierSet]:
    pins: dict[str, SpecifierSet] = {}
    if not req_path.exists():
        return pins
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            # index-url or similar options
            continue
        # Drop inline comments
        line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
        try:
            req = Requirement(line)
        except Exception:
            continue
        name = canonicalize_name(req.name)
        pins[name] = req.specifier or SpecifierSet()
    return pins


def _load_pins_from_pyproject(pyproject_path: Path) -> dict[str, SpecifierSet]:
    pins: dict[str, SpecifierSet] = {}
    if not pyproject_path.exists() or tomllib is None:
        return pins
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", []) or []
    for dep in deps:
        try:
            req = Requirement(dep)
        except Exception:
            continue
        name = canonicalize_name(req.name)
        pins[name] = req.specifier or SpecifierSet()
    return pins


def _merge_pins(*dicts: dict[str, SpecifierSet]) -> dict[str, SpecifierSet]:
    merged: dict[str, SpecifierSet] = {}
    for d in dicts:
        merged.update(d)
    return merged


@dataclass
class Finding:
    level: str  # "error" | "warn" | "info"
    code: str
    message: str
    details: dict[str, Any] | None = None


def _canonical_dist_for_import(mod: str) -> str | None:
    # Explicit mapping first
    if mod in IMPORT_TO_DIST:
        return canonicalize_name(IMPORT_TO_DIST[mod])
    # Try installed distribution mapping
    try:
        mapping = importlib_metadata.packages_distributions()  # type: ignore[attr-defined]
        candidates = mapping.get(mod)
        if candidates:
            # Choose the shortest name (usually canonical)
            cand = sorted(candidates, key=len)[0]
            return canonicalize_name(cand)
    except Exception:
        pass
    return None


def _version_of(dist: str) -> str | None:
    try:
        return importlib_metadata.version(dist)
    except Exception:
        return None


def check_pins(imports: dict[str, set[Path]], pins: dict[str, SpecifierSet]) -> list[Finding]:
    findings: list[Finding] = []
    for mod, files in sorted(imports.items()):
        if _is_stdlib(mod):
            continue
        dist = _canonical_dist_for_import(mod)
        if dist is None:
            # Unknown module; attempt import to decide if third-party
            is_tp = not _is_stdlib(mod)
            if is_tp:
                findings.append(
                    Finding(
                        "warn",
                        "unmapped-module",
                        f"Cannot map import '{mod}' to a distribution; ensure it is pinned.",
                        {"module": mod, "files": [str(p) for p in files]},
                    )
                )
            continue
        if dist not in pins:
            findings.append(
                Finding(
                    "error",
                    "unPinned",
                    f"Imported third-party module '{mod}' maps to '{dist}' which is not pinned.",
                    {"module": mod, "dist": dist, "files": [str(p) for p in files]},
                )
            )
            continue
        # If pinned, ensure installed satisfies it (if installed)
        installed = _version_of(dist)
        spec = pins[dist]
        if installed is not None and spec and not spec.contains(installed, prereleases=True):
            findings.append(
                Finding(
                    "error",
                    "version-mismatch",
                    f"Installed {dist}=={installed} does not satisfy spec '{spec}'.",
                    {"dist": dist, "installed": installed, "spec": str(spec)},
                )
            )
    return findings


def check_abi_stacks() -> list[Finding]:
    findings: list[Finding] = []

    # Torch stack compatibility
    torch_v = _version_of("torch")
    ta_v = _version_of("torchaudio")
    tv_v = _version_of("torchvision")
    np_v = _version_of("numpy")

    def _base(v: str | None) -> tuple[str, str | None]:
        if not v:
            return ("", None)
        parts = v.split("+")
        core = parts[0]
        local = parts[1] if len(parts) > 1 else None
        return core, local

    t_core, t_local = _base(torch_v)
    ta_core, ta_local = _base(ta_v)
    tv_core, tv_local = _base(tv_v)

    def _mm(v: str | None) -> str:
        if not v:
            return ""
        try:
            ver = Version(v)
            return f"{ver.major}.{ver.minor}"
        except InvalidVersion:
            return v.split(".")[:2] and ".".join(v.split(".")[:2]) or ""

    if torch_v and ta_v and _mm(torch_v) != _mm(ta_v):
        findings.append(
            Finding(
                "error",
                "torch-stack-mismatch",
                f"torch {torch_v} and torchaudio {ta_v} minor versions differ.",
            )
        )
    if torch_v and tv_v:
        # Heuristic: torchvision minor should track torch minor with known mapping
        # 2.4 ↔ 0.19, 2.3 ↔ 0.18, 2.2 ↔ 0.17
        mapping = {"2.4": "0.19", "2.3": "0.18", "2.2": "0.17"}
        torch_mm = _mm(torch_v)
        tv_expect = mapping.get(torch_mm)
        tv_mm = _mm(tv_v)
        if tv_expect and tv_mm != tv_expect:
            findings.append(
                Finding(
                    "error",
                    "torchvision-mismatch",
                    f"torch {torch_v} expects torchvision ~{tv_expect}.x but found {tv_v}.",
                )
            )

    # Ensure CPU-only local tags (avoid CUDA wheels)
    for name, local in ("torch", t_local), ("torchaudio", ta_local), ("torchvision", tv_local):
        if local and "cu" in local.lower():
            findings.append(
                Finding(
                    "error",
                    "cuda-wheel-present",
                    f"{name} has CUDA local tag '+{local}' but environment must be CPU-only.",
                )
            )

    # onnxruntime: ensure CPU variant and providers
    try:
        gpu_ver = _version_of("onnxruntime-gpu")
        if gpu_ver is not None:
            findings.append(
                Finding(
                    "error",
                    "onnxruntime-gpu-installed",
                    "onnxruntime-gpu is installed; use CPU-only onnxruntime.",
                    {"onnxruntime-gpu": gpu_ver},
                )
            )
    except Exception:
        pass

    try:
        import onnxruntime as ort

        dev = getattr(ort, "get_device", lambda: "UNKNOWN")()
        providers = getattr(ort, "get_available_providers", lambda: [])()
        if dev != "CPU":
            findings.append(Finding("error", "ort-not-cpu", f"onnxruntime device is {dev}"))
        bad = [p for p in providers if "CUDA" in p or "ROCM" in p]
        if bad:
            findings.append(
                Finding(
                    "error",
                    "ort-gpu-providers",
                    f"GPU providers present in onnxruntime: {bad}",
                )
            )
    except Exception as exc:
        findings.append(Finding("error", "ort-check-failed", f"onnxruntime check failed: {exc}"))

    # faster-whisper ↔ ctranslate2 presence and pin
    fw_v = _version_of("faster-whisper")
    ct2_v = _version_of("ctranslate2")
    if fw_v and not ct2_v:
        findings.append(
            Finding("error", "ct2-missing", "ctranslate2 required by faster-whisper is missing.")
        )
    if fw_v and ct2_v:
        # Minimal floor mapping; adjust if contract changes
        # Known good pins from AGENTS_CLOUD.md: faster-whisper==1.1.0, ctranslate2==4.6.0
        try:
            if Version(ct2_v) < Version("4.0.0"):
                findings.append(
                    Finding(
                        "error",
                        "ct2-too-old",
                        f"ctranslate2 {ct2_v} is older than required baseline 4.0.0.",
                    )
                )
        except InvalidVersion:
            pass

    # soundfile/libsndfile check
    try:
        import soundfile as sf

        fmts = sf.available_formats()
        if not fmts:
            findings.append(
                Finding(
                    "error",
                    "sndfile-missing",
                    "soundfile loaded but libsndfile formats empty (likely missing backend).",
                )
            )
    except Exception as exc:
        findings.append(Finding("error", "soundfile-import", f"soundfile import failed: {exc}"))

    # pydub ↔ ffmpeg binary presence (warn only)
    try:
        from pydub.utils import which as _which

        if not _which("ffmpeg"):
            findings.append(
                Finding(
                    "warn",
                    "ffmpeg-not-found",
                    "pydub cannot find ffmpeg on PATH; some utilities may be unavailable.",
                )
            )
    except Exception:
        # pydub optional
        pass

    # numpy presence (simple sanity)
    if np_v is None:
        findings.append(Finding("error", "numpy-missing", "numpy is not installed"))

    return findings


def generate_report(graph: dict[str, set[Path]], pins: dict[str, SpecifierSet]) -> dict[str, Any]:
    pin_findings = check_pins(graph, pins)
    abi_findings = check_abi_stacks()
    return {
        "imports": {k: sorted(str(p) for p in v) for k, v in sorted(graph.items())},
        "findings": [f.__dict__ for f in pin_findings + abi_findings],
        "pins": {k: str(v) for k, v in sorted(pins.items())},
    }


def _load_all_pins() -> dict[str, SpecifierSet]:
    req = _load_pins_from_requirements(REPO_ROOT / "requirements.txt")
    pyp = _load_pins_from_pyproject(REPO_ROOT / "pyproject.toml")
    return _merge_pins(req, pyp)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DiaRemot dependency reality check")
    parser.add_argument("--graph", action="store_true", help="print import graph")
    parser.add_argument("--json", dest="as_json", action="store_true", help="emit JSON report")
    parser.add_argument("--strict", action="store_true", help="exit 1 on warnings as well")
    args = parser.parse_args(argv)

    graph = build_import_map(SRC_ROOT)
    pins = _load_all_pins()
    report = generate_report(graph, pins)

    errors = [f for f in report["findings"] if f["level"] == "error"]
    warns = [f for f in report["findings"] if f["level"] == "warn"]

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        if args.graph:
            # ASCII-only for Windows consoles
            print("== Import graph (module <- files) ==")
            for mod, files in sorted(report["imports"].items()):
                print(f"{mod}:")
                for f in files:
                    print(f"  - {f}")
            print("")
        print("== Findings ==")
        for f in report["findings"]:
            details = f.get("details")
            suffix = f" :: {details}" if details else ""
            print(f"[{f['level'].upper()}] {f['code']}: {f['message']}{suffix}")

    if errors:
        return 1
    if warns and args.strict:
        return 1
    return 0 if not warns else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
