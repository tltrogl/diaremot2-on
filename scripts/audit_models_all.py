from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


# Canonical targets this pipeline actually uses
CANONICAL = {
    "silero_vad": {
        "patterns": ("silero_vad.onnx", os.path.join("silero", "vad.onnx")),
    },
    "ecapa": {
        "patterns": (
            os.path.join("ecapa-onnx", "ecapa_tdnn.onnx"),
            os.path.join("Diarization", "ecapa-onnx", "ecapa_tdnn.onnx"),
            "ecapa_tdnn.onnx",
        ),
    },
    "panns_cnn14": {
        "patterns": ("panns_cnn14.onnx", "cnn14.onnx"),
        "label_candidates": ("labels.csv", "class_labels_indices.csv"),
    },
    "ser8": {
        "patterns": (os.path.join("affect", "ser8"),),  # directory hint
        "file_glob": ".onnx",
    },
    "affect_vad": {
        "patterns": (os.path.join("affect", "vad"),),
        "file_glob": ".onnx",
    },
    "faster_whisper": {
        "patterns": (os.path.join("faster-whisper", "tiny.en"),),  # directory unit
        "dir_unit": True,
    },
}


@dataclass
class FileInfo:
    path: Path
    size: int
    sha256: str
    kind: str  # e.g. 'silero_vad', 'ecapa', 'panns_cnn14', 'ser8', 'affect_vad', 'faster_whisper', or 'other'


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name


def classify(path: Path) -> str:
    p = str(path).replace("\\", "/").lower()
    # Exact dirs handled separately (faster-whisper)
    if p.endswith(".onnx"):
        if "/silero/" in p or p.endswith("/silero_vad.onnx"):
            return "silero_vad"
        if "ecapa" in p and p.endswith("ecapa_tdnn.onnx"):
            return "ecapa"
        if "cnn14" in p or "panns_cnn14" in p:
            return "panns_cnn14"
        if "/affect/ser8/" in p or "/ser8/" in p or "/ser/" in p:
            return "ser8"
        if "/affect/vad/" in p:
            return "affect_vad"
    return "other"


def collect_onnx(root: Path) -> list[FileInfo]:
    items: list[FileInfo] = []
    for p in iter_files(root):
        if not p.is_file():
            continue
        kind = classify(p)
        if kind == "other":
            # Include only model-like binaries for 'other'
            if not p.suffix.lower() in {".onnx", ".bin", ".json", ".vocab", ".model"}:
                continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        # Avoid hashing non-onnx large data unless it's faster-whisper (handled separately)
        digest = sha256sum(p) if p.suffix.lower() == ".onnx" else ""
        items.append(FileInfo(path=p, size=size, sha256=digest, kind=kind))
    return items


def dir_hash(dirpath: Path) -> str:
    """Compute a content hash for a directory (file names + contents)."""
    h = hashlib.sha256()
    for p in sorted([q for q in dirpath.rglob("*") if q.is_file()]):
        rel = str(p.relative_to(dirpath)).encode()
        h.update(rel)
        try:
            with p.open("rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
        except OSError:
            continue
    return h.hexdigest()


def find_fw_dirs(root: Path) -> list[tuple[Path, str, int]]:
    out: list[tuple[Path, str, int]] = []
    # Look for faster-whisper/<name> containing model.bin
    base = root / "faster-whisper"
    if base.exists():
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "model.bin").exists():
                dh = dir_hash(sub)
                size = sum((f.stat().st_size for f in sub.rglob("*") if f.is_file()), 0)
                out.append((sub, dh, size))
    return out


def best_by_group(files: list[FileInfo]) -> list[FileInfo]:
    # Group by sha256 for ONNX; pick preferred by quantized name then size
    groups: dict[str, list[FileInfo]] = {}
    for fi in files:
        key = fi.sha256 if fi.sha256 else f"{fi.kind}:{fi.path.name}:{fi.size}"
        groups.setdefault(key, []).append(fi)

    picks: list[FileInfo] = []
    for _key, group in groups.items():
        if len(group) == 1:
            picks.append(group[0])
            continue
        def rank(g: FileInfo) -> tuple[int, int]:
            name = g.path.name.lower()
            quant = int("int8" in name or "uint8" in name or "q8" in name)
            return (quant, g.size)
        group_sorted = sorted(group, key=rank, reverse=True)
        picks.append(group_sorted[0])
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit all models under a root; keep used, suggest duplicates to delete.")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DIAREMOT_MODEL_DIR", "/srv/models")), help="Models root directory")
    ap.add_argument("--json", action="store_true", help="Print JSON summary")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"Models root not found: {root}")
        raise SystemExit(2)

    # Collect ONNX-like files
    files = collect_onnx(root)

    # Choose preferred per duplicate group only for the kinds we care about
    keep_candidates = [f for f in files if f.kind in {"silero_vad", "ecapa", "panns_cnn14", "ser8", "affect_vad"}]
    keep_picks = best_by_group(keep_candidates)
    keep_paths = {str(p.path.resolve()) for p in keep_picks}

    # Ensure label file near CNN14
    for kp in keep_picks:
        if kp.kind != "panns_cnn14":
            continue
        # Try labels nearby
        for name in CANONICAL["panns_cnn14"]["label_candidates"]:
            candidate = kp.path.parent / name
            if candidate.exists():
                keep_paths.add(str(candidate.resolve()))

    # Faster-Whisper directories: keep tiny.en if present; dedupe directories by dir hash
    fw_dirs = find_fw_dirs(root)
    # Prefer tiny.en; if not present, keep the largest dir once per hash
    by_hash: dict[str, list[tuple[Path, str, int]]] = {}
    for sub, h, size in fw_dirs:
        by_hash.setdefault(h, []).append((sub, h, size))
    for h, group in by_hash.items():
        preferred: Optional[Path] = None
        for sub, _hh, _size in group:
            if sub.name.lower() == "tiny.en":
                preferred = sub
                break
        if preferred is None:
            preferred = sorted(group, key=lambda x: x[2], reverse=True)[0][0]
        # Keep the entire directory contents
        for f in preferred.rglob("*"):
            if f.is_file():
                keep_paths.add(str(f.resolve()))

    all_files = sorted(str(p.resolve()) for p in root.rglob("*") if p.is_file())
    delete = [p for p in all_files if p not in keep_paths]
    keep = sorted(keep_paths)

    report = {
        "root": str(root),
        "keep": keep,
        "delete": delete,
        "stats": {
            "total_files": len(all_files),
            "keep_files": len(keep),
            "delete_files": len(delete),
            "fw_dirs": [str(d[0]) for d in fw_dirs],
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Models root: {report['root']}")
        print(f"Total files: {report['stats']['total_files']}")
        print("\nKeep (in use / preferred):")
        for k in report["keep"]:
            print("  ", k)
        if report["delete"]:
            print("\nCandidates to delete:")
            for d in report["delete"]:
                print("  ", d)


if __name__ == "__main__":
    main()

