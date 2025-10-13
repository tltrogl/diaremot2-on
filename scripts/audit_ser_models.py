from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


SER_PATTERNS = (
    "ser8",
    os.path.join("affect", "ser8"),
)


@dataclass
class ModelInfo:
    path: Path
    size: int
    sha256: str
    n_outputs: Optional[int]
    n_classes: Optional[int]
    valid_ser8: bool


def iter_files(root: Path) -> Iterator[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".onnx"):
                yield Path(dirpath) / name


def is_ser_candidate(path: Path) -> bool:
    p = str(path).lower()
    return any(s in p for s in SER_PATTERNS) or "ser" in p


def sha256sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_inspect_onnx(path: Path) -> tuple[Optional[int], Optional[int]]:
    """Return (n_outputs, n_classes) when discoverable; otherwise (None, None).

    Uses onnxruntime if available to read output shapes. Falls back to graph inspection via onnx.
    """
    n_outputs: Optional[int] = None
    n_classes: Optional[int] = None
    try:
        import onnxruntime as ort  # type: ignore

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # noqa: FBT003
        outs = sess.get_outputs()
        n_outputs = len(outs)
        # Try to infer classes from the last dim of the first output
        if outs:
            shape = outs[0].shape  # may contain dynamic dims
            if isinstance(shape, (list, tuple)) and shape:
                last = shape[-1]
                if isinstance(last, int):
                    n_classes = last
    except Exception:
        try:
            import onnx  # type: ignore

            m = onnx.load(str(path))
            n_outputs = len(m.graph.output)
            # Try to read value info (may be absent)
            if m.graph.output:
                vi = m.graph.output[0]
                t = vi.type.tensor_type
                if t.HasField("shape") and t.shape.dim:
                    last = t.shape.dim[-1]
                    if last.HasField("dim_value"):
                        n_classes = int(last.dim_value)
        except Exception:
            pass
    return n_outputs, n_classes


def collect(root: Path) -> list[ModelInfo]:
    results: List[ModelInfo] = []
    for p in iter_files(root):
        if not is_ser_candidate(p):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        digest = sha256sum(p)
        n_out, n_cls = try_inspect_onnx(p)
        valid = (n_cls == 8) or (n_cls is None and "ser8" in str(p).lower())
        results.append(ModelInfo(path=p, size=size, sha256=digest, n_outputs=n_out, n_classes=n_cls, valid_ser8=valid))
    return results


def pick_preferred(infos: list[ModelInfo]) -> ModelInfo:
    # Heuristics:
    # 1) Prefer int8/uint8 quantized models by filename substring
    # 2) Else prefer files under an "affect/ser8" directory
    # 3) Else prefer larger file (often higher precision)
    def rank(mi: ModelInfo) -> tuple[int, int, int]:
        name = mi.path.name.lower()
        quant = int("int8" in name or "uint8" in name or "q8" in name)
        in_canonical = int(os.path.join("affect", "ser8") in str(mi.path).lower())
        return (quant, in_canonical, mi.size)

    return sorted(infos, key=rank, reverse=True)[0]


def group_by_hash(infos: list[ModelInfo]) -> dict[str, list[ModelInfo]]:
    out: dict[str, list[ModelInfo]] = {}
    for mi in infos:
        out.setdefault(mi.sha256, []).append(mi)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit SER (speech emotion) ONNX models and list duplicates.")
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("DIAREMOT_MODEL_DIR", "/srv/models")), help="Models root directory (default: $DIAREMOT_MODEL_DIR or /srv/models)")
    ap.add_argument("--json", action="store_true", help="Print JSON summary (keep and delete lists)")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"Models root not found: {root}")
        raise SystemExit(2)

    infos = collect(root)
    if not infos:
        print("No SER ONNX candidates found.")
        raise SystemExit(0)

    # Filter to only valid SER-8 or unknown-class-but-ser8-path
    valid = [mi for mi in infos if mi.valid_ser8]
    if not valid:
        print("Found SER ONNX files, but none look like 8-class models. Listing all candidates:")
        valid = infos

    # Group by content hash
    groups = group_by_hash(valid)

    keep: list[str] = []
    delete: list[str] = []
    picks: list[ModelInfo] = []
    for _h, group in groups.items():
        if not group:
            continue
        if len(group) == 1:
            picks.append(group[0])
            continue
        chosen = pick_preferred(group)
        picks.append(chosen)
        for g in group:
            if g.path != chosen.path:
                delete.append(str(g.path))

    keep = sorted(str(p.path) for p in picks)

    report = {
        "root": str(root),
        "keep": keep,
        "delete": sorted(delete),
        "candidates": [
            {
                "path": str(mi.path),
                "size": mi.size,
                "sha256": mi.sha256,
                "n_outputs": mi.n_outputs,
                "n_classes": mi.n_classes,
                "valid_ser8": mi.valid_ser8,
            }
            for mi in sorted(valid, key=lambda x: str(x.path))
        ],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Models root: {report['root']}")
        print("\nKeep (recommended):")
        for k in report["keep"]:
            print("  ", k)
        if report["delete"]:
            print("\nDuplicates to delete:")
            for d in report["delete"]:
                print("  ", d)
        print("\nAll SER candidates:")
        for c in report["candidates"]:
            print(f"  {c['path']}  size={c['size']}  classes={c['n_classes']}  hash={c['sha256'][:10]}...")


if __name__ == "__main__":
    main()

