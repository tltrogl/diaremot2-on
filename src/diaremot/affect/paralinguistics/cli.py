"""Command-line interface for paralinguistic feature extraction."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .benchmark import benchmark_performance_v2
from .config import get_config_preset
from .environment import LIBROSA_AVAILABLE, librosa
from .features import compute_segment_features_v2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paralinguistic Feature Extraction")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--text", type=str, help="Corresponding text")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end", type=float, help="End time (seconds)")
    parser.add_argument(
        "--preset",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality", "research"],
        help="Configuration preset",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args(argv)

    cfg = get_config_preset(args.preset)

    if args.benchmark:
        duration = 10.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        test_audio = (
            0.5 * np.sin(2 * np.pi * 200 * t)
            + 0.3 * np.sin(2 * np.pi * 400 * t)
            + 0.1 * np.random.normal(0, 1, len(t))
        ).astype(np.float32)
        test_text = "This is a test sentence for benchmarking the paralinguistic feature extraction system."
        results = benchmark_performance_v2(test_audio, sr, test_text, cfg=cfg)

        print("\n" + "=" * 60)
        print("PARALINGUISTICS BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Success Rate: {results.get('success_rate', 0):.1%}")
        print(f"Mean Processing Time: {results.get('mean_time_sec', 0):.3f}s")
        print(f"Performance Rating: {results.get('performance_rating', 'unknown')}")
        print(f"Features Extracted: {results.get('mean_features', 0):.0f}")

        libs = results.get("libraries_available", {})
        print("\nLibrary Status:")
        print(f"  Librosa: {'Y' if libs.get('librosa') else 'N'}")
        print(f"  SciPy: {'Y' if libs.get('scipy') else 'N'}")
        print(f"  Parselmouth: {'Y' if libs.get('parselmouth') else 'N'}")

        if args.output:
            with open(args.output, "w") as handle:
                json.dump(results, handle, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
        return 0

    if not args.audio:
        print("Please specify --audio for processing or --benchmark for testing")
        print("Use --help for full usage information")
        return 1

    try:
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio file loading")
        audio, sr = librosa.load(args.audio, sr=None)
        end_time = args.end if args.end else len(audio) / sr
        text: str | None
        if args.text and not args.text.strip():
            text = ""
        elif args.text and not args.text.endswith(('.txt', '.json')):
            text = args.text
        elif args.text:
            with open(args.text) as handle:
                text = handle.read().strip()
        else:
            text = ""

        print(f"Processing audio segment: {args.start:.2f}s to {end_time:.2f}s")
        start_time = time.time()
        features = compute_segment_features_v2(audio, sr, args.start, end_time, text, cfg)
        processing_time = time.time() - start_time

        print(f"\nProcessing completed in {processing_time:.3f}s")
        print("\nFeature Summary:")
        print("-" * 40)

        text_features = [
            "wpm",
            "sps",
            "filler_count",
            "repetition_count",
            "false_start_count",
            "disfluency_rate",
        ]
        audio_features = ["pause_count", "pause_ratio", "pitch_med_hz", "loudness_dbfs_med"]
        voice_features = ["vq_jitter_pct", "vq_shimmer_db", "vq_hnr_db", "vq_reliable"]

        print("Text Analysis:")
        for feat in text_features:
            if feat in features:
                value = features[feat]
                if isinstance(value, float) and not np.isnan(value):
                    print(f"  {feat}: {value:.2f}")
                elif isinstance(value, int):
                    print(f"  {feat}: {value}")

        print("\nAudio Analysis:")
        for feat in audio_features:
            if feat in features:
                value = features[feat]
                if isinstance(value, float) and not np.isnan(value):
                    print(f"  {feat}: {value:.2f}")

        print("\nVoice Quality:")
        for feat in voice_features:
            if feat in features:
                value = features[feat]
                if feat == "vq_reliable":
                    print(f"  {feat}: {value}")
                elif isinstance(value, float) and not np.isnan(value):
                    print(f"  {feat}: {value:.2f}")

        if args.output:
            with open(args.output, "w") as handle:
                json.dump(features, handle, indent=2, default=str)
            print(f"\nFull results saved to: {args.output}")
        return 0
    except Exception as exc:
        print(f"Error processing audio: {exc}")
        return 1


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())
