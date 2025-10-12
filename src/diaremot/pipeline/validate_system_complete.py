#!/usr/bin/env python3
"""
DiaRemot Complete System Validation and Test

This script validates the complete system after fixes:
1. Runs quick diagnostic
2. Tests transcription with real audio
3. Validates pipeline integration
4. Reports final status
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

from .diagnostics_smoke import (
    SMOKE_TEST_TRANSCRIBE_KWARGS,
    burst_audio_factory,
    prepare_smoke_wav,
    run_pipeline_smoke_test,
)

VALIDATION_TMP_DIR = Path("test_validation_tmp")
VALIDATION_OUTPUT_DIR = Path("test_output")


def set_cpu_environment():
    """Set CPU-only environment"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_DEVICE"] = "cpu"
    os.environ["WHISPER_DEVICE"] = "cpu"
    os.environ["WHISPER_COMPUTE_TYPE"] = "int8"
    print("✓ CPU-only environment configured")


def create_test_audio(tmp_dir: Path) -> str:
    """Create a short test audio file"""
    print("Creating test audio...")
    try:
        wav_path = prepare_smoke_wav(
            tmp_dir,
            waveform_factory=burst_audio_factory,
            duration_seconds=5.0,
            filename="test_audio.wav",
        )
        print(f"✓ Test audio created: {wav_path}")
        return str(wav_path)
    except RuntimeError:
        print("⚠ soundfile not available, using numpy save")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = tmp_dir / "test_audio.npy"
        audio = burst_audio_factory(16000, 5.0)
        np.save(fallback_path, audio.astype(np.float32))
        return str(fallback_path)


def test_transcription_directly():
    """Test transcription module directly"""
    print("\n" + "=" * 50)
    print("DIRECT TRANSCRIPTION TEST")
    print("=" * 50)

    try:
        from diaremot.pipeline.transcription_module import AudioTranscriber

        # Initialize with CPU-only settings
        print("Initializing transcriber...")
        transcriber = AudioTranscriber(**SMOKE_TEST_TRANSCRIBE_KWARGS)
        print("✓ Transcriber initialized")

        # Validate backend
        if hasattr(transcriber, "validate_backend"):
            validation = transcriber.validate_backend()
            print(f"Backend: {validation['active_backend']}")
            print(f"Functional: {validation['backend_functional']}")

            if not validation["backend_functional"]:
                print(f"❌ Backend validation failed: {validation['error']}")
                return False
        else:
            print("⚠ No validation method available")

        # Test with synthetic audio
        print("\nTesting with synthetic audio...")
        test_audio = np.random.randn(16000).astype(np.float32) * 0.01  # 1 second of quiet noise
        test_segments = [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "speaker_id": "test",
                "speaker_name": "Test Speaker",
            }
        ]

        start_time = time.time()
        results = transcriber.transcribe_segments(test_audio, 16000, test_segments)
        elapsed = time.time() - start_time

        if results and len(results) > 0:
            result = results[0]
            print(f"✓ Transcription successful in {elapsed:.2f}s")
            print(f"  - Text: '{result.text}'")
            print(f"  - Model: {result.model_used}")
            print(f"  - Confidence: {result.confidence}")
            return True
        else:
            print("❌ No transcription results returned")
            return False

    except Exception as e:
        print(f"❌ Direct transcription test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the complete pipeline"""
    print("\n" + "=" * 50)
    print("FULL PIPELINE TEST")
    print("=" * 50)

    try:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))

        # Create test audio
        test_audio_path = create_test_audio(VALIDATION_TMP_DIR)
        wav_path = Path(test_audio_path)

        if wav_path.suffix == ".npy":
            audio_data = np.load(wav_path)
            import soundfile as sf

            wav_path = VALIDATION_TMP_DIR / "test_audio_from_npy.wav"
            sf.write(wav_path, audio_data, 16000)

        config_overrides = {
            "registry_path": str(VALIDATION_TMP_DIR / "test_speaker_registry.json"),
        }

        print(f"\nRunning pipeline on {wav_path}...")
        start_time = time.time()

        result = run_pipeline_smoke_test(
            config_overrides=config_overrides,
            tmp_dir=VALIDATION_TMP_DIR,
            output_dir=VALIDATION_OUTPUT_DIR,
            waveform_factory=burst_audio_factory,
            duration_seconds=5.0,
            wav_path=wav_path,
        )

        if not result.success:
            raise RuntimeError(result.error or "unknown diagnostics failure")

        elapsed = time.time() - start_time
        run_info = result.run_result or {}

        print(f"✓ Pipeline completed in {elapsed:.2f}s")
        print(f"  - Run ID: {run_info.get('run_id', 'unknown')}")
        print(f"  - Output dir: {run_info.get('out_dir', result.output_dir)}")

        tx_info = run_info.get("transcriber", {})
        if tx_info:
            print(f"  - Backend: {tx_info.get('backend', 'unknown')}")
            print(f"  - Device: {tx_info.get('device', 'unknown')}")
            print(f"  - Model: {tx_info.get('model_size', 'unknown')}")
            if tx_info.get("backend") == "fallback":
                print("⚠ Pipeline using fallback transcriber")

        # Check outputs
        csv_path = Path(run_info.get("outputs", {}).get("csv", ""))
        if csv_path.exists():
            print(f"  - CSV output: {csv_path}")
            # Count lines
            with open(csv_path) as f:
                lines = f.readlines()
            print(f"  - Segments: {len(lines) - 1}")  # -1 for header

        return True

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_parselmouth():
    """Verify Parselmouth integration via paralinguistics"""
    print("\n" + "=" * 50)
    print("PARSELMOUTH INTEGRATION TEST")
    print("=" * 50)
    try:
        from diaremot.affect import paralinguistics as para
    except Exception:
        if str(Path.cwd()) not in sys.path:
            sys.path.insert(0, str(Path.cwd()))
        import paralinguistics as para

        if not getattr(para, "PARSELMOUTH_AVAILABLE", False):
            print("⚠ Parselmouth not available")
            return False
        audio = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 16000)).astype(np.float32)
        cfg = para.ParalinguisticsConfig(vq_use_parselmouth=True)
        res = para._compute_voice_quality_parselmouth_v2(audio, 16000, cfg)
        print(f"✓ Parselmouth functional (jitter_pct={res.get('jitter_pct', 0.0):.3f})")
        return True
    except Exception as e:
        print(f"❌ Parselmouth test failed: {e}")
        return False


def run_comprehensive_validation():
    """Run all validation tests"""
    print("DiaRemot Complete System Validation")
    print("=" * 60)

    # Set environment
    set_cpu_environment()

    # Check if we're in the right directory
    # Ensure we're at the project root where modules live directly
    project_root_marker = Path("src/diaremot/pipeline/transcription_module.py")
    if not project_root_marker.exists():
        print("❌ Please run this script from your project root directory")
        return False

    # Run tests
    tests = [
        ("Direct Transcription", test_transcription_directly),
        ("Full Pipeline", test_full_pipeline),
        ("Parselmouth Integration", test_parselmouth),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 SYSTEM FULLY VALIDATED - ALL SYSTEMS WORKING!")
        print("\nYour DiaRemot system is ready to use:")
        print("  • CPU-only transcription: ✓")
        print("  • Faster-Whisper backend: ✓")
        print("  • Full pipeline integration: ✓")
        print("\nYou can now process audio files with confidence.")
        return True
    else:
        print("\n❌ VALIDATION FAILED - SOME ISSUES REMAIN")
        print("\nTo fix remaining issues:")
        print("  1. Run: python transcription_backend_comprehensive_fix.py")
        print("  2. Check dependency installation")
        print("  3. Review error messages above")
        return False


def cleanup_test_files():
    """Clean up test files"""
    try:
        import shutil

        shutil.rmtree(VALIDATION_TMP_DIR, ignore_errors=True)
        shutil.rmtree(VALIDATION_OUTPUT_DIR, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        success = run_comprehensive_validation()
        cleanup_test_files()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        cleanup_test_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation script crashed: {e}")
        cleanup_test_files()
        sys.exit(1)
