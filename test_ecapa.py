#!/usr/bin/env python3
"""Quick ECAPA ONNX model loading test."""
import sys
import os

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diaremot.pipeline.speaker_diarization import _ECAPAWrapper

print("Testing ECAPA ONNX model loading...")
try:
    e = _ECAPAWrapper()
    print(f"✓ Session loaded: {e.session is not None}")
    print(f"  Input name: {e.input_name}")
    print(f"  Output name: {e.output_name}")
    
    if e.session is None:
        print("\n✗ ECAPA model failed to load!")
        print("  Check ECAPA_ONNX_PATH environment variable")
        print("  Or ensure model exists in default locations")
        sys.exit(1)
    else:
        print("\n✓ ECAPA ready")
        sys.exit(0)
        
except Exception as ex:
    print(f"✗ ECAPA load failed: {ex}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
