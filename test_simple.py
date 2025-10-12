import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from diaremot.pipeline.speaker_diarization import _ECAPAWrapper
    print("Import OK")
    e = _ECAPAWrapper()
    print(f"Session loaded: {e.session is not None}")
    print(f"Model path searched")
except Exception as ex:
    print(f"ERROR: {ex}")
    import traceback
    traceback.print_exc()
