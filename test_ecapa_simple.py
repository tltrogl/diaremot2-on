import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

output = []
try:
    from diaremot.pipeline.speaker_diarization import _ECAPAWrapper
    output.append("Import OK")
    e = _ECAPAWrapper()
    output.append(f"Session loaded: {e.session is not None}")
    output.append(f"Input: {e.input_name}")
    output.append(f"Output: {e.output_name}")
    
    if e.session:
        # Test embedding extraction
        import numpy as np
        import librosa
        # Generate 1.5s of noise
        sr = 16000
        wav = np.random.randn(int(1.5 * sr)).astype(np.float32) * 0.01
        result = e.embed_batch([wav], sr)
        if result and result[0] is not None:
            emb = result[0]
            output.append(f"Embedding shape: {emb.shape}")
            output.append(f"Embedding mean: {emb.mean():.6f}")
            output.append(f"Embedding std: {emb.std():.6f}")
            output.append(f"Embedding norm: {np.linalg.norm(emb):.6f}")
        else:
            output.append("Embedding extraction FAILED")
    else:
        output.append("ECAPA session is None")
except Exception as ex:
    output.append(f"ERROR: {ex}")
    import traceback
    output.append(traceback.format_exc())

# Write to file
with open("D:\\diaremot\\diaremot2-on\\ecapa_test_output.txt", "w") as f:
    f.write("\n".join(output))
