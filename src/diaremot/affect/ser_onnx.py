import os

import numpy as np
import onnxruntime as ort

ID2LABEL = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]


def _pick_provider():
    # DiaRemot is CPU-only. Ignore GPU/DML providers even if present.
    force = os.getenv("ORT_PROVIDER")
    if force:
        return [force]
    return ["CPUExecutionProvider"]


class SEROnnx:
    def __init__(self, onnx_path=None, threads=4):
        onnx_path = onnx_path or os.getenv(
            "DIAREMOT_SER_ONNX", r"D:\models\Affect\ser8-onnx-int8\ser8.int8.onnx"
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = _pick_provider()
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        print(f"[SER] ONNX loaded via providers={providers}: {onnx_path}")

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def predict_16k_f32(self, wav_f32: np.ndarray):
        if wav_f32.ndim > 1:
            wav_f32 = wav_f32.mean(axis=1)
        logits = self.sess.run([self.out], {self.inp: wav_f32[None, :].astype(np.float32)})[0][0]
        p = self._softmax(logits)
        i = int(p.argmax())
        return ID2LABEL[i], {ID2LABEL[j]: float(p[j]) for j in range(len(p))}
