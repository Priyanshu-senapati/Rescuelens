"""ONNX models: YOLO vehicle detector, ConvNeXt classifier (+ unknown-car score), classical baseline (numpy)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from . import classical

VEHICLE_CLASSES = (2, 5, 7)          # COCO car, bus, truck


def _session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2      # deployment box has 2 vCPUs
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Detector:
    """YOLO11 ONNX export: input 1x3x640x640 RGB/255, output 1x84x8400 (cx, cy, w, h, 80 class scores)."""

    def __init__(self, path: Path, size: int = 640, conf: float = 0.25):
        self.session, self.size, self.conf = _session(path), size, conf

    def __call__(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        pad_y, pad_x = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        out = self.session.run(None, {self.session.get_inputs()[0].name: x})[0][0]      # 84 x 8400
        scores = out[[4 + c for c in VEHICLE_CLASSES]].max(0)
        keep = scores > self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[:4, keep]
        scores = scores[keep]
        x1, y1 = (cx - bw / 2 - pad_x) / s, (cy - bh / 2 - pad_y) / s
        boxes = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw / s, bh / s)]
        idx = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self.conf, 0.45)
        found = []
        for i in np.array(idx).ravel():
            bx, by, bw_, bh_ = boxes[i]
            b = (int(max(0, bx)), int(max(0, by)), int(min(w, bx + bw_)), int(min(h, by + bh_)))
            if b[2] - b[0] >= 16 and b[3] - b[1] >= 16:      # drop slivers (e.g. boxes in the letterbox padding)
                found.append((b, float(scores[i])))
        if not found:
            return []
        # main vehicle = biggest of the confident boxes; the rest follow
        area = lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1])
        main = max([f for f in found if f[1] >= 0.4] or found, key=area)
        return [main] + [f for f in found if f is not main]


class Classifier:
    def __init__(self, model_dir: Path):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.class_names: list[str] = cfg["class_names"]
        self.size, self.temperature, self.k = cfg["img"], cfg["temperature"], cfg["knn_k"]
        self.unknown_threshold, self.hv_threshold = cfg["unknown_threshold"], cfg.get("hv_threshold", 0.1)
        self.mean, self.std = np.array(cfg["mean"], np.float32), np.array(cfg["std"], np.float32)
        int8 = model_dir / "classifier_int8.onnx"
        use_int8 = cfg.get("use_int8", True) and int8.exists()     # notebook sets use_int8 = False if INT8 lost > 1 point
        self.precision = "int8" if use_int8 else "fp32"
        self.session = _session(int8 if use_int8 else model_dir / "classifier_fp32.onnx")
        self.bank = np.load(model_dir / "train_emb.npy").astype(np.float32)          # L2-normalised train embeddings

    def __call__(self, car: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (calibrated class probabilities, unknown-car score)."""
        # same resize as training (torchvision Resize on a PIL image = PIL bilinear with antialiasing)
        x = np.asarray(Image.fromarray(car).resize((self.size, self.size), Image.BILINEAR), dtype=np.float32) / 255
        x = np.ascontiguousarray(((x - self.mean) / self.std).transpose(2, 0, 1)[None], dtype=np.float32)
        logits, emb = self.session.run(None, {"input": x})
        e = emb[0] / max(1e-8, float(np.linalg.norm(emb[0])))
        sims = np.sort(self.bank @ e)[-self.k:]
        return softmax(logits[0] / self.temperature), float(1 - sims.mean())


class Baseline:
    """Classical-only classifier: HOG + colour + LBP + geometry -> PCA -> logistic regression (trained in notebook 2)."""

    def __init__(self, path: Path):
        w = np.load(path)
        self.mean, self.scale, self.pca_mean = w["mean"], w["scale"], w["pca_mean"]
        self.components, self.coef, self.bias = w["components"], w["coef"], w["bias"]

    def __call__(self, car: np.ndarray) -> np.ndarray:
        z = ((classical.baseline_features(car) - self.mean) / self.scale - self.pca_mean) @ self.components.T
        return softmax(self.coef @ z + self.bias)
