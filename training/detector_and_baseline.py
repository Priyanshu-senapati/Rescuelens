"""RescueLens - detector evaluation (IoU, mAP) + YOLO ONNX export + classical-only baseline classifier.

Runs on Kaggle or locally. CPU is fine (GPU makes the detector part faster). Needs training/rl_data.py next to it.

Kaggle cell (Internet on, same Stanford Cars input as training):
    !wget -q https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/<sha>/training/rl_data.py
    !wget -q -O det.py https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/<sha>/training/detector_and_baseline.py
    %env QUICK=1
    %run -i det.py

Local (Windows, from the repo root):
    set DATA_ROOT=C:\\path\\to\\stanford_cars & set EXPORT_DIR=models & set QUICK=1
    training\\.venv\\Scripts\\python training\\detector_and_baseline.py
Outputs: yolo11s.onnx, baseline.npz, baseline_probs.npz, det_metrics.json
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
REPO_RAW = "https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/main"
if ON_KAGGLE:
    if __name__ == "__main__":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim"], check=False)
        urllib.request.urlretrieve(f"{REPO_RAW}/backend/app/classical.py", "classical.py")
    sys.path.insert(0, str(Path.cwd()))
else:
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent / "backend" / "app")]

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import classical                                   # the exact feature code the app uses
from rl_data import iou, load_stanford_cars, split_items

QUICK = os.environ.get("QUICK", "1") != "0"
SEED = 42
VAL_FRAC = 0.1
DET_N = 120 if QUICK else int(os.environ.get("DET_N", 8041 if torch.cuda.is_available() else 3000))
ROI_N = 80 if QUICK else 2000
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo11s.pt")   # or models/yolo11s_cars.pt (fine-tuned)
SKIP_BASELINE = os.environ.get("SKIP_BASELINE") == "1"        # only re-measure the detector, keep the classical results
IN = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
EXPORT = Path(os.environ.get("EXPORT_DIR", "/kaggle/working/export"))
WORKERS = int(os.environ.get("WORKERS", min(os.cpu_count() or 2, 4 if ON_KAGGLE else 8)))
assert cv2.__version__.startswith("4."), f"need OpenCV 4.x for HOG, got {cv2.__version__}"


def contour_box(path: str) -> tuple:
    """Classical localisation (no deep learning): largest edge contour -> bounding box."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    c = classical.body_contour(classical.edges(classical.enhance(rgb)))
    if c is None:
        return (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0]))
    x, y, w, h = cv2.boundingRect(c)
    return (float(x), float(y), float(x + w), float(y + h))


def feature(job: tuple) -> np.ndarray:
    path, bx = job
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return classical.baseline_features(classical.crop(rgb, tuple(int(v) for v in bx)))


def topk(probs, y, k):
    return float((np.argsort(-probs, 1)[:, :k] == y[:, None]).any(1).mean())


def main():
    EXPORT.mkdir(parents=True, exist_ok=True)
    train_items, test_items, _ = load_stanford_cars(IN)
    tr_items, va_items, te_items = split_items(train_items, test_items, VAL_FRAC, SEED, QUICK)
    rng = np.random.RandomState(SEED)

    # ---- deep-learning detector: YOLO11s, COCO-pretrained (car / bus / truck) or fine-tuned on Stanford Cars boxes
    from ultralytics import YOLO
    det = YOLO(YOLO_WEIGHTS)
    classes = [2, 5, 7] if len(det.names) == 80 else None
    det_idx = rng.choice(len(te_items), min(DET_N, len(te_items)), replace=False)
    device = 0 if torch.cuda.is_available() else "cpu"
    preds, ious, t0 = [], [], time.time()        # preds: (image id, confidence, box)
    for k, i in enumerate(det_idx):
        r = det.predict(str(te_items[i][0]), classes=classes, conf=0.05, imgsz=640, device=device, verbose=False)[0]
        boxes, confs = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        for b, c in zip(boxes, confs):
            preds.append((k, float(c), tuple(b)))
        ious.append(iou(tuple(boxes[confs.argmax()]), te_items[i][1]) if len(boxes) else 0.0)
    det_ms = (time.time() - t0) / len(det_idx) * 1000
    gts = [te_items[i][1] for i in det_idx]

    def average_precision(thr):
        """All-point interpolated AP, one ground-truth box per image."""
        matched, tp = set(), []
        for k, _, b in sorted(preds, key=lambda p: -p[1]):
            hit = k not in matched and iou(b, gts[k]) >= thr
            tp.append(1 if hit else 0)
            if hit:
                matched.add(k)
        tp = np.array(tp, float)
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec, prec = ctp / len(gts), ctp / np.maximum(ctp + cfp, 1e-9)
        mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        j = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[j + 1] - mrec[j]) * mpre[j + 1]))

    aps = {t: average_precision(t) for t in np.round(np.arange(0.5, 0.96, 0.05), 2)}
    metrics_path = EXPORT / "det_metrics.json"
    M = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    if classes is None and "detection" in M and "COCO" in M["detection"].get("model", ""):
        M["detection_pretrained"] = M["detection"]          # keep the before/after for the ablation table
    kind = "COCO pretrained, car/bus/truck" if classes else "fine-tuned on Stanford Cars train boxes"
    M["detection"] = {"model": f"yolo11s ({kind})", "n": len(det_idx),
                       "mean_iou": float(np.mean(ious)), "iou50_rate": float(np.mean(np.array(ious) >= 0.5)),
                       "map50": aps[0.5], "map50_95": float(np.mean(list(aps.values()))),
                       "ms_per_image": det_ms, "device": str(device)}
    print(json.dumps(M["detection"], indent=1), flush=True)
    exported = det.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    shutil.copy(exported, EXPORT / "yolo11s.onnx")
    if SKIP_BASELINE:
        metrics_path.write_text(json.dumps(M, indent=1))
        print("DONE (detector only)", flush=True)
        return

    # ---- classical localisation for comparison
    roi_idx = rng.choice(len(te_items), min(ROI_N, len(te_items)), replace=False)
    with Pool(WORKERS) as pool:
        roi_boxes = pool.map(contour_box, [str(te_items[i][0]) for i in roi_idx])
    roi_ious = [iou(b, te_items[i][1]) for b, i in zip(roi_boxes, roi_idx)]
    M["classical_roi"] = {"n": len(roi_idx), "mean_iou": float(np.mean(roi_ious)),
                          "iou50_rate": float(np.mean(np.array(roi_ious) >= 0.5))}
    print(json.dumps(M["classical_roi"], indent=1), flush=True)

    # ---- classical-only classifier: HOG + HSV colour + LBP texture + geometry -> PCA -> logistic regression
    def features(items):
        with Pool(WORKERS) as pool:
            return np.stack(pool.map(feature, [(str(p), bx) for p, bx, _ in items], chunksize=64))

    t0 = time.time()
    X_tr, X_va, X_te = features(tr_items), features(va_items), features(te_items)
    y_tr, y_va, y_te = (np.array([it[2] for it in items]) for items in (tr_items, va_items, te_items))
    print(f"features {X_tr.shape} in {time.time() - t0:.0f}s", flush=True)
    scaler = StandardScaler().fit(X_tr)
    pca = PCA(n_components=min(256, len(X_tr) - 1), random_state=SEED).fit(scaler.transform(X_tr))
    clf = LogisticRegression(C=0.5, max_iter=400).fit(pca.transform(scaler.transform(X_tr)), y_tr)

    def probs(X):  # QUICK subsets may not contain all 196 classes
        out = np.zeros((len(X), 196), np.float32)
        out[:, clf.classes_] = clf.predict_proba(pca.transform(scaler.transform(X)))
        return out

    P_va, P_te = probs(X_va), probs(X_te)
    M["classical_baseline"] = {"val_top1": topk(P_va, y_va, 1), "test_top1": topk(P_te, y_te, 1),
                               "test_top5": topk(P_te, y_te, 5), "feature_dim": int(X_tr.shape[1]), "pca": int(pca.n_components_)}
    print(json.dumps(M["classical_baseline"], indent=1), flush=True)

    # weights for the app (numpy): z = ((x - mean) / scale - pca_mean) @ components.T ; logits = coef @ z + bias
    coef = np.zeros((196, clf.coef_.shape[1]), np.float32); coef[clf.classes_] = clf.coef_
    bias = np.full(196, -1e4, np.float32); bias[clf.classes_] = clf.intercept_
    np.savez_compressed(EXPORT / "baseline.npz", mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
                        pca_mean=pca.mean_.astype(np.float32), components=pca.components_.astype(np.float32), coef=coef, bias=bias)
    np.savez_compressed(EXPORT / "baseline_probs.npz", val=P_va.astype(np.float16), val_labels=y_va,
                        test=P_te.astype(np.float16), test_labels=y_te)
    (EXPORT / "det_metrics.json").write_text(json.dumps(M, indent=1))
    for f in sorted(EXPORT.iterdir()):
        print(f"{f.name:24s} {f.stat().st_size / 2**20:7.1f} MB")
    print("DONE" + (" (QUICK - now run with QUICK=0)" if QUICK else ""), flush=True)


if __name__ == "__main__":
    main()
