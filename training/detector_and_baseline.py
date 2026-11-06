# %% [markdown]
# # RescueLens - detector evaluation + classical baseline (Kaggle)
#
# Runs fine on **CPU** (Accelerator = None), so it can run at the same time as the training notebook.
# 1. Settings: Internet = **On**. Add the **same Stanford Cars input** as the training notebook.
# 2. Run all with `QUICK = True` (a few minutes). Then `QUICK = False` -> Save Version -> Save & Run All (Commit).
# 3. Download the `export/` folder from the Output tab: `yolo11s.onnx`, `baseline.npz`, `baseline_probs.npz`, `det_metrics.json`.

# %%
import os
QUICK = os.environ.get("QUICK", "1") != "0"   # QUICK=1 (default): smoke test. QUICK=0: full run
SEED = 42
VAL_FRAC = 0.1
DET_N = 3000          # test images for detector metrics on CPU (all 8041 if a GPU is on)
ROI_N = 2000          # test images for the classical ROI (contour) localisation comparison
REPO_RAW = "https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/main"

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxslim"], check=False)

import csv, json, os, shutil, time, urllib.request
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IN = Path("/kaggle/input")
EXPORT = Path("/kaggle/working/export"); EXPORT.mkdir(parents=True, exist_ok=True)
WORKERS = os.cpu_count() or 2
urllib.request.urlretrieve(f"{REPO_RAW}/backend/app/classical.py", "/kaggle/working/classical.py")
sys.path.insert(0, "/kaggle/working")
import classical                                        # the exact feature code the app uses
if QUICK:
    DET_N, ROI_N = 120, 80
assert cv2.__version__.startswith("4."), f"need OpenCV 4.x for HOG, got {cv2.__version__}"

# %% [markdown]
# ## Load dataset (same loader + same split as the training notebook)

# %%
def find(pattern):
    hits = sorted(IN.rglob(pattern))
    return hits[0] if hits else None

def mat_rows(path):
    arr = np.atleast_1d(sio.loadmat(path, squeeze_me=True)["annotations"])
    return [{k: a[k] for k in arr.dtype.names} for a in arr]

def box(r):
    return tuple(float(r[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))

def split_of(p):
    """Nearest parent folder naming car_ims / test / train (ignores Kaggle's owner/slug prefix)."""
    for part in reversed(p.relative_to(IN).parts[:-1]):
        s = part.lower()
        if "car_ims" in s: return "car_ims"
        if "test" in s: return "test"
        if "train" in s: return "train"
    return None

train_imgs, test_imgs, carims = {}, {}, {}
for p in IN.rglob("*.jpg"):
    target = {"car_ims": carims, "test": test_imgs, "train": train_imgs}.get(split_of(p))
    if target is not None:
        target[p.name] = p

tr_mat, te_mat, full_mat = find("cars_train_annos.mat"), find("cars_test_annos_withlabels*.mat"), find("cars_annos.mat")
train_items, test_items = [], []
if tr_mat is not None and te_mat is not None and train_imgs and test_imgs:
    for r in mat_rows(tr_mat):
        if str(r["fname"]) in train_imgs: train_items.append((train_imgs[str(r["fname"])], box(r), int(r["class"]) - 1))
    for r in mat_rows(te_mat):
        if str(r["fname"]) in test_imgs: test_items.append((test_imgs[str(r["fname"])], box(r), int(r["class"]) - 1))
elif find("anno_train.csv") is not None and find("anno_test.csv") is not None and train_imgs and test_imgs:
    for csv_path, imgs, out in ((find("anno_train.csv"), train_imgs, train_items), (find("anno_test.csv"), test_imgs, test_items)):
        for row in csv.reader(open(csv_path)):
            if len(row) >= 6 and row[0] in imgs:
                out.append((imgs[row[0]], tuple(float(v) for v in row[1:5]), int(row[5]) - 1))
elif full_mat is not None and carims:
    for r in mat_rows(full_mat):
        name = Path(str(r["relative_im_path"])).name
        if name in carims: (test_items if int(r["test"]) else train_items).append((carims[name], box(r), int(r["class"]) - 1))
else:
    raise SystemExit("Stanford Cars images + labels not found - add the same input as the training notebook.")
print(f"train {len(train_items)} | test {len(test_items)}")

tr_idx, va_idx = train_test_split(np.arange(len(train_items)), test_size=VAL_FRAC,
                                  stratify=[it[2] for it in train_items], random_state=SEED)
tr_items, va_items, te_items = [train_items[i] for i in tr_idx], [train_items[i] for i in va_idx], list(test_items)
if QUICK:
    tr_items, va_items, te_items = tr_items[:600], va_items[:200], te_items[:300]

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

# %% [markdown]
# ## Deep-learning detector: YOLO11s (COCO-pretrained), vehicle classes car / bus / truck

# %%
from ultralytics import YOLO

det = YOLO("yolo11s.pt")
rng = np.random.RandomState(SEED)
det_idx = rng.choice(len(te_items), min(DET_N, len(te_items)), replace=False)
device = 0 if torch.cuda.is_available() else "cpu"
preds, ious, t0 = [], [], time.time()          # preds: (image id, confidence, box)
for k, i in enumerate(det_idx):
    r = det.predict(str(te_items[i][0]), classes=[2, 5, 7], conf=0.05, imgsz=640, device=device, verbose=False)[0]
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
        if hit: matched.add(k)
    tp = np.array(tp, float); ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / len(gts), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    j = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[j + 1] - mrec[j]) * mpre[j + 1]))

aps = {t: average_precision(t) for t in np.round(np.arange(0.5, 0.96, 0.05), 2)}
M = {"detection": {"model": "yolo11s (COCO pretrained, car/bus/truck)", "n": len(det_idx),
                   "mean_iou": float(np.mean(ious)), "iou50_rate": float(np.mean(np.array(ious) >= 0.5)),
                   "map50": aps[0.5], "map50_95": float(np.mean(list(aps.values()))),
                   "ms_per_image": det_ms, "device": str(device)}}
print(json.dumps(M["detection"], indent=1))

exported = det.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
shutil.copy(exported, EXPORT / "yolo11s.onnx")

# %% [markdown]
# ## Classical localisation (no deep learning): largest edge contour -> bounding box

# %%
def contour_box(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    c = classical.body_contour(classical.edges(classical.enhance(rgb)))
    if c is None:
        return (0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0]))
    x, y, w, h = cv2.boundingRect(c)
    return (float(x), float(y), float(x + w), float(y + h))

roi_idx = rng.choice(len(te_items), min(ROI_N, len(te_items)), replace=False)
with Pool(WORKERS) as pool:
    roi_boxes = pool.map(contour_box, [str(te_items[i][0]) for i in roi_idx])
roi_ious = [iou(b, te_items[i][1]) for b, i in zip(roi_boxes, roi_idx)]
M["classical_roi"] = {"n": len(roi_idx), "mean_iou": float(np.mean(roi_ious)), "iou50_rate": float(np.mean(np.array(roi_ious) >= 0.5))}
print(json.dumps(M["classical_roi"], indent=1))

# %% [markdown]
# ## Classical-only classifier: HOG + HSV colour + LBP texture + geometry -> PCA -> logistic regression

# %%
def feature(job):
    path, bx = job
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return classical.baseline_features(classical.crop(rgb, tuple(int(v) for v in bx)))

def features(items):
    with Pool(WORKERS) as pool:
        return np.stack(pool.map(feature, [(str(p), bx) for p, bx, _ in items], chunksize=64))

t0 = time.time()
X_tr, X_va, X_te = features(tr_items), features(va_items), features(te_items)
y_tr, y_va, y_te = (np.array([it[2] for it in items]) for items in (tr_items, va_items, te_items))
print(f"features {X_tr.shape} in {time.time() - t0:.0f}s")

scaler = StandardScaler().fit(X_tr)
pca = PCA(n_components=min(256, len(X_tr) - 1), random_state=SEED).fit(scaler.transform(X_tr))
Z = lambda X: pca.transform(scaler.transform(X))
clf = LogisticRegression(C=0.5, max_iter=400).fit(Z(X_tr), y_tr)

def topk(probs, y, k):
    return float((np.argsort(-probs, 1)[:, :k] == y[:, None]).any(1).mean())

def full(P):   # QUICK subsets may not contain all 196 classes
    out = np.zeros((len(P), 196), np.float32); out[:, clf.classes_] = P; return out

P_va, P_te = full(clf.predict_proba(Z(X_va))), full(clf.predict_proba(Z(X_te)))
M["classical_baseline"] = {"val_top1": topk(P_va, y_va, 1), "test_top1": topk(P_te, y_te, 1),
                           "test_top5": topk(P_te, y_te, 5), "feature_dim": int(X_tr.shape[1]), "pca": int(pca.n_components_)}
print(json.dumps(M["classical_baseline"], indent=1))

# weights for the app (pure numpy at inference): z = ((x - mean) / scale - pca_mean) @ components.T ; logits = z @ coef.T + b
coef = np.zeros((196, clf.coef_.shape[1]), np.float32); coef[clf.classes_] = clf.coef_
bias = np.full(196, -1e4, np.float32); bias[clf.classes_] = clf.intercept_
np.savez_compressed(EXPORT / "baseline.npz", mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
                    pca_mean=pca.mean_.astype(np.float32), components=pca.components_.astype(np.float32), coef=coef, bias=bias)
np.savez_compressed(EXPORT / "baseline_probs.npz", val=P_va.astype(np.float16), val_labels=y_va,
                    test=P_te.astype(np.float16), test_labels=y_te)
(EXPORT / "det_metrics.json").write_text(json.dumps(M, indent=1))
for f in sorted(EXPORT.iterdir()):
    print(f"{f.name:24s} {f.stat().st_size / 2**20:7.1f} MB")
print("DONE" + (" (QUICK - now set QUICK = False and Save & Run All)" if QUICK else ""))
