# %% [markdown]
# # RescueLens - classifier training (Kaggle)
#
# 1. **Settings** (right panel): Accelerator = **GPU T4 x2** (not P100: recent PyTorch may not support it), Internet = **On**.
# 2. **Add Input** -> search `stanford cars` -> add a copy that has the images **and**
#    `cars_test_annos_withlabels.mat` (test labels). If the first cell says test labels are missing,
#    add a second dataset that contains that file. The loader searches all of `/kaggle/input`.
# 3. Run all with `QUICK = True` (about 5 min smoke test).
# 4. If it finishes, set `QUICK = False` -> **Save Version -> Save & Run All (Commit)**.
#    It keeps running if you close the tab. Takes about 2.5-3 h on a T4.
# 5. When it's done: version page -> **Output** -> download the `export/` folder.

# %%
import os
QUICK = os.environ.get("QUICK", "1") != "0"   # QUICK=1 (default): smoke test, few hundred images, 1 epoch. QUICK=0: full run
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k_384"
IMG = 384
EPOCHS = 25
BATCH = 32
LR = 2e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_FRAC = 0.1
TIME_BUDGET_MIN = 170            # stop training early (keeping the best epoch) after this many minutes
SEED = 42
ROBUST_N = 2000                  # test images used for robustness tests
ONNX_EVAL_N = 300                # test images used to compare FP32 vs INT8 on CPU
CLASSNAMES_URL = "https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/main/data/classnames.txt"

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9", "onnx", "onnxruntime", "onnxscript"], check=False)

import json, math, os, random, time, urllib.request
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
assert torch.cuda.is_available(), "No GPU. Settings -> Accelerator -> GPU."
DEV = "cuda"
WORKERS = os.cpu_count() or 2
IN = Path("/kaggle/input")
EXPORT = Path("/kaggle/working/export"); EXPORT.mkdir(parents=True, exist_ok=True)
CACHE = Path("/tmp/cache")
if QUICK:
    EPOCHS, ROBUST_N, ONNX_EVAL_N = 1, 100, 40
print("torch", torch.__version__, "| timm", timm.__version__, "| GPU", torch.cuda.get_device_name(0), "| workers", WORKERS)

# %% [markdown]
# ## Find the dataset (works with the devkit layout or the full `cars_annos.mat` layout)

# %%
def find(pattern):
    hits = sorted(IN.rglob(pattern))
    return hits[0] if hits else None

def mat_rows(path):
    arr = np.atleast_1d(sio.loadmat(path, squeeze_me=True)["annotations"])
    return [{k: a[k] for k in arr.dtype.names} for a in arr]

def box(r):
    return tuple(float(r[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))

meta = find("cars_meta.mat")
if meta is not None:
    CLASS_NAMES = [str(x) for x in sio.loadmat(meta, squeeze_me=True)["class_names"]]
else:
    CLASS_NAMES = urllib.request.urlopen(CLASSNAMES_URL).read().decode().strip().splitlines()
assert len(CLASS_NAMES) == 196, f"expected 196 class names, got {len(CLASS_NAMES)}"

# index every jpg by (split folder, file name); train and test both use names like 00001.jpg
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
train_items, test_items = [], []          # (image path, box, label 0-195)
if tr_mat is not None and te_mat is not None and train_imgs and test_imgs:
    for r in mat_rows(tr_mat):
        if str(r["fname"]) in train_imgs:
            train_items.append((train_imgs[str(r["fname"])], box(r), int(r["class"]) - 1))
    for r in mat_rows(te_mat):
        if str(r["fname"]) in test_imgs:
            test_items.append((test_imgs[str(r["fname"])], box(r), int(r["class"]) - 1))
elif find("anno_train.csv") is not None and find("anno_test.csv") is not None and train_imgs and test_imgs:
    import csv   # "stanford-car-dataset-by-classes-folder" layout: rows = fname, x1, y1, x2, y2, class (1-based)
    for csv_path, imgs, out in ((find("anno_train.csv"), train_imgs, train_items), (find("anno_test.csv"), test_imgs, test_items)):
        for row in csv.reader(open(csv_path)):
            if len(row) >= 6 and row[0] in imgs:
                out.append((imgs[row[0]], tuple(float(v) for v in row[1:5]), int(row[5]) - 1))
elif full_mat is not None and carims:
    for r in mat_rows(full_mat):
        name = Path(str(r["relative_im_path"])).name
        if name in carims:
            (test_items if int(r["test"]) else train_items).append((carims[name], box(r), int(r["class"]) - 1))
else:
    found = sorted(str(p.relative_to(IN)) for p in IN.rglob("*.mat"))
    raise SystemExit(
        "Could not find images + labels.\n"
        "Need either (cars_train_annos.mat + cars_test_annos_withlabels.mat + cars_train/ + cars_test/ images)\n"
        "or (cars_annos.mat + car_ims/ images).\n"
        f".mat files found: {found}\n"
        f"jpgs found: train={len(train_imgs)} test={len(test_imgs)} car_ims={len(carims)}\n"
        "Fix: Add Input -> add a Stanford Cars copy that includes cars_test_annos_withlabels.mat."
    )
print(f"train {len(train_items)} (expect 8144) | test {len(test_items)} (expect 8041)")
assert len(train_items) > 8000 and len(test_items) > 8000, "Dataset incomplete - check the inputs above"

labels_all = [it[2] for it in train_items]
tr_idx, va_idx = train_test_split(np.arange(len(train_items)), test_size=VAL_FRAC, stratify=labels_all, random_state=SEED)
tr_items = [train_items[i] for i in tr_idx]
va_items = [train_items[i] for i in va_idx]
te_items = list(test_items)
if QUICK:
    tr_items, va_items, te_items = tr_items[:600], va_items[:200], te_items[:300]
print(f"split -> train {len(tr_items)} | val {len(va_items)} | test {len(te_items)}")

# %% [markdown]
# ## Crop each car (ground-truth box + 8% margin) and cache small copies so training isn't I/O bound

# %%
CACHE_SIDE = int(IMG * 1.15)

def crop_resize(job):
    src, bx, dst, crop = job
    if os.path.exists(dst):
        return
    im = Image.open(src).convert("RGB")
    if crop:
        x1, y1, x2, y2 = bx
        mw, mh = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
        im = im.crop((int(max(0, x1 - mw)), int(max(0, y1 - mh)), int(min(im.width, x2 + mw)), int(min(im.height, y2 + mh))))
    s = CACHE_SIDE / min(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    im.save(dst, quality=95)

def cache(items, split, crop=True):
    d = CACHE / split; d.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), bx, str(d / f"{i}.jpg"), crop) for i, (p, bx, _) in enumerate(items)]
    with Pool(WORKERS) as pool:
        pool.map(crop_resize, jobs, chunksize=64)
    return [j[2] for j in jobs], [it[2] for it in items]

t0 = time.time()
tr_paths, tr_labels = cache(tr_items, "train")
va_paths, va_labels = cache(va_items, "val")
te_paths, te_labels = cache(te_items, "test")
tf_paths, _ = cache(te_items, "test_full", crop=False)
print(f"cached in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Model, data, augmentation

# %%
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=196, drop_path_rate=0.1)
MEAN, STD = list(model.pretrained_cfg["mean"]), list(model.pretrained_cfg["std"])
model = model.to(DEV).to(memory_format=torch.channels_last)

class RandomDark:
    """Simulates night / underexposed scenes."""
    def __init__(self, p): self.p = p
    def __call__(self, im):
        return ImageEnhance.Brightness(im).enhance(random.uniform(0.25, 0.7)) if random.random() < self.p else im

train_tf = T.Compose([
    T.RandomResizedCrop(IMG, scale=(0.6, 1.0), ratio=(0.75, 2.0)),
    T.RandomHorizontalFlip(),
    RandomDark(0.2),
    T.ColorJitter(0.2, 0.2, 0.2, 0.02),
    T.ToTensor(), T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),       # partial occlusion
])
eval_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(self, paths, labels, tf, degrade=None):
        self.paths, self.labels, self.tf, self.degrade = paths, labels, tf, degrade
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.degrade is not None:
            im = self.degrade(im, i)
        return self.tf(im), self.labels[i]

dl_tr = DataLoader(DS(tr_paths, tr_labels, train_tf), BATCH, shuffle=True, num_workers=WORKERS,
                   pin_memory=True, drop_last=True, persistent_workers=True)

@torch.no_grad()
def predict(ds, tta=False, bs=64):
    """Returns (logits float32 [N,196], embeddings float16 [N,D])."""
    model.eval()
    L, E = [], []
    for x, _ in DataLoader(ds, bs, shuffle=False, num_workers=WORKERS, pin_memory=True):
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            e = model.forward_head(model.forward_features(x), pre_logits=True)
            lg = model.get_classifier()(e)
            if tta:
                lg = (lg + model(torch.flip(x, dims=[3]))) / 2
        L.append(lg.float().cpu()); E.append(e.half().cpu())
    return torch.cat(L).numpy(), torch.cat(E).numpy()

def topk(logits, labels, k):
    return float((np.argsort(-logits, 1)[:, :k] == np.asarray(labels)[:, None]).any(1).mean())

# %% [markdown]
# ## Train (saves the best epoch by validation accuracy)

# %%
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps, warm = EPOCHS * len(dl_tr), len(dl_tr)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
scaler = torch.amp.GradScaler("cuda")
crit = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
va_ds = DS(va_paths, va_labels, eval_tf)

best, history, t_start = -1.0, [], time.time()
for ep in range(EPOCHS):
    model.train(); t_ep, run_loss = time.time(), 0.0
    for x, y in dl_tr:
        x = x.to(DEV, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run_loss += loss.item()
    va_acc = topk(predict(va_ds)[0], va_labels, 1)
    mins = (time.time() - t_start) / 60
    history.append({"epoch": ep + 1, "loss": run_loss / len(dl_tr), "val_top1": va_acc, "minutes": round(mins, 1)})
    if va_acc > best:
        best = va_acc
        torch.save(model.state_dict(), EXPORT / "best.pt")
    eta = (time.time() - t_ep) * (EPOCHS - ep - 1) / 60
    print(f"epoch {ep + 1}/{EPOCHS} loss {run_loss / len(dl_tr):.3f} val_top1 {va_acc:.4f} best {best:.4f} | {mins:.0f} min, ETA {eta:.0f} min", flush=True)
    if mins > TIME_BUDGET_MIN:
        print("time budget reached - stopping early with the best epoch so far"); break

model.load_state_dict(torch.load(EXPORT / "best.pt", map_location=DEV, weights_only=True))
print(f"best val top-1 {best:.4f}")

# %% [markdown]
# ## Test accuracy, crop vs no-crop, flip TTA

# %%
te_ds = DS(te_paths, te_labels, eval_tf)
te_logits, te_emb = predict(te_ds)
te_logits_tta, _ = predict(te_ds, tta=True)
tf_logits, _ = predict(DS(tf_paths, te_labels, eval_tf))
va_logits, va_emb = predict(va_ds)
tr_logits, tr_emb = predict(DS(tr_paths, tr_labels, eval_tf))

M = {"history": history, "classification": {
    "top1": topk(te_logits, te_labels, 1), "top5": topk(te_logits, te_labels, 5),
    "top1_tta": topk(te_logits_tta, te_labels, 1), "top5_tta": topk(te_logits_tta, te_labels, 5),
    "top1_no_crop": topk(tf_logits, te_labels, 1), "val_top1": best, "n_test": len(te_labels)}}
print(json.dumps(M["classification"], indent=1))

# %% [markdown]
# ## Calibration (temperature scaling on val), unknown-car score, high-voltage safety metrics

# %%
def fit_temperature(logits, labels):
    t = torch.ones(1, requires_grad=True)
    L, y = torch.tensor(logits), torch.tensor(labels)
    opt_t = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    def closure():
        opt_t.zero_grad()
        loss = F.cross_entropy(L / t.clamp(min=0.05), y); loss.backward(); return loss
    opt_t.step(closure)
    return float(t.detach().clamp(min=0.05))

def softmax(lg, temp=1.0):
    z = lg / temp; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); acc = pred == np.asarray(labels); out = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)

TEMP = fit_temperature(va_logits, va_labels)
te_p = softmax(te_logits, TEMP)
M["calibration"] = {"temperature": TEMP, "ece_before": ece(softmax(te_logits), te_labels), "ece_after": ece(te_p, te_labels)}

# unknown-car score = 1 - mean cosine similarity to the 5 nearest training cars
KNN_K = 5
bank = torch.tensor(tr_emb, dtype=torch.float32, device=DEV); bank = F.normalize(bank, dim=1)
def unknown_score(emb):
    q = F.normalize(torch.tensor(emb, dtype=torch.float32, device=DEV), dim=1)
    return (1 - (q @ bank.T).topk(KNN_K, dim=1).values.mean(1)).cpu().numpy()
va_unk, te_unk = unknown_score(va_emb), unknown_score(te_emb)
UNKNOWN_THRESHOLD = float(np.percentile(va_unk, 95))     # 5% of known cars flagged; retune with real unknown photos
M["unknown"] = {"threshold": UNKNOWN_THRESHOLD, "k": KNN_K, "known_test_flagged_rate": float((te_unk > UNKNOWN_THRESHOLD).mean())}

HV = {"BMW ActiveHybrid 5 Sedan 2012", "Chevrolet Silverado 1500 Hybrid Crew Cab 2012", "Chevrolet Tahoe Hybrid SUV 2012",
      "Chevrolet Malibu Hybrid Sedan 2010", "GMC Yukon Hybrid SUV 2012", "Hyundai Sonata Hybrid Sedan 2012",
      "Fisker Karma Sedan 2012", "Nissan Leaf Hatchback 2012", "Tesla Model S Sedan 2012"}
hv_idx = np.array([i for i, n in enumerate(CLASS_NAMES) if n in HV]); assert len(hv_idx) == 9, hv_idx
p_hv = te_p[:, hv_idx].sum(1); is_hv = np.isin(np.asarray(te_labels), hv_idx)
M["safety"] = {"n_hv_test": int(is_hv.sum()), "by_threshold": [
    {"threshold": t, "hv_miss_rate": float((p_hv[is_hv] < t).mean()) if is_hv.any() else None,
     "hv_false_alarm_rate": float((p_hv[~is_hv] >= t).mean())} for t in (0.05, 0.1, 0.2, 0.3, 0.5)],
    "argmax_hv_miss_rate": float((~np.isin(te_p.argmax(1)[is_hv], hv_idx)).mean()) if is_hv.any() else None}

# accuracy when the system only answers on its most confident cases
order = np.argsort(-te_p.max(1)); correct = (te_p.argmax(1) == np.asarray(te_labels))[order]
M["risk_coverage"] = [{"coverage": c, "accuracy": float(correct[: max(1, int(c * len(correct)))].mean())}
                      for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)]

# most-confused pairs, for the failure-case write-up
pred = te_p.argmax(1); pairs = {}
for t, p in zip(te_labels, pred):
    if t != p: pairs[(CLASS_NAMES[t], CLASS_NAMES[p])] = pairs.get((CLASS_NAMES[t], CLASS_NAMES[p]), 0) + 1
M["top_confusions"] = [{"true": a, "predicted": b, "count": c} for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]]
print(json.dumps({k: M[k] for k in ("calibration", "unknown", "safety")}, indent=1))

# %% [markdown]
# ## Robustness: darkness, haze, blur, low resolution, occlusion (+ does CLAHE rescue dark photos?)

# %%
def clahe(im):
    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

def occlude(im, frac, i):
    r = random.Random(i); w, h = im.size; s = math.sqrt(frac)
    bw, bh = int(w * s), int(h * s); x, y = r.randint(0, w - bw), r.randint(0, h - bh)
    im = im.copy(); ImageDraw.Draw(im).rectangle([x, y, x + bw, y + bh], fill=(128, 128, 128)); return im

def lowres(im, side):
    s = side / min(im.size); return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR).resize(im.size, Image.BILINEAR)

CONDITIONS = {
    "dark":      [lambda im, i, v=v: ImageEnhance.Brightness(im).enhance(v) for v in (0.6, 0.35, 0.15)],
    "dark+clahe": [lambda im, i, v=v: clahe(ImageEnhance.Brightness(im).enhance(v)) for v in (0.6, 0.35, 0.15)],
    "haze":      [lambda im, i, a=a: Image.blend(im, Image.new("RGB", im.size, (200, 200, 200)), a) for a in (0.25, 0.45, 0.65)],
    "blur":      [lambda im, i, r=r: im.filter(ImageFilter.GaussianBlur(r)) for r in (1.5, 3, 5)],
    "low_res":   [lambda im, i, s=s: lowres(im, s) for s in (96, 64, 40)],
    "occlusion": [lambda im, i, f=f: occlude(im, f, i) for f in (0.15, 0.3, 0.45)],
}
sub = np.random.RandomState(SEED).choice(len(te_paths), min(ROBUST_N, len(te_paths)), replace=False)
sp, sl = [te_paths[i] for i in sub], [te_labels[i] for i in sub]
M["robustness"] = [{"condition": "clean", "severity": 0, "top1": topk(predict(DS(sp, sl, eval_tf))[0], sl, 1)}]
for name, fns in CONDITIONS.items():
    for sev, fn in enumerate(fns, 1):
        acc = topk(predict(DS(sp, sl, eval_tf, degrade=fn))[0], sl, 1)
        M["robustness"].append({"condition": name, "severity": sev, "top1": acc})
        print(f"{name:10s} sev {sev}: {acc:.4f}", flush=True)
M["robustness_n"] = len(sp)

# %% [markdown]
# ## Export ONNX (FP32) + INT8, compare size / CPU latency / accuracy

# %%
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

class ExportNet(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        e = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
        return self.m.get_classifier()(e), e

net = ExportNet(model.float().eval().to("cpu").to(memory_format=torch.contiguous_format))
fp32_path, int8_path = EXPORT / "classifier_fp32.onnx", EXPORT / "classifier_int8.onnx"
kw = dict(input_names=["input"], output_names=["logits", "embedding"],
          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
dummy = torch.randn(1, 3, IMG, IMG)
try:
    torch.onnx.export(net, dummy, str(fp32_path), dynamo=False, **kw)
except TypeError:
    torch.onnx.export(net, dummy, str(fp32_path), **kw)
# MatMul/Gemm only: ConvNeXt's heavy layers are Linear; quantizing Conv needs ConvInteger, which ORT CPU lacks for int8 weights
quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])

def ort_eval(path, paths, labels):
    """Accuracy, CPU ms/image and embeddings from an ONNX model, exactly as the app will run it."""
    so = ort.SessionOptions(); so.intra_op_num_threads = 2          # match the 2-vCPU deployment box
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    xs = [eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).numpy() for p in paths]
    for x in xs[:3]: sess.run(None, {"input": x})                     # warm-up
    t0 = time.time(); outs = [sess.run(None, {"input": x}) for x in xs]; ms = (time.time() - t0) / len(xs) * 1000
    preds = np.array([int(o[0].argmax()) for o in outs])
    return float(np.mean(preds == np.array(labels))), ms, np.stack([o[1][0] for o in outs]).astype(np.float32)

n = ONNX_EVAL_N
fp32_acc, fp32_ms, fp32_te_emb = ort_eval(fp32_path, te_paths[:n], te_labels[:n])
int8_acc, int8_ms, int8_te_emb = ort_eval(int8_path, te_paths[:n], te_labels[:n])
USE_INT8 = int8_acc >= fp32_acc - 0.01            # only ship INT8 if it costs < 1 point
deploy_path, deploy_te_emb = (int8_path, int8_te_emb) if USE_INT8 else (fp32_path, fp32_te_emb)
M["efficiency"] = {"params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                   "fp32_mb": fp32_path.stat().st_size / 2**20, "int8_mb": int8_path.stat().st_size / 2**20,
                   "fp32_ms_cpu2": fp32_ms, "int8_ms_cpu2": int8_ms, "fp32_top1_subset": fp32_acc, "int8_top1_subset": int8_acc,
                   "subset_n": n, "deployed": "int8" if USE_INT8 else "fp32"}

# re-set the unknown-car threshold with the ONNX model the app actually runs (quantization shifts embeddings)
calib_n = min(len(va_paths), 40 if QUICK else 400)
_, _, va_emb_deploy = ort_eval(deploy_path, va_paths[:calib_n], va_labels[:calib_n])
UNKNOWN_THRESHOLD = float(np.percentile(unknown_score(va_emb_deploy), 95))
M["unknown"].update(threshold=UNKNOWN_THRESHOLD, calibrated_on=M["efficiency"]["deployed"], calib_n=calib_n,
                    known_test_flagged_rate=float((unknown_score(deploy_te_emb) > UNKNOWN_THRESHOLD).mean()))
print(json.dumps({"efficiency": M["efficiency"], "unknown": M["unknown"]}, indent=1))

# %% [markdown]
# ## Save everything the back end and the eval file need

# %%
np.save(EXPORT / "train_emb.npy", F.normalize(torch.tensor(tr_emb, dtype=torch.float32), dim=1).numpy().astype(np.float16))
np.save(EXPORT / "train_labels.npy", np.asarray(tr_labels, dtype=np.int16))
np.savez_compressed(EXPORT / "logits.npz", val_logits=va_logits, val_labels=va_labels, test_logits=te_logits,
                    test_labels=te_labels, test_full_logits=tf_logits, test_emb=te_emb, test_unknown=te_unk)
(EXPORT / "config.json").write_text(json.dumps({
    "model_name": MODEL_NAME, "img": IMG, "mean": MEAN, "std": STD, "class_names": CLASS_NAMES,
    "temperature": TEMP, "knn_k": KNN_K, "unknown_threshold": UNKNOWN_THRESHOLD, "hv_threshold": 0.1,
    "use_int8": USE_INT8, "quick": QUICK}, indent=1))
(EXPORT / "metrics.json").write_text(json.dumps(M, indent=1))
for f in sorted(EXPORT.iterdir()):
    print(f"{f.name:28s} {f.stat().st_size / 2**20:8.1f} MB")
print("DONE" + (" (QUICK smoke test - now set QUICK = False and Save & Run All)" if QUICK else ""))
