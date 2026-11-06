"""Brand expert: a second network that sees only zoomed-in parts of the car (grille, badge, lights, wheels) and predicts
the make (49 brands). At test time it votes over a fixed 3x3 grid of half-size windows; its brand vote is fused with the
whole-car classifier's summed make probabilities. Logos are tiny in a whole-car 384 px view; in a half-size window at
224 px they are several times larger.

    python training/train_brand_parts.py     env: DATA_ROOT, EXPORT_DIR (models), CACHE_DIR (training/.cache), QUICK
Writes EXPORT_DIR/brand_parts_fp32.onnx, brand_parts_int8.onnx, brand_parts.json, brand_metrics.json
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_data import DS, RandomDark, cache_items, load_stanford_cars, occlude, split_items  # noqa: E402

QUICK = os.environ.get("QUICK", "1") != "0"
IN = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
EXPORT = Path(os.environ.get("EXPORT_DIR", "models"))
CACHE = Path(os.environ.get("CACHE_DIR", "training/.cache"))
WHOLE_CKPT = Path(os.environ.get("WHOLE_CKPT", "models_small/best.pt"))
WHOLE_NAME = os.environ.get("WHOLE_NAME", "convnext_small.fb_in22k_ft_in1k_384")
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k"
PART, GRID = 224, 448                       # 3x3 windows of 224 inside the car crop squashed to 448 (stride 112)
EPOCHS = 1 if QUICK else int(os.environ.get("EPOCHS", 12))
BATCH, LR, SEED = 64, 3e-4, 42
WORKERS = int(os.environ.get("WORKERS", 6))
EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"      # re-run evaluation + fusion choice on the saved brand_parts.pt
TWO_WORD = ("AM General", "Aston Martin", "Land Rover")


def make_of(label: str) -> str:
    return " ".join(label.split()[:2]) if label.startswith(TWO_WORD) else label.split()[0]


def grid_windows(im: Image.Image) -> list[Image.Image]:
    """Same 9 windows the app uses: squash the car crop to 448x448, take 224x224 windows at offsets 0/112/224."""
    sq = im.resize((GRID, GRID), Image.BILINEAR)
    return [sq.crop((x, y, x + PART, y + PART)) for y in (0, 112, 224) for x in (0, 112, 224)]


class GridDS(Dataset):
    """Returns the 9 grid windows of one car crop as a [9, 3, 224, 224] tensor (optionally occluded first)."""

    def __init__(self, paths, labels, tf, degrade=None):
        self.paths, self.labels, self.tf, self.degrade = paths, labels, tf, degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.degrade is not None:
            im = self.degrade(im, i)
        return torch.stack([self.tf(w) for w in grid_windows(im)]), self.labels[i]


def occ30(im, i):
    return occlude(im, i, 0.3)


def fuse(rule: str, w: float, whole: np.ndarray, details: np.ndarray) -> np.ndarray:
    """linear: (1-w)*whole + w*details. product: whole^(1-w) * details^w, renormalised - a confident expert dominates
    an unsure one, which is the damaged-car case (the whole-car view goes flat, the badge is still readable)."""
    if rule == "linear":
        return (1 - w) * whole + w * details
    p = np.exp((1 - w) * np.log(whole + 1e-9) + w * np.log(details + 1e-9))
    return p / p.sum(-1, keepdims=True)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda"
    EXPORT.mkdir(parents=True, exist_ok=True)
    train_items, test_items, class_names = load_stanford_cars(IN)
    tr_items, va_items, te_items = split_items(train_items, test_items, 0.1, SEED, QUICK)
    makes = sorted({make_of(n) for n in class_names})
    class_make = np.array([makes.index(make_of(n)) for n in class_names])
    side = int(384 * 1.15)                  # same cache as the classifier (crops already on disk)
    tr_paths, tr_cls = cache_items(tr_items, CACHE / "train", True, side, WORKERS)
    va_paths, va_cls = cache_items(va_items, CACHE / "val", True, side, WORKERS)
    te_paths, te_cls = cache_items(te_items, CACHE / "test", True, side, WORKERS)
    tr_mk, va_mk, te_mk = (class_make[np.array(c)] for c in (tr_cls, va_cls, te_cls))
    print(f"{len(makes)} makes | train {len(tr_paths)} val {len(va_paths)} test {len(te_paths)}", flush=True)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(makes), drop_path_rate=0.1)
    mean, std = list(model.pretrained_cfg["mean"]), list(model.pretrained_cfg["std"])
    model = model.to(dev).to(memory_format=torch.channels_last)
    train_tf = T.Compose([T.RandomResizedCrop(PART, scale=(0.06, 0.35), ratio=(0.5, 2.0)), T.RandomHorizontalFlip(),
                          RandomDark(0.2), T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    dl = DataLoader(DS(tr_paths, list(tr_mk), train_tf), BATCH, shuffle=True, drop_last=True, num_workers=WORKERS,
                    pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    steps, warm = EPOCHS * len(dl), len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    @torch.no_grad()
    def grid_logits(paths, labels, degrade=None):
        """Mean over the 9 windows of the per-window logits -> [N, makes]."""
        model.eval()
        out = []
        for x, _ in DataLoader(GridDS(paths, list(labels), eval_tf, degrade), 16, num_workers=WORKERS, pin_memory=True):
            b = x.shape[0]
            x = x.view(-1, 3, PART, PART).to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(x).float()
            out.append(lg.view(b, 9, -1).mean(1).cpu())
        return torch.cat(out).numpy()

    eval_only = EVAL_ONLY and (EXPORT / "brand_parts.pt").exists()
    best, t0 = -1.0, time.time()
    for ep in range(0 if eval_only else EPOCHS):
        model.train()
        for x, y in dl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        acc = float((grid_logits(va_paths, va_mk).argmax(1) == va_mk).mean())
        if acc > best:
            best = acc
            torch.save(model.state_dict(), EXPORT / "brand_parts.pt")
        print(f"epoch {ep + 1}/{EPOCHS} val make acc (9-window vote) {acc:.4f} best {best:.4f} | {(time.time() - t0) / 60:.0f} min", flush=True)
    model.load_state_dict(torch.load(EXPORT / "brand_parts.pt", map_location=dev, weights_only=True))
    if eval_only:
        best = float((grid_logits(va_paths, va_mk).argmax(1) == va_mk).mean())

    # ---- temperature for the parts vote (validation), then the fusion weight (validation)
    def softmax(z, t=1.0):
        z = z / t; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    va_lg = grid_logits(va_paths, va_mk)
    temps = np.linspace(0.3, 3.0, 28)
    temp = float(min(temps, key=lambda t: -np.log(softmax(va_lg, t)[np.arange(len(va_mk)), va_mk] + 1e-9).mean()))
    to_make = np.zeros((len(class_names), len(makes))); to_make[np.arange(len(class_names)), class_make] = 1

    # whole-car make probabilities: the deployed classifier's calibrated class probabilities summed per make
    cfg = json.loads((EXPORT / "config.json").read_text())
    lg = np.load(EXPORT / "logits.npz")
    whole_va = softmax(lg["val_logits"], cfg["temperature"]) @ to_make
    whole_te = softmax(lg["test_logits"], cfg["temperature"]) @ to_make
    assert len(whole_te) == len(te_mk) and (class_make[lg["test_labels"]] == te_mk).all(), "classifier logits order differs"
    parts_va, parts_te = softmax(va_lg, temp), softmax(grid_logits(te_paths, te_mk), temp)
    acc = lambda p, y: float((p.argmax(1) == y).mean())

    # ---- damage-like condition: 30% of the car hidden; whole-car classifier re-run on the same occluded crops
    whole = timm.create_model(WHOLE_NAME, pretrained=False, num_classes=len(class_names))
    whole.load_state_dict(torch.load(WHOLE_CKPT, map_location="cpu", weights_only=True))
    whole = whole.to(dev).eval().to(memory_format=torch.channels_last)
    wtf = T.Compose([T.Resize((cfg["img"], cfg["img"])), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    def whole_makes(paths, labels, degrade):
        out = []
        with torch.no_grad():
            for x, _ in DataLoader(DS(paths, list(labels), wtf, degrade=degrade), 32, num_workers=WORKERS):
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(whole(x.to(dev).to(memory_format=torch.channels_last)).float().cpu())
        return softmax(torch.cat(out).numpy(), cfg["temperature"]) @ to_make

    whole_occ_va, parts_occ_va = whole_makes(va_paths, va_mk, occ30), softmax(grid_logits(va_paths, va_mk, degrade=occ30), temp)
    # fusion rule + weight chosen on validation, clean and occluded counted equally (crash photos are rarely clean)
    rule, weight = max(((r, float(w)) for r in ("linear", "product") for w in np.linspace(0, 1, 11)),
                       key=lambda rw: acc(fuse(*rw, whole_va, parts_va), va_mk) + acc(fuse(*rw, whole_occ_va, parts_occ_va), va_mk))
    sub = np.random.RandomState(SEED).choice(len(te_paths), min(80 if QUICK else 2000, len(te_paths)), replace=False)
    sp, smk = [te_paths[i] for i in sub], te_mk[sub]
    whole_occ, parts_occ = whole_makes(sp, smk, occ30), softmax(grid_logits(sp, smk, degrade=occ30), temp)

    M = {"makes": len(makes), "parts_temperature": temp, "fusion_rule": rule, "fusion_weight": weight, "val_parts_acc": best,
         "test_clean": {"whole": acc(whole_te, te_mk), "parts": acc(parts_te, te_mk), "fused": acc(fuse(rule, weight, whole_te, parts_te), te_mk), "n": len(te_mk)},
         "test_occluded_30": {"whole": acc(whole_occ, smk), "parts": acc(parts_occ, smk), "fused": acc(fuse(rule, weight, whole_occ, parts_occ), smk), "n": len(smk)}}
    print(json.dumps(M, indent=1), flush=True)

    # ---- ONNX (+ INT8 for MatMul/Gemm, same as the classifier) for the app
    from onnxruntime.quantization import QuantType, quantize_dynamic
    net = model.float().eval().to("cpu").to(memory_format=torch.contiguous_format)
    fp32, int8 = EXPORT / "brand_parts_fp32.onnx", EXPORT / "brand_parts_int8.onnx"
    kw = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(net, torch.randn(1, 3, PART, PART), str(fp32), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, torch.randn(1, 3, PART, PART), str(fp32), **kw)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    (EXPORT / "brand_parts.json").write_text(json.dumps({"makes": makes, "part": PART, "grid": GRID, "mean": mean, "std": std,
                                                         "temperature": temp, "rule": rule, "weight": weight, "quick": QUICK}, indent=1))
    (EXPORT / "brand_metrics.json").write_text(json.dumps(M, indent=1))
    print("DONE" + (" (QUICK)" if QUICK else ""), flush=True)


if __name__ == "__main__":
    main()
