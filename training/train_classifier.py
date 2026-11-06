"""RescueLens - fine-grained classifier training (ConvNeXt-Tiny on Stanford Cars) + all evaluation numbers.

Runs on Kaggle (Linux) or locally (Windows / Linux) with a CUDA GPU. Needs training/rl_data.py next to it.

Kaggle cell (GPU T4, Internet on, Stanford Cars input added):
    !wget -q https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/<sha>/training/rl_data.py
    !wget -q -O train.py https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/<sha>/training/train_classifier.py
    %env QUICK=1
    %run -i train.py

Local (Windows, from the repo root):
    set DATA_ROOT=C:\\path\\to\\stanford_cars & set EXPORT_DIR=models & set CACHE_DIR=training\\.cache & set QUICK=1
    training\\.venv\\Scripts\\python training\\train_classifier.py

QUICK=1 (default) = smoke test on a few hundred images, 1 epoch. QUICK=0 = full run.
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
if ON_KAGGLE and __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=1.0.9", "onnx", "onnxruntime", "onnxscript"], check=False)

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, str(Path.cwd() if ON_KAGGLE else Path(__file__).resolve().parent))   # where rl_data.py lives
from rl_data import CONDITIONS, DS, RandomDark, cache_items, load_stanford_cars, split_items

QUICK = os.environ.get("QUICK", "1") != "0"
MODEL_NAME = "convnext_tiny.fb_in22k_ft_in1k_384"
IMG = 384
EPOCHS = 1 if QUICK else int(os.environ.get("EPOCHS", 25))
BATCH = 32
LR = 2e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_FRAC = 0.1
TIME_BUDGET_MIN = float(os.environ.get("TIME_BUDGET_MIN", 170))   # stop early (keeping the best epoch) after this
SEED = 42
ROBUST_N = 100 if QUICK else 2000          # test images used for robustness tests
ONNX_EVAL_N = 40 if QUICK else 300         # test images used to compare FP32 vs INT8 on CPU
IN = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
EXPORT = Path(os.environ.get("EXPORT_DIR", "/kaggle/working/export"))
CACHE = Path(os.environ.get("CACHE_DIR", "/tmp/cache"))
WORKERS = int(os.environ.get("WORKERS", min(os.cpu_count() or 2, 4 if ON_KAGGLE else 6)))
HV = {"BMW ActiveHybrid 5 Sedan 2012", "Chevrolet Silverado 1500 Hybrid Crew Cab 2012", "Chevrolet Tahoe Hybrid SUV 2012",
      "Chevrolet Malibu Hybrid Sedan 2010", "GMC Yukon Hybrid SUV 2012", "Hyundai Sonata Hybrid Sedan 2012",
      "Fisker Karma Sedan 2012", "Nissan Leaf Hatchback 2012", "Tesla Model S Sedan 2012"}


def topk(logits, labels, k):
    return float((np.argsort(-logits, 1)[:, :k] == np.asarray(labels)[:, None]).any(1).mean())


def softmax(lg, temp=1.0):
    z = lg / temp
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = pred == np.asarray(labels)
    out = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def fit_temperature(logits, labels):
    t = torch.ones(1, requires_grad=True)
    L, y = torch.tensor(logits), torch.tensor(np.asarray(labels))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / t.clamp(min=0.05), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=0.05))


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    assert torch.cuda.is_available(), "No CUDA GPU found."
    dev = "cuda"
    EXPORT.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "| timm", timm.__version__, "| GPU", torch.cuda.get_device_name(0),
          "| workers", WORKERS, "| QUICK", QUICK, flush=True)

    # ---- data
    train_items, test_items, class_names = load_stanford_cars(IN)
    tr_items, va_items, te_items = split_items(train_items, test_items, VAL_FRAC, SEED, QUICK)
    side = int(IMG * 1.15)
    t0 = time.time()
    tr_paths, tr_labels = cache_items(tr_items, CACHE / "train", True, side, WORKERS)
    va_paths, va_labels = cache_items(va_items, CACHE / "val", True, side, WORKERS)
    te_paths, te_labels = cache_items(te_items, CACHE / "test", True, side, WORKERS)
    tf_paths, _ = cache_items(te_items, CACHE / "test_full", False, side, WORKERS)
    print(f"cached crops in {time.time() - t0:.0f}s", flush=True)

    # ---- model + transforms
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=196, drop_path_rate=0.1)
    mean, std = list(model.pretrained_cfg["mean"]), list(model.pretrained_cfg["std"])
    model = model.to(dev).to(memory_format=torch.channels_last)
    train_tf = T.Compose([
        T.RandomResizedCrop(IMG, scale=(0.6, 1.0), ratio=(0.75, 2.0)),
        T.RandomHorizontalFlip(),
        RandomDark(0.2),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(), T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),       # partial occlusion
    ])
    eval_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(mean, std)])

    def loader(ds, train=False, persistent=False, bs=None):
        return DataLoader(ds, bs or (BATCH if train else 64), shuffle=train, drop_last=train, num_workers=WORKERS,
                          pin_memory=True, persistent_workers=persistent and WORKERS > 0)

    @torch.no_grad()
    def predict(dl, tta=False):
        """Returns (logits float32 [N,196], embeddings float16 [N,D])."""
        model.eval()
        L, E = [], []
        for x, _ in dl:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                e = model.forward_head(model.forward_features(x), pre_logits=True)
                lg = model.get_classifier()(e)
                if tta:
                    lg = (lg + model(torch.flip(x, dims=[3]))) / 2
            L.append(lg.float().cpu()); E.append(e.half().cpu())
        return torch.cat(L).numpy(), torch.cat(E).numpy()

    # ---- train (keeps the best epoch by validation accuracy)
    dl_tr = loader(DS(tr_paths, tr_labels, train_tf), train=True, persistent=True)
    dl_va = loader(DS(va_paths, va_labels, eval_tf), persistent=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps, warm = EPOCHS * len(dl_tr), len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    best, history, t_start = -1.0, [], time.time()
    eval_only = os.environ.get("EVAL_ONLY") == "1" and (EXPORT / "best.pt").exists()   # re-run evaluation without retraining
    for ep in range(0 if eval_only else EPOCHS):
        model.train()
        t_ep, run_loss = time.time(), 0.0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()
        va_acc = topk(predict(dl_va)[0], va_labels, 1)
        mins = (time.time() - t_start) / 60
        history.append({"epoch": ep + 1, "loss": run_loss / len(dl_tr), "val_top1": va_acc, "minutes": round(mins, 1)})
        if va_acc > best:
            best = va_acc
            torch.save(model.state_dict(), EXPORT / "best.pt")
        eta = (time.time() - t_ep) * (EPOCHS - ep - 1) / 60
        print(f"epoch {ep + 1}/{EPOCHS} loss {run_loss / len(dl_tr):.3f} val_top1 {va_acc:.4f} best {best:.4f} | "
              f"{mins:.0f} min, ETA {eta:.0f} min", flush=True)
        if mins > TIME_BUDGET_MIN:
            print("time budget reached - stopping early with the best epoch so far", flush=True)
            break
    model.load_state_dict(torch.load(EXPORT / "best.pt", map_location=dev, weights_only=True))
    if eval_only:
        best = topk(predict(dl_va)[0], va_labels, 1)
    print(f"best val top-1 {best:.4f}", flush=True)

    # ---- test accuracy, crop vs no-crop, flip TTA
    te_ds = DS(te_paths, te_labels, eval_tf)
    te_logits, te_emb = predict(loader(te_ds))
    te_logits_tta, _ = predict(loader(te_ds), tta=True)
    tf_logits, _ = predict(loader(DS(tf_paths, te_labels, eval_tf)))
    va_logits, va_emb = predict(dl_va)
    tr_logits, tr_emb = predict(loader(DS(tr_paths, tr_labels, eval_tf)))
    M = {"history": history, "classification": {
        "top1": topk(te_logits, te_labels, 1), "top5": topk(te_logits, te_labels, 5),
        "top1_tta": topk(te_logits_tta, te_labels, 1), "top5_tta": topk(te_logits_tta, te_labels, 5),
        "top1_no_crop": topk(tf_logits, te_labels, 1), "val_top1": best, "n_test": len(te_labels)}}
    print(json.dumps(M["classification"], indent=1), flush=True)

    # ---- calibration, unknown-car score, high-voltage safety metrics
    temp = fit_temperature(va_logits, va_labels)
    te_p = softmax(te_logits, temp)
    M["calibration"] = {"temperature": temp, "ece_before": ece(softmax(te_logits), te_labels), "ece_after": ece(te_p, te_labels)}

    knn_k = 5
    bank = F.normalize(torch.tensor(tr_emb, dtype=torch.float32, device=dev), dim=1)

    def unknown_score(emb):
        """1 - mean cosine similarity to the 5 nearest training cars."""
        q = F.normalize(torch.tensor(np.asarray(emb, np.float32), device=dev), dim=1)
        return (1 - (q @ bank.T).topk(knn_k, dim=1).values.mean(1)).cpu().numpy()

    va_unk, te_unk = unknown_score(va_emb), unknown_score(te_emb)
    unknown_threshold = float(np.percentile(va_unk, 95))       # re-set below with the deployed ONNX model
    M["unknown"] = {"threshold": unknown_threshold, "k": knn_k, "known_test_flagged_rate": float((te_unk > unknown_threshold).mean())}

    hv_idx = np.array([i for i, n in enumerate(class_names) if n in HV])
    assert len(hv_idx) == 9, hv_idx
    p_hv = te_p[:, hv_idx].sum(1)
    is_hv = np.isin(np.asarray(te_labels), hv_idx)
    M["safety"] = {"n_hv_test": int(is_hv.sum()), "by_threshold": [
        {"threshold": t, "hv_miss_rate": float((p_hv[is_hv] < t).mean()) if is_hv.any() else None,
         "hv_false_alarm_rate": float((p_hv[~is_hv] >= t).mean())} for t in (0.05, 0.1, 0.2, 0.3, 0.5)],
        "argmax_hv_miss_rate": float((~np.isin(te_p.argmax(1)[is_hv], hv_idx)).mean()) if is_hv.any() else None}

    order = np.argsort(-te_p.max(1))
    correct = (te_p.argmax(1) == np.asarray(te_labels))[order]
    M["risk_coverage"] = [{"coverage": c, "accuracy": float(correct[: max(1, int(c * len(correct)))].mean())}
                          for c in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)]
    pairs = {}
    for t, p in zip(te_labels, te_p.argmax(1)):
        if t != p:
            pairs[(class_names[t], class_names[p])] = pairs.get((class_names[t], class_names[p]), 0) + 1
    M["top_confusions"] = [{"true": a, "predicted": b, "count": c}
                           for (a, b), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]]
    print(json.dumps({k: M[k] for k in ("calibration", "unknown", "safety")}, indent=1), flush=True)

    # ---- robustness: darkness, haze, blur, low resolution, occlusion (+ does CLAHE rescue dark photos?)
    sub = np.random.RandomState(SEED).choice(len(te_paths), min(ROBUST_N, len(te_paths)), replace=False)
    sp, sl = [te_paths[i] for i in sub], [te_labels[i] for i in sub]
    M["robustness"] = [{"condition": "clean", "severity": 0, "top1": topk(predict(loader(DS(sp, sl, eval_tf)))[0], sl, 1)}]
    for name, fns in CONDITIONS.items():
        for sev, fn in enumerate(fns, 1):
            acc = topk(predict(loader(DS(sp, sl, eval_tf, degrade=fn)))[0], sl, 1)
            M["robustness"].append({"condition": name, "severity": sev, "top1": acc})
            print(f"{name:10s} sev {sev}: {acc:.4f}", flush=True)
    M["robustness_n"] = len(sp)

    # ---- ONNX export (FP32) + INT8; size, CPU latency and accuracy exactly as the app runs them
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    class ExportNet(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            e = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
            return self.m.get_classifier()(e), e

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    net = ExportNet(model.float().eval().to("cpu").to(memory_format=torch.contiguous_format))
    fp32_path, int8_path = EXPORT / "classifier_fp32.onnx", EXPORT / "classifier_int8.onnx"
    kw = dict(input_names=["input"], output_names=["logits", "embedding"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    dummy = torch.randn(1, 3, IMG, IMG)
    try:
        torch.onnx.export(net, dummy, str(fp32_path), dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(net, dummy, str(fp32_path), **kw)
    # MatMul/Gemm only: ConvNeXt's heavy layers are Linear; quantizing Conv needs ConvInteger, which ORT CPU lacks for int8
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])

    def ort_eval(path, paths, labels):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2           # match the 2-vCPU deployment box
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        xs = [eval_tf(Image.open(p).convert("RGB")).unsqueeze(0).numpy() for p in paths]
        for x in xs[:3]:
            sess.run(None, {"input": x})      # warm-up
        t0 = time.time()
        outs = [sess.run(None, {"input": x}) for x in xs]
        ms = (time.time() - t0) / len(xs) * 1000
        preds = np.array([int(o[0].argmax()) for o in outs])
        return float(np.mean(preds == np.array(labels))), ms, np.stack([o[1][0] for o in outs]).astype(np.float32)

    n = ONNX_EVAL_N
    fp32_acc, fp32_ms, fp32_te_emb = ort_eval(fp32_path, te_paths[:n], te_labels[:n])
    int8_acc, int8_ms, int8_te_emb = ort_eval(int8_path, te_paths[:n], te_labels[:n])
    use_int8 = int8_acc >= fp32_acc - 0.01    # only ship INT8 if it costs < 1 point
    deploy_path, deploy_te_emb = (int8_path, int8_te_emb) if use_int8 else (fp32_path, fp32_te_emb)
    M["efficiency"] = {"params_m": params_m, "fp32_mb": fp32_path.stat().st_size / 2**20, "int8_mb": int8_path.stat().st_size / 2**20,
                       "fp32_ms_cpu2": fp32_ms, "int8_ms_cpu2": int8_ms, "fp32_top1_subset": fp32_acc,
                       "int8_top1_subset": int8_acc, "subset_n": n, "deployed": "int8" if use_int8 else "fp32"}
    # unknown-car threshold re-set with the ONNX model the app actually runs (quantization shifts embeddings)
    calib_n = min(len(va_paths), 40 if QUICK else 400)
    _, _, va_emb_deploy = ort_eval(deploy_path, va_paths[:calib_n], va_labels[:calib_n])
    unknown_threshold = float(np.percentile(unknown_score(va_emb_deploy), 95))
    M["unknown"].update(threshold=unknown_threshold, calibrated_on=M["efficiency"]["deployed"], calib_n=calib_n,
                        known_test_flagged_rate=float((unknown_score(deploy_te_emb) > unknown_threshold).mean()))
    print(json.dumps({"efficiency": M["efficiency"], "unknown": M["unknown"]}, indent=1), flush=True)

    # ---- save everything the back end and the eval file need
    np.save(EXPORT / "train_emb.npy", bank.cpu().numpy().astype(np.float16))
    np.save(EXPORT / "train_labels.npy", np.asarray(tr_labels, dtype=np.int16))
    np.savez_compressed(EXPORT / "logits.npz", val_logits=va_logits, val_labels=va_labels, test_logits=te_logits,
                        test_labels=te_labels, test_full_logits=tf_logits, test_emb=te_emb, test_unknown=te_unk)
    (EXPORT / "config.json").write_text(json.dumps({
        "model_name": MODEL_NAME, "img": IMG, "mean": mean, "std": std, "class_names": class_names,
        "temperature": temp, "knn_k": knn_k, "unknown_threshold": unknown_threshold, "hv_threshold": 0.1,
        "use_int8": use_int8, "quick": QUICK}, indent=1))
    (EXPORT / "metrics.json").write_text(json.dumps(M, indent=1))
    for f in sorted(EXPORT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 2**20:8.1f} MB")
    print("DONE" + (" (QUICK smoke test - now run with QUICK=0)" if QUICK else ""), flush=True)


if __name__ == "__main__":
    main()
