"""Merge both Kaggle exports into models/metrics_public.json (for /api/metrics) and eval_P5_<team>.md.

python tools/build_report.py --train path/to/export1 --det path/to/export2 --team T01 --usns "USN1,USN2,USN3"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def top1(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(1) == y).mean())


def topk(p: np.ndarray, y: np.ndarray, k: int) -> float:
    return float((np.argsort(-p, 1)[:, :k] == y[:, None]).any(1).mean())


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.2f}%"


def hybrid_fusion(train: Path, det: Path, temp: float) -> dict | None:
    """Late fusion of deep + classical probabilities; weight picked on validation only."""
    lg, bp = np.load(train / "logits.npz"), det / "baseline_probs.npz"
    if not bp.exists():
        return None
    bp = np.load(bp)
    if not np.array_equal(lg["val_labels"], bp["val_labels"]) or not np.array_equal(lg["test_labels"], bp["test_labels"]):
        print("WARNING: notebook splits differ - skipping fusion")
        return None
    pv, pt = softmax(lg["val_logits"] / temp), softmax(lg["test_logits"] / temp)
    cv_, ct = bp["val"].astype(np.float32), bp["test"].astype(np.float32)
    w = max(np.linspace(0, 0.5, 11), key=lambda w: top1((1 - w) * pv + w * cv_, lg["val_labels"]))
    fused = (1 - w) * pt + w * ct
    return {"weight": float(w), "top1": top1(fused, lg["test_labels"]), "top5": topk(fused, lg["test_labels"], 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--det", type=Path, required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--usns", required=True)
    ap.add_argument("--out", type=Path, default=Path("models"))
    a = ap.parse_args()

    m = json.loads((a.train / "metrics.json").read_text())
    d = json.loads((a.det / "det_metrics.json").read_text())
    cfg = json.loads((a.train / "config.json").read_text())
    c, e, cal, saf = m["classification"], m["efficiency"], m["calibration"], m["safety"]
    base, detm, roi = d.get("classical_baseline", {}), d["detection"], d.get("classical_roi", {})
    pre = d.get("detection_pretrained")              # present once the detector has been fine-tuned
    fusion = hybrid_fusion(a.train, a.det, cfg["temperature"])
    hybrid_top1 = max(c["top1"], fusion["top1"]) if fusion else c["top1"]
    hybrid_top5 = fusion["top5"] if fusion and fusion["top1"] >= c["top1"] else c["top5"]
    at = {r["threshold"]: r for r in saf["by_threshold"]}
    chosen = at.get(cfg["hv_threshold"], saf["by_threshold"][1])
    rob = {(r["condition"], r["severity"]): r["top1"] for r in m["robustness"]}
    unk_path = a.out / "unknown_eval.json"           # from tools/eval_unknown.py, if it was run
    unk = json.loads(unk_path.read_text()) if unk_path.exists() else {}
    brand_path = a.out / "brand_metrics.json"         # from training/train_brand_parts.py, if trained
    brand = json.loads(brand_path.read_text()) if brand_path.exists() else None

    public = {
        "classification": {"top1": c["top1"], "top5": c["top5"], "top1_no_crop": c["top1_no_crop"], "top1_tta": c["top1_tta"]},
        "detection": {"mean_iou": detm["mean_iou"], "map50": detm["map50"], "map50_95": detm["map50_95"]},
        "efficiency": {"fp32_mb": e["fp32_mb"], "int8_mb": e["int8_mb"], "fp32_ms": e["fp32_ms_cpu2"],
                       "int8_ms": e["int8_ms_cpu2"], "int8_top1": e["int8_top1_subset"], "fp32_top1": e["fp32_top1_subset"]},
        "robustness": m["robustness"],
        "calibration": {"ece_before": cal["ece_before"], "ece_after": cal["ece_after"], "temperature": cal["temperature"]},
        "safety": {"hv_threshold": cfg["hv_threshold"], "hv_miss_rate": chosen["hv_miss_rate"],
                   "hv_false_alarm_rate": chosen["hv_false_alarm_rate"], "argmax_hv_miss_rate": saf["argmax_hv_miss_rate"],
                   "known_flagged_unknown_rate": m["unknown"]["known_test_flagged_rate"],
                   "unknown_catch_rate": unk.get("unknown_catch_rate"), "by_threshold": saf["by_threshold"]},
        "approaches": [
            {"name": "classical", "top1": base.get("test_top1"), "top5": base.get("test_top5")},
            {"name": "deep", "top1": c["top1_no_crop"], "top5": None},
            {"name": "hybrid", "top1": hybrid_top1, "top5": hybrid_top5},
        ],
        "localisation": {"deep_mean_iou": detm["mean_iou"], "classical_mean_iou": roi.get("mean_iou")},
        "risk_coverage": m["risk_coverage"],
        "top_confusions": m["top_confusions"][:10],
        "fusion": fusion,
        "brand_experts": brand,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metrics_public.json").write_text(json.dumps(public, indent=1))

    dark = [(s, rob.get(("dark", s)), rob.get(("dark+clahe", s))) for s in (1, 2, 3)]
    lines = [
        a.usns,
        "",
        f"# Evaluation - P5 Intelligent Fine-Grained Vehicle Recognition - Team {a.team}",
        "",
        f"Test set: official Stanford Cars test split ({c['n_test']} images, never used for training or tuning). "
        "Validation: stratified 10% of the official train split.",
        "",
        "## 1. Classification performance",
        "| Metric | Value |", "|---|---|",
        f"| Top-1 accuracy | {pct(c['top1'])} |", f"| Top-5 accuracy | {pct(c['top5'])} |",
        f"| Top-1 with flip TTA | {pct(c['top1_tta'])} |",
        "",
        "## 2. Object localization (IoU)",
        "| Method | Mean IoU | IoU >= 0.5 |", "|---|---|---|",
        f"| {detm['model']} | {detm['mean_iou']:.3f} | {pct(detm['iou50_rate'])} |",
        *([f"| {pre['model']} | {pre['mean_iou']:.3f} | {pct(pre['iou50_rate'])} |"] if pre else []),
        f"| Largest edge contour (classical) | {roi.get('mean_iou', float('nan')):.3f} | {pct(roi.get('iou50_rate'))} |",
        f"(n = {detm['n']} / {roi.get('n')} random test images)",
        "",
        "## 3. Object detection (mAP)",
        "| Metric | Value |", "|---|---|",
        f"| mAP@0.5 | {detm['map50']:.3f} |", f"| mAP@0.5:0.95 | {detm['map50_95']:.3f} |",
        *([f"| mAP@0.5 / @0.5:0.95 before fine-tuning (COCO-pretrained) | {pre['map50']:.3f} / {pre['map50_95']:.3f} |"] if pre else []),
        "",
        "## 4. Model efficiency",
        "| Model | Size | CPU latency (2 threads) | Top-1 (subset) |", "|---|---|---|---|",
        f"| Classifier FP32 (ONNX) | {e['fp32_mb']:.1f} MB | {e['fp32_ms_cpu2']:.0f} ms | {pct(e['fp32_top1_subset'])} |",
        f"| Classifier INT8 (dynamic quantization) | {e['int8_mb']:.1f} MB | {e['int8_ms_cpu2']:.0f} ms | {pct(e['int8_top1_subset'])} |",
        f"| YOLO11s detector | - | {detm['ms_per_image']:.0f} ms ({detm['device']}) | - |",
        f"Parameters: {e['params_m']:.1f} M. Subset = first {e['subset_n']} test images.",
        "",
        "## 5. Robustness (top-1 under degraded conditions)",
        f"{m['robustness_n']} random test images.", "",
        "| Condition | Severity | Top-1 |", "|---|---|---|",
        *[f"| {r['condition']} | {r['severity']} | {pct(r['top1'])} |" for r in m["robustness"]],
        "",
        "Classical CLAHE before the network on dark photos: " +
        ", ".join(f"severity {s}: {pct(x)} -> {pct(y)}" for s, x, y in dark),
        "",
        "## 6. Confidence and threshold analysis",
        f"Temperature scaling: T = {cal['temperature']:.3f}, ECE {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}.",
        "",
        "| Answer only when most confident | Accuracy |", "|---|---|",
        *[f"| top {int(r['coverage'] * 100)}% of photos | {pct(r['accuracy'])} |" for r in m["risk_coverage"]],
        "",
        "High-voltage (hybrid/EV) warning threshold on summed calibrated probability:", "",
        "| Threshold | HV miss rate | False alarm rate |", "|---|---|---|",
        *[f"| {r['threshold']} | {pct(r['hv_miss_rate'])} | {pct(r['hv_false_alarm_rate'])} |" for r in saf["by_threshold"]],
        f"Plain top-1 (argmax) would miss {pct(saf['argmax_hv_miss_rate'])} of hybrid/EV test cars ({saf['n_hv_test']} images).",
        f"Unknown-car check: {pct(m['unknown']['known_test_flagged_rate'])} of known test cars wrongly flagged unknown; "
        f"{pct(unk.get('unknown_catch_rate'))} of {unk.get('n_vehicle_found', 0)} photos of cars outside the 196 classes "
        "(Indian / post-2012 models) correctly answered 'unknown'.",
        "",
        "## Approach comparison (classical vs deep learning vs hybrid)",
        "| Approach | What it is | Top-1 | Top-5 |", "|---|---|---|---|",
        f"| Classical | HOG + HSV colour + LBP + geometry -> PCA -> logistic regression | {pct(base.get('test_top1'))} | {pct(base.get('test_top5'))} |",
        f"| Deep learning | ConvNeXt-T on the whole photo | {pct(c['top1_no_crop'])} | - |",
        f"| Hybrid | detector crop + CLAHE when dark + calibrated fusion + safety layer | {pct(hybrid_top1)} | {pct(hybrid_top5)} |",
        f"Fusion weight on classical probabilities (chosen on validation): {fusion['weight'] if fusion else 'n/a'}.",
        "",
        "## Ablation (each row removes or swaps one component)",
        "| Variant | Top-1 |", "|---|---|",
        f"| Full hybrid | {pct(hybrid_top1)} |",
        f"| - classical fusion | {pct(c['top1'])} |",
        f"| - detection crop (whole photo) | {pct(c['top1_no_crop'])} |",
        f"| + flip TTA | {pct(c['top1_tta'])} |",
        f"| FP32 instead of INT8 (subset) | {pct(e['fp32_top1_subset'])} vs {pct(e['int8_top1_subset'])} |",
        "",
        *([
            "## Brand experts (make accuracy, 49 brands)",
            f"Whole-car classifier summed per brand vs a close-up expert that votes over 9 zoomed windows (badge, grille, "
            f"lights); combined with the {brand['fusion_rule']} rule, weight {brand['fusion_weight']:.1f} (chosen on validation, clean + occluded).",
            "", "| Condition | Whole car | Close-ups | Combined |", "|---|---|---|---|",
            f"| Clean test ({brand['test_clean']['n']}) | {pct(brand['test_clean']['whole'])} | {pct(brand['test_clean']['parts'])} | {pct(brand['test_clean']['fused'])} |",
            f"| 30% of the vehicle hidden ({brand['test_occluded_30']['n']}) | {pct(brand['test_occluded_30']['whole'])} | {pct(brand['test_occluded_30']['parts'])} | {pct(brand['test_occluded_30']['fused'])} |",
            "",
        ] if brand else []),
        "## Most-confused classes (failure-case candidates)",
        "| True | Predicted | Count |", "|---|---|---|",
        *[f"| {t['true']} | {t['predicted']} | {t['count']} |" for t in m["top_confusions"][:10]],
    ]
    out_md = Path(f"eval_P5_{a.team}.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {a.out / 'metrics_public.json'} and {out_md}")


if __name__ == "__main__":
    main()
