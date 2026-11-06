"""Photo -> API response: detect -> classical analysis -> classify -> safety decision (shape: docs/api-contract.md)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import classical
from .safety import DISCLAIMER, CarClass, Thresholds, decide, load_classes

DetectFn = Callable[[np.ndarray], list]
ClassifyFn = Callable[[np.ndarray], tuple]


@dataclass(frozen=True)
class Engine:
    detector: DetectFn                       # rgb -> [((x1, y1, x2, y2), score), ...] main vehicle first
    classifier: ClassifyFn                   # rgb crop -> (calibrated probs, unknown score)
    baseline: Callable[[np.ndarray], np.ndarray] | None
    classes: list[CarClass]
    thresholds: Thresholds
    night_dual_pass: bool = True             # dark photos: also classify a CLAHE-boosted copy and average the two
    brand: Callable[[np.ndarray], np.ndarray] | None = None   # close-up brand expert: car crop -> make probabilities
    brand_makes: tuple[str, ...] = ()
    brand_weight: float = 0.0                                  # share of the close-up vote, chosen on validation
    brand_rule: str = "linear"                                 # linear average or product of the two votes


def load_engine(model_dir: Path) -> Engine:
    from .models import Baseline, BrandParts, Classifier, Detector   # onnxruntime only needed when real models exist
    clf = Classifier(model_dir)
    baseline_path = model_dir / "baseline.npz"
    brand = BrandParts(model_dir) if (model_dir / "brand_parts.json").exists() else None
    return Engine(detector=Detector(model_dir / "yolo11s.onnx"), classifier=clf,
                  baseline=Baseline(baseline_path) if baseline_path.exists() else None,
                  classes=load_classes(clf.class_names),
                  thresholds=Thresholds(hv=clf.hv_threshold, unknown=clf.unknown_threshold),
                  brand=brand, brand_makes=tuple(brand.makes) if brand else (), brand_weight=brand.weight if brand else 0.0,
                  brand_rule=brand.rule if brand else "linear")


def fuse_brand(probs: np.ndarray, classes: list[CarClass], makes: tuple[str, ...], details: np.ndarray,
               weight: float, rule: str = "linear") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-weights class probabilities so each make's total becomes the combined brand vote: linear
    (1 - w) * whole + w * close-up, or product whole^(1-w) * close-up^w (a confident expert dominates an unsure one).
    Ratios inside a make are kept, so the model ranking within a brand is the whole-car network's."""
    index = {m: i for i, m in enumerate(makes)}
    cls_make = np.array([index[c.make] for c in classes])
    whole = np.bincount(cls_make, weights=probs, minlength=len(makes))
    if rule == "product":
        target = np.exp((1 - weight) * np.log(whole + 1e-9) + weight * np.log(details + 1e-9))
        target = target / target.sum()
    else:
        target = (1 - weight) * whole + weight * details
    scale = np.divide(target, whole, out=np.zeros_like(target), where=whole > 0)
    fused = probs * scale[cls_make]
    return fused / fused.sum(), whole, target


def _top_makes(p: np.ndarray, makes: tuple[str, ...], k: int = 3) -> list[dict]:
    return [{"make": makes[i], "prob": round(float(p[i]), 4)} for i in np.argsort(-p)[:k]]


def _top(probs: np.ndarray, classes: list[CarClass]) -> dict:
    i = int(probs.argmax())
    return {"label": classes[i].label, "prob": round(float(probs[i]), 4)}


def _ms(a: float, b: float) -> int:
    return int(round((b - a) * 1000))


def analyze(rgb: np.ndarray, eng: Engine) -> dict:
    t0 = time.perf_counter()
    h, w = rgb.shape[:2]
    base = {"request_id": uuid.uuid4().hex[:12], "image": {"width": w, "height": h}, "disclaimer": DISCLAIMER}

    dets = eng.detector(rgb)
    found_after_boost = False
    if not dets:                             # hybrid: dark scenes often detect after CLAHE
        dets = eng.detector(classical.enhance(rgb))
        found_after_boost = bool(dets)
    t_det = time.perf_counter()

    if not dets:
        return {**base, "status": "no_vehicle", "headline": "No vehicle found - try a clearer photo of the car",
                "detection": None, "other_boxes": [], "prediction": None, "hierarchy": [], "resolved_level": "none",
                "top5": [], "safety": None, "unknown": None, "quality": None, "reliability": None, "steps": [],
                "features": None, "comparison": None, "brand_evidence": None,
                "timing_ms": {"detect": _ms(t0, t_det), "classify": 0, "classical": 0, "total": _ms(t0, time.perf_counter())}}

    (box, score), others = dets[0], dets[1:6]
    cl = classical.analyse(rgb, box)
    t_cl = time.perf_counter()

    car = classical.crop(rgb, box)
    probs, unknown_score = eng.classifier(car)
    night = None
    if eng.night_dual_pass and cl["quality"]["brightness"] < 30:   # low light: second opinion from a boosted copy
        p_boost, u_boost = eng.classifier(classical.crop(classical.enhance(rgb), box))
        night = (float(probs.max()), float(((probs + p_boost) / 2).max()))
        probs, unknown_score = (probs + p_boost) / 2, min(unknown_score, u_boost)
    deep_probs, _ = eng.classifier(rgb)      # plain deep learning: whole photo, no crop, no enhancement
    brand_evidence = None
    if eng.brand is not None:                # second expert: zoomed windows (badge, grille, lights) vote on the brand
        details = eng.brand(car)
        probs, whole, combined = fuse_brand(probs, eng.classes, eng.brand_makes, details, eng.brand_weight, eng.brand_rule)
        brand_evidence = {"whole_car": _top_makes(whole, eng.brand_makes), "details": _top_makes(details, eng.brand_makes),
                          "combined": _top_makes(combined, eng.brand_makes), "weight": round(eng.brand_weight, 2),
                          "conflict": bool(whole.argmax() != details.argmax())}
    t_clf = time.perf_counter()

    decision = decide(probs, unknown_score, eng.classes, eng.thresholds, cl["quality"]["issues"])
    if night:
        decision["reliability"]["reasons"].append(
            f"Low light: also read a contrast-boosted copy (top answer {night[0]:.0%} -> {night[1]:.0%})")
    if brand_evidence and brand_evidence["conflict"]:
        decision["reliability"]["reasons"].append(
            f"Close-up details point to {brand_evidence['details'][0]['make']}, the whole-car view to {brand_evidence['whole_car'][0]['make']}")
    steps = cl["steps"]
    if found_after_boost:
        steps[0] = {**steps[0], "caption": steps[0]["caption"] + " (found only after contrast boost)"}

    return {
        **base, **decision,
        "detection": {"box": [int(v) for v in box], "score": round(float(score), 4)},
        "other_boxes": [[int(v) for v in b] for b, _ in others],
        "quality": cl["quality"], "steps": steps, "features": cl["features"],
        "comparison": {
            "classical": _top(eng.baseline(car), eng.classes) if eng.baseline else None,
            "deep": _top(deep_probs, eng.classes),
            "hybrid": {**_top(probs, eng.classes), "status": decision["status"]},
        },
        "brand_evidence": brand_evidence,
        "timing_ms": {"detect": _ms(t0, t_det), "classical": _ms(t_det, t_cl), "classify": _ms(t_cl, t_clf),
                      "total": _ms(t0, time.perf_counter())},
    }
