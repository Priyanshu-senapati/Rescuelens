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
    clahe_when_dark: bool = True             # hybrid: boost contrast before classifying dark photos


def load_engine(model_dir: Path) -> Engine:
    from .models import Baseline, Classifier, Detector   # onnxruntime only needed when real models exist
    clf = Classifier(model_dir)
    baseline_path = model_dir / "baseline.npz"
    return Engine(detector=Detector(model_dir / "yolo11s.onnx"), classifier=clf,
                  baseline=Baseline(baseline_path) if baseline_path.exists() else None,
                  classes=load_classes(clf.class_names),
                  thresholds=Thresholds(hv=clf.hv_threshold, unknown=clf.unknown_threshold))


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
                "features": None, "comparison": None,
                "timing_ms": {"detect": _ms(t0, t_det), "classify": 0, "classical": 0, "total": _ms(t0, time.perf_counter())}}

    (box, score), others = dets[0], dets[1:6]
    cl = classical.analyse(rgb, box)
    t_cl = time.perf_counter()

    car = classical.crop(rgb, box)
    dark = cl["quality"]["brightness"] < 30
    hybrid_input = classical.crop(classical.enhance(rgb), box) if dark and eng.clahe_when_dark else car
    probs, unknown_score = eng.classifier(hybrid_input)
    deep_probs, _ = eng.classifier(rgb)      # plain deep learning: whole photo, no crop, no enhancement
    t_clf = time.perf_counter()

    decision = decide(probs, unknown_score, eng.classes, eng.thresholds, cl["quality"]["issues"])
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
        "timing_ms": {"detect": _ms(t0, t_det), "classical": _ms(t_det, t_cl), "classify": _ms(t_cl, t_clf),
                      "total": _ms(t0, time.perf_counter())},
    }
