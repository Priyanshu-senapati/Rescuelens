"""Photo -> API response: detect -> classical analysis -> classify -> safety decision (shape: docs/api-contract.md)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import classical
from .safety import DISCLAIMER, CarClass, Thresholds, _safety, decide, load_classes

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


def _normalise_product(dists: list[np.ndarray]) -> np.ndarray:
    """Product of independent views' distributions (computed in logs), renormalised. With one view it is that view."""
    logp = np.sum([np.log(d + 1e-12) for d in dists], axis=0)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def _named(v: dict, eng: Engine) -> tuple[str, str | None] | None:
    """(make, family or None) that one photo names with confidence, else None. Unfamiliar-looking photos count too:
    a crushed front rarely names a brand with confidence (4 of 45 real crash photos), and when it does, a clash is real."""
    t = eng.thresholds
    make_p: dict[str, float] = {}
    family_p: dict[str, float] = {}
    for p, c in zip(v["probs"], eng.classes):
        make_p[c.make] = make_p.get(c.make, 0.0) + float(p)
        family_p[c.family] = family_p.get(c.family, 0.0) + float(p)
    make, family = max(make_p, key=make_p.get), max(family_p, key=family_p.get)
    if make_p[make] < t.make:
        return None
    return make, family if family_p[family] >= t.family and family.startswith(make) else None


def _disagreement(views: list[dict | None], eng: Engine) -> tuple[str, bool] | None:
    """(reason, different makes?) when two photos confidently name different vehicles. Different makes are almost
    certainly two vehicles: multiplying them would let the clearer one silently win. Different families of one make
    (Audi A4 front vs S4 rear) are often look-alike trims of one car, so they are only reported."""
    named = [(i, n) for i, v in enumerate(views) if v is not None and (n := _named(v, eng))]
    pairs = [(i, a, j, b) for k, (i, a) in enumerate(named) for j, b in named[k + 1:]]
    for i, (make_i, fam_i), j, (make_j, fam_j) in sorted(pairs, key=lambda q: q[1][0] == q[3][0]):
        if make_i != make_j or (fam_i and fam_j and fam_i != fam_j):
            return f"Photo {i + 1} reads as {fam_i or make_i}, photo {j + 1} as {fam_j or make_j}", make_i != make_j
    return None


def _view(rgb: np.ndarray, eng: Engine) -> dict | None:
    """Evidence from one photo of the vehicle: detection, classical analysis, class probabilities, brand close-ups."""
    t0 = time.perf_counter()
    dets = eng.detector(rgb)
    found_after_boost = False
    if not dets:                             # hybrid: dark scenes often detect after CLAHE
        dets = eng.detector(classical.enhance(rgb))
        found_after_boost = bool(dets)
    t_det = time.perf_counter()
    if not dets:
        return None
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
    details = eng.brand(car) if eng.brand is not None else None   # zoomed windows (badge, grille, lights)
    return {"rgb": rgb, "box": box, "score": score, "others": others, "cl": cl, "car": car, "probs": probs,
            "unknown": float(unknown_score), "night": night, "details": details, "found_after_boost": found_after_boost,
            "ms": {"detect": _ms(t0, t_det), "classical": _ms(t_det, t_cl), "classify": _ms(t_cl, time.perf_counter())}}


def analyze(rgb: np.ndarray, eng: Engine) -> dict:
    return analyze_views([rgb], eng)


def analyze_views(rgbs: list[np.ndarray], eng: Engine) -> dict:
    """1-3 photos of the same vehicle (e.g. a crushed front and an intact rear). Each view is analysed on its own; class
    probabilities are multiplied across views, so an uninformative view barely moves the answer and an intact view can
    decide it. The first photo with a vehicle is the one shown (box, pipeline steps, comparison)."""
    t0 = time.perf_counter()
    h, w = rgbs[0].shape[:2]
    base = {"request_id": uuid.uuid4().hex[:12], "image": {"width": w, "height": h}, "disclaimer": DISCLAIMER}
    views = [_view(rgb, eng) for rgb in rgbs]
    found = [v for v in views if v is not None]
    summaries = [{"index": i, "vehicle_found": v is not None,
                  "top": _top(v["probs"], eng.classes)["label"] if v else None,
                  "prob": _top(v["probs"], eng.classes)["prob"] if v else None,
                  "unknown_score": round(v["unknown"], 4) if v else None,
                  "quality_issues": v["cl"]["quality"]["issues"] if v else []} for i, v in enumerate(views)]
    if not found:
        return {**base, "status": "no_vehicle", "headline": "No vehicle found - try a clearer photo of the car",
                "detection": None, "other_boxes": [], "prediction": None, "hierarchy": [], "resolved_level": "none",
                "top5": [], "safety": None, "unknown": None, "quality": None, "reliability": None, "steps": [],
                "features": None, "comparison": None, "brand_evidence": None, "views": summaries,
                "timing_ms": {"detect": _ms(t0, time.perf_counter()), "classify": 0, "classical": 0,
                              "total": _ms(t0, time.perf_counter())}}

    main = found[0]
    if main["rgb"] is not rgbs[0]:
        h, w = main["rgb"].shape[:2]
        base["image"] = {"width": w, "height": h}
    clearest = min(found, key=lambda v: v["unknown"])       # the view that looks most like a car we know
    probs = _normalise_product([v["probs"] for v in found])
    unknown_score = clearest["unknown"]
    deep_probs, _ = eng.classifier(main["rgb"])             # plain deep learning: whole photo, no crop, no enhancement
    brand_evidence = None
    if eng.brand is not None:
        details = _normalise_product([v["details"] for v in found])
        probs, whole, combined = fuse_brand(probs, eng.classes, eng.brand_makes, details, eng.brand_weight, eng.brand_rule)
        brand_evidence = {"whole_car": _top_makes(whole, eng.brand_makes), "details": _top_makes(details, eng.brand_makes),
                          "combined": _top_makes(combined, eng.brand_makes), "weight": round(eng.brand_weight, 2),
                          "conflict": bool(whole.argmax() != details.argmax())}

    decision = decide(probs, unknown_score, eng.classes, eng.thresholds, clearest["cl"]["quality"]["issues"])
    reasons = decision["reliability"]["reasons"]
    clash = _disagreement(views, eng)
    if clash:
        reasons.insert(0, clash[0])
    if clash and clash[1]:                                  # never merge two brands into one confident answer
        decision = {**decision, "status": "unknown", "resolved_level": "none",
                    "headline": "These photos seem to show different vehicles - photograph one vehicle only",
                    "safety": _safety(probs, eng.classes, True, eng.thresholds),
                    "reliability": {"score": 0.0, "reasons": reasons}}
    if len(found) > 1:
        reasons.append(f"Combined {len(found)} photos of the vehicle; the clearest one decides whether it looks familiar")
        reasons.extend(f"Photo {i + 1} on its own was not recognised; the answer rests on the other photos"
                       for i, v in enumerate(views) if v is not None and v["unknown"] > eng.thresholds.unknown
                       and v is not clearest)
    for v in found:
        if v["night"]:
            reasons.append(f"Low light: also read a contrast-boosted copy (top answer {v['night'][0]:.0%} -> {v['night'][1]:.0%})")
    if brand_evidence and brand_evidence["conflict"]:
        reasons.append(f"Close-up details point to {brand_evidence['details'][0]['make']}, "
                       f"the whole-car view to {brand_evidence['whole_car'][0]['make']}")
    steps = main["cl"]["steps"]
    if main["found_after_boost"]:
        steps[0] = {**steps[0], "caption": steps[0]["caption"] + " (found only after contrast boost)"}

    return {
        **base, **decision,
        "detection": {"box": [int(v) for v in main["box"]], "score": round(float(main["score"]), 4)},
        "other_boxes": [[int(v) for v in b] for b, _ in main["others"]],
        "quality": main["cl"]["quality"], "steps": steps, "features": main["cl"]["features"],
        "comparison": {
            "classical": _top(eng.baseline(main["car"]), eng.classes) if eng.baseline else None,
            "deep": _top(deep_probs, eng.classes),
            "hybrid": {**_top(probs, eng.classes), "status": decision["status"]},
        },
        "brand_evidence": brand_evidence,
        "views": summaries,
        "timing_ms": {k: sum(v["ms"][k] for v in found) for k in ("detect", "classical", "classify")}
                     | {"total": _ms(t0, time.perf_counter())},
    }
