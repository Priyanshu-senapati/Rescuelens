"""Turns class probabilities into what is safe to conclude: identified / ambiguous / unknown + high-voltage advice."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
TWO_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
DISCLAIMER = ("Decision-support prototype. Vehicle identification is probabilistic. "
              "Never rely on it alone for cutting, lifting or disabling a vehicle.")
HV_ACTIONS = ["Do not cut orange high-voltage cables", "Immobilise the vehicle and switch off the ignition",
              "Look for HYBRID / EV badges before cutting", "Confirm the vehicle with VIN or plate"]


@dataclass(frozen=True)
class CarClass:
    label: str
    make: str
    model: str
    year: int
    body: str
    family: str
    powertrain: str          # hybrid | ev | combustion
    guide: str | None


@dataclass(frozen=True)
class Thresholds:
    identify: float = 0.6    # top class prob needed to name the exact vehicle
    family: float = 0.6      # family prob needed to name the family
    make: float = 0.5        # below this the model has no real idea -> unknown
    hv: float = 0.05         # summed hybrid/EV prob that triggers the high-voltage warning (chosen on miss vs false-alarm table)
    unknown: float = 0.5     # unknown-car score threshold (set from training config)


def parse_label(label: str, hazards: dict) -> CarClass:
    words = label.split()
    n = 2 if label.startswith(TWO_WORD_MAKES) else 1
    make, model_words, year = " ".join(words[:n]), words[n:-1], int(words[-1])
    family_words = model_words[:2] if len(model_words) > 1 and model_words[1] == "Series" else model_words[:1]
    hz = hazards.get(label, {})
    return CarClass(label=label, make=make, model=" ".join(model_words), year=year, body=words[-2],
                    family=" ".join([make, *family_words]), powertrain=hz.get("powertrain", "combustion"),
                    guide=hz.get("guide"))


def load_classes(names: list[str] | None = None) -> list[CarClass]:
    """Class list in model-output order. `names` comes from the model config; defaults to data/classnames.txt."""
    if names is None:
        names = (DATA_DIR / "classnames.txt").read_text(encoding="utf-8").strip().splitlines()
    hazards = json.loads((DATA_DIR / "hazards.json").read_text(encoding="utf-8"))["high_voltage"]
    return [parse_label(n, hazards) for n in names]


def _summary(c: CarClass, p: float) -> dict:
    return {"label": c.label, "make": c.make, "model": c.model, "year": c.year, "prob": round(float(p), 4),
            "powertrain": c.powertrain}


def _safety(probs: np.ndarray, classes: list[CarClass], unknown: bool, t: Thresholds) -> dict:
    if unknown:
        return {"hv_probability": None, "hv_warning": True, "level": "caution",
                "message": "Vehicle not recognised. Powertrain unknown - treat as possibly high-voltage.",
                "actions": ["Do not use this prediction for rescue decisions",
                            "Identify the vehicle manually (VIN, plate, badges)",
                            "Check for HYBRID / EV badges and orange cables"], "guide": None}
    hv_idx = [i for i, c in enumerate(classes) if c.powertrain != "combustion"]
    p_hv = float(probs[hv_idx].sum())
    if p_hv < t.hv:
        return {"hv_probability": round(p_hv, 4), "hv_warning": False, "level": "info",
                "message": "No high-voltage system expected for this vehicle.",
                "actions": ["Standard airbag and seat-belt pretensioner precautions apply",
                            "Confirm the vehicle with VIN or plate"], "guide": None}
    top_hv = sorted(hv_idx, key=lambda i: -probs[i])
    with_guide = next((classes[i] for i in top_hv if classes[i].guide), None)
    certain = classes[int(probs.argmax())].powertrain != "combustion" and probs.max() >= t.identify
    message = ("Hybrid / EV identified: HIGH VOLTAGE system present." if certain
               else f"Hybrid possible ({p_hv:.0%}). Treat as HIGH VOLTAGE until verified.")
    return {"hv_probability": round(p_hv, 4), "hv_warning": True, "level": "danger", "message": message,
            "actions": HV_ACTIONS,
            "guide": {"title": f"{with_guide.label} - Emergency Response Guide", "url": with_guide.guide} if with_guide else None}


def decide(probs: np.ndarray, unknown_score: float, classes: list[CarClass], t: Thresholds,
           quality_issues: list[str] | None = None) -> dict:
    """probs = calibrated softmax over the classes. Returns the decision part of the API response."""
    order = np.argsort(-probs)
    top = classes[order[0]]
    p_top = float(probs[order[0]])
    p_family = float(sum(probs[i] for i, c in enumerate(classes) if c.family == top.family))
    p_make = float(sum(probs[i] for i, c in enumerate(classes) if c.make == top.make))
    is_ood = unknown_score > t.unknown
    unknown = is_ood or p_make < t.make

    if unknown:
        status, level = "unknown", "none"
        headline = ("Unknown vehicle - not one of the 196 supported models" if is_ood
                    else "Cannot identify this vehicle with enough confidence")
    elif p_top >= t.identify:
        status, level, headline = "identified", "vehicle", top.label
    elif p_family >= t.family:
        years = sorted({classes[i].year for i in order[:5] if classes[i].family == top.family and probs[i] >= 0.05})
        year_txt = str(years[0]) if len(years) == 1 else " or ".join(map(str, years))
        status, level = "ambiguous", "family"
        headline = f"{top.family} family ({year_txt}) - exact variant uncertain"
    else:
        status, level, headline = "ambiguous", "make", f"{top.make} - model uncertain"

    safety = _safety(probs, classes, unknown, t)
    if status == "ambiguous" and safety["hv_warning"]:
        headline += " - hybrid cannot be ruled out"

    second = classes[order[1]]
    reasons = list(quality_issues or [])
    if second.family == top.family and probs[order[1]] >= 0.15:
        reasons.append("Top two answers are a known look-alike pair")
    if is_ood:
        reasons.append("Looks unlike every car in the training set")

    return {
        "status": status, "headline": headline, "resolved_level": level,
        "prediction": {"label": top.label, "make": top.make, "model": top.model, "year": top.year, "body": top.body,
                       "confidence": round(p_top, 4)},
        "hierarchy": [{"level": "make", "name": top.make, "prob": round(p_make, 4)},
                      {"level": "family", "name": top.family, "prob": round(p_family, 4)},
                      {"level": "vehicle", "name": top.label, "prob": round(p_top, 4)}],
        "top5": [_summary(classes[i], probs[i]) for i in order[:5]],
        "safety": safety,
        "unknown": {"score": round(float(unknown_score), 4), "threshold": t.unknown, "is_unknown": bool(is_ood)},
        "reliability": {"score": round(0.0 if unknown else p_top, 4), "reasons": reasons},
        "disclaimer": DISCLAIMER,
    }
