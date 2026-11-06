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
HV_ACTIONS = ["Do not cut orange high-voltage cables (blue on some 36 V hybrids)", "Immobilise the vehicle and switch off the ignition",
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
    guide_title: str | None = None
    battery: str | None = None
    hv_cables: str | None = None


NO_LOCATION = {"location_vehicle": None, "battery": None, "hv_cables": None}


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
                    guide=hz.get("guide"), guide_title=hz.get("guide_title"), battery=hz.get("battery"),
                    hv_cables=hz.get("hv_cables"))


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
                            "Check for HYBRID / EV badges and orange cables"], "guide": None, **NO_LOCATION}
    hv_idx = [i for i, c in enumerate(classes) if c.powertrain != "combustion"]
    p_hv = float(probs[hv_idx].sum())
    if p_hv < t.hv:
        return {"hv_probability": round(p_hv, 4), "hv_warning": False, "level": "info",
                "message": "No high-voltage system expected for this vehicle.",
                "actions": ["Standard airbag and seat-belt pretensioner precautions apply",
                            "Confirm the vehicle with VIN or plate"], "guide": None, **NO_LOCATION}
    top_hv = sorted(hv_idx, key=lambda i: -probs[i])
    with_guide = next((classes[i] for i in top_hv if classes[i].guide), None)
    likeliest = classes[top_hv[0]]           # only this car's location: another model's layout could mislead
    certain = classes[int(probs.argmax())].powertrain != "combustion" and probs.max() >= t.identify
    message = ("Hybrid / EV identified: HIGH VOLTAGE system present." if certain
               else f"Hybrid possible ({p_hv:.0%}). Treat as HIGH VOLTAGE until verified.")
    return {"hv_probability": round(p_hv, 4), "hv_warning": True, "level": "danger", "message": message,
            "actions": HV_ACTIONS,
            "guide": {"title": with_guide.guide_title or f"{with_guide.label} - Emergency Response Guide",
                      "url": with_guide.guide} if with_guide else None,
            "location_vehicle": likeliest.label if likeliest.battery or likeliest.hv_cables else None,
            "battery": likeliest.battery, "hv_cables": likeliest.hv_cables}


def decide(probs: np.ndarray, unknown_score: float, classes: list[CarClass], t: Thresholds,
           quality_issues: list[str] | None = None) -> dict:
    """probs = calibrated softmax over the classes. Returns the decision part of the API response.

    Brand first: the make is the one with the most summed probability, then the strongest family inside that make, then
    the likeliest vehicle inside that family. A single top class can't overrule the combined evidence for another brand
    (e.g. BMW 3 Series 16% vs Audi S5 12% + S4 6% + A5 + ... on a damaged Audi)."""
    order = np.argsort(-probs)
    make_p: dict[str, float] = {}
    family_p: dict[str, float] = {}
    for i, c in enumerate(classes):
        make_p[c.make] = make_p.get(c.make, 0.0) + float(probs[i])
        family_p[c.family] = family_p.get(c.family, 0.0) + float(probs[i])
    make = max(make_p, key=make_p.get)                                       # ties: first in class order
    family = max((c.family for c in classes if c.make == make), key=family_p.get)
    top_i = max((i for i, c in enumerate(classes) if c.family == family), key=lambda i: probs[i])
    top, p_top = classes[top_i], float(probs[top_i])
    p_make, p_family = make_p[make], family_p[family]
    single, p_single = classes[order[0]], float(probs[order[0]])            # raw top-1, before brand reasoning
    is_ood = unknown_score > t.unknown
    unknown = is_ood or p_make < t.make

    if unknown:
        status, level = "unknown", "none"
        if is_ood and quality_issues and p_make >= t.make:     # the photo, not the car, is the likely problem
            headline = f"Low-confidence match: {top.label} - photo quality prevents a reliable identification"
        elif is_ood:
            headline = "Not recognised - damaged, unusual or not one of the 196 supported models"
        else:
            headline = "Cannot identify this vehicle with enough confidence"
    elif p_top >= t.identify:
        status, level, headline = "identified", "vehicle", top.label
    elif p_family >= t.family:
        years = sorted({classes[i].year for i in order[:5] if classes[i].family == family and probs[i] >= 0.05})
        year_txt = str(years[0]) if len(years) == 1 else " or ".join(map(str, years))
        status, level = "ambiguous", "family"
        headline = f"{family} family ({year_txt}) - exact variant uncertain"
    else:
        status, level, headline = "ambiguous", "make", f"{make} - model uncertain"

    safety = _safety(probs, classes, unknown, t)
    if status == "ambiguous" and safety["hv_warning"]:
        headline += " - hybrid cannot be ruled out"

    second = classes[order[1]]
    reasons = list(quality_issues or [])
    if second.family == single.family and probs[order[1]] >= 0.15:
        reasons.append("Top two answers are a known look-alike pair")
    if single.make != make:
        reasons.append(f"The single top guess ({single.label}, {p_single:.0%}) is not a {make}: "
                       f"the {make} models together add up to {p_make:.0%}")
    if status != "identified" and p_single < 0.3:
        reasons.append(f"No single model stands out (top answer only {p_single:.0%})")
    if is_ood:
        reasons.append("Looks unlike the training photos: damage, an unusual angle, or a model outside the 196")

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
