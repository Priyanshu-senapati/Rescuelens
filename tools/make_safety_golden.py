"""Regenerates docs/fixtures/safety_golden.json: what backend/app/safety.py answers for a fixed set of cases.

python tools/make_safety_golden.py      (run after changing the safety rules on purpose, then port the change to TypeScript)
The Python and TypeScript engines are both tested against this file, so they can't quietly disagree on a safety rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.safety import Thresholds, decide, load_classes  # noqa: E402

OUT = ROOT / "docs" / "fixtures" / "safety_golden.json"
THRESHOLDS = {"identify": 0.6, "family": 0.6, "make": 0.5, "hv": 0.05, "unknown": 0.5}
NAMED = [
    ("confident petrol", {"BMW M3 Coupe 2012": 0.94}, 0.1),
    ("confident EV", {"Tesla Model S Sedan 2012": 0.95}, 0.1),
    ("confident hybrid, no location clash", {"Chevrolet Malibu Hybrid Sedan 2010": 0.9}, 0.1),
    ("sonata or sonata hybrid", {"Hyundai Sonata Hybrid Sedan 2012": 0.48, "Hyundai Sonata Sedan 2012": 0.43}, 0.2),
    ("no single hybrid wins", {"Chevrolet Tahoe Hybrid SUV 2012": 0.25, "GMC Yukon Hybrid SUV 2012": 0.20, "Chevrolet Avalanche Crew Cab 2012": 0.30}, 0.2),
    ("two years of one family", {"Audi S4 Sedan 2012": 0.5, "Audi S4 Sedan 2007": 0.4}, 0.1),
    ("make only", {"Audi S5 Coupe 2012": 0.4, "Audi A5 Coupe 2012": 0.35}, 0.1),
    ("unseen car", {"Toyota Camry Sedan 2012": 0.93}, 0.9),
    ("flat distribution", {}, 0.1),
    ("just under the warning", {"BMW M3 Coupe 2012": 0.9, "Tesla Model S Sedan 2012": 0.03}, 0.1),
    ("just over the warning", {"BMW M3 Coupe 2012": 0.85, "Tesla Model S Sedan 2012": 0.07}, 0.1),
    ("brand beats the single top guess", {"BMW 3 Series Sedan 2012": 0.16, "Audi S5 Coupe 2012": 0.14, "Audi S4 Sedan 2012": 0.12,
                                          "Audi A5 Coupe 2012": 0.11, "Audi S5 Convertible 2012": 0.08, "Audi S4 Sedan 2007": 0.06}, 0.2),
]


def main() -> None:
    classes = load_classes()
    index = {c.label: i for i, c in enumerate(classes)}
    t = Thresholds(**THRESHOLDS)
    cases = []
    for name, named, score in NAMED:
        p = np.zeros(len(classes))
        for label, v in named.items():
            p[index[label]] = v
        rest = p == 0
        ramp = np.arange(1, rest.sum() + 1, dtype=float)   # all different: engines may break exact ties differently
        p[rest] = (1 - p.sum()) * ramp / ramp.sum()
        cases.append({"name": name, "probs": [round(float(v), 8) for v in p], "unknown_score": score, "expected": decide(p, score, classes, t)})
    OUT.write_text(json.dumps({"thresholds": THRESHOLDS, "class_names": [c.label for c in classes], "cases": cases}, indent=None), encoding="utf-8")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
