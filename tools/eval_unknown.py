"""Run the real app pipeline on photos of cars that are NOT in Stanford Cars (Indian / post-2012 models)
and report how often RescueLens says "unknown" instead of guessing.

python tools/eval_unknown.py private_photos/unknown            (uses ./models)
Writes models/unknown_eval.json, which tools/build_report.py picks up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.main import load_rgb            # noqa: E402  (same image loading as the live API)
from app.pipeline import analyze, load_engine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    a = ap.parse_args()

    eng = load_engine(a.models)
    photos = sorted(p for p in a.folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"})
    assert photos, f"no photos in {a.folder}"
    rows = []
    for p in photos:
        r = analyze(load_rgb(p.read_bytes()), eng)
        rows.append({"photo": p.name, "status": r["status"], "unknown_score": (r["unknown"] or {}).get("score"),
                     "top1": (r["prediction"] or {}).get("label"), "confidence": (r["prediction"] or {}).get("confidence")})
        print(f"{r['status']:11s} score={rows[-1]['unknown_score']}  top1={rows[-1]['top1']}  ({p.name})")

    cars = [r for r in rows if r["status"] != "no_vehicle"]
    scores = np.array([r["unknown_score"] for r in cars], float)
    summary = {
        "n_photos": len(rows), "n_vehicle_found": len(cars),
        "unknown_catch_rate": float(np.mean([r["status"] == "unknown" for r in cars])) if cars else None,
        "ood_score_catch_rate": float(np.mean(scores > eng.thresholds.unknown)) if cars else None,
        "threshold": eng.thresholds.unknown,
        "catch_rate_if_threshold": {f"{t:.2f}": float(np.mean(scores > t)) for t in np.linspace(0.1, 0.6, 11)} if cars else {},
        "rows": rows,
    }
    (a.models / "unknown_eval.json").write_text(json.dumps(summary, indent=1))
    print(f"\nunknown catch rate: {summary['unknown_catch_rate']} over {len(cars)} photos with a vehicle "
          f"(threshold {eng.thresholds.unknown:.3f})")


if __name__ == "__main__":
    main()
