"""Real crashed / damaged cars (Wikimedia Commons, brand known from the photo title): does the close-up brand expert
fix the brand decision? Runs the live pipeline twice per photo - whole-car only and with the expert.

python tools/eval_crashed.py private_photos/crashed          (uses ./models; writes models/crashed_eval.json)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.main import load_rgb               # noqa: E402  (same image loading as the live API)
from app.pipeline import analyze, load_engine  # noqa: E402

BRANDS = {"mercedes-benz": "Mercedes-Benz", "land-rover": "Land Rover", "bmw": "BMW"}


def brand_of(photo: Path) -> str:
    slug = photo.stem.rsplit("_", 1)[0]
    return BRANDS.get(slug, slug.capitalize())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    a = ap.parse_args()
    with_expert = load_engine(a.models)
    assert with_expert.brand is not None, "brand expert files not found in the models folder"
    whole_only = dataclasses.replace(with_expert, brand=None)
    photos = sorted(a.folder.glob("*.jpg"))
    rows = []
    for p in photos:
        rgb, truth = load_rgb(p.read_bytes()), brand_of(p)
        r0, r1 = analyze(rgb, whole_only), analyze(rgb, with_expert)
        if r1["status"] == "no_vehicle":
            continue
        b0, b1 = r0["hierarchy"][0]["name"], r1["hierarchy"][0]["name"]
        rows.append({"photo": p.name, "truth": truth, "whole_only": b0, "with_expert": b1, "status": r1["status"],
                     "headline": r1["headline"], "details_vote": (r1["brand_evidence"] or {}).get("details", [{}])[0].get("make")})
        print(f"{p.name:24s} truth={truth:14s} whole={b0:14s} expert={b1:14s} {r1['status']}")
    acc = lambda key: sum(r[key] == r["truth"] for r in rows) / max(1, len(rows))
    summary = {"n": len(rows), "brand_acc_whole_only": acc("whole_only"), "brand_acc_with_expert": acc("with_expert"),
               "fixed": sum(r["whole_only"] != r["truth"] and r["with_expert"] == r["truth"] for r in rows),
               "broken": sum(r["whole_only"] == r["truth"] and r["with_expert"] != r["truth"] for r in rows), "rows": rows}
    (a.models / "crashed_eval.json").write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
