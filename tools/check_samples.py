"""Pre-test the demo samples against the real models before showing them to anyone.

python tools/check_samples.py                  (uses ./models)

A file named ambiguous-* must answer "ambiguous" and unknown-* must answer "unknown"; the others just print what they answer.
Writes nothing. Exits 1 if a named sample misbehaves: swap the photo, don't change the label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.main import load_rgb            # noqa: E402  (same image loading as the live API)
from app.pipeline import analyze, load_engine  # noqa: E402

EXPECTED = {"ambiguous": "ambiguous", "unknown": "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--samples", type=Path, default=ROOT / "frontend" / "public" / "samples")
    a = ap.parse_args()

    eng = load_engine(a.models)
    wrong = 0
    for p in sorted(a.samples.glob("*.jpg")):
        r = analyze(load_rgb(p.read_bytes()), eng)
        want = EXPECTED.get(p.stem.split("-")[0])
        verdict = "" if want is None else ("OK" if r["status"] == want else f"WRONG, expected {want}")
        wrong += verdict.startswith("WRONG")
        hv = (r["safety"] or {}).get("hv_warning")
        print(f"{p.name:28s} {r['status']:11s} hv={hv!s:5s} {r['headline'][:60]:60s} {verdict}")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
