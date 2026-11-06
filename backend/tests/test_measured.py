"""The Model report must never show numbers that disagree with the evaluation file judges read."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = json.loads((ROOT / "data" / "measured.json").read_text(encoding="utf-8"))
EVAL = (ROOT / "eval_P5_T50.md").read_text(encoding="utf-8")


def as_rate(pct: str) -> float:
    return round(float(pct) / 100, 4)


def test_threshold_table_matches_the_eval_file():
    rows = re.findall(r"^\| (0\.\d+) \| ([\d.]+)% \| ([\d.]+)% \|$", EVAL, re.M)
    assert [(float(t), as_rate(a), as_rate(b)) for t, a, b in rows] == [
        (r["threshold"], r["hv_miss_rate"], r["hv_false_alarm_rate"]) for r in M["hv_thresholds"]
    ]


def test_top1_miss_rate_and_unknown_result_match_the_eval_file():
    assert as_rate(re.search(r"\(argmax\) would miss ([\d.]+)%", EVAL).group(1)) == M["argmax_hv_miss_rate"]
    flagged, caught, photos = re.search(r"([\d.]+)% of known test cars wrongly flagged unknown; ([\d.]+)% of (\d+) photos", EVAL).groups()
    assert as_rate(flagged) == M["unknown"]["known_flagged_rate"]
    assert as_rate(caught) == M["unknown"]["catch_rate"]
    assert int(photos) == M["unknown"]["photos"]
    assert round(M["unknown"]["caught"] / M["unknown"]["photos"], 4) == M["unknown"]["catch_rate"]
