"""The TypeScript port for offline mode is tested against the same file (frontend/e2e/offline-safety.spec.ts)."""
import json
from pathlib import Path

import numpy as np
import pytest

from app.safety import Thresholds, decide, load_classes

GOLDEN = json.loads((Path(__file__).resolve().parents[2] / "docs" / "fixtures" / "safety_golden.json").read_text(encoding="utf-8"))
CLASSES = load_classes(GOLDEN["class_names"])


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_python_engine_still_gives_the_golden_answer(case):
    got = decide(np.array(case["probs"]), case["unknown_score"], CLASSES, Thresholds(**GOLDEN["thresholds"]))
    assert json.loads(json.dumps(got)) == case["expected"]
