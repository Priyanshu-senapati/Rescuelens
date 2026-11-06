import numpy as np

from app.pipeline import fuse_brand
from app.safety import Thresholds, decide, load_classes

CLASSES = load_classes()
MAKES = tuple(sorted({c.make for c in CLASSES}))      # same order as training/train_brand_parts.py
IDX = {c.label: i for i, c in enumerate(CLASSES)}


def damaged_audi_whole_car() -> np.ndarray:
    """The reported failure: the whole-car view leans BMW, the Audi evidence is there but spread out."""
    named = {"BMW 3 Series Sedan 2012": 0.16, "BMW M3 Coupe 2012": 0.24, "Audi S5 Coupe 2012": 0.12, "Audi S4 Sedan 2012": 0.06}
    p = np.full(len(CLASSES), 0.42 / (len(CLASSES) - len(named)))
    for label, v in named.items():
        p[IDX[label]] = v
    return p / p.sum()


def close_up(audi: float, bmw: float) -> np.ndarray:
    d = np.full(len(MAKES), (1 - audi - bmw) / (len(MAKES) - 2))
    d[MAKES.index("Audi")], d[MAKES.index("BMW")] = audi, bmw
    return d


def test_fusion_sets_the_make_totals_and_keeps_the_ranking_inside_a_make():
    p = damaged_audi_whole_car()
    fused, whole, target = fuse_brand(p, CLASSES, MAKES, close_up(0.8, 0.1), 0.5)
    audi = [i for i, c in enumerate(CLASSES) if c.make == "Audi"]
    assert abs(fused.sum() - 1) < 1e-9
    assert abs(fused[audi].sum() - target[MAKES.index("Audi")]) < 1e-6
    assert fused[IDX["Audi S5 Coupe 2012"]] > fused[IDX["Audi S4 Sedan 2012"]]


def test_close_up_evidence_turns_the_damaged_audi_back_into_an_audi():
    p = damaged_audi_whole_car()
    assert decide(p, 0.2, CLASSES, Thresholds())["hierarchy"][0]["name"] == "BMW"
    fused, _, _ = fuse_brand(p, CLASSES, MAKES, close_up(0.9, 0.05), 0.5)
    after = decide(fused, 0.2, CLASSES, Thresholds())
    assert after["hierarchy"][0]["name"] == "Audi" and after["prediction"]["make"] == "Audi"


def test_zero_weight_leaves_the_whole_car_answer_untouched():
    p = damaged_audi_whole_car()
    fused, _, _ = fuse_brand(p, CLASSES, MAKES, close_up(0.9, 0.05), 0.0)
    assert np.allclose(fused, p)


def test_product_rule_lets_a_confident_close_up_overrule_a_flat_whole_car_view():
    p = damaged_audi_whole_car()
    fused, _, target = fuse_brand(p, CLASSES, MAKES, close_up(0.9, 0.05), 0.5, rule="product")
    assert abs(fused.sum() - 1) < 1e-9 and abs(target.sum() - 1) < 1e-9
    assert decide(fused, 0.2, CLASSES, Thresholds())["hierarchy"][0]["name"] == "Audi"
