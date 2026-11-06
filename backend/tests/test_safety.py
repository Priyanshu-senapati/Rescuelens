import numpy as np
import pytest

from app.safety import Thresholds, decide, load_classes, parse_label

CLASSES = load_classes()
IDX = {c.label: i for i, c in enumerate(CLASSES)}
T = Thresholds(identify=0.6, family=0.6, make=0.5, hv=0.1, unknown=0.5)


def probs(**named: float) -> np.ndarray:
    """Probability vector with the given classes; the rest spread evenly over all other classes."""
    p = np.zeros(len(CLASSES))
    for label, v in named.items():
        p[IDX[label.replace("_", " ")]] = v
    rest = 1 - p.sum()
    others = p == 0
    p[others] = rest / others.sum()
    return p


def test_all_196_labels_parse_and_nine_are_high_voltage():
    assert len(CLASSES) == 196
    assert all(1990 <= c.year <= 2012 for c in CLASSES)
    assert sum(c.powertrain != "combustion" for c in CLASSES) == 9


@pytest.mark.parametrize("label,make,model,family", [
    ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage Coupe", "Aston Martin V8"),
    ("AM General Hummer SUV 2000", "AM General", "Hummer SUV", "AM General Hummer"),
    ("BMW 1 Series Coupe 2012", "BMW", "1 Series Coupe", "BMW 1 Series"),
    ("Hyundai Sonata Hybrid Sedan 2012", "Hyundai", "Sonata Hybrid Sedan", "Hyundai Sonata"),
])
def test_parse_label(label, make, model, family):
    c = parse_label(label, {})
    assert (c.make, c.model, c.family) == (make, model, family)


def test_confident_petrol_car_is_identified_without_hv_warning():
    r = decide(probs(**{"BMW M3 Coupe 2012": 0.94}), 0.1, CLASSES, T)
    assert r["status"] == "identified" and r["resolved_level"] == "vehicle"
    assert r["safety"]["hv_warning"] is False


def test_sonata_vs_sonata_hybrid_is_ambiguous_and_warns_high_voltage():
    p = probs(**{"Hyundai Sonata Hybrid Sedan 2012": 0.48, "Hyundai Sonata Sedan 2012": 0.43})
    r = decide(p, 0.2, CLASSES, T)
    assert r["status"] == "ambiguous" and r["resolved_level"] == "family"
    assert r["safety"]["hv_warning"] is True and r["safety"]["level"] == "danger"
    assert r["safety"]["guide"] is not None
    assert "look-alike" in " ".join(r["reliability"]["reasons"])


def test_hv_warning_uses_summed_probability_not_top1():
    # no single hybrid wins, but together they are 45%: must still warn
    p = probs(**{"Chevrolet Tahoe Hybrid SUV 2012": 0.25, "GMC Yukon Hybrid SUV 2012": 0.20,
                 "Chevrolet Avalanche Crew Cab 2012": 0.30})
    r = decide(p, 0.2, CLASSES, T)
    assert r["prediction"]["label"] == "Chevrolet Avalanche Crew Cab 2012"
    assert r["safety"]["hv_warning"] is True


def test_out_of_distribution_car_is_unknown_and_treated_as_possibly_hv():
    r = decide(probs(**{"Toyota Camry Sedan 2012": 0.93}), 0.9, CLASSES, T)
    assert r["status"] == "unknown" and r["resolved_level"] == "none"
    assert r["safety"]["hv_probability"] is None and r["safety"]["hv_warning"] is True
    assert r["safety"]["level"] == "caution"


def test_low_make_confidence_is_unknown_even_if_in_distribution():
    r = decide(probs(), 0.1, CLASSES, T)       # flat distribution: model has no idea
    assert r["status"] == "unknown"


def test_hierarchy_probabilities_never_increase_with_depth():
    r = decide(probs(**{"Audi S4 Sedan 2012": 0.5, "Audi S4 Sedan 2007": 0.4}), 0.1, CLASSES, T)
    make, family, vehicle = (h["prob"] for h in r["hierarchy"])
    assert make >= family >= vehicle
    assert r["status"] == "ambiguous" and "2007" in r["headline"] and "2012" in r["headline"]
