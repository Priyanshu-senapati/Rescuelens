import numpy as np

from app.models import vehicle_scores


def test_coco_head_uses_only_car_bus_truck():
    out = np.zeros((84, 3), np.float32)
    out[4 + 0, 0] = 0.9            # person: must be ignored
    out[4 + 2, 1] = 0.8            # car
    out[4 + 7, 2] = 0.7            # truck
    assert np.allclose(vehicle_scores(out), [0.0, 0.8, 0.7])


def test_single_class_finetuned_head_uses_its_class():
    out = np.zeros((5, 2), np.float32)
    out[4] = [0.3, 0.95]
    assert np.allclose(vehicle_scores(out), [0.3, 0.95])
