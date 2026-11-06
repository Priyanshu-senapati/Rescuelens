import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.pipeline import Engine, analyze, analyze_views
from app.safety import Thresholds, load_classes

FIXTURES = Path(__file__).resolve().parents[2] / "docs" / "fixtures"
CLASSES = load_classes()
IDX = {c.label: i for i, c in enumerate(CLASSES)}


def car_photo() -> np.ndarray:
    img = np.full((400, 600, 3), 60, np.uint8)
    img[150:330, 100:500] = (180, 180, 190)
    return img


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray(car_photo()).save(buf, format="PNG")
    return buf.getvalue()


def sonata_classifier(_: np.ndarray):
    p = np.full(len(CLASSES), 0.05 / (len(CLASSES) - 2))
    p[IDX["Hyundai Sonata Hybrid Sedan 2012"]], p[IDX["Hyundai Sonata Sedan 2012"]] = 0.5, 0.45
    return p, 0.2


def engine(detections: list) -> Engine:
    return Engine(detector=lambda rgb: detections, classifier=sonata_classifier, baseline=None,
                  classes=CLASSES, thresholds=Thresholds())


def test_pipeline_output_has_exactly_the_contract_fields():
    out = analyze(car_photo(), engine([((100, 150, 500, 330), 0.9)]))
    fixture = json.loads((FIXTURES / "ambiguous.json").read_text(encoding="utf-8"))
    assert set(out) == set(fixture)
    assert out["status"] == "ambiguous" and out["safety"]["hv_warning"] is True
    assert out["comparison"]["hybrid"]["status"] == "ambiguous"


def test_no_vehicle_is_a_normal_result_with_the_same_fields():
    out = analyze(car_photo(), engine([]))
    assert out["status"] == "no_vehicle" and out["prediction"] is None
    assert set(out) == set(json.loads((FIXTURES / "identified.json").read_text(encoding="utf-8")))


client = TestClient(create_app())      # no models locally -> mock mode


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_missing_file_is_400():
    r = client.post("/api/analyze")
    assert r.status_code == 400 and r.json()["error"]["code"] == "missing_file"


def test_not_an_image_is_400():
    r = client.post("/api/analyze", files={"file": ("x.jpg", b"definitely not a jpeg", "image/jpeg")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "not_image"


def test_too_large_is_413():
    r = client.post("/api/analyze", files={"file": ("big.jpg", b"0" * (10 * 2**20 + 1), "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


def test_offline_files_are_off_by_default_then_whitelisted(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.MODEL_DIR", tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "metrics.json").write_text("{}")
    assert client.get("/offline/config.json").status_code == 404         # weights stay private unless the team opts in

    monkeypatch.setenv("OFFLINE_MODELS", "1")
    assert client.get("/offline/config.json").status_code == 200
    assert client.head("/offline/config.json").status_code == 200
    for name in ("metrics.json", "..%2Fapp%2Fmain.py", "classifier_int8.onnx"):  # not whitelisted, traversal, missing
        assert client.get(f"/offline/{name}").status_code == 404


def test_valid_photo_returns_a_result():
    r = client.post("/api/analyze", files={"file": ("car.png", png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["status"] in {"identified", "ambiguous", "unknown", "no_vehicle"}


def test_dark_photo_gets_a_second_opinion_from_a_contrast_boosted_copy():
    calls = []

    def counting_classifier(crop):
        calls.append(float(crop.mean()))
        return sonata_classifier(crop)

    dark = (car_photo() * 0.1).astype(np.uint8)
    eng = Engine(detector=lambda rgb: [((100, 150, 500, 330), 0.9)], classifier=counting_classifier, baseline=None,
                 classes=CLASSES, thresholds=Thresholds())
    out = analyze(dark, eng)
    assert len(calls) == 3                                  # crop, boosted crop, whole photo (comparison panel)
    assert calls[1] > calls[0]                              # the second pass really is brighter
    assert any("contrast-boosted" in x for x in out["reliability"]["reasons"])


def test_a_second_photo_of_an_intact_side_can_decide_the_answer():
    def flat_wreck(_):                                       # crushed front: no idea, looks unfamiliar
        return np.full(len(CLASSES), 1 / len(CLASSES)), 0.9

    def clear_rear(_):
        p = np.full(len(CLASSES), 0.02 / (len(CLASSES) - 1))
        p[IDX["Hyundai Sonata Hybrid Sedan 2012"]] = 0.98
        return p, 0.1

    views = iter([flat_wreck, clear_rear, flat_wreck])       # view 1, view 2, then the comparison panel's whole photo
    eng = Engine(detector=lambda rgb: [((100, 150, 500, 330), 0.9)], classifier=lambda x: next(views)(x),
                 baseline=None, classes=CLASSES, thresholds=Thresholds())
    out = analyze_views([car_photo(), car_photo()], eng)
    assert out["status"] == "identified" and out["prediction"]["label"] == "Hyundai Sonata Hybrid Sedan 2012"
    assert out["safety"]["hv_warning"] is True
    assert [v["vehicle_found"] for v in out["views"]] == [True, True]
    assert out["views"][0]["unknown_score"] == 0.9 and out["unknown"]["score"] == 0.1
    assert any("Combined 2 photos" in x for x in out["reliability"]["reasons"])
    assert "Photo 1 on its own was not recognised; the answer rests on the other photos" in out["reliability"]["reasons"]


def test_a_photo_without_a_vehicle_is_skipped_not_fatal():
    dets = iter([[], [], [((100, 150, 500, 330), 0.9)]])      # photo 1: nothing, even after boost; photo 2: a car
    eng = Engine(detector=lambda rgb: next(dets), classifier=sonata_classifier, baseline=None,
                 classes=CLASSES, thresholds=Thresholds())
    out = analyze_views([car_photo(), car_photo()], eng)
    assert out["status"] == "ambiguous" and [v["vehicle_found"] for v in out["views"]] == [False, True]


def test_up_to_three_photos_accepted_four_rejected():
    photo = ("car.png", png_bytes(), "image/png")
    assert client.post("/api/analyze", files=[("file", photo)] * 3).status_code == 200
    r = client.post("/api/analyze", files=[("file", photo)] * 4)
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_files"


def test_photos_of_two_different_vehicles_are_not_merged_into_one_answer():
    def confident(label):
        def clf(_):
            p = np.full(len(CLASSES), 0.02 / (len(CLASSES) - 1))
            p[IDX[label]] = 0.98
            return p, 0.1
        return clf

    views = iter([confident("Hyundai Sonata Hybrid Sedan 2012"), confident("BMW M3 Coupe 2012"),
                  confident("Hyundai Sonata Hybrid Sedan 2012")])
    eng = Engine(detector=lambda rgb: [((100, 150, 500, 330), 0.9)], classifier=lambda x: next(views)(x),
                 baseline=None, classes=CLASSES, thresholds=Thresholds())
    out = analyze_views([car_photo(), car_photo()], eng)
    assert out["status"] == "unknown" and "different vehicles" in out["headline"]
    assert out["safety"]["hv_warning"] is True and out["safety"]["level"] == "caution"
    assert out["reliability"]["reasons"][0] == "Photo 1 reads as Hyundai Sonata, photo 2 as BMW M3"
    assert out["comparison"]["hybrid"]["status"] == "unknown"


def test_look_alike_families_of_one_make_are_reported_not_refused():
    def confident(label):
        def clf(_):
            p = np.full(len(CLASSES), 0.3 / (len(CLASSES) - 1))
            p[IDX[label]] = 0.7
            return p, 0.1
        return clf

    views = iter([confident("Audi S4 Sedan 2012"), confident("Audi S5 Coupe 2012"), confident("Audi S4 Sedan 2012")])
    eng = Engine(detector=lambda rgb: [((100, 150, 500, 330), 0.9)], classifier=lambda x: next(views)(x),
                 baseline=None, classes=CLASSES, thresholds=Thresholds())
    out = analyze_views([car_photo(), car_photo()], eng)
    assert "different vehicles" not in out["headline"] and out["hierarchy"][0]["name"] == "Audi"
    assert out["reliability"]["reasons"][0] == "Photo 1 reads as Audi S4, photo 2 as Audi S5"
