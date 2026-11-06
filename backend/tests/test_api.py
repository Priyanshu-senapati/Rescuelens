import io
import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.pipeline import Engine, analyze
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
