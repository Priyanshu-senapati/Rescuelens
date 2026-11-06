import numpy as np

from app.classical import analyse, baseline_features, dominant_colours


def car_like(brightness: int = 150) -> np.ndarray:
    img = np.full((400, 600, 3), 40, np.uint8)
    img[150:330, 100:500] = (brightness, 30, 30)            # red-ish body
    for cx in (180, 420):                                   # two dark wheels
        yy, xx = np.ogrid[:400, :600]
        img[(yy - 320) ** 2 + (xx - cx) ** 2 < 45 ** 2] = 10
    return img


def test_analyse_returns_all_steps_and_quality():
    out = analyse(car_like(), (100, 150, 500, 370))
    assert [s["key"] for s in out["steps"]] == ["original", "enhanced", "edges", "contours", "roi", "wheels", "colour"]
    assert all(s["image"].startswith("data:image/jpeg;base64,") for s in out["steps"])
    assert set(out["quality"]) >= {"brightness", "contrast", "sharpness", "resolution_px", "cut_off", "issues"}


def test_dark_photo_is_flagged_low_light():
    dark = (car_like() * 0.15).astype(np.uint8)
    assert "Low light" in analyse(dark, (100, 150, 500, 370))["quality"]["issues"]


def test_baseline_vector_has_fixed_length_for_any_crop_size():
    a = baseline_features(car_like()[150:370, 100:500])
    b = baseline_features(car_like()[0:90, 0:300])
    assert a.shape == b.shape and a.dtype == np.float32


def test_dominant_colour_of_red_block_is_red():
    img = np.zeros((100, 100, 3), np.uint8); img[:] = (200, 30, 30)
    assert dominant_colours(img, k=2)[0]["name"] == "red"
