"""Classical computer vision: enhancement, edges, contours, colour / texture / geometry features, image quality.

Also used by training/classical_baseline.ipynb (downloaded from GitHub) so baseline training and the app
compute exactly the same features.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

Box = tuple[int, int, int, int]

PALETTE = {  # name -> RGB, for naming dominant colours
    "black": (20, 20, 20), "white": (240, 240, 240), "grey": (128, 128, 128), "silver": (192, 192, 192),
    "red": (200, 30, 30), "maroon": (110, 20, 30), "blue": (30, 70, 200), "navy": (20, 30, 90),
    "green": (40, 140, 60), "yellow": (230, 200, 40), "orange": (230, 120, 30), "brown": (110, 70, 40),
    "beige": (210, 190, 150), "gold": (180, 150, 70),
}
_PALETTE_LAB = {n: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float) for n, c in PALETTE.items()}


def crop(rgb: np.ndarray, box: Box, margin: float = 0.08) -> np.ndarray:
    """Same crop rule as training: box + 8% margin, clamped to the image."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    mw, mh = margin * (x2 - x1), margin * (y2 - y1)
    xa, ya, xb, yb = int(max(0, x1 - mw)), int(max(0, y1 - mh)), int(min(w, x2 + mw)), int(min(h, y2 + mh))
    return rgb[ya:yb, xa:xb]


def enhance(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on the lightness channel: lifts dark regions without blowing out highlights."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def edges(rgb: np.ndarray) -> np.ndarray:
    """Canny with thresholds picked from the median intensity (no hand-tuned constants per photo)."""
    gray = cv2.bilateralFilter(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 7, 50, 50)
    med = float(np.median(gray))
    return cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))


def body_contour(edge_img: np.ndarray) -> np.ndarray | None:
    """Largest outer contour after closing gaps between edges: a rough body outline."""
    closed = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def wheels(rgb: np.ndarray) -> list[tuple[int, int, int]]:
    """Hough circles in the lower part of the car crop; keeps the two most plausible wheels."""
    h, w = rgb.shape[:2]
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    top = int(h * 0.4)
    circles = cv2.HoughCircles(gray[top:], cv2.HOUGH_GRADIENT, dp=1.2, minDist=w * 0.25, param1=120, param2=40,
                               minRadius=int(h * 0.08), maxRadius=int(h * 0.3))
    if circles is None:
        return []
    found = [(int(x), int(y) + top, int(r)) for x, y, r in circles[0][:6]]
    if len(found) < 2:
        return found
    # best pair = similar height in the image, far apart horizontally
    pairs = [(a, b) for i, a in enumerate(found) for b in found[i + 1:]]
    a, b = max(pairs, key=lambda p: abs(p[0][0] - p[1][0]) - 2 * abs(p[0][1] - p[1][1]))
    return sorted([a, b])


def dominant_colours(rgb: np.ndarray, k: int = 3) -> list[dict]:
    """k-means on the central 70% of the crop (less background)."""
    h, w = rgb.shape[:2]
    core = rgb[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    px = cv2.resize(core, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
    _, labels, centres = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                    3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    out = []
    for c, s in sorted(zip(centres, shares), key=lambda cs: -cs[1]):
        lab = cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        name = min(_PALETTE_LAB, key=lambda n: np.linalg.norm(_PALETTE_LAB[n] - lab))
        out.append({"hex": "#%02x%02x%02x" % tuple(int(v) for v in c), "name": name, "share": round(float(s), 3)})
    return out


def lbp_hist(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Local binary pattern texture histogram (8 neighbours, radius 1)."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        code |= ((g[1 + dy:g.shape[0] - 1 + dy, 1 + dx:g.shape[1] - 1 + dx] >= c).astype(np.uint8) << bit)
    hist = np.bincount((code // (256 // bins)).ravel(), minlength=bins).astype(np.float32)
    return hist / max(1.0, hist.sum())


_HOG = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9)


def baseline_features(car: np.ndarray) -> np.ndarray:
    """Feature vector for the classical-only classifier: HOG shape + HSV colour + LBP texture + geometry."""
    small = cv2.resize(car, (128, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hog = _HOG.compute(gray).ravel()
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    colour = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    colour /= max(1.0, colour.sum())
    h, w = car.shape[:2]
    geo = np.array([w / max(1, h)], dtype=np.float32)
    return np.concatenate([hog, colour, lbp_hist(gray), geo]).astype(np.float32)


def quality(rgb: np.ndarray, car: np.ndarray, box: Box) -> dict:
    """Image-quality measurements (0-100, higher = better) + plain-English issues."""
    gray = cv2.cvtColor(car, cv2.COLOR_RGB2GRAY)
    norm = cv2.resize(gray, (256, max(1, int(256 * gray.shape[0] / max(1, gray.shape[1])))))
    brightness = float(cv2.cvtColor(car, cv2.COLOR_RGB2HSV)[..., 2].mean() / 255 * 100)
    contrast = float(min(100.0, gray.std() / 64 * 100))
    sharpness = float(100 * np.clip(np.log10(cv2.Laplacian(norm, cv2.CV_64F).var() + 1) / 3, 0, 1))
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    cut_off = x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3
    resolution = int(min(car.shape[:2]))
    issues = [msg for bad, msg in [
        (brightness < 30, "Low light"), (contrast < 25, "Low contrast or haze"), (sharpness < 45, "Blurry"),
        (resolution < 120, "Low resolution"), (cut_off, "Vehicle cut off at image edge")] if bad]
    return {"brightness": round(brightness), "contrast": round(contrast), "sharpness": round(sharpness),
            "resolution_px": resolution, "cut_off": bool(cut_off), "issues": issues}


def _data_url(rgb: np.ndarray, max_side: int = 480) -> str:
    s = max_side / max(rgb.shape[:2])
    if s < 1:
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def analyse(rgb: np.ndarray, box: Box) -> dict:
    """Runs the whole classical pipeline on one detected car. Returns steps, features, quality, baseline vector."""
    car = crop(rgb, box)
    enhanced = enhance(rgb)
    car_enh = crop(enhanced, box)
    edge_img = edges(car_enh)
    contour = body_contour(edge_img)
    wheel_list = wheels(car_enh)
    colours = dominant_colours(car)

    boxed = rgb.copy()
    cv2.rectangle(boxed, box[:2], box[2:], (255, 80, 0), max(2, rgb.shape[1] // 300))
    contour_img = car_enh.copy()
    if contour is not None:
        cv2.drawContours(contour_img, [contour], -1, (0, 255, 120), 2)
    wheel_img = car_enh.copy()
    for x, y, r in wheel_list:
        cv2.circle(wheel_img, (x, y), r, (255, 200, 0), 2)
    swatch = np.zeros((60, 300, 3), np.uint8)
    x = 0
    for c in colours:
        width = int(round(c["share"] * 300))
        swatch[:, x:x + width] = [int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)]
        x += width

    ch, cw = car.shape[:2]
    wheelbase = abs(wheel_list[1][0] - wheel_list[0][0]) / max(1, cw) if len(wheel_list) == 2 else None
    extent = solidity = None
    if contour is not None and cv2.contourArea(contour) > 0:
        bx, by, bw, bh = cv2.boundingRect(contour)
        extent = cv2.contourArea(contour) / max(1, bw * bh)
        solidity = cv2.contourArea(contour) / max(1.0, cv2.contourArea(cv2.convexHull(contour)))

    return {
        "steps": [
            {"key": "original", "title": "Original", "caption": "Uploaded photo with the detected vehicle", "image": _data_url(boxed)},
            {"key": "enhanced", "title": "Contrast boost", "caption": "CLAHE on the lightness channel", "image": _data_url(enhanced)},
            {"key": "edges", "title": "Edges", "caption": "Canny edges, thresholds from median brightness", "image": _data_url(edge_img[..., None].repeat(3, 2))},
            {"key": "contours", "title": "Contours", "caption": "Largest outer contour = body outline", "image": _data_url(contour_img)},
            {"key": "roi", "title": "Vehicle region", "caption": "Crop sent to the classifier", "image": _data_url(car)},
            {"key": "wheels", "title": "Wheels", "caption": "Hough circles -> wheelbase ratio", "image": _data_url(wheel_img)},
            {"key": "colour", "title": "Colour", "caption": "Dominant colours (k-means)", "image": _data_url(swatch)},
        ],
        "features": {
            "dominant_colours": colours, "aspect_ratio": round(cw / max(1, ch), 3), "wheel_count": len(wheel_list),
            "wheelbase_ratio": round(wheelbase, 3) if wheelbase else None,
            "extent": round(extent, 3) if extent else None, "solidity": round(solidity, 3) if solidity else None,
        },
        "quality": quality(rgb, car, box),
        "baseline_vector": baseline_features(car),
    }
