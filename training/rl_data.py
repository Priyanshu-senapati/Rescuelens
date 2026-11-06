"""Shared by train_classifier.py and detector_and_baseline.py: Stanford Cars loading, the train/val split,
crop caching, the training Dataset and the robustness degradations.

Everything a DataLoader / Pool worker touches lives here (importable module, no lambdas) so the scripts also
run on Windows, where workers are spawned and must re-import what they use.
"""
from __future__ import annotations

import csv
import math
import os
import random
import urllib.request
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASSNAMES_URL = "https://raw.githubusercontent.com/Priyanshu-senapati/Rescuelens/main/data/classnames.txt"
Item = tuple  # (image path, (x1, y1, x2, y2), label 0-195)


# ---------------------------------------------------------------- dataset loading

def _find(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


def _mat_rows(path: Path) -> list[dict]:
    arr = np.atleast_1d(sio.loadmat(path, squeeze_me=True)["annotations"])
    return [{k: a[k] for k in arr.dtype.names} for a in arr]


def _box(r: dict) -> tuple:
    return tuple(float(r[k]) for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))


def _split_of(p: Path, root: Path) -> str | None:
    """Nearest parent folder naming car_ims / test / train (ignores Kaggle's owner/slug prefix)."""
    for part in reversed(p.relative_to(root).parts[:-1]):
        s = part.lower()
        if "car_ims" in s:
            return "car_ims"
        if "test" in s:
            return "test"
        if "train" in s:
            return "train"
    return None


def load_stanford_cars(root: Path) -> tuple[list[Item], list[Item], list[str]]:
    """Supports the devkit (.mat) layout, the 'by classes folder' (.csv) layout and the full cars_annos.mat layout."""
    meta = _find(root, "cars_meta.mat")
    if meta is not None:
        names = [str(x) for x in sio.loadmat(meta, squeeze_me=True)["class_names"]]
    else:
        names = urllib.request.urlopen(CLASSNAMES_URL).read().decode().strip().splitlines()
    assert len(names) == 196, f"expected 196 class names, got {len(names)}"

    imgs: dict[str, dict[str, Path]] = {"train": {}, "test": {}, "car_ims": {}}
    for p in root.rglob("*.jpg"):
        split = _split_of(p, root)
        if split:
            imgs[split][p.name] = p      # train and test both use names like 00001.jpg, so keep them apart

    train, test = [], []
    tr_mat, te_mat = _find(root, "cars_train_annos.mat"), _find(root, "cars_test_annos_withlabels*.mat")
    tr_csv, te_csv, full_mat = _find(root, "anno_train.csv"), _find(root, "anno_test.csv"), _find(root, "cars_annos.mat")
    if tr_mat and te_mat and imgs["train"] and imgs["test"]:
        for mat, split, out in ((tr_mat, "train", train), (te_mat, "test", test)):
            for r in _mat_rows(mat):
                if str(r["fname"]) in imgs[split]:
                    out.append((imgs[split][str(r["fname"])], _box(r), int(r["class"]) - 1))
    elif tr_csv and te_csv and imgs["train"] and imgs["test"]:
        for path, split, out in ((tr_csv, "train", train), (te_csv, "test", test)):
            with open(path, newline="") as fh:
                for row in csv.reader(fh):       # fname, x1, y1, x2, y2, class (1-based)
                    if len(row) >= 6 and row[0] in imgs[split]:
                        out.append((imgs[split][row[0]], tuple(float(v) for v in row[1:5]), int(row[5]) - 1))
    elif full_mat and imgs["car_ims"]:
        for r in _mat_rows(full_mat):
            name = Path(str(r["relative_im_path"])).name
            if name in imgs["car_ims"]:
                (test if int(r["test"]) else train).append((imgs["car_ims"][name], _box(r), int(r["class"]) - 1))
    else:
        found = sorted(str(p.relative_to(root)) for p in root.rglob("*.mat")) + sorted(str(p.relative_to(root)) for p in root.rglob("*.csv"))
        raise SystemExit(
            f"Stanford Cars images + labels not found under {root}.\n"
            f"annotation files found: {found}\n"
            f"jpgs found: train={len(imgs['train'])} test={len(imgs['test'])} car_ims={len(imgs['car_ims'])}\n"
            "Fix: add/extract a Stanford Cars copy that includes test labels "
            "(e.g. Kaggle 'stanford-car-dataset-by-classes-folder').")
    print(f"train {len(train)} (expect 8144) | test {len(test)} (expect 8041)", flush=True)
    assert len(train) > 8000 and len(test) > 8000, "Dataset incomplete"
    return train, test, names


def split_items(train: list[Item], test: list[Item], val_frac: float, seed: int, quick: bool):
    """Stratified train/val split of the official train set; the official test set stays untouched."""
    tr_idx, va_idx = train_test_split(np.arange(len(train)), test_size=val_frac,
                                      stratify=[it[2] for it in train], random_state=seed)
    tr, va, te = [train[i] for i in tr_idx], [train[i] for i in va_idx], list(test)
    if quick:
        tr, va, te = tr[:600], va[:200], te[:300]
    print(f"split -> train {len(tr)} | val {len(va)} | test {len(te)}", flush=True)
    return tr, va, te


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- crop cache

def crop_resize(job: tuple) -> None:
    """Crop the car (box + 8% margin, same rule as the app) and save a small copy."""
    src, bx, dst, crop, side = job
    if os.path.exists(dst):
        return
    im = Image.open(src).convert("RGB")
    if crop:
        x1, y1, x2, y2 = bx
        mw, mh = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
        im = im.crop((int(max(0, x1 - mw)), int(max(0, y1 - mh)), int(min(im.width, x2 + mw)), int(min(im.height, y2 + mh))))
    s = side / min(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    im.save(dst, quality=95)


def cache_items(items: list[Item], folder: Path, crop: bool, side: int, workers: int) -> tuple[list[str], list[int]]:
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), bx, str(folder / f"{i}.jpg"), crop, side) for i, (p, bx, _) in enumerate(items)]
    with Pool(workers) as pool:
        pool.map(crop_resize, jobs, chunksize=64)
    return [j[2] for j in jobs], [it[2] for it in items]


# ---------------------------------------------------------------- dataset + augmentation

class RandomDark:
    """Simulates night / underexposed scenes."""

    def __init__(self, p: float):
        self.p = p

    def __call__(self, im: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(im).enhance(random.uniform(0.25, 0.7)) if random.random() < self.p else im


class DS(Dataset):
    def __init__(self, paths, labels, tf, degrade=None):
        self.paths, self.labels, self.tf, self.degrade = paths, labels, tf, degrade

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        if self.degrade is not None:
            im = self.degrade(im, i)
        return self.tf(im), self.labels[i]


# ---------------------------------------------------------------- robustness degradations (im, index) -> im

def clahe(im: Image.Image) -> Image.Image:
    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def dark(im, i, v):
    return ImageEnhance.Brightness(im).enhance(v)


def dark_clahe(im, i, v):
    return clahe(dark(im, i, v))


def haze(im, i, a):
    return Image.blend(im, Image.new("RGB", im.size, (200, 200, 200)), a)


def blur(im, i, r):
    return im.filter(ImageFilter.GaussianBlur(r))


def low_res(im, i, side):
    s = side / min(im.size)
    small = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    return small.resize(im.size, Image.BILINEAR)


def occlude(im, i, frac):
    r = random.Random(i)
    w, h = im.size
    s = math.sqrt(frac)
    bw, bh = int(w * s), int(h * s)
    x, y = r.randint(0, w - bw), r.randint(0, h - bh)
    im = im.copy()
    ImageDraw.Draw(im).rectangle([x, y, x + bw, y + bh], fill=(128, 128, 128))
    return im


CONDITIONS = {
    "dark": [partial(dark, v=v) for v in (0.6, 0.35, 0.15)],
    "dark+clahe": [partial(dark_clahe, v=v) for v in (0.6, 0.35, 0.15)],
    "haze": [partial(haze, a=a) for a in (0.25, 0.45, 0.65)],
    "blur": [partial(blur, r=r) for r in (1.5, 3, 5)],
    "low_res": [partial(low_res, side=s) for s in (96, 64, 40)],
    "occlusion": [partial(occlude, frac=f) for f in (0.15, 0.3, 0.45)],
}
