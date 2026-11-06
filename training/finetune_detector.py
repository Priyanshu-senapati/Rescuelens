"""Fine-tune YOLO11s on the Stanford Cars training boxes (one class: vehicle) and export it to ONNX.

Only the train split is used (validation = the same stratified 10% split as the classifier); the official test split is
never touched, so detector_and_baseline.py can still report honest IoU / mAP on it.

    python training/finetune_detector.py        env: DATA_ROOT, EXPORT_DIR (default models), EPOCHS (20), QUICK (1)
Writes EXPORT_DIR/yolo11s_cars.pt and EXPORT_DIR/yolo11s_cars.onnx.
"""
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_data import load_stanford_cars, split_items  # noqa: E402

QUICK = os.environ.get("QUICK", "1") != "0"
IN = Path(os.environ.get("DATA_ROOT", "/kaggle/input"))
EXPORT = Path(os.environ.get("EXPORT_DIR", "models"))
WORK = Path(os.environ.get("YOLO_WORK", "training/.yolo_cars"))
EPOCHS = 1 if QUICK else int(os.environ.get("EPOCHS", 20))


def build_split(items, split: str) -> None:
    """YOLO layout: images/<split>/x.jpg (hard links, no copies) + labels/<split>/x.txt ("0 cx cy w h", normalised)."""
    img_dir, lbl_dir = WORK / "images" / split, WORK / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for path, (x1, y1, x2, y2), _ in items:
        dst = img_dir / path.name
        if not dst.exists():
            try:
                os.link(path, dst)
            except OSError:
                shutil.copy(path, dst)
        w, h = Image.open(path).size
        clamp = lambda v: min(1.0, max(0.0, v))
        cx, cy = clamp((x1 + x2) / 2 / w), clamp((y1 + y2) / 2 / h)
        bw, bh = clamp((x2 - x1) / w), clamp((y2 - y1) / h)
        (lbl_dir / f"{path.stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main() -> None:
    from ultralytics import YOLO

    train, test, _ = load_stanford_cars(IN)
    tr, va, _ = split_items(train, test, 0.1, 42, QUICK)
    build_split(tr, "train")
    build_split(va, "val")
    data = WORK / "cars.yaml"
    data.write_text(f"path: {WORK.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vehicle\n")

    model = YOLO("yolo11s.pt")   # COCO-pretrained start, so it keeps knowing what vehicles look like
    model.train(data=str(data), epochs=EPOCHS, imgsz=640, batch=16, device=0, workers=4, seed=42,
                project=str((WORK / "runs").resolve()), name="ft", exist_ok=True, plots=False)
    best = WORK / "runs" / "ft" / "weights" / "best.pt"
    EXPORT.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, EXPORT / "yolo11s_cars.pt")
    onnx = YOLO(str(best)).export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    shutil.copy(onnx, EXPORT / "yolo11s_cars.onnx")
    print("DONE" + (" (QUICK)" if QUICK else ""), flush=True)


if __name__ == "__main__":
    main()
