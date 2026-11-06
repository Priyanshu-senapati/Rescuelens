"""Downloads the trained model files into MODEL_DIR at image build time.

MODEL_BASE_URL = folder URL that holds the files, e.g. https://huggingface.co/<user>/rescuelens-models/resolve/main
Missing files are reported; the app then starts in mock mode (fixture responses) instead of crashing.
"""
import os
import sys
import urllib.request
from pathlib import Path

FILES = ["config.json", "classifier_int8.onnx", "train_emb.npy", "yolo11s.onnx", "baseline.npz", "metrics_public.json"]

base = os.environ.get("MODEL_BASE_URL", "").rstrip("/")
out = Path(os.environ.get("MODEL_DIR", "models"))
out.mkdir(parents=True, exist_ok=True)
if not base:
    print("MODEL_BASE_URL not set - no models downloaded, app will run in mock mode")
    sys.exit(0)
for name in FILES:
    try:
        urllib.request.urlretrieve(f"{base}/{name}", out / name)
        print(f"ok      {name} ({(out / name).stat().st_size / 2**20:.1f} MB)")
    except OSError as exc:
        print(f"MISSING {name}: {exc}")
