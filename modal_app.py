"""Deploys RescueLens (FastAPI + React build + ONNX models) on Modal: one always-warm 2-CPU container.

    modal deploy modal_app.py        # prints the public https URL

Before deploying: build the UI (cd frontend && npm ci && npm run build) and have the model files in models/.
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent
MODEL_FILES = ["config.json", "classifier_int8.onnx", "train_emb.npy", "yolo11s.onnx", "baseline.npz", "metrics_public.json"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "backend" / "requirements.txt"))
    .env({"MODEL_DIR": "/root/models", "FRONTEND_DIST": "/root/frontend/dist", "DATA_DIR": "/root/data",
          "APP_VERSION": "modal"})
    .add_local_dir(ROOT / "backend" / "app", "/root/backend/app", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "data", "/root/data")
    .add_local_dir(ROOT / "docs" / "fixtures", "/root/docs/fixtures")
    .add_local_dir(ROOT / "frontend" / "dist", "/root/frontend/dist")
)
for name in MODEL_FILES:
    image = image.add_local_file(ROOT / "models" / name, f"/root/models/{name}")

app = modal.App("rescuelens", image=image)


# min_containers=1 keeps one container warm so judges never hit a cold start (~$2.5/day of the free $30 credit)
@app.function(cpu=2.0, memory=2048, min_containers=1, scaledown_window=1200, timeout=120)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/backend")
    from app.main import app as fastapi_app   # the same FastAPI app the Dockerfile runs

    return fastapi_app
