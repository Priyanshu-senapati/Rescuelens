"""FastAPI app: POST /api/analyze, GET /api/metrics, GET /api/health, and the built React UI at /."""
from __future__ import annotations

import io
import itertools
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .pipeline import analyze_views, load_engine

try:                                         # iPhone photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", ROOT / "frontend" / "dist"))
FIXTURES = ROOT / "docs" / "fixtures"
MAX_VIEWS = 3                                # e.g. crushed front + rear + side of one vehicle
MAX_BYTES = 10 * 2**20
OFFLINE_FILES = {"config.json", "classifier_int8.onnx", "train_emb.npy", "yolo11s.onnx"}
MAX_SIDE = 1600                              # bigger photos are shrunk first: faster, same accuracy
Image.MAX_IMAGE_PIXELS = 60_000_000          # refuse decompression bombs
log = logging.getLogger("rescuelens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def load_rgb(data: bytes) -> np.ndarray:
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ApiError(400, "not_image", "That file isn't a readable image. Try a JPEG or PNG photo.") from exc
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.asarray(im)


def create_app() -> FastAPI:
    app = FastAPI(title="RescueLens", version=os.environ.get("APP_VERSION", "dev"))
    try:
        engine = load_engine(MODEL_DIR)
        log.info("models loaded from %s (%s)", MODEL_DIR, engine.classifier.precision)
    except Exception as exc:                 # missing / corrupt model files must not take the site down
        engine = None
        log.warning("models not available (%s) - serving fixture responses (mock mode)", exc)
    mocks = itertools.cycle([json.loads((FIXTURES / f"{n}.json").read_text(encoding="utf-8"))
                             for n in ("identified", "ambiguous", "unknown")])
    busy = threading.Semaphore(2)            # 2 vCPUs: don't let a burst of uploads thrash the box

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse({"error": {"code": "internal", "message": "Something went wrong analysing that photo."}},
                            status_code=500)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": engine is not None, "mode": "live" if engine else "mock",
                "version": app.version}

    @app.post("/api/analyze")
    def analyze_photo(file: list[UploadFile] | None = File(None)) -> dict:
        if not file:
            raise ApiError(400, "missing_file", "No photo received. Attach it as the 'file' field.")
        if len(file) > MAX_VIEWS:
            raise ApiError(400, "too_many_files", f"Send at most {MAX_VIEWS} photos of the same vehicle.")
        rgbs = []
        for f in file:
            data = f.file.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ApiError(413, "too_large", "That photo is over 10 MB. Try a smaller one.")
            rgbs.append(load_rgb(data))
        if engine is None:
            return {**next(mocks), "mock": True}
        with busy:
            result = analyze_views(rgbs, engine)
        log.info("analyze views=%d status=%s total_ms=%s", len(rgbs), result["status"], result["timing_ms"]["total"])
        return result

    @app.get("/api/metrics")
    def metrics() -> dict:
        path = MODEL_DIR / "metrics_public.json"
        if not path.exists():
            raise ApiError(404, "not_ready", "Evaluation metrics are not available yet.")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.api_route("/offline/{name}", methods=["GET", "HEAD"])
    def offline_file(name: str) -> FileResponse:
        path = MODEL_DIR / name
        # opt-in, because this makes the model weights publicly downloadable; whitelist: only what on-device mode needs
        if os.environ.get("OFFLINE_MODELS") != "1" or name not in OFFLINE_FILES or not path.exists():
            raise ApiError(404, "not_ready", "Offline files are not available on this server.")
        return FileResponse(path)

    if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="ui")
    else:
        @app.get("/")
        def root() -> dict:
            return {"service": "RescueLens API", "docs": "/docs", "health": "/api/health"}

    return app


app = create_app()
