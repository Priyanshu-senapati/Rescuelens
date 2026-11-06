# RescueLens

Uncertainty-aware vehicle identification for crash response (Computer Vision Hackathon 2026, P5).

**Live app: https://priyanshu-senapati--rescuelens-web.us-east.modal.run** (phone or laptop; deployed on Modal from `modal_app.py`)

Photo or live camera -> detect the vehicle -> identify make / model / year (196 Stanford Cars classes) ->
decide what is safe to conclude: **identified**, **ambiguous** (e.g. hybrid can't be ruled out) or **unknown**.
Point a phone at a car and the answer locks in once 3 frames agree; a high-voltage warning shows at once, with where the battery
and cables are, from the manufacturer's guide.

![Architecture](docs/architecture.png)

## Results (official Stanford Cars test split, 8,041 images)

| | |
|---|---|
| Top-1 / Top-5 accuracy | **94.55% / 99.27%** (ConvNeXt-Small, 384 px, detector crop; 94.70% with flip TTA) |
| Vehicle localisation (YOLO11s) | mean IoU **0.886**, mAP@0.5 **0.892**, mAP@0.5:0.95 0.825 |
| Classical-only vs deep (whole photo) vs hybrid | 13.7% vs 91.7% vs **94.6%** top-1 |
| Model size / CPU latency | 189.4 MB -> **54.3 MB** INT8, 739 ms -> **302 ms** (2 threads), no accuracy loss on a 300-image check |
| Calibration | ECE 0.098 -> **0.011** after temperature scaling |
| Missed hybrids / EVs | **1.2%** with the summed-probability warning vs 4.3% using the top-1 label |
| Unseen car models (30 models, 60 photos) | **96.7%** answered "unknown"; 8.3% of known test cars wrongly flagged |

Full tables (robustness, ablation, thresholds, failure cases): [`eval_P5_T50.md`](eval_P5_T50.md).

| Folder | What |
|---|---|
| `training/` | Training + evaluation scripts (Kaggle or local GPU): classifier, detector eval, classical baseline |
| `backend/` | FastAPI inference service (ONNX Runtime) |
| `frontend/` | React UI |
| `data/` | class names, high-voltage class list |
| `docs/` | API contract, fixtures, design doc |

**Build plan / handover (pending work, dropped features): [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)** · Jobs for people, not code: [docs/field-kit.md](docs/field-kit.md) · Viva sheets: [docs/viva/](docs/viva/) · High-voltage sources: [docs/hazard-sources.md](docs/hazard-sources.md) · Design document + architecture diagram: [docs/design.md](docs/design.md) · API: [docs/api-contract.md](docs/api-contract.md) · Evaluation: `eval_P5_T50.md`

## Run it

```bash
docker build -t rescuelens --build-arg MODEL_BASE_URL=https://huggingface.co/<user>/rescuelens-models/resolve/main .
docker run -p 7860:7860 rescuelens          # open http://localhost:7860
```

Without `MODEL_BASE_URL` the app starts in mock mode (fixture responses), which is handy for UI work.

Back-end tests: `cd backend && pip install -r requirements.txt -r requirements-dev.txt && pytest`

Browser tests: `cd frontend && npm ci && npx playwright install chromium && npm test`. CI runs both on every push.

Pre-test the demo samples on the real models: `python tools/check_samples.py`

## Reproduce the numbers

1. Get Stanford Cars with test labels (e.g. Kaggle `jutrera/stanford-car-dataset-by-classes-folder`) and extract it.
2. Python 3.11 + CUDA PyTorch, then `pip install timm onnx onnxruntime onnxscript scikit-learn scipy "opencv-python-headless<5" pillow ultralytics`.
3. From the repo root (set `DATA_ROOT` to the extracted folder, `EXPORT_DIR=models`, `QUICK=0`):
   `python training/train_classifier.py` (about 45 min on an RTX 5070 Ti laptop GPU, 2.5 h on a Kaggle T4) and
   `python training/detector_and_baseline.py`. On Windows, `training\run_local.ps1` starts them as background processes.
4. `python tools/build_report.py --train models --det models --team T50 --usns "..."` writes `eval_P5_T50.md`.
   Seed 42 everywhere; the validation split is a stratified 10% of the official train split.
