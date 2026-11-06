---
title: RescueLens
emoji: 🚒
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# RescueLens

Uncertainty-aware vehicle identification for crash response (Computer Vision Hackathon 2026, P5).

Photo of a vehicle -> detect it -> identify make / model / year (196 Stanford Cars classes) ->
decide what is safe to conclude: **identified**, **ambiguous** (e.g. hybrid can't be ruled out) or **unknown**.

## Results (official Stanford Cars test split, 8,041 images)

| | |
|---|---|
| Top-1 / Top-5 accuracy | **94.18% / 99.05%** (ConvNeXt-Tiny, 384 px, detector crop) |
| Vehicle localisation (YOLO11s) | mean IoU **0.886**, mAP@0.5 **0.892**, mAP@0.5:0.95 0.825 |
| Classical-only vs deep (whole photo) vs hybrid | 13.7% vs 90.9% vs **94.2%** top-1 |
| Model size / CPU latency | 106.8 MB -> **32.4 MB** INT8, 324 ms -> **127 ms** (2 threads), 94.3% -> 94.0% |
| Calibration | ECE 0.125 -> **0.017** after temperature scaling |
| Missed hybrids / EVs | **0.9%** with the summed-probability warning vs 4.6% using the top-1 label |

Full tables (robustness, ablation, thresholds, failure cases): [`eval_P5_T50.md`](eval_P5_T50.md).

| Folder | What |
|---|---|
| `training/` | Training + evaluation scripts (Kaggle or local GPU): classifier, detector eval, classical baseline |
| `backend/` | FastAPI inference service (ONNX Runtime) |
| `frontend/` | React UI |
| `data/` | class names, high-voltage class list |
| `docs/` | API contract, fixtures, design doc |

Design document + architecture diagram: [docs/design.md](docs/design.md) · API: [docs/api-contract.md](docs/api-contract.md) · Evaluation: `eval_P5_T50.md`

## Run it

```bash
docker build -t rescuelens --build-arg MODEL_BASE_URL=https://huggingface.co/<user>/rescuelens-models/resolve/main .
docker run -p 7860:7860 rescuelens          # open http://localhost:7860
```

Without `MODEL_BASE_URL` the app starts in mock mode (fixture responses), which is handy for UI work.

Back-end tests: `cd backend && pip install -r requirements.txt -r requirements-dev.txt && pytest`

## Reproduce the numbers

1. Get Stanford Cars with test labels (e.g. Kaggle `jutrera/stanford-car-dataset-by-classes-folder`) and extract it.
2. Python 3.11 + CUDA PyTorch, then `pip install timm onnx onnxruntime onnxscript scikit-learn scipy "opencv-python-headless<5" pillow ultralytics`.
3. From the repo root (set `DATA_ROOT` to the extracted folder, `EXPORT_DIR=models`, `QUICK=0`):
   `python training/train_classifier.py` (about 45 min on an RTX 5070 Ti laptop GPU, 2.5 h on a Kaggle T4) and
   `python training/detector_and_baseline.py`. On Windows, `training\run_local.ps1` starts them as background processes.
4. `python tools/build_report.py --train models --det models --team T50 --usns "..."` writes `eval_P5_T50.md`.
   Seed 42 everywhere; the validation split is a stratified 10% of the official train split.
