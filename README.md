<div align="center">

# RescueLens

### Point a phone at a crashed car. Know what it is, where the high voltage runs, and when not to trust the answer.

[![Live app](https://img.shields.io/badge/live_app-open_on_your_phone-e8651a?style=for-the-badge)](https://priyanshu-senapati--rescuelens-web.us-east.modal.run)
[![CI](https://img.shields.io/github/actions/workflow/status/Priyanshu-senapati/Rescuelens/ci.yml?branch=main&style=for-the-badge&label=tests)](https://github.com/Priyanshu-senapati/Rescuelens/actions/workflows/ci.yml)

![Top-1 94.55%](https://img.shields.io/badge/top--1-94.55%25-2f5d8a)
![Unknown cars caught 58 of 60](https://img.shields.io/badge/unknown_cars_caught-58_of_60-2f5d8a)
![Hybrids missed 1.15%](https://img.shields.io/badge/hybrids_and_EVs_missed-1.15%25-e8651a)
![CPU answer 302 ms](https://img.shields.io/badge/classifier_on_CPU-302_ms-2e7d4f)

<img src="docs/img/live-lock.gif" alt="Live camera mode: edges are drawn over the car, corner ticks lock on, and a high-voltage verdict with the battery location rises into view" width="800">

<sub>Real answers from the deployed model, with this repository's guide data. The camera feed is simulated from a sample photo.</sub>

</div>

---

## Most classifiers are built to be right. RescueLens is built to know when being wrong is dangerous.

Before cutting a trapped person out of a car, rescuers need to know exactly what they are cutting. Hybrids and EVs hide high-voltage batteries and cables behind bodywork that can look identical to the petrol version. Today's tools need a number plate, VIN or QR code first, and at a crash those can be crushed, hidden or missing.

**India had 4,80,583 road accidents and 1,72,890 deaths in 2023** ([MoRTH](https://data.opencity.in/dataset/road-accidents-in-india-2023)), while yearly EV registrations grew **from 1.74 lakh to 19.68 lakh** between FY 2019-20 and FY 2024-25 ([Lok Sabha, source: Vahan](https://heavyindustries.gov.in/sites/default/files/2026-03/lsauq_2913_0.pdf)). Sources and caveats are in the [design document](docs/design.md).

RescueLens identifies a vehicle from how it looks, across 196 make, model and year classes, and then decides what is safe to conclude.

## One model, three honest answers

![Three results from the real model: identified Tesla Model S with a high-voltage warning, a partly hidden Chevrolet where a hybrid cannot be ruled out, and a Tata Nexon answered as not recognised](docs/img/three-answers.png)

<sub>Real answers from the deployed model, with this repository's guide data.</sub>

| Answer | When | What the responder is told |
|---|---|---|
| **Identified** | Sure of the exact model and year | The vehicle, plus where its battery and cables are if it's a hybrid or EV |
| **Not certain** | Sure of the brand or family only | The part it is sure of. If a hybrid is possible at all, treat it as high voltage |
| **Not recognised** | The car looks unlike anything it trained on | It refuses to guess. Treat as possibly high voltage and identify it by hand |

## What it does

- **Live camera that locks.** A frame every 1.5 s, with a live edge drawing and corner ticks on the detected vehicle. Three agreeing frames lock the answer.
- **Warnings never wait.** A high-voltage warning shows on the first frame that raises it, and a car flickering between "warning" and "no warning" can never lock as safe.
- **Summed-probability safety rule.** No single hybrid has to win. If all 9 hybrid/EV classes together reach 5%, the warning fires.
- **A rescue sheet that draws itself.** A technical drawing marks where the battery sits and how the cables run, read from manufacturer guides for all 9 models, with [every source and caveat listed](docs/hazard-sources.md).
- **Unknown-car detection.** Softmax confidence can stay high on cars it has never seen, so it also measures distance to the training cars.
- **Automatic stress sweep.** One button runs your photo through the real model at 8 rising levels of darkness, smoke, blur or occlusion, plots the confidence curve live and says exactly where the answer broke.
- **It shows its working.** While the model runs, the photo dims into a blueprint of its real edge map. When the answer lands, the drawing snaps onto the detected vehicle.
- **Three approaches, side by side.** Classical computer vision, deep learning and the hybrid on the same photo, with every classical step visible.
- **Works without signal (opt-in).** An installable app can run the models on the device, under the same safety rules, pinned to the server by a shared test file.
- **Fails safely.** A crash shows safe advice and a "Start again" button, never a blank page.

![The automatic stress sweep on the real model: the answer holds as blur rises, then the curve drops and the app says where it changed](docs/img/stress-sweep.png)

## Try these on the live app

Answers from the real deployed model, 6 Nov 2026.

| Sample | Answer | Why it's worth showing |
|---|---|---|
| Clear daylight photo | **Identified:** BMW M3 Coupe 2012, 98% | The easy case, with the full pipeline on show |
| Hybrid that looks like a petrol car | **Identified:** Hyundai Sonata Hybrid 2012, 99%, high voltage | The whole pitch in one photo |
| Taken at night | **Identified:** Audi S4 Sedan 2012, 92% | Darkness barely hurts accuracy |
| Partly hidden | **Not certain:** Chevrolet, hybrid can't be ruled out | Uncertainty changes the advice |
| Electric car | **Identified:** Tesla Model S 2012, 98%, high voltage | Finds the EV and warns about it |
| Look-alike trim: S5 or A5? | **Not certain:** Audi, model uncertain | Its most frequent mix-up, and it says so |
| Indian car outside the 196 models | **Not recognised** | Refuses to guess at a Tata Nexon |

## The numbers

Official Stanford Cars test split: 8,041 images, never used for training or tuning.

| | |
|---|---|
| **Top-1 / top-5 accuracy** | **94.55% / 99.27%** (ConvNeXt-Small, 384 px, detector crop) |
| **Finding the vehicle** (YOLO11s) | mean IoU **0.886**, mAP@0.5 **0.892**, mAP@0.5:0.95 0.825 |
| **Classical vs deep vs hybrid** | 13.7% vs 91.7% vs **94.6%** top-1 |
| **Missed hybrids and EVs** | **1.15%** with the summed rule vs 4.32% trusting the top answer, with 0.69% false alarms |
| **Unseen car models** (30 models, 60 photos) | **96.7%** answered "unknown"; 8.3% of known cars wrongly flagged |
| **Calibration** | ECE 0.098 -> **0.011** after temperature scaling |
| **Model size / CPU latency** | 189.4 MB -> **54.3 MB** INT8, 739 ms -> **302 ms** on 2 threads, no accuracy loss on a 300-image check |

Robustness, ablation, thresholds and failure cases: [`eval_P5_T50.md`](eval_P5_T50.md).

<details>
<summary><b>See the Model report page</b></summary>

![The Model report page with accuracy figures, the high-voltage threshold table and the unknown-car result](docs/img/model-report.png)

</details>

## How it works

![Architecture: photo or camera frame, validation, YOLO11s detector, crop, classical CV and ConvNeXt-Small classifier, calibrated probabilities and unknown score, safety engine, then the UI](docs/architecture.png)

Reasoning behind each choice: [design document](docs/design.md).

## Run it

```bash
docker build -t rescuelens --build-arg MODEL_BASE_URL=<folder-with-the-model-files> .
docker run -p 7860:7860 rescuelens          # open http://localhost:7860
```

Without `MODEL_BASE_URL` the app starts in mock mode with fixture responses. For UI work with no back end:

```bash
cd frontend && npm ci && VITE_MOCK=1 npm run dev
```

The live deployment runs on Modal: `modal deploy modal_app.py`. The camera needs https or localhost.

### Tests

```bash
cd backend && pip install -r requirements.txt -r requirements-dev.txt && pytest
```

```bash
cd frontend && npm ci && npx playwright install chromium && npm test
```

CI runs both on every push. Check the demo samples against the real models with `python tools/check_samples.py`.

### Reproduce the numbers

1. Get Stanford Cars with test labels (e.g. Kaggle `jutrera/stanford-car-dataset-by-classes-folder`) and extract it.
2. Python 3.11 + CUDA PyTorch, then `pip install timm onnx onnxruntime onnxscript scikit-learn scipy "opencv-python-headless<5" pillow ultralytics`.
3. From the repo root (set `DATA_ROOT` to the extracted folder, `EXPORT_DIR=models`, `QUICK=0`):
   `python training/train_classifier.py` (about 45 min on an RTX 5070 Ti laptop GPU, 2.5 h on a Kaggle T4) and
   `python training/detector_and_baseline.py`. On Windows, `training\run_local.ps1` starts them as background processes.
4. `python tools/build_report.py --train models --det models --team T50 --usns "..."` writes `eval_P5_T50.md`.
   Seed 42 everywhere; the validation split is a stratified 10% of the official train split.

## Find your way around

| Folder | What |
|---|---|
| `training/` | Training and evaluation scripts: classifier, detector, classical baseline |
| `backend/` | FastAPI inference service (ONNX Runtime) and the safety engine |
| `frontend/` | React UI: live camera, results, stress test, Model report, offline mode |
| `data/` | Class names, hybrid/EV guides and locations, measured safety numbers |
| `tools/` | Report builder, unknown-car evaluation, sample checker |
| `docs/` | Design, API, sources, viva sheets, field kit |

**Docs:** [Design](docs/design.md) · [API contract](docs/api-contract.md) · [High-voltage sources](docs/hazard-sources.md) · [Build plan](docs/BUILD_PLAN.md) · [Field kit](docs/field-kit.md) · [Viva sheets](docs/viva/) · [Evaluation](eval_P5_T50.md) · [Photo credits](frontend/public/samples/CREDITS.md)

## What it can't do

- It was trained on **US-market cars up to 2012**, so most cars on Indian roads today will be answered "unknown", on purpose. It doesn't recognise them.
- There are **no crashed vehicles in the data**. Robustness is measured with simulated darkness, haze, blur, low resolution and occlusion.
- **Weak spots:** severe blur (40% top-1), very low resolution (53%) and heavy occlusion (66%).
- **Offline mode has not been measured** with the real models on a phone.
- **No responder has reviewed the high-voltage text yet.**

> **Decision-support prototype.** Vehicle identification is probabilistic. Never rely on it alone for cutting, lifting or disabling a vehicle. It reads no number plates or faces, and uploads are processed in memory and not kept.

## Team T50 "Clover"

Computer Vision Hackathon 2026, problem statement P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System.

Priyanshu Kumar Senapati · Saatwik Kumar Yadav · Rachana Bhaskar Gowda · R Muni Disnitha
