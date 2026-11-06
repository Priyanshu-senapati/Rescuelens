# RescueLens - Design Document

**Team T50 "Clover"** · Computer Vision Hackathon 2026 · Problem statement **P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System**

> All numbers quoted here come from `eval_P5_T50.md` (official Stanford Cars test split, never used for training or tuning).

## 1. Problem

P5 asks for a system that finds a vehicle in a photo and names its make, model and year among 196 visually
similar Stanford Cars classes, combining classical computer vision with deep learning, handling low confidence,
poor lighting, occlusion and unusual viewpoints.

We applied it to a setting where getting this wrong has consequences: **crash response**. Before cutting a trapped
person out of a car, rescuers need the exact model and generation to know where airbags, reinforced steel and,
for hybrids and EVs, high-voltage batteries and cables are. Existing tools (Euro RESCUE / ANCAP RESCUE rescue sheets,
Crash Recovery System, manufacturer QR stickers, the AURA concept) all need the vehicle to be identified **first** -
by number plate, VIN, QR code or manual search. At a crash those can be missing, crushed or unreadable.

**RescueLens identifies the vehicle from how it looks, and - more importantly - says when it is not sure.**

## 2. Goals and non-goals

| Goals | Non-goals |
|---|---|
| Detect the vehicle and draw its bounding box | Number-plate recognition, tracking, traffic events (excluded by P5) |
| Make / model / year with calibrated confidence | Replacing official rescue documentation |
| Classical vs deep vs hybrid comparison, ablation, failure analysis | Recognising cars outside the 196 classes (we detect and flag them instead) |
| Turn uncertainty into a safe action (identified / ambiguous / unknown) | Damage assessment |
| Live web app usable from a phone, answer in seconds | |

## 3. Architecture

```mermaid
flowchart LR
    U[Photo upload<br/>phone or laptop] --> V[Validate + EXIF rotate<br/>+ downscale]
    V --> D[YOLO11s detector<br/>ONNX, car/bus/truck]
    D -- no vehicle --> D2[Retry on CLAHE-enhanced photo]
    D2 -- still none --> NV[status: no_vehicle]
    D --> R[ROI crop<br/>box + 8% margin]
    D2 --> R
    R --> C[Classical CV<br/>CLAHE, Canny, contours,<br/>Hough wheels, colour k-means,<br/>quality scores]
    R --> G{Dark?<br/>brightness < 30}
    G -- yes --> E[CLAHE enhance] --> M
    G -- no --> M[ConvNeXt-Tiny classifier<br/>ONNX INT8, 196 classes]
    M --> P[Temperature-calibrated<br/>probabilities]
    M --> K[Embedding -> kNN distance<br/>to training cars = unknown score]
    C --> B[Classical baseline<br/>HOG+colour+LBP -> PCA -> LogReg]
    P --> S[Safety engine]
    K --> S
    C -- quality issues --> S
    S --> O[identified / ambiguous / unknown<br/>+ high-voltage advice]
    O --> UI[React UI]
    B --> UI
```

| Unit | Code | Does | Depends on |
|---|---|---|---|
| Detector | `backend/app/models.py::Detector` | Letterbox, ONNX inference, NMS, picks the main vehicle | `yolo11s.onnx` |
| Classical CV | `backend/app/classical.py` | Enhancement, edges, contours, wheels, colour, texture, quality, baseline features, step images | OpenCV |
| Classifier | `backend/app/models.py::Classifier` | Preprocess like training, logits + embedding, temperature, unknown score | `classifier_int8.onnx`, `config.json`, `train_emb.npy` |
| Classical baseline | `backend/app/models.py::Baseline` | Numpy re-implementation of the trained scaler -> PCA -> logistic regression | `baseline.npz` |
| Safety engine | `backend/app/safety.py` | Hierarchy (make -> family -> vehicle), 3 result states, summed high-voltage probability, actions | `data/hazards.json` |
| Pipeline / API | `backend/app/pipeline.py`, `main.py` | Orchestration, timing, errors, mock mode, serves UI | all of the above |
| UI | `frontend/` | Upload / camera, result, pipeline steps, stress test, comparison dashboard | `docs/api-contract.md` |
| Training | `training/` | Classifier training + all metrics; detector eval + classical baseline | Stanford Cars |

## 4. Data

* **Stanford Cars**: 16,185 images, 196 classes (make, model, year), one ground-truth box per image.
  8,144 train / 8,041 test.
* **Split**: stratified 90/10 train/validation from the official train set (seed 42). The official test set is used
  once, for the final numbers only.
* **Preprocessing**: crop the ground-truth box + 8% margin (same rule the app uses on detector boxes), resize to 384x384.
* **Augmentation**: random resized crop, horizontal flip, random darkening (night), colour jitter, random erasing
  (occlusion).
* **Known gaps**: newest model year is 2012, US-market cars only, no damaged vehicles. Handled by the unknown check
  and stated as limitations.

## 5. Models and why

| Choice | Why |
|---|---|
| **ConvNeXt-Tiny**, ImageNet-22k pretrained, fine-tuned at 384 px | Strong fine-grained transfer at 28 M parameters; small enough for CPU inference after INT8 |
| **YOLO11s** COCO-pretrained (car/bus/truck) | Stanford images are single-vehicle, so a pretrained detector already localises well; no labelled crash data to fine-tune on |
| **Classical baseline**: HOG + HSV histogram + LBP + aspect ratio -> PCA(256) -> logistic regression | Required comparison; uses exactly the same feature code as the app |
| **Temperature scaling** on validation | Raw softmax is over/under-confident; calibrated probabilities make thresholds meaningful |
| **kNN embedding distance** for unknown cars | Softmax confidence stays high on unseen models (e.g. Indian cars); distance to known cars does not |
| **Dynamic INT8** (MatMul/Gemm) via ONNX Runtime | ~3-4x smaller, faster on the 2-vCPU host; shipped only if it loses < 1 point of accuracy |

## 6. Hybrid integration (where classical CV earns its place)

1. **Localisation**: detector crop vs whole image (ablation: crop vs no crop).
2. **Enhancement**: CLAHE before classification on dark photos, and as a detector retry when nothing is found.
3. **Measurement**: brightness, contrast, sharpness (Laplacian variance), resolution and cut-off checks explain
   *why* a prediction is unreliable ("Low light", "Blurry").
4. **Late fusion**: classical-baseline probabilities fused with deep probabilities (weight chosen on validation).
5. **Interpretability**: every classical stage is returned as an image and shown in the UI.

## 7. Low-confidence handling (the safety engine)

* **Hierarchy**: probabilities summed to family (e.g. "Hyundai Sonata") and make, so the system answers at the deepest
  level it is sure of.
* **States**: `identified` (vehicle prob >= 0.6) · `ambiguous` (family or make level) · `unknown`
  (unknown score above threshold or make prob < 0.5).
* **High voltage**: P(hybrid/EV) = sum of the 9 hybrid/EV class probabilities. Warn at >= 0.10 - because missing a hybrid
  is far worse than a false alarm. Unknown vehicles are treated as *possibly* high-voltage.
* Thresholds come from the validation set; the threshold table (miss rate vs false alarms) is in the eval file.

## 8. Evaluation

See `eval_P5_T50.md`: top-1/top-5, IoU, mAP@0.5 and @0.5:0.95, model size + latency (FP32 vs INT8),
robustness under darkness / haze / blur / low resolution / occlusion, calibration (ECE), risk-coverage,
high-voltage miss / false-alarm rates, classical vs deep vs hybrid, ablation, most-confused pairs.

## 9. Failure cases

1. **Same body, different powertrain**: Hyundai Sonata Sedan 2012 vs Sonata Hybrid Sedan 2012 - near-identical
   exterior. Handled by the summed high-voltage probability instead of the top-1 label.
2. **Same model, different generation**: e.g. Audi S4 Sedan 2007 vs 2012. The year label stands for a generation;
   the system backs off to the family and lists both years.
3. **Rebadged twins**: Chevrolet Express vs GMC Savana vans, Chevrolet Tahoe vs GMC Yukon hybrids - same vehicle,
   different badge.
4. **Out-of-distribution cars** (post-2012 or non-US models): would get a confident wrong label from softmax alone;
   caught by the unknown score (rate reported in the eval file).
(Final examples with images are taken from the most-confused pairs in the eval file.)

## 10. Deployment

One Docker image (React build + FastAPI + ONNX Runtime) on Hugging Face Spaces (2 vCPU, 16 GB). GitHub `main` is
mirrored to the Space by a GitHub Action. Model files live in the `rescuelens-models` repo and are fetched at build
time. Inputs are validated (type, 10 MB limit, decompression-bomb guard, EXIF rotation, HEIC); errors return a JSON
error code shown by the UI; without model files the app serves fixture responses instead of crashing.

## 11. Limitations

* Trained on 2012-and-older US-market cars; anything newer is flagged unknown rather than identified.
* No crashed vehicles in the data; robustness is measured with simulated darkness, haze, blur, low resolution and occlusion.
* The rescue layer links official guides for the hybrid/EV classes only; it is a prototype, not certified.

## 12. Ethics and safety

* **Decision support only**: every result carries a disclaimer; the UI never hides uncertainty.
* **Conservative by design**: unknown vehicles and ambiguous hybrids default to "treat as high-voltage".
* **Privacy**: no number plates or faces are read or stored; uploads are processed in memory and not kept.
* **Dual use**: fine-grained vehicle recognition can support surveillance; we deliberately excluded plate reading and
  tracking and scoped the application to rescue.

## 13. Related work

Euro NCAP Euro RESCUE and ANCAP RESCUE (rescue-sheet apps, manual model search / QR), Bliksund Crash Recovery System
(plate/VIN lookup), Mercedes-Benz Rescue Assist (QR), AURA (James Dyson Award 2026 concept: plate recognition + AR),
Motorola Assist (police footage search). None identify the vehicle from its appearance for rescue, or turn
identification uncertainty into a safety decision.

## 14. Team

| Member | USN | Owns |
|---|---|---|
| Priyanshu Kumar Senapati | 1RUA24CSE0337 | Classifier training, calibration, safety metrics |
| Saatwik Kumar Yadav | 1RUA24CSE0390 | (fill in) |
| Rachana Bhaskar Gowda | 1RUA24CSE0348 | (fill in) |
| R Muni Disnitha | 1RUA24CSE0345 | (fill in) |
