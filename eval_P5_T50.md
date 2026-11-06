1RUA24CSE0337,1RUA24CSE0390,1RUA24CSE0348,1RUA24CSE0345

# Evaluation - P5 Intelligent Fine-Grained Vehicle Recognition - Team T50

Test set: official Stanford Cars test split (8041 images, never used for training or tuning). Validation: stratified 10% of the official train split.

## 1. Classification performance
| Metric | Value |
|---|---|
| Top-1 accuracy | 94.18% |
| Top-5 accuracy | 99.05% |
| Top-1 with flip TTA | 94.22% |

## 2. Object localization (IoU)
| Method | Mean IoU | IoU >= 0.5 |
|---|---|---|
| YOLO11s (deep) | 0.886 | 92.81% |
| Largest edge contour (classical) | 0.639 | 76.20% |
(n = 8041 / 2000 random test images)

## 3. Object detection (mAP)
| Metric | Value |
|---|---|
| mAP@0.5 | 0.892 |
| mAP@0.5:0.95 | 0.825 |

## 4. Model efficiency
| Model | Size | CPU latency (2 threads) | Top-1 (subset) |
|---|---|---|---|
| Classifier FP32 (ONNX) | 106.8 MB | 324 ms | 94.33% |
| Classifier INT8 (dynamic quantization) | 32.4 MB | 127 ms | 94.00% |
| YOLO11s detector | - | 39 ms (0) | - |
Parameters: 28.0 M. Subset = first 300 test images.

## 5. Robustness (top-1 under degraded conditions)
2000 random test images.

| Condition | Severity | Top-1 |
|---|---|---|
| clean | 0 | 94.00% |
| dark | 1 | 94.10% |
| dark | 2 | 94.20% |
| dark | 3 | 94.20% |
| dark+clahe | 1 | 93.80% |
| dark+clahe | 2 | 93.80% |
| dark+clahe | 3 | 93.70% |
| haze | 1 | 94.15% |
| haze | 2 | 94.15% |
| haze | 3 | 94.25% |
| blur | 1 | 90.80% |
| blur | 2 | 72.40% |
| blur | 3 | 43.65% |
| low_res | 1 | 92.20% |
| low_res | 2 | 85.75% |
| low_res | 3 | 55.25% |
| occlusion | 1 | 91.80% |
| occlusion | 2 | 83.60% |
| occlusion | 3 | 61.95% |

Classical CLAHE before the network on dark photos: severity 1: 94.10% -> 93.80%, severity 2: 94.20% -> 93.80%, severity 3: 94.20% -> 93.70%

## 6. Confidence and threshold analysis
Temperature scaling: T = 0.695, ECE 0.1248 -> 0.0173.

| Answer only when most confident | Accuracy |
|---|---|
| top 100% of photos | 94.18% |
| top 95% of photos | 96.47% |
| top 90% of photos | 97.53% |
| top 80% of photos | 98.10% |
| top 70% of photos | 98.42% |
| top 60% of photos | 98.65% |
| top 50% of photos | 98.71% |

High-voltage (hybrid/EV) warning threshold on summed calibrated probability:

| Threshold | HV miss rate | False alarm rate |
|---|---|---|
| 0.05 | 0.86% | 0.71% |
| 0.1 | 1.73% | 0.35% |
| 0.2 | 2.59% | 0.18% |
| 0.3 | 4.03% | 0.14% |
| 0.5 | 5.48% | 0.08% |
Plain top-1 (argmax) would miss 4.61% of hybrid/EV test cars (347 images).
Unknown-car check: 5.00% of known test cars wrongly flagged unknown; n/a of 0 photos of cars outside the 196 classes (Indian / post-2012 models) correctly answered 'unknown'.

## Approach comparison (classical vs deep learning vs hybrid)
| Approach | What it is | Top-1 | Top-5 |
|---|---|---|---|
| Classical | HOG + HSV colour + LBP + geometry -> PCA -> logistic regression | 13.72% | 34.03% |
| Deep learning | ConvNeXt-T on the whole photo | 90.86% | - |
| Hybrid | detector crop + CLAHE when dark + calibrated fusion + safety layer | 94.18% | 99.05% |
Fusion weight on classical probabilities (chosen on validation): 0.0.

## Ablation (each row removes or swaps one component)
| Variant | Top-1 |
|---|---|
| Full hybrid | 94.18% |
| - classical fusion | 94.18% |
| - detection crop (whole photo) | 90.86% |
| + flip TTA | 94.22% |
| FP32 instead of INT8 (subset) | 94.33% vs 94.00% |

## Most-confused classes (failure-case candidates)
| True | Predicted | Count |
|---|---|---|
| Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 | 14 |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 10 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 10 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 9 |
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 9 |
| Dodge Sprinter Cargo Van 2009 | Mercedes-Benz Sprinter Van 2012 | 8 |
| GMC Savana Van 2012 | Chevrolet Express Van 2007 | 8 |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 8 |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 8 |
| Bugatti Veyron 16.4 Coupe 2009 | Bugatti Veyron 16.4 Convertible 2009 | 8 |
