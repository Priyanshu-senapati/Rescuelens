1RUA24CSE0337,1RUA24CSE0390,1RUA24CSE0348,1RUA24CSE0345

# Evaluation - P5 Intelligent Fine-Grained Vehicle Recognition - Team T50

Test set: official Stanford Cars test split (8041 images, never used for training or tuning). Validation: stratified 10% of the official train split.

## 1. Classification performance
| Metric | Value |
|---|---|
| Top-1 accuracy | 94.55% |
| Top-5 accuracy | 99.27% |
| Top-1 with flip TTA | 94.70% |

## 2. Object localization (IoU)
| Method | Mean IoU | IoU >= 0.5 |
|---|---|---|
| yolo11s (fine-tuned on Stanford Cars train boxes) | 0.954 | 99.85% |
| yolo11s (COCO pretrained, car/bus/truck) | 0.886 | 92.81% |
| Largest edge contour (classical) | 0.639 | 76.20% |
(n = 8041 / 2000 random test images)

## 3. Object detection (mAP)
| Metric | Value |
|---|---|
| mAP@0.5 | 0.999 |
| mAP@0.5:0.95 | 0.941 |
| mAP@0.5 / @0.5:0.95 before fine-tuning (COCO-pretrained) | 0.892 / 0.825 |

## 4. Model efficiency
| Model | Size | CPU latency (2 threads) | Top-1 (subset) |
|---|---|---|---|
| Classifier FP32 (ONNX) | 189.4 MB | 739 ms | 93.67% |
| Classifier INT8 (dynamic quantization) | 54.3 MB | 302 ms | 93.67% |
| YOLO11s detector | - | 26 ms (0) | - |
Parameters: 49.6 M. Subset = first 300 test images.

## 5. Robustness (top-1 under degraded conditions)
2000 random test images.

| Condition | Severity | Top-1 |
|---|---|---|
| clean | 0 | 94.40% |
| dark | 1 | 94.30% |
| dark | 2 | 94.30% |
| dark | 3 | 93.75% |
| dark+clahe | 1 | 94.35% |
| dark+clahe | 2 | 94.25% |
| dark+clahe | 3 | 93.90% |
| haze | 1 | 94.45% |
| haze | 2 | 94.40% |
| haze | 3 | 94.00% |
| blur | 1 | 90.55% |
| blur | 2 | 69.05% |
| blur | 3 | 40.40% |
| low_res | 1 | 90.80% |
| low_res | 2 | 83.10% |
| low_res | 3 | 52.60% |
| occlusion | 1 | 91.25% |
| occlusion | 2 | 84.05% |
| occlusion | 3 | 66.10% |

Classical CLAHE before the network on dark photos: severity 1: 94.30% -> 94.35%, severity 2: 94.30% -> 94.25%, severity 3: 93.75% -> 93.90%

## 6. Confidence and threshold analysis
Temperature scaling: T = 0.768, ECE 0.0984 -> 0.0108.

| Answer only when most confident | Accuracy |
|---|---|
| top 100% of photos | 94.55% |
| top 95% of photos | 96.48% |
| top 90% of photos | 97.58% |
| top 80% of photos | 98.34% |
| top 70% of photos | 98.35% |
| top 60% of photos | 98.34% |
| top 50% of photos | 98.38% |

High-voltage (hybrid/EV) warning threshold on summed calibrated probability:

| Threshold | HV miss rate | False alarm rate |
|---|---|---|
| 0.05 | 1.15% | 0.69% |
| 0.1 | 2.02% | 0.38% |
| 0.2 | 2.88% | 0.30% |
| 0.3 | 3.75% | 0.18% |
| 0.5 | 5.48% | 0.10% |
Plain top-1 (argmax) would miss 4.32% of hybrid/EV test cars (347 images).
Unknown-car check: 8.33% of known test cars wrongly flagged unknown; 96.67% of 60 photos of cars outside the 196 classes (Indian / post-2012 models) correctly answered 'unknown'.

## Approach comparison (classical vs deep learning vs hybrid)
| Approach | What it is | Top-1 | Top-5 |
|---|---|---|---|
| Classical | HOG + HSV colour + LBP + geometry -> PCA -> logistic regression | 13.72% | 34.03% |
| Deep learning | ConvNeXt-T on the whole photo | 91.66% | - |
| Hybrid | detector crop + CLAHE when dark + calibrated fusion + safety layer | 94.55% | 99.27% |
Fusion weight on classical probabilities (chosen on validation): 0.0.

## Ablation (each row removes or swaps one component)
| Variant | Top-1 |
|---|---|
| Full hybrid | 94.55% |
| - classical fusion | 94.55% |
| - detection crop (whole photo) | 91.66% |
| + flip TTA | 94.70% |
| FP32 instead of INT8 (subset) | 93.67% vs 93.67% |

## Most-confused classes (failure-case candidates)
| True | Predicted | Count |
|---|---|---|
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 15 |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 13 |
| Bentley Continental GT Coupe 2007 | Bentley Continental GT Coupe 2012 | 11 |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 9 |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 9 |
| Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 | 8 |
| Chevrolet Silverado 1500 Regular Cab 2012 | Chevrolet Silverado 2500HD Regular Cab 2012 | 8 |
| Audi TTS Coupe 2012 | Audi TT Hatchback 2011 | 8 |
| Audi 100 Sedan 1994 | Audi V8 Sedan 1994 | 7 |
| BMW 6 Series Convertible 2007 | BMW M6 Convertible 2010 | 7 |
