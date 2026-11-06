# Priyanshu Kumar Senapati (1RUA24CSE0337)

**Owns:** Classifier training, calibration, safety metrics, deployment. **Demo slot:** 0:00-0:50: problem and the clear sample.

## Your part

| Question | Short answer |
|---|---|
| How was it trained? | ConvNeXt-Small fine-tuned at 384 px on ground-truth crops (box + 8% margin), with flips, darkening, jitter and random erasing. Stratified 90/10 validation split, seed 42. The test split was used once, for final numbers. |
| Why does the crop help? | Whole photo 91.66%, crop 94.55%: the network spends its pixels on the car rather than the background. |
| How is it deployed? | One warm 2-CPU Modal container with FastAPI and ONNX Runtime. Without model files the app falls back to sample answers instead of crashing. |

**Your failure case:** Audi S5 taken for A5 (15 photos, our most frequent error): the trims differ mainly in grille, bumpers and badges.

## Everyone must be able to answer

| Question | Short answer |
|---|---|
| Why ConvNeXt at 384 px? | Fine-grained classes differ in small details, so resolution matters. ConvNeXt-Small (ImageNet-22k pretrained) beat Tiny on validation, 94.23% to 93.74%. Test: 94.55% top-1, and 96.7% of unknown cars caught against Tiny's 88.3%. |
| Why fine-tune YOLO? | COCO weights already found the single car (mean IoU 0.886) but boxes were loose. Fine-tuned on the Stanford **train** boxes as one class: IoU 0.954, mAP@0.5 0.999, mAP@0.5:0.95 0.941. No crash data exists to tune on, so that gap remains. |
| What did CLAHE do? | Before the classifier it did not help (within +/-0.15 points on dark photos), because training already darkens images. So it is off there. It still retries the detector and drives the pipeline view. We kept that as a measured result. |
| Temperature scaling and ECE? | Divide logits by T = 0.768, fitted on validation, so "80% sure" means about 80% right. ECE fell from 0.098 to 0.011. Accuracy doesn't change, but thresholds become meaningful. |
| The summed-probability rule? | Add up the probability of all 9 hybrid/EV classes and warn at 0.05 or above. It misses 1.15% with 0.69% false alarms, against 4.32% missed if we trust the top answer. A miss is far worse than a false alarm. |
| How does the unknown check work? | 1 minus the mean cosine similarity to the 5 nearest training embeddings. Threshold: 95th percentile on validation. 58 of 60 unseen cars caught; 8.3% of known cars wrongly flagged. |
| Why dynamic INT8, MatMul/Gemm only? | ConvNeXt's heavy layers are Linear, and quantising Conv needs ConvInteger, which ONNX Runtime CPU lacks for int8. 189.4 MB to 54.3 MB, 739 ms to 302 ms, no accuracy loss on a 300-image check. |
| What are the limits? | US-market cars up to 2012, no crashed vehicles in the data, and simulated degradations only. Weak spots: severe blur 40%, very low resolution 53%, heavy occlusion 66%. It is decision support, never a replacement for official documents. |
| Is the classical part useful? | Classical alone gets 13.7%. Fusion weight came out 0 on validation, and we report that. Its value: the detector retry, the quality reasons behind low confidence, and the visible pipeline. The hybrid gain is the crop (+2.9 points), calibration and safety rules. |

**If you don't know:** say so, then say where the number lives (`eval_P5_T50.md`). Never guess a number.
