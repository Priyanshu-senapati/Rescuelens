# R Muni Disnitha (1RUA24CSE0345)

**Owns:** Evidence, documentation and demo. **Demo slot:** 2:20-3:00: Model report and closing line.

## Your part

| Question | Short answer |
|---|---|
| Where do your numbers come from? | eval_P5_T50.md: the official 8,041-image test split, never used for tuning. A test fails if the report page disagrees with it. |
| What was the ablation result? | Remove the crop: 94.55% to 91.66%. Remove classical fusion: no change. Add flip TTA: 94.70%. FP32 and INT8 match on the subset. |
| How honest is the unknown number? | 60 photos is small, and they also influenced our choice of model. That is why the parking-lot photos matter: a second, independent test. |
| What if the demo fails? | A recorded backup video follows the same 3-minute script, kept on two laptops and a phone. |

**Your failure case:** Heavy blur: top-1 falls to 40%, but the "Blurry" check explains why confidence is low instead of hiding it.

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
