# Rachana Bhaskar Gowda (1RUA24CSE0348)

**Owns:** Safety content and real-world validation. **Demo slot:** 0:50-1:20: the hybrid that looks like a petrol car.

## Your part

| Question | Short answer |
|---|---|
| Where do battery locations come from? | Read from manufacturer guides for all 9 models, with pages listed in docs/hazard-sources.md. Only the most likely hybrid's location is shown, and it names that car. |
| How far can they be trusted? | Seven links are copies hosted by fire services or rescue sites, the Malibu guide is older than the 2010 car, and the BMW sheet is German. No responder has reviewed the text yet. |
| Any surprise in the guides? | The Malibu Hybrid's cables are blue and 36-42 V, not orange, so "look for orange" alone would be wrong. The app's advice now says so. |
| Why does India need this? | 4,80,583 accidents and 1,72,890 deaths in 2023 (MoRTH). EV registrations 1.74 lakh in FY20 to 19.68 lakh in FY25 (Lok Sabha, Vahan). Most EVs are two- and three-wheelers, which we don't cover. |

**Your failure case:** Sonata vs Sonata Hybrid: same body, different powertrain. Summing probabilities cuts missed hybrids from 4.3% to 1.2%.

## Everyone must be able to answer

| Question | Short answer |
|---|---|
| Why ConvNeXt at 384 px? | Fine-grained classes differ in small details, so resolution matters. ConvNeXt-Small (ImageNet-22k pretrained) beat Tiny on validation, 94.23% to 93.74%. Test: 94.55% top-1, and 96.7% of unknown cars caught against Tiny's 88.3%. |
| Why YOLO pretrained, not fine-tuned? | Stanford photos hold one vehicle, so COCO YOLO11s already finds it: mean IoU 0.886, mAP@0.5 0.892. We have no labelled crash data to tune on. Fine-tuning on Stanford boxes is listed as future work. |
| What did CLAHE do? | Before the classifier it did not help (within +/-0.15 points on dark photos), because training already darkens images. So it is off there. It still retries the detector and drives the pipeline view. We kept that as a measured result. |
| Temperature scaling and ECE? | Divide logits by T = 0.768, fitted on validation, so "80% sure" means about 80% right. ECE fell from 0.098 to 0.011. Accuracy doesn't change, but thresholds become meaningful. |
| The summed-probability rule? | Add up the probability of all 9 hybrid/EV classes and warn at 0.05 or above. It misses 1.15% with 0.69% false alarms, against 4.32% missed if we trust the top answer. A miss is far worse than a false alarm. |
| How does the unknown check work? | 1 minus the mean cosine similarity to the 5 nearest training embeddings. Threshold: 95th percentile on validation. 58 of 60 unseen cars caught; 8.3% of known cars wrongly flagged. |
| Why dynamic INT8, MatMul/Gemm only? | ConvNeXt's heavy layers are Linear, and quantising Conv needs ConvInteger, which ONNX Runtime CPU lacks for int8. 189.4 MB to 54.3 MB, 739 ms to 302 ms, no accuracy loss on a 300-image check. |
| What are the limits? | US-market cars up to 2012, no crashed vehicles in the data, and simulated degradations only. Weak spots: severe blur 40%, very low resolution 53%, heavy occlusion 66%. It is decision support, never a replacement for official documents. |
| Is the classical part useful? | Classical alone gets 13.7%. Fusion weight came out 0 on validation, and we report that. Its value: the detector retry, the quality reasons behind low confidence, and the visible pipeline. The hybrid gain is the crop (+2.9 points), calibration and safety rules. |

**If you don't know:** say so, then say where the number lives (`eval_P5_T50.md`). Never guess a number.
