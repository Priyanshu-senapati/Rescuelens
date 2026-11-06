# Saatwik Kumar Yadav (1RUA24CSE0390)

**Owns:** Product and UI: live camera, crash handling, browser tests, CI. **Demo slot:** 1:20-2:20: live camera and stress test.

## Your part

| Question | Short answer |
|---|---|
| How does the lock work? | One frame every 1.5 s, one request at a time. 3 agreeing frames in a row lock it. For an unknown car the closest match is ignored; for an ambiguous one only the level it is sure of counts. |
| How is live mode kept safe? | The high-voltage state is part of what must agree, so a flickering warning can never lock as safe. A warning shows on the first frame that raises it and stays for 3 frames. |
| Is the edge overlay real? | Yes. Sobel plus non-maximum suppression (the core of Canny) runs in the browser about 12 times a second, brightest inside the detector box. It is visual feedback; the answer comes from the server. |
| What is the automatic sweep? | It re-renders the photo at 8 rising levels of one condition, sends each to the real model and plots usable confidence, marking the first level where the answer changes. Real calls, nothing precomputed. |
| Is the rescue sheet accurate? | It marks the region the manufacturer's guide describes on a generic body shape. Not to scale, and the app says it isn't a guide for where to cut. |
| What if something breaks? | An error boundary shows "Start again" plus safe advice. Blocked camera and lost server each have their own message. Browser tests cover these and run on every push. |

**Your failure case:** Mahindra Thar answered as Jeep Wrangler: it is styled on the Wrangler, so the unknown check can miss close look-alikes.

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
