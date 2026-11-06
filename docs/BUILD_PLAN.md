# RescueLens - build plan and handover

Team T50 "Clover" · P5 · **Submission deadline: Sat 7 Nov 2026, 14:00 IST** (aim to submit by 13:30; -5 points per 15 min late).
Last updated: Fri 6 Nov 2026, 20:15 IST.

This file is the single source of truth for what is done, what is pending, and what was dropped and should be added
back. Work top to bottom: section 2 is the critical path to a valid submission.

---

## 1. Where things stand

### Done and pushed to `main`
| Area | State | Where |
|---|---|---|
| Classifier (run 1) | ConvNeXt-Tiny 384 px, **94.18% top-1 / 99.05% top-5** on the 8,041-image test split | `models/` (local only, not in git) |
| Detector | YOLO11s COCO-pretrained, mean IoU **0.886**, mAP@0.5 **0.892**, mAP@0.5:0.95 0.825 | `models/yolo11s.onnx` |
| Classical baseline | HOG+HSV+LBP+geometry -> PCA -> LogReg, 13.7% / 34.0% | `models/baseline.npz` |
| INT8 | 106.8 MB -> 32.4 MB, 324 -> 127 ms on 2 CPU threads, 94.3% -> 94.0% (subset) | `models/classifier_int8.onnx` |
| Safety layer | identified / ambiguous / unknown, summed hybrid/EV probability (threshold **0.05**: 0.9% missed vs 4.6% top-1, 0.7% false alarms) | `backend/app/safety.py` |
| Unknown cars | **88.3%** of 60 unseen models (Wikimedia Commons) answered "unknown", 5% of known cars wrongly flagged | `tools/eval_unknown.py`, `private_photos/unknown/` (git-ignored) |
| Back end | FastAPI + ONNX Runtime, mock mode without models, input validation, 22 tests passing | `backend/` |
| Front end | React + Vite + Tailwind: verdict, high-voltage banner, confidence ladder, top-5, quality reasons, pipeline steps + before/after, stress-test sliders, 3-approach comparison, model report | `frontend/` |
| Docs | design doc + mermaid architecture diagram (matches code), API contract, README results + reproduction steps | `docs/`, `README.md` |
| Eval file | `eval_P5_T50.md`, first line = the 4 USNs, sections in P5 metric order | repo root |
| Deploy config | `modal_app.py` (Modal), `Dockerfile` (any Docker host) | repo root |

### Running / in progress when this was written
- **Run 2 (ConvNeXt-Small, 384 px)**: best val 94.23% (run 1: 93.74%), **test top-1 94.55% / top-5 99.27%**
  (run 1: 94.18% / 99.05%), whole-photo 91.66%. **Run 2 is the better model -> switch to it (section 2.1).**
  Its robustness/ONNX/INT8 steps were still running and write everything to `models_small/`
  (log: `training/train_classifier_models_small.log`, wait for `DONE`).
- Local server with the run-1 model: `http://localhost:8765` (started from `backend/` with `MODEL_DIR=models`).

---

## 2. Critical path to submission (must do, in order)

### 2.1 Pick the final classifier (10 min)
1. Open `training/train_classifier_models_small.log`, check it ends with `DONE`, read `"top1"` in `models_small/metrics.json`.
2. Run 2 test top-1 is 0.9455 > 0.9418, so **switch to run 2** once its log says `DONE`:
   ```powershell
   cd C:\Users\priyanshu\cv\Rescuelens
   # keep the detector + baseline files from run 1, replace the classifier files
   copy models_small\config.json, models_small\classifier_int8.onnx, models_small\classifier_fp32.onnx, `
        models_small\train_emb.npy, models_small\train_labels.npy, models_small\logits.npz, models_small\metrics.json models\
   ```
3. Set the high-voltage threshold in `models\config.json` to `0.05` (run 2 was started before that change, so it wrote 0.1).
4. Re-measure the unknown-car rate and rebuild the report:
   ```powershell
   backend\.venv\Scripts\python tools\eval_unknown.py private_photos\unknown --models models
   backend\.venv\Scripts\python tools\build_report.py --train models --det models --team T50 --usns "1RUA24CSE0337,1RUA24CSE0390,1RUA24CSE0348,1RUA24CSE0345" --out models
   ```
5. Update the numbers in `README.md` (results table) and `docs/design.md` (sections 5, 7, 9) if the model changed. If run 2
   wins, the model name in the README/design doc becomes ConvNeXt-Small (50 M params) and the latency/size numbers come from
   `models\metrics.json -> efficiency`.
6. Commit + push.

### 2.2 Deploy the live URL on Modal (20 min)
Hugging Face now requires PRO for Docker/Gradio Spaces, so we host on Modal (free $30/month credit, no card).
1. Sign up at modal.com with GitHub (account owner: Priyanshu).
2. Log in once on the laptop: `training\.venv\Scripts\modal.exe setup` (opens the browser, saves a token in your user folder).
3. Build the UI and deploy:
   ```powershell
   cd C:\Users\priyanshu\cv\Rescuelens\frontend; npm ci; npm run build; cd ..
   training\.venv\Scripts\modal.exe deploy modal_app.py
   ```
   The command prints the public URL (`https://<workspace>--rescuelens-web.modal.run`).
4. Check on a phone **and** a laptop: upload page loads, each sample gives an answer in under 10 s, "Model report" shows numbers.
5. Put the URL in `README.md` (top) and in the submission form.
6. `min_containers=1` keeps it warm (about $2.5/day of credit). After judging, run `modal app stop rescuelens`.

**Backup if Modal fails:** run the app on the laptop and expose it with a Cloudflare quick tunnel
(`cloudflared tunnel --url http://localhost:8765`, free, no account; the laptop must stay on and online). Download
`cloudflared` only from Cloudflare's official GitHub releases.

### 2.3 Make the Dockerfile reproducible for judges (15 min)
`models/` is not in git (100 MB GitHub limit). Publish the 6 deployment files as a GitHub Release and point the Dockerfile at it:
```powershell
gh release create v1.0 models\config.json models\classifier_int8.onnx models\train_emb.npy models\yolo11s.onnx `
   models\baseline.npz models\metrics_public.json --title "RescueLens model files" --notes "ONNX models for the Dockerfile"
```
Then set `ARG MODEL_BASE_URL="https://github.com/Priyanshu-senapati/Rescuelens/releases/download/v1.0"` in the `Dockerfile`
and update the "Run it" section of the README. (Decide as a team; it makes the model weights public.)

### 2.4 Documentation gaps (30 min, anyone)
- [ ] `docs/design.md` section 14: fill in each member's role (who explains what in the viva).
- [ ] Ask the organisers whether a specific **design-doc template** exists (the rubric mentions "every template section");
      if yes, map our sections onto it.
- [ ] Ask the organisers which rubric weights are right (text says "others carry 15", table says 25/20/15/15/25).
- [ ] Add 3 failure-case screenshots to the design doc (from the app: an "ambiguous" stress-test result, the Thar->Wrangler
      unknown miss, a heavy-blur failure).
- [ ] Re-read `eval_P5_T50.md` once the final model is chosen; the order must follow the P5 metrics table.

### 2.5 Demo and viva prep (45 min, whole team, Saturday morning)
3-minute script:
1. (0:00-0:20) Problem: crash scenes hide the plate/VIN/QR; rescue sheets need the exact model; hybrids hide high voltage.
2. (0:20-0:50) Clear sample -> identified, box, pipeline steps.
3. (0:50-1:20) "Hybrid that looks like a petrol car": classical-only says Sonata (petrol), RescueLens says Sonata **Hybrid**
   + high-voltage banner + official guide.
4. (1:20-1:50) Stress test: slide blur/hidden until the status flips to "Not certain" and the advice changes.
5. (1:50-2:20) Judge's own photo (likely an Indian car) -> "Not recognised, treat as possibly high-voltage".
6. (2:20-2:50) Model report: 94.2% top-1, crop +3.3 points, INT8 4x smaller, 0.9% missed hybrids, 88% unknown caught.
7. (2:50-3:00) "Most classifiers are built to be right; RescueLens is built to know when being wrong is dangerous."

Viva questions every member should answer: why ConvNeXt + why 384 px; why YOLO pretrained (not fine-tuned); what CLAHE did
(and why we turned it off for the classifier); temperature scaling and ECE; the summed-probability rule and its threshold;
how the unknown check works (kNN cosine distance on embeddings, 95th percentile of validation); INT8 dynamic quantization and
why only MatMul/Gemm; one failure case each.

### 2.6 Submission checklist
- [ ] Live URL works from a phone on mobile data
- [ ] GitHub link: https://github.com/Priyanshu-senapati/Rescuelens (public, `main` up to date)
- [ ] Architecture diagram + design doc: `docs/design.md`
- [ ] `eval_P5_T50.md` (first line = USNs, comma-separated)
- [ ] Ablation + failure analysis: `eval_P5_T50.md` + `docs/design.md` section 9
- [ ] No secrets in the repo (`git grep -i -E "token|secret|password"` before submitting)

---

## 3. Training session at 22:00 (optional accuracy push)
Everything runs from the repo root on the laptop GPU (RTX 5070 Ti, 12 GB). Dataset: `C:\Users\priyanshu\cv\data`.
`training\run_local.ps1` starts a script as a background process; logs go to `training\<script>_<EXPORT_DIR>.log`.

| Experiment | Command (PowerShell) | Time | Expected |
|---|---|---|---|
| ConvNeXt-Small at 448 px | `$env:MODEL_NAME="convnext_small.fb_in22k_ft_in1k_384"; $env:EXPORT_DIR="models_s448"; $env:BATCH="16"; $env:TIME_BUDGET_MIN="80"` then edit `IMG = 448` in `train_classifier.py` and `.\training\run_local.ps1 -Script train_classifier.py -Quick 0` | ~60 min + 20 min eval | +0.3-0.8 top-1 (fine-grained likes resolution) |
| Longer schedule | `$env:EPOCHS="40"` on the best model | +50% time | small gain, val plateaued after ~15 epochs |
| Tiny + Small ensemble | average the two models' logits in `tools/build_report.py` style; app would run both (2x CPU time) | 30 min code | +0.3-0.7 top-1 |
| Fine-tune YOLO on Stanford boxes | see 4.1 | ~30 min | IoU 0.886 -> likely > 0.95, mAP up |

Rules: never evaluate on the test split while choosing settings (use val); keep `SEED=42`; after any new model, redo 2.1.

---

## 4. Dropped features to add back (priority order)

### Worth it for the scoring rubric
1. **Fine-tune the detector on Stanford Cars boxes.** Convert `anno_train.csv` boxes to YOLO format (one class "car"), `yolo train
   model=yolo11s.pt data=cars.yaml epochs=20 imgsz=640`, export ONNX, re-run `detector_and_baseline.py` metrics with the new
   weights. Directly raises IoU / mAP (25-point "Evaluation metrics" criterion).
2. **Reliability model ("is this answer wrong?").** Train a logistic regression on validation features (top-1 prob, top-1 minus
   top-2 margin, unknown score, brightness, contrast, sharpness, box size) to predict errors; use it for `reliability.score`
   instead of the raw probability and report its AUROC + a risk-coverage curve. Classical features then add measurable value.
3. **"Where the model looked" heat-map.** Occlusion-sensitivity map (slide a grey patch, record the drop in the top class)
   computed in ONNX, or Grad-CAM in PyTorch at training time for the failure-case figures. Strong viva/demo visual.
4. **Unknown-threshold tuning with a real ROC.** We now have 60 unknown photos: plot catch rate vs known-car false-flag rate,
   pick the threshold on that curve (currently the 95th percentile of validation distances). Collect a second, held-out set of
   unknown photos for an honest test number.
5. **Official emergency response guides for the other 8 hybrid/EV classes** in `data/hazards.json` (only the Sonata Hybrid
   has one). Manufacturer PDFs: Tesla, Nissan, GM (Silverado/Tahoe/Malibu/Yukon hybrids), BMW, Fisker.
6. **Calibration plot (reliability diagram)** on the Model report page, plus the threshold table for the high-voltage rule.
7. **Demo samples:** add one pre-tested "ambiguous" sample and one unknown-car sample (from the Wikimedia set; licences allow it,
   keep attribution) to `frontend/public/samples/` and `SAMPLES` in `frontend/src/components/Uploader.tsx`.

### Product features (beyond the rubric)
8. **Offline on-device mode.** Run the INT8 classifier + YOLO in the browser with onnxruntime-web (WebGPU/WASM) as an
   installable PWA: rural crash sites often have no signal. Biggest "real product" upgrade.
9. **Rescue-sheet overlay.** For models with an official sheet, draw the hazard zones (battery, HV cables, airbags) roughly
   aligned to the detected box; mark clearly as approximate.
10. **Live camera mode.** Classify video frames and lock the answer once confidence is stable over several frames.
11. **Multiple vehicles.** Classify every detected vehicle (currently only the main one; others are drawn faint).
12. **AR view** (from the original idea): overlay hazards on the live camera feed. Large effort; only after 8-10.

### Engineering hygiene
13. React error boundary, Playwright end-to-end test of upload -> result, `eslint`/`prettier`.
14. Rate limiting on `/api/analyze` (Modal handles scale, but abuse protection is still good practice).
15. CI: GitHub Action that runs `pytest` and `npm run build` on every push; a Modal deploy action with `MODAL_TOKEN_ID` /
    `MODAL_TOKEN_SECRET` repo secrets.
16. HEIC photos on Android/desktop: the browser cannot preview them; convert server-side and return a preview image.
17. Remove the leftover `ui-fallback` branch and the `Rescuelens-ui` worktree once nobody needs them
    (`git worktree remove ..\Rescuelens-ui`, `git push origin --delete ui-fallback`).

---

## 5. Known issues and limits (put these in the viva answers)
- Stanford Cars stops at 2012 and is US-market only; newer/Indian cars are flagged unknown (88% of the time), not identified.
- No crashed vehicles in the data; robustness is measured with simulated darkness, haze, blur, low resolution and occlusion.
- Heavy blur (44% top-1), a 40 px vehicle (55%) and 45% occlusion (62%) are the real weak spots.
- The classical-feature fusion weight came out 0 on validation: the hybrid gain comes from the detector crop (+3.3 points),
  calibration and the safety rules, not from mixing in the classical classifier.
- Decision-support prototype only; it never replaces official rescue documentation.

## 6. Paths and environments (Windows laptop)
| What | Where |
|---|---|
| Repo | `C:\Users\priyanshu\cv\Rescuelens` |
| Dataset (Kaggle `jutrera/stanford-car-dataset-by-classes-folder`) | `C:\Users\priyanshu\cv\data` |
| Training venv (Python 3.11, CUDA PyTorch 2.11, timm, ultralytics, modal) | `training\.venv` |
| Back-end venv (Python 3.14, tests) | `backend\.venv` |
| Run-1 models + metrics | `models\` (git-ignored) |
| Run-2 models + metrics | `models_small\` (git-ignored) |
| Unknown-car photos (Wikimedia Commons, evaluation only) | `private_photos\unknown\` (git-ignored) |
| Crop cache for training | `training\.cache\` (git-ignored) |
| Back-end tests | `cd backend; .venv\Scripts\python -m pytest -q tests` |
| Local app | `cd backend; $env:MODEL_DIR="..\models"; $env:FRONTEND_DIST="..\frontend\dist"; .venv\Scripts\python -m uvicorn app.main:app --port 8765` |
