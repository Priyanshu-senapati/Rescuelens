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

| Folder | What |
|---|---|
| `training/` | Kaggle notebooks: classifier training, detector eval, classical baseline |
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
