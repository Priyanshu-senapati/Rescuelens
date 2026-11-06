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

API: [docs/api-contract.md](docs/api-contract.md)
