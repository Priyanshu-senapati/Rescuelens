# RescueLens API contract

Front end and back end both follow this file. Change it only by agreement.
Example responses: `docs/fixtures/identified.json`, `ambiguous.json`, `unknown.json`.

Same origin: the React build is served by FastAPI at `/`, the API lives under `/api`.
In development, Vite proxies `/api` to `http://localhost:8000`.

## `POST /api/analyze`

`multipart/form-data`, one field: `file` (JPEG, PNG, WebP or HEIC, max 10 MB).

### 200 response

| Field | Type | Meaning |
|---|---|---|
| `request_id` | string | for logs |
| `status` | `"identified" \| "ambiguous" \| "unknown" \| "no_vehicle"` | the 3 result states + nothing found |
| `headline` | string | one-line verdict to show big |
| `image` | `{width, height}` | original size in px (after EXIF rotation) |
| `detection` | `{box:[x1,y1,x2,y2], score}` or `null` | box in original pixel coords; `null` when `no_vehicle` |
| `other_boxes` | `[[x1,y1,x2,y2], ...]` | other cars found (drawn faint, not classified) |
| `prediction` | `{label, make, model, year, body, confidence}` or `null` | top-1 class. Present even when `unknown` (UI must mark it untrusted) |
| `hierarchy` | `[{level, name, prob}]` | `level` in `make`, `family`, `vehicle`, broadest first |
| `resolved_level` | `"vehicle" \| "family" \| "make" \| "none"` | deepest level we are confident about |
| `top5` | `[{label, make, model, year, prob, powertrain}]` | `powertrain` in `hybrid`, `ev`, `combustion` |
| `safety` | object | see below |
| `unknown` | `{score, threshold, is_unknown}` | distance from known cars; above threshold = not in our 196 |
| `quality` | object | see below |
| `reliability` | `{score, reasons:[string]}` | estimated chance the top-1 is right (0-1) + plain-English reasons ("Low light", "Look-alike pair") |
| `steps` | `[{key, title, caption, image}]` | classical pipeline images as data URLs, in order: `original`, `enhanced`, `edges`, `contours`, `roi`, `wheels`, `colour` |
| `features` | `{dominant_colours:[{hex, name, share}], aspect_ratio, wheel_count, wheelbase_ratio}` | classical measurements |
| `comparison` | `{classical, deep, hybrid}` each `{label, prob}`; `hybrid` also has `status` | same photo through the 3 approaches |
| `timing_ms` | `{detect, classify, classical, total}` | latency |
| `disclaimer` | string | always shown |

`safety`:

| Field | Type | Meaning |
|---|---|---|
| `hv_probability` | number 0-1 | summed calibrated probability of all hybrid/EV classes |
| `hv_warning` | bool | true = show the red HIGH VOLTAGE banner |
| `level` | `"danger" \| "caution" \| "info"` | banner colour: red / amber / neutral |
| `message` | string | banner text |
| `actions` | `[string]` | short checklist |
| `guide` | `{title, url}` or `null` | official emergency response guide when we have one |
| `location_vehicle` | string or `null` | which vehicle `battery` and `hv_cables` describe: the most likely hybrid/EV. The UI must name it, because the answer may be uncertain |
| `battery` | string or `null` | where the high-voltage battery is, from that vehicle's manufacturer guide (`docs/hazard-sources.md`) |
| `hv_cables` | string or `null` | how to recognise the high-voltage cables and where they run, from the same guide |

`quality` (all scores 0-100, higher = better):
`brightness`, `contrast`, `sharpness`, `resolution_px` (shorter side of the car crop), `cut_off` (bool: car touches image edge), `issues` ([string]).

### Errors

`4xx/5xx` body: `{"error": {"code": string, "message": string}}`

| code | HTTP | when |
|---|---|---|
| `not_image` | 400 | file isn't a readable image |
| `too_large` | 413 | over 10 MB |
| `missing_file` | 400 | no `file` field |
| `internal` | 500 | anything else (message is safe to show) |

`no_vehicle` is NOT an error: it's a 200 with `status:"no_vehicle"`, `detection:null`, `prediction:null`.

## `GET /api/metrics`

The evaluation numbers for the dashboard (same numbers as `eval_P5_<TeamID>.md`):

```json
{
  "classification": {"top1": 0.0, "top5": 0.0, "top1_no_crop": 0.0, "top1_tta": 0.0},
  "detection": {"mean_iou": 0.0, "map50": 0.0, "map50_95": 0.0},
  "efficiency": {"fp32_mb": 0.0, "int8_mb": 0.0, "fp32_ms": 0.0, "int8_ms": 0.0, "int8_top1": 0.0},
  "robustness": [{"condition": "dark", "severity": 1, "top1": 0.0}],
  "calibration": {"ece_before": 0.0, "ece_after": 0.0, "temperature": 1.0},
  "safety": {"hv_miss_rate": 0.0, "hv_false_alarm_rate": 0.0, "unknown_catch_rate": 0.0},
  "approaches": [{"name": "classical", "top1": 0.0, "top5": 0.0}, {"name": "deep", "top1": 0.0, "top5": 0.0}, {"name": "hybrid", "top1": 0.0, "top5": 0.0}],
  "ablation": [{"variant": "full system", "top1": 0.0}],
  "risk_coverage": [{"coverage": 1.0, "accuracy": 0.0}]
}
```

## `GET /api/health`

`{"status": "ok", "model_loaded": true, "version": "..."}`
