# Front-end brief (owner: teammate building the UI)

Deadline: code freeze Sat 7 Nov, 11:30 IST. Judges open the live URL on **phones and laptops**,
upload their own photo, and expect an answer within ~10 s with clear loading and error states.
Professional, interactive, mobile-first. It must not look like a default Gradio/Streamlit app.

## Stack and layout

- `frontend/`: React + Vite + TypeScript + Tailwind (shadcn/ui components are fine).
- `npm run build` must output to `frontend/dist/`. FastAPI serves that folder at `/` (same origin, no CORS).
- API base path: `/api` (see `docs/api-contract.md`). Vite dev server proxies `/api` to `http://localhost:8000`.
- **Mock mode:** when `VITE_MOCK=1`, skip the network and return `docs/fixtures/*.json`
  (identified / ambiguous / unknown), so the UI can be built before the model exists.
  Pick the fixture by sample-image name, or cycle through them.
- No secrets, no external analytics, no CDN fonts that can fail offline (bundle the font).

## Screens (one page app, tabs or sections)

1. **Identify:** drag-and-drop + "Take photo" (`<input type="file" accept="image/*" capture="environment">`),
   plus a strip of sample photos (night, partly hidden, unknown car) in `frontend/public/samples/`.
2. **Result** (the money screen):
   - Photo with the detection box drawn on it (scale `detection.box` from original px to displayed size); `other_boxes` faint.
   - Big status card from `status`: green IDENTIFIED / amber AMBIGUOUS / red-grey UNKNOWN / neutral NO VEHICLE.
     Show `headline` large.
   - Safety banner from `safety` (`level` danger = red, caution = amber, info = neutral), its `actions` as a checklist,
     `guide` as a button when present. **High voltage banner must be impossible to miss.**
   - Hierarchy: make → family → vehicle, each with a probability bar; dim levels below `resolved_level`.
   - Top-5 list with bars, hybrid/EV chips from `powertrain`.
   - "Why this confidence?" card: `reliability.reasons` + `quality` scores as small meters.
   - When `status` is `unknown`: show the top guess crossed out / greyed with "Do not use for rescue decisions".
   - Always-visible `disclaimer` (small, not hidden).
3. **How it saw it:** horizontal stepper over `steps[]` (image + title + caption). Tap to enlarge.
   Before/after slider between `original` and `enhanced`.
4. **Stress test** (the interactive demo moment): sliders for Darkness, Haze/smoke, Blur, Occlusion (draws a grey block).
   Apply them **client-side on a canvas** to the uploaded photo, then re-POST the canvas as JPEG
   (debounce ~600 ms, cancel in-flight requests). Show how `status`, confidence and the safety banner change.
5. **Compare:** `comparison` for this photo (classical vs deep vs hybrid side by side) + metrics dashboard from
   `GET /api/metrics` (top-1/top-5 per approach, robustness chart, FP32 vs INT8 size + latency, risk-coverage curve).
   Charts: Recharts is fine.

## States (graded explicitly)

- Loading: skeleton + step text ("Detecting vehicle… Classifying… Running safety checks…").
- Errors from `{"error": {code, message}}`: friendly message + retry. Client-side checks before upload:
  not an image, > 10 MB.
- `no_vehicle`: friendly "No vehicle found — try a clearer photo of the car".
- Network/timeout (> 20 s): message + retry button.
- First request after the server wakes can be slow: show "Waking the server…" after 5 s.

## Look and feel

- Serious emergency-services tool: dark UI, high contrast, one accent colour, safety colours only for meaning
  (red = high voltage / danger, amber = ambiguous, green = identified).
- Big type for the verdict, numbers in tabular figures, generous spacing, works one-handed on a phone.
- Accessible: colour is never the only signal (icons + words), focus states, alt text, 44 px touch targets.
- Header: "RescueLens — uncertainty-aware vehicle identification for crash response".
