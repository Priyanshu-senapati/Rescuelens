// On-device inference: the same detector -> crop -> classifier -> safety decision as backend/app, running in the browser.
// Differences from the server, all stated in docs/design.md: no CLAHE retry, no classical pipeline or comparison, canvas resize.
import * as ort from "onnxruntime-web/wasm";
// Relative paths on purpose: the package's "exports" map doesn't list these files, and they must be same-origin to work offline.
import wasmUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url";
import mjsUrl from "../../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url";
import hazardFile from "../../../data/hazards.json";
import type { Result } from "../types";
import { vehicleRows, vehicleScore } from "./detection";
import { DISCLAIMER, decide, parseLabel, type CarClass, type Hazard, type Thresholds } from "./safety";
import { modelFile } from "./store";

const DET_SIZE = 640;
const DET_CONF = 0.25;
const NMS_IOU = 0.45;
const MAX_SIDE = 1600;

type Box = [number, number, number, number];
type Detection = { box: Box; score: number };

interface Engine {
  detector: ort.InferenceSession;
  classifier: ort.InferenceSession;
  bank: Float32Array;
  dim: number;
  k: number;
  size: number;
  temperature: number;
  mean: number[];
  std: number[];
  classes: CarClass[];
  thresholds: Thresholds;
}

/** Reads a little-endian float .npy matrix. */
export function parseNpy(buf: ArrayBuffer): { data: Float32Array; rows: number; cols: number } {
  const view = new DataView(buf);
  const major = view.getUint8(6);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 + headerLen : 12 + headerLen;
  const header = new TextDecoder().decode(new Uint8Array(buf, major === 1 ? 10 : 12, headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const shape = /'shape':\s*\((\d+),\s*(\d+)/.exec(header);
  if (!shape) throw new Error("Unexpected embedding file shape");
  const rows = +shape[1];
  const cols = +shape[2];
  const body = buf.slice(start);
  if (descr === "<f4") return { data: new Float32Array(body), rows, cols };
  if (descr === "<f8") return { data: Float32Array.from(new Float64Array(body)), rows, cols };
  if (descr === "<f2") {
    const h = new Uint16Array(body);
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const s = h[i] & 0x8000 ? -1 : 1;
      const e = (h[i] >> 10) & 0x1f;
      const f = h[i] & 0x3ff;
      out[i] = s * (e === 0 ? f * 2 ** -24 : e === 31 ? Infinity : (1 + f / 1024) * 2 ** (e - 15));
    }
    return { data: out, rows, cols };
  }
  throw new Error(`Unsupported embedding type ${descr}`);
}

let loading: Promise<Engine> | null = null;

export function loadEngine(): Promise<Engine> {
  loading ??= (async () => {
    ort.env.wasm.wasmPaths = { wasm: new URL(wasmUrl, location.href).href, mjs: new URL(mjsUrl, location.href).href };
    ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which the host doesn't set
    const cfg = JSON.parse(new TextDecoder().decode(await modelFile("config.json")));
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: ["wasm"] };
    const detector = await ort.InferenceSession.create(new Uint8Array(await modelFile("yolo11s.onnx")), opts);
    const classifier = await ort.InferenceSession.create(new Uint8Array(await modelFile("classifier_int8.onnx")), opts);
    const emb = parseNpy(await modelFile("train_emb.npy"));
    const hazards = hazardFile.high_voltage as unknown as Record<string, Hazard>;
    return {
      detector, classifier, bank: emb.data, dim: emb.cols, k: cfg.knn_k, size: cfg.img, temperature: cfg.temperature, mean: cfg.mean, std: cfg.std,
      classes: (cfg.class_names as string[]).map((n) => parseLabel(n, hazards)),
      thresholds: { identify: 0.6, family: 0.6, make: 0.5, hv: cfg.hv_threshold ?? 0.05, unknown: cfg.unknown_threshold },
    };
  })().catch((err) => {
    loading = null; // a failed load can be retried
    throw err;
  });
  return loading;
}

function iou(a: Box, b: Box): number {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter || 1);
}

async function detect(eng: Engine, photo: HTMLCanvasElement): Promise<Detection[]> {
  const { width: w, height: h } = photo;
  const s = DET_SIZE / Math.max(w, h);
  const nw = Math.round(w * s);
  const nh = Math.round(h * s);
  const padX = Math.floor((DET_SIZE - nw) / 2);
  const padY = Math.floor((DET_SIZE - nh) / 2);
  const c = document.createElement("canvas");
  c.width = c.height = DET_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, DET_SIZE, DET_SIZE);
  ctx.drawImage(photo, padX, padY, nw, nh);
  const px = ctx.getImageData(0, 0, DET_SIZE, DET_SIZE).data;
  const n = DET_SIZE * DET_SIZE;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[i] = px[i * 4] / 255;
    x[n + i] = px[i * 4 + 1] / 255;
    x[2 * n + i] = px[i * 4 + 2] / 255;
  }
  const res = await eng.detector.run({ [eng.detector.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, DET_SIZE, DET_SIZE]) });
  const out = res[eng.detector.outputNames[0]];
  const data = out.data as Float32Array; // 1 x (4 + classes) x N: cx, cy, w, h, class scores
  const rows = vehicleRows(out.dims[1]);
  const count = out.dims[2];
  const found: Detection[] = [];
  for (let i = 0; i < count; i++) {
    const score = vehicleScore(data, count, i, rows);
    if (score <= DET_CONF) continue;
    const cx = (data[i] - padX) / s;
    const cy = (data[count + i] - padY) / s;
    const bw = data[2 * count + i] / s;
    const bh = data[3 * count + i] / s;
    found.push({ box: [Math.max(0, cx - bw / 2), Math.max(0, cy - bh / 2), Math.min(w, cx + bw / 2), Math.min(h, cy + bh / 2)], score });
  }
  found.sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const d of found) {
    if (d.box[2] - d.box[0] < 16 || d.box[3] - d.box[1] < 16) continue;
    if (kept.every((k) => iou(k.box, d.box) <= NMS_IOU)) kept.push(d);
  }
  if (!kept.length) return [];
  // main vehicle = biggest of the confident boxes, as on the server
  const area = (d: Detection) => (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]);
  const pool = kept.filter((d) => d.score >= 0.4);
  const main = (pool.length ? pool : kept).reduce((a, b) => (area(b) > area(a) ? b : a));
  return [main, ...kept.filter((d) => d !== main)];
}

async function classify(eng: Engine, photo: HTMLCanvasElement, box: Box): Promise<{ probs: Float32Array; unknownScore: number }> {
  const mw = 0.08 * (box[2] - box[0]);
  const mh = 0.08 * (box[3] - box[1]);
  const xa = Math.max(0, Math.floor(box[0] - mw));
  const ya = Math.max(0, Math.floor(box[1] - mh));
  const xb = Math.min(photo.width, Math.floor(box[2] + mw));
  const yb = Math.min(photo.height, Math.floor(box[3] + mh));
  const size = eng.size;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(photo, xa, ya, xb - xa, yb - ya, 0, 0, size, size);
  const px = ctx.getImageData(0, 0, size, size).data;
  const n = size * size;
  const x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++)
    for (let ch = 0; ch < 3; ch++) x[ch * n + i] = (px[i * 4 + ch] / 255 - eng.mean[ch]) / eng.std[ch];

  const res = await eng.classifier.run({ input: new ort.Tensor("float32", x, [1, 3, size, size]) });
  const logits = res[eng.classifier.outputNames[0]].data as Float32Array;
  const emb = res[eng.classifier.outputNames[1]].data as Float32Array;

  let peak = -Infinity;
  for (const v of logits) peak = Math.max(peak, v);
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += probs[i] = Math.exp((logits[i] - peak) / eng.temperature);
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;

  // unknown-car score: 1 - mean cosine similarity to the k nearest training embeddings
  const norm = Math.max(1e-8, Math.hypot(...emb));
  const rows = eng.bank.length / eng.dim;
  const best: number[] = [];
  for (let r = 0; r < rows; r++) {
    let sim = 0;
    const off = r * eng.dim;
    for (let d = 0; d < eng.dim; d++) sim += eng.bank[off + d] * emb[d];
    sim /= norm;
    if (best.length < eng.k) {
      best.push(sim);
      best.sort((a, b) => a - b);
    } else if (sim > best[0]) {
      best[0] = sim;
      best.sort((a, b) => a - b);
    }
  }
  return { probs, unknownScore: 1 - best.reduce((a, b) => a + b, 0) / best.length };
}

async function toCanvas(photo: Blob): Promise<HTMLCanvasElement> {
  const bmp = await createImageBitmap(photo, { imageOrientation: "from-image" });
  const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  c.getContext("2d")!.drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close();
  return c;
}

export async function analyzeLocally(photo: Blob): Promise<Result> {
  const t0 = performance.now();
  const eng = await loadEngine();
  const canvas = await toCanvas(photo);
  const dets = await detect(eng, canvas);
  const tDet = performance.now();
  const base = {
    request_id: `local-${Date.now().toString(36)}`, image: { width: canvas.width, height: canvas.height }, quality: null, steps: [],
    features: null, comparison: null, offline: true,
  };
  if (!dets.length)
    return {
      ...base, status: "no_vehicle", headline: "No vehicle found - try a clearer photo of the car", detection: null, other_boxes: [], prediction: null,
      hierarchy: [], resolved_level: "none", top5: [], safety: null, unknown: null, reliability: null, disclaimer: DISCLAIMER,
      timing_ms: { detect: Math.round(tDet - t0), classify: 0, classical: 0, total: Math.round(performance.now() - t0) },
    };
  const [main, ...others] = dets;
  const { probs, unknownScore } = await classify(eng, canvas, main.box);
  const decision = decide(probs, unknownScore, eng.classes, eng.thresholds);
  const t1 = performance.now();
  return {
    ...base, ...decision,
    detection: { box: main.box.map(Math.round) as Box, score: main.score },
    other_boxes: others.slice(0, 5).map((d) => d.box.map(Math.round)),
    timing_ms: { detect: Math.round(tDet - t0), classify: Math.round(t1 - tDet), classical: 0, total: Math.round(t1 - t0) },
  };
}
