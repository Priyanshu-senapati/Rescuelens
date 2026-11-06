import { offlineReady } from "./offline/store";
import type { Metrics, Result } from "./types";

const MOCK = import.meta.env.VITE_MOCK === "1";
const TIMEOUT_MS = 25_000;

export class ApiError extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

async function answerOnDevice(photo: Blob): Promise<Result> {
  try {
    const { analyzeLocally } = await import("./offline/engine"); // only loaded when needed: onnxruntime is large
    return await analyzeLocally(photo);
  } catch (err) {
    console.error("On-device analysis failed", err);
    throw new ApiError("offline", "This device couldn't analyse the photo offline. Identify the vehicle by VIN, plate and badges.");
  }
}

async function mockResult(name: string): Promise<Result> {
  // mock mode (VITE_MOCK=1): fixture chosen by file name, so the UI works without the back end
  const n = name.toLowerCase();
  const mod = n.includes("unknown")
    ? await import("../../docs/fixtures/unknown.json")
    : n.includes("sonata") || n.includes("ambiguous")
      ? await import("../../docs/fixtures/ambiguous.json")
      : await import("../../docs/fixtures/identified.json");
  await new Promise((r) => setTimeout(r, 700));
  return { ...(mod.default as unknown as Result), mock: true };
}

export async function analyze(photo: Blob, name: string, signal?: AbortSignal): Promise<Result> {
  if (MOCK) return mockResult(name);
  if (!navigator.onLine && (await offlineReady())) return answerOnDevice(photo);
  const body = new FormData();
  body.append("file", photo, name);
  // one controller + timer instead of AbortSignal.any/timeout, which older iPhones (iOS < 17.4) lack
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, TIMEOUT_MS);
  const forward = () => ctrl.abort();
  signal?.addEventListener("abort", forward);
  const stopped = () => {
    if (signal?.aborted) return new DOMException("Aborted", "AbortError");
    if (timedOut) return new ApiError("timeout", "The server took too long to answer. Try again in a moment.");
    return null;
  };
  try {
    let res: Response;
    try {
      res = await fetch("/api/analyze", { method: "POST", body, signal: ctrl.signal });
    } catch {
      const stop = stopped();
      if (stop instanceof DOMException) throw stop;
      if (await offlineReady()) return answerOnDevice(photo); // signal dropped or too slow: don't leave the responder waiting
      throw stop ?? new ApiError("network", "Couldn't reach the server. Check your connection and try again.");
    }
    const data: unknown = await res.json().catch(() => null);
    const stop = stopped();
    if (stop) throw stop;
    if (!res.ok) {
      const e = (data as { error?: { code?: string; message?: string } } | null)?.error;
      throw new ApiError(e?.code ?? "internal", e?.message ?? `The server returned an error (${res.status}).`);
    }
    if (data === null) throw new ApiError("internal", "The server sent an answer that couldn't be read. Try again.");
    return data as Result;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forward);
  }
}

export async function fetchMetrics(): Promise<Metrics | null> {
  try {
    const res = await fetch("/api/metrics");
    return res.ok ? ((await res.json()) as Metrics) : null;
  } catch {
    return null;
  }
}
