import type { Metrics, Result } from "./types";

const MOCK = import.meta.env.VITE_MOCK === "1";
const TIMEOUT_MS = 25_000;

export class ApiError extends Error {
  constructor(public code: string, message: string) {
    super(message);
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
  const body = new FormData();
  body.append("file", photo, name);
  const timeout = AbortSignal.timeout(TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch("/api/analyze", {
      method: "POST",
      body,
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError" && signal?.aborted) throw err;
    if (err instanceof DOMException && (err.name === "TimeoutError" || timeout.aborted))
      throw new ApiError("timeout", "The server took too long to answer. Try again in a moment.");
    throw new ApiError("network", "Couldn't reach the server. Check your connection and try again.");
  }
  const data: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const e = (data as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiError(e?.code ?? "internal", e?.message ?? `The server returned an error (${res.status}).`);
  }
  return data as Result;
}

export async function fetchMetrics(): Promise<Metrics | null> {
  const res = await fetch("/api/metrics").catch(() => null);
  if (!res || !res.ok) return null;
  return (await res.json()) as Metrics;
}
