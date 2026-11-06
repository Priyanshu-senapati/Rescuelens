// Offline downloads live in Cache Storage, so a page reload or lost signal doesn't lose them.
const MODEL_CACHE = "rescuelens-models-v1";
const FILES = ["config.json", "classifier_int8.onnx", "train_emb.npy", "yolo11s.onnx"];
const SAMPLES = ["clear", "sonata-hybrid", "night", "hidden", "electric", "ambiguous-audi-s5", "unknown-tata-nexon"];
const SHELL_CACHE = "rescuelens-shell-v1";
const url = (name: string) => `/offline/${name}`;

/** True when this server offers the model files, so the option is worth showing. */
export async function offlineAvailable(): Promise<boolean> {
  if (!("caches" in window) || !("serviceWorker" in navigator)) return false;
  try {
    const res = await fetch(url("config.json"));
    return res.ok && (res.headers.get("content-type") ?? "").includes("json");
  } catch {
    return false;
  }
}

export async function offlineReady(): Promise<boolean> {
  try {
    const cache = await caches.open(MODEL_CACHE);
    return (await Promise.all(FILES.map((f) => cache.match(url(f))))).every(Boolean);
  } catch {
    return false;
  }
}

export async function modelFile(name: string): Promise<ArrayBuffer> {
  const hit = await (await caches.open(MODEL_CACHE)).match(url(name));
  if (!hit) throw new Error(`${name} has not been downloaded`);
  return hit.arrayBuffer();
}

export async function totalDownloadBytes(): Promise<number | null> {
  try {
    const sizes = await Promise.all(FILES.map((f) => fetch(url(f), { method: "HEAD" }).then((r) => Number(r.headers.get("content-length")))));
    return sizes.every((s) => s > 0) ? sizes.reduce((a, b) => a + b, 0) : null;
  } catch {
    return null;
  }
}

/** Downloads the models and everything the page needs, then proves the engine loads. Leaves nothing half-done on failure. */
export async function prepareOffline(onProgress: (fraction: number | null) => void): Promise<void> {
  await navigator.serviceWorker.ready;
  await navigator.storage?.persist?.().catch(() => false);
  const total = await totalDownloadBytes();
  const cache = await caches.open(MODEL_CACHE);
  let done = 0;
  try {
    for (const f of FILES) {
      const res = await fetch(url(f));
      if (!res.ok || !res.body) throw new Error(`Couldn't download ${f}`);
      const chunks: Uint8Array[] = [];
      const reader = res.body.getReader();
      for (;;) {
        const { done: end, value } = await reader.read();
        if (end) break;
        chunks.push(value);
        done += value.length;
        onProgress(total ? Math.min(1, done / total) : null);
      }
      await cache.put(url(f), new Response(new Blob(chunks as BlobPart[]), { headers: { "content-type": res.headers.get("content-type") ?? "application/octet-stream" } }));
    }
    onProgress(null);
    const { loadEngine } = await import("./engine");
    await loadEngine();
    const shell = await caches.open(SHELL_CACHE);
    const loaded = performance.getEntriesByType("resource").map((e) => e.name).filter((n) => n.startsWith(location.origin) && !n.includes("/api/") && !n.includes("/offline/"));
    const needed = ["/", ...SAMPLES.map((s) => `/samples/${s}.jpg`), ...loaded].map((u) => new URL(u, location.origin).href);
    await shell.addAll([...new Set(needed)]);
  } catch (err) {
    await caches.delete(MODEL_CACHE);
    throw err;
  }
}
