// Keeps the app shell and downloaded models usable without signal. Online, the network always wins, so deployments show up at once.
const SHELL = "rescuelens-shell-v1";
const MODELS = "rescuelens-models-v1";

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const key of await caches.keys()) if (key !== SHELL && key !== MODELS) await caches.delete(key);
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin || url.pathname.startsWith("/api/")) return;
  if (url.pathname.startsWith("/offline/")) return; // models are read directly from Cache Storage by the page

  event.respondWith(
    (async () => {
      const cache = await caches.open(SHELL);
      try {
        const res = await fetch(req);
        // only top up what offline mode has already stored, so users who never opt in don't fill their storage
        if (res.ok && (await cache.match(req.mode === "navigate" ? "/" : req))) await cache.put(req.mode === "navigate" ? "/" : req, res.clone());
        return res;
      } catch (err) {
        const hit = await cache.match(req.mode === "navigate" ? "/" : req);
        if (hit) return hit;
        throw err;
      }
    })(),
  );
});
