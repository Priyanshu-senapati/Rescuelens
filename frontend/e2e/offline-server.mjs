// Serves the production build plus stand-in models, standing in for the FastAPI host in the offline browser test.
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "../dist");
const stubs = resolve(here, "stub-models");
const port = Number(process.argv[2] ?? 5197);
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".jpg": "image/jpeg", ".png": "image/png", ".woff2": "font/woff2", ".webmanifest": "application/manifest+json" };

createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (path.replaceAll("\\", "/").startsWith("/api/")) {
    res.writeHead(503, { "content-type": "application/json" }).end('{"error":{"code":"internal","message":"No server in this test."}}');
    return;
  }
  const off = path.replaceAll("\\", "/").match(/^\/offline\/([\w.]+)$/);
  let file = off ? join(stubs, off[1]) : join(dist, path);
  if (!off && (!existsSync(file) || statSync(file).isDirectory())) file = join(dist, "index.html");
  if (!existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": TYPES[extname(file)] ?? "application/octet-stream", "content-length": statSync(file).size });
  if (req.method === "HEAD") return res.end();
  createReadStream(file).pipe(res);
}).listen(port);
