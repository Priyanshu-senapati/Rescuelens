import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Page, Route } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));

export function fixture(name: "identified" | "ambiguous" | "unknown"): Record<string, unknown> {
  return JSON.parse(readFileSync(resolve(here, `../../docs/fixtures/${name}.json`), "utf-8"));
}

export const samplePhoto = resolve(here, "../public/samples/clear.jpg");

/** Answers POST /api/analyze with the given bodies in turn; the last one repeats. Returns the requests it saw. */
export async function stubAnalyze(page: Page, ...bodies: unknown[]) {
  const seen: { contentType: string; size: number }[] = [];
  await page.route("**/api/analyze", async (route: Route) => {
    const req = route.request();
    seen.push({ contentType: req.headers()["content-type"] ?? "", size: req.postDataBuffer()?.length ?? 0 });
    const body = bodies[Math.min(seen.length - 1, bodies.length - 1)];
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  return seen;
}
