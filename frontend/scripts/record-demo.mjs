// Records the backup demo video from a running RescueLens.
//   npm run demo:record -- https://your-live-url
// Saves a captioned .webm in frontend/demo-output/. The camera scene uses a sample photo as a simulated feed, and says so on screen.
import { copyFileSync, mkdirSync, readdirSync, renameSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const url = process.argv[2];
if (!url) {
  console.error("Usage: npm run demo:record -- https://your-live-url");
  process.exit(1);
}

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(here, "../demo-output");
mkdirSync(out, { recursive: true });
const feed = join(out, "feed.mjpeg");
// Chromium's fake camera only accepts some image sizes: this one works, clear.jpg comes through as a 2x2 video.
copyFileSync(resolve(here, "../public/samples/sonata-hybrid.jpg"), feed);

const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${feed.replaceAll("\\", "/")}`],
});
const context = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  permissions: ["camera"],
  recordVideo: { dir: out, size: { width: 1280, height: 720 } },
});
const page = await context.newPage();

// Captions stay on screen so the video works without narration, for example in a noisy hall.
await page.addInitScript(() => {
  window.__caption = (text) => {
    let el = document.getElementById("__caption");
    if (!el) {
      el = document.createElement("div");
      el.id = "__caption";
      el.style.cssText = "position:fixed;left:50%;bottom:16px;transform:translateX(-50%);z-index:99;max-width:88%;padding:8px 16px;border-radius:6px;background:#000c;color:#fff;font:600 19px Arial,sans-serif;text-align:center;pointer-events:none";
      document.documentElement.append(el);
    }
    el.textContent = text;
  };
});

const caption = (text) => page.evaluate((t) => window.__caption(t), text);
const wait = (s) => page.waitForTimeout(s * 1000);
const answer = () => page.getByRole("button", { name: "Check another vehicle" }).waitFor({ timeout: 60_000 });
const resultIn = () => page.locator("section[aria-busy='false']").first().waitFor({ timeout: 60_000 });
const again = async () => {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.getByRole("button", { name: "Check another vehicle" }).click();
};
const sample = async (name) => {
  await page.getByRole("button", { name }).click();
  await answer();
  await resultIn();
};

await page.goto(url);
await caption("Crash scenes hide the plate, VIN and QR code. Hybrids hide high voltage.");
await wait(8);
await caption("Same shape, different danger: where the official guides put the battery");
await wait(8);

await caption("A clear photo: RescueLens finds the vehicle and names it");
await sample(/Clear daylight/);
await wait(10);
await page.evaluate(() => document.querySelector("h2")?.scrollIntoView({ behavior: "smooth" }));
await caption("Every classical computer-vision step is visible");
await wait(8);
await again();

await caption("A hybrid that looks like a petrol car");
await sample(/Hybrid that looks like/);
await caption("High-voltage warning, where the battery is, and the official guide");
await wait(20);
await again();

await caption("Live camera (simulated feed here): the answer locks once 3 frames agree");
await page.getByRole("button", { name: /Point the camera/ }).click();
await page.getByRole("button", { name: "See the full answer" }).waitFor({ timeout: 60_000 });
await wait(9);
await page.getByRole("button", { name: "See the full answer" }).click();
await answer();
await again();
await sample(/Clear daylight/);

await page.getByRole("heading", { name: "Stress test" }).scrollIntoViewIfNeeded();
await caption("Automatic sweep: the real model at 8 rising levels of occlusion");
await page.getByRole("button", { name: "Sweep occlusion" }).click();
const curve = page.getByRole("img", { name: /Usable confidence/ });
await curve.waitFor({ timeout: 30_000 });
await curve.evaluate((el) => el.scrollIntoView({ behavior: "smooth", block: "center" })); // keep the curve in shot as it grows
const verdict = page.getByRole("status").filter({ hasText: /Held|Still/ });
await verdict.waitFor({ timeout: 90_000 });
await verdict.evaluate((el) => el.scrollIntoView({ behavior: "smooth", block: "center" }));
await caption("It says exactly where the answer breaks");
await wait(9);
await again();

await caption("A car it has never seen: it says so instead of guessing");
await sample(/Indian car outside/);
await wait(16);
await page.evaluate(() => window.scrollTo(0, 0));

await page.getByRole("button", { name: "Model report" }).click();
await caption("Measured on 8,041 test photos the models never saw");
await wait(10);
await page.getByRole("heading", { name: /high-voltage warning fires/ }).scrollIntoViewIfNeeded();
await caption("1.15% of hybrids missed, against 4.32% if we trusted the top answer");
await wait(10);
await page.getByRole("heading", { name: /cars it never saw/ }).scrollIntoViewIfNeeded();
await caption("58 of 60 unseen cars correctly called unknown");
await wait(10);
await caption("Most classifiers are built to be right. RescueLens is built to know when being wrong is dangerous.");
await wait(7);

await context.close();
await browser.close();
const video = readdirSync(out).find((f) => f.endsWith(".webm") && f !== "rescuelens-demo.webm");
if (video) renameSync(join(out, video), join(out, "rescuelens-demo.webm"));
console.log(`Saved ${join(out, "rescuelens-demo.webm")}. Watch it all the way through before relying on it.`);
