import { expect, test } from "@playwright/test";
import { samplePhoto } from "./helpers";

const SITE = "http://localhost:5197/";

test("after opting in, the app reloads and answers with no connection at all", async ({ page, context }) => {
  test.setTimeout(90_000);
  await page.goto(SITE);
  await page.evaluate(() => navigator.serviceWorker.ready);
  await page.reload(); // so the service worker controls this page and sees every asset

  await page.getByRole("button", { name: "Make available offline" }).click();
  await expect(page.getByText(/Ready without signal/)).toBeVisible({ timeout: 60_000 });

  await context.setOffline(true);
  await page.reload();
  await expect(page.getByRole("button", { name: "Choose a photo" })).toBeVisible();
  await expect(page.getByText(/Ready without signal/)).toBeVisible();

  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);
  await expect(page.getByText(/Answered on this device/)).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("Tesla Model S Sedan 2012", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("High voltage", { exact: true })).toBeVisible();
  await expect(page.getByText(/Below the floor, under a floor pan/)).toBeVisible();

  // the demo samples are stored too
  await page.getByRole("button", { name: "Check another vehicle" }).click();
  await page.getByRole("button", { name: /Electric car/ }).click();
  await expect(page.getByText(/Answered on this device/)).toBeVisible({ timeout: 45_000 });
  await context.setOffline(false);
});

test("without opting in, losing the connection gives the normal error and nothing downloads", async ({ page, context }) => {
  const asked: string[] = [];
  page.on("request", (r) => r.url().includes("/offline/") && asked.push(r.url().split("/offline/")[1]));
  await page.goto(SITE);
  await expect(page.getByRole("button", { name: "Make available offline" })).toBeVisible();
  expect(asked.filter((f) => f.endsWith(".onnx") || f.endsWith(".npy")).length).toBeLessThanOrEqual(3); // size checks only
  await context.setOffline(true);
  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);
  await expect(page.getByText("Couldn't analyse this photo")).toBeVisible();
  await context.setOffline(false);
});
