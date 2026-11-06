import { expect, test } from "@playwright/test";
import { fixture, stubAnalyze } from "./helpers";

const identified = fixture("identified");
const noVehicle = { ...identified, status: "no_vehicle", detection: null, prediction: null, hierarchy: [], top5: [], safety: null };

test.describe.configure({ timeout: 45_000 });

test("live scan locks after 3 agreeing frames and hands over the full answer", async ({ page }) => {
  const seen = await stubAnalyze(page, noVehicle, identified, identified, identified);
  await page.goto("/");
  await page.getByRole("button", { name: /Point the camera/ }).click();
  await expect(page.getByRole("dialog", { name: "Live camera scan" })).toBeVisible();

  await expect(page.getByRole("button", { name: "See the full answer" })).toBeVisible({ timeout: 20_000 });
  expect(seen).toHaveLength(4); // no further frames after the lock
  expect(seen[0].contentType).toContain("multipart/form-data");

  await page.getByRole("button", { name: "See the full answer" }).click();
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(page.getByRole("button", { name: "Check another vehicle" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Stress test" })).toBeVisible();
});

test("a high-voltage warning shows before the answer locks", async ({ page }) => {
  await stubAnalyze(page, fixture("ambiguous"), noVehicle);
  await page.goto("/");
  await page.getByRole("button", { name: /Point the camera/ }).click();
  await expect(page.getByRole("alert")).toContainText("Treat as HIGH VOLTAGE");
  await expect(page.getByRole("button", { name: "See the full answer" })).toBeHidden();
});

test("a blocked camera explains itself and offers a photo instead", async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("denied", "NotAllowedError"));
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Point the camera/ }).click();
  await expect(page.getByRole("heading", { name: "Camera access is off" })).toBeVisible();
  await page.getByRole("button", { name: "Use a photo instead" }).click();
  await expect(page.getByRole("button", { name: "Choose a photo" })).toBeVisible();
});

test("losing the server stops the scan with a way forward", async ({ page }) => {
  await page.route("**/api/analyze", (r) => r.abort());
  await page.goto("/");
  await page.getByRole("button", { name: /Point the camera/ }).click();
  await expect(page.getByRole("heading", { name: "Lost contact with the server" })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("closing the camera releases it", async ({ page }) => {
  await stubAnalyze(page, noVehicle);
  await page.addInitScript(() => {
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (c) => {
      const s = await real(c);
      (window as unknown as { __stream: MediaStream }).__stream = s;
      return s;
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Point the camera/ }).click();
  await expect(page.getByText(/No vehicle in view|Point the camera at the vehicle/)).toBeVisible();
  await page.getByRole("button", { name: "Close the camera" }).click();
  await expect(page.getByRole("dialog")).toBeHidden();
  const states = await page.evaluate(() =>
    (window as unknown as { __stream: MediaStream }).__stream.getTracks().map((t) => t.readyState),
  );
  expect(states).toEqual(["ended"]);
});
