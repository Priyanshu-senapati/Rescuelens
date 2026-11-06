import { expect, test } from "@playwright/test";
import { fixture, samplePhoto, stubAnalyze } from "./helpers";

test("uploading a photo sends it to the API and shows the verdict", async ({ page }) => {
  const seen = await stubAnalyze(page, fixture("identified"));
  await page.goto("/");
  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);

  await expect(page.getByText("Identified", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Check another vehicle" })).toBeVisible();
  expect(seen).toHaveLength(1);
  expect(seen[0].contentType).toContain("multipart/form-data");
  expect(seen[0].size).toBeGreaterThan(1000);
});

test("picking a demo sample gives a result", async ({ page }) => {
  await stubAnalyze(page, fixture("ambiguous"));
  await page.goto("/");
  await page.getByRole("button", { name: /Hybrid that looks like a petrol car/ }).click();
  await expect(page.getByText("Not certain", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("High voltage", { exact: true })).toBeVisible();
});

test("a server error is explained and can be retried", async ({ page }) => {
  let calls = 0;
  await page.route("**/api/analyze", (route) => {
    calls++;
    if (calls === 1)
      return route.fulfill({
        status: 500,
        contentType: "application/json",
        body: JSON.stringify({ error: { code: "internal", message: "Something went wrong analysing that photo." } }),
      });
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(fixture("identified")) });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);
  await expect(page.getByText("Couldn't analyse this photo")).toBeVisible();
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByText("Identified", { exact: true }).first()).toBeVisible();
});

test("a malformed answer shows the crash screen, and Start again recovers", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await stubAnalyze(page, { status: "not-a-real-status" });
  await page.goto("/");
  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);

  await expect(page.getByRole("heading", { name: "RescueLens hit a problem" })).toBeVisible();
  await expect(page.getByText(/possibly high-voltage/)).toBeVisible();
  await page.getByRole("button", { name: "Start again" }).click();
  await expect(page.getByRole("button", { name: "Choose a photo" })).toBeVisible();
  expect(errors.join("\n")).toContain("RescueLens UI crashed"); // proves the boundary, not a lucky render, handled it
});
