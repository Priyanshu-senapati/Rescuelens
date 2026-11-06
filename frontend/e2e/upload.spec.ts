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

test("an unsure answer suggests another side; the extra photo is sent with the first and can be removed", async ({ page }) => {
  const files: number[] = [];
  const combined = {
    ...fixture("identified"),
    views: [
      { index: 0, vehicle_found: true, top: "Toyota Camry Sedan 2012", prob: 0.21, unknown_score: 0.9, quality_issues: [] },
      { index: 1, vehicle_found: true, top: "BMW M3 Coupe 2012", prob: 0.94, unknown_score: 0.1, quality_issues: [] },
    ],
  };
  const answers = [fixture("unknown"), combined, fixture("unknown")];
  await page.route("**/api/analyze", (route) => {
    files.push((route.request().postDataBuffer()?.toString("latin1").match(/name="file"/g) ?? []).length);
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(answers[files.length - 1]) });
  });
  await page.goto("/");
  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);
  await expect(page.getByText("Front crushed or badge hidden?")).toBeVisible();

  await page.locator('input[type="file"]').last().setInputFiles(samplePhoto);
  await expect(page.getByText("Identified", { exact: true }).first()).toBeVisible();
  await expect(page.getByAltText("Photo 2 of this vehicle")).toBeVisible();
  expect(files).toEqual([1, 2]);
  await expect(page.getByText(/Photo 1 was not recognised on its\s+own/)).toBeVisible(); // the answer says which photo it rests on
  await expect(page.getByText("Photo 2 alone: BMW M3 Coupe 2012 94%")).toBeVisible();

  await page.getByRole("button", { name: "Remove photo 2" }).click();
  await expect(page.getByText("Front crushed or badge hidden?")).toBeVisible();
  expect(files).toEqual([1, 2, 1]);
});
