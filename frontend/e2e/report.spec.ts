import { expect, test } from "@playwright/test";

test("the Model report shows the threshold table and the unknown-car result even before metrics are deployed", async ({ page }) => {
  await page.route("**/api/metrics", (r) => r.fulfill({ status: 404, body: "{}" }));
  await page.goto("/");
  await page.getByRole("button", { name: "Model report" }).click();

  await expect(page.getByRole("heading", { name: "Why the high-voltage warning fires at 5%" })).toBeVisible();
  const row = page.getByRole("row", { name: /0\.05/ });
  await expect(row).toContainText("In use");
  await expect(row).toContainText("1.15%");
  await expect(row).toContainText("0.69%");
  await expect(page.getByRole("heading", { name: "58 of 60 cars it never saw were called unknown" })).toBeVisible();
  await expect(page.getByText("88.3%")).toBeVisible();
});

test("the two new samples load and give the matching answer", async ({ page }) => {
  await page.route("**/api/analyze", async (route) => {
    const body = route.request().postDataBuffer()?.toString("latin1") ?? "";
    const name = body.includes("unknown-tata-nexon") ? "unknown" : "ambiguous";
    const { readFileSync } = await import("node:fs");
    await route.fulfill({ status: 200, contentType: "application/json", body: readFileSync(new URL(`../../docs/fixtures/${name}.json`, import.meta.url)) });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Indian car outside/ }).click();
  await expect(page.getByText("Not recognised", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Check another vehicle" }).click();
  await page.getByRole("button", { name: /Look-alike trim/ }).click();
  await expect(page.getByText("Not certain", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Behind the rear seat, in the trunk area. 270 V.")).toBeVisible();
});
