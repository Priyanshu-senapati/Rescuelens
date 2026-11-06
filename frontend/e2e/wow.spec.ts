import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { firstChange, sweepSummary, type SweepPoint } from "../src/components/SweepChart";
import { fixture, stubAnalyze } from "./helpers";

const here = dirname(fileURLToPath(import.meta.url));
const hazards = JSON.parse(readFileSync(resolve(here, "../../data/hazards.json"), "utf-8")).high_voltage;

test("every high-voltage model has a schematic, and only stated cable routes are drawn", () => {
  expect(Object.keys(hazards)).toHaveLength(9);
  for (const [label, h] of Object.entries<{ schematic?: { cable: string; cable_colour?: string } }>(hazards)) {
    expect(h.schematic, label).toBeTruthy();
    if (h.schematic!.cable === "underbody") expect(h.schematic!.cable_colour, label).toBeTruthy();
  }
  // the two routes the guides state; the Malibu's is blue
  expect(hazards["Chevrolet Malibu Hybrid Sedan 2010"].schematic.cable_colour).toBe("blue");
  expect(hazards["Hyundai Sonata Hybrid Sedan 2012"].schematic.cable_colour).toBe("orange");
});

const point = (level: number, status: SweepPoint["status"], key: string, hv = false): SweepPoint => ({ level, status, key, hv, score: 0.5, headline: key });

test("the sweep names the exact point the answer changed", () => {
  const steady = [point(0, "identified", "a"), point(0.5, "identified", "a"), point(1, "identified", "a")];
  expect(firstChange(steady)).toBe(-1);
  expect(sweepSummary(steady, "blur")).toContain("even at 100% blur");

  const flips = [point(0, "identified", "a"), point(0.25, "identified", "a"), point(0.5, "ambiguous", "b", true)];
  expect(firstChange(flips)).toBe(2);
  expect(sweepSummary(flips, "blur")).toBe('Held "Identified" up to 25% blur. At 50% the answer changed to "Not certain", and the high-voltage warning came on.');
});

test("the home page cycles the rescue sheet and can be paused or driven by hand", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Same shape. Different danger." })).toBeVisible();
  await page.getByRole("button", { name: "Malibu" }).click();
  await expect(page.getByText(/In the trunk, directly behind the rear passenger seats/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Pause" })).toBeHidden(); // choosing a model stops the cycle
  await page.waitForTimeout(4500);
  await expect(page.getByText(/In the trunk, directly behind the rear passenger seats/)).toBeVisible();
});

test("a result draws the rescue sheet with the guide's text", async ({ page }) => {
  await stubAnalyze(page, fixture("ambiguous"));
  await page.goto("/");
  await page.getByRole("button", { name: /Hybrid that looks like/ }).click();
  await expect(page.getByRole("img", { name: /Hyundai Sonata Hybrid Sedan 2012 with the high-voltage battery marked/ })).toBeVisible();
  await expect(page.getByText(/not a guide for where to cut/)).toBeVisible();
});

test("the automatic sweep asks the model at every level, plots it and says where it broke", async ({ page }) => {
  const identified = fixture("identified");
  const ambiguous = fixture("ambiguous");
  let calls = 0;
  await page.route("**/api/analyze", (route) => {
    calls++;
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(calls < 5 ? identified : ambiguous) });
  });
  await page.goto("/");
  await page.getByRole("button", { name: /Clear daylight/ }).click();
  await page.getByRole("button", { name: "Sweep blur" }).click();
  await expect(page.getByRole("status").filter({ hasText: /answer changed to "Not certain"/ })).toBeVisible({ timeout: 20_000 });
  expect(calls).toBe(9); // the first analysis plus 8 levels
  await expect(page.getByRole("img", { name: /Usable confidence as blur increases/ })).toBeVisible();

  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByText("Run a sweep or move a slider to see the new answer.")).toBeVisible();
});
