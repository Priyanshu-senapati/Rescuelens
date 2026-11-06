import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import { vehicleRows, vehicleScore } from "../src/offline/detection";
import { decide, parseLabel, type Hazard } from "../src/offline/safety";

const here = dirname(fileURLToPath(import.meta.url));
const read = (p: string) => JSON.parse(readFileSync(resolve(here, p), "utf-8"));
const golden = read("../../docs/fixtures/safety_golden.json");
const hazards = read("../../data/hazards.json").high_voltage as Record<string, Hazard>;
const classes = (golden.class_names as string[]).map((n) => parseLabel(n, hazards));

/** Rounds so a last-digit difference between Python and JavaScript floats doesn't count, while any real change does. */
function close(actual: unknown, expected: unknown, path: string) {
  if (typeof expected === "number") return expect(Math.abs((actual as number) - expected), path).toBeLessThan(0.0002);
  if (Array.isArray(expected)) {
    expect((actual as unknown[]).length, path).toBe(expected.length);
    return expected.forEach((v, i) => close((actual as unknown[])[i], v, `${path}[${i}]`));
  }
  if (expected && typeof expected === "object") {
    expect(Object.keys(actual as object).sort(), path).toEqual(Object.keys(expected).sort());
    return Object.entries(expected).forEach(([k, v]) => close((actual as Record<string, unknown>)[k], v, `${path}.${k}`));
  }
  expect(actual, path).toBe(expected);
}

test("offline detector scores match the back end for both detector heads", () => {
  const coco = new Array(84 * 3).fill(0);
  coco[4 * 3 + 0] = 0.9; // person: must be ignored
  coco[(4 + 2) * 3 + 1] = 0.8; // car
  coco[(4 + 7) * 3 + 2] = 0.7; // truck
  const rows = vehicleRows(84);
  expect([0, 1, 2].map((i) => vehicleScore(coco, 3, i, rows))).toEqual([0, 0.8, 0.7]);

  const tuned = [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.95]; // 5 rows x 2 anchors
  expect([0, 1].map((i) => vehicleScore(tuned, 2, i, vehicleRows(5)))).toEqual([0.3, 0.95]);
});

for (const c of golden.cases) {
  test(`on-device safety matches the server: ${c.name}`, () => {
    close(decide(c.probs, c.unknown_score, classes, golden.thresholds), c.expected, c.name);
  });
}
