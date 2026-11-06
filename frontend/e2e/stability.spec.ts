import { expect, test } from "@playwright/test";
import { START, advance, stabilityKey } from "../src/live/stability";
import type { Result } from "../src/types";
import { fixture } from "./helpers";

const load = (n: "identified" | "ambiguous" | "unknown") => fixture(n) as unknown as Result;
const noVehicle = { ...load("identified"), status: "no_vehicle", safety: null, prediction: null, hierarchy: [] } as unknown as Result;
const run = (...frames: Result[]) => frames.reduce(advance, START);

test("three matching frames lock, two do not", () => {
  const a = load("identified");
  expect(run(a, a).locked).toBe(false);
  expect(run(a, a, a).locked).toBe(true);
});

test("a different answer restarts the count", () => {
  const a = load("identified");
  const b = { ...a, prediction: { ...a.prediction!, label: "Some Other Car 2012" } };
  const s = run(a, a, b);
  expect(s.locked).toBe(false);
  expect(s.streak).toBe(1);
});

test("losing the vehicle restarts the count and never locks", () => {
  const a = load("identified");
  expect(run(a, a, noVehicle, a).streak).toBe(1);
  expect(run(noVehicle, noVehicle, noVehicle, noVehicle).locked).toBe(false);
});

test("a car flickering between warning and no warning can never lock as safe", () => {
  const a = load("identified");
  const warn = { ...a, safety: { ...a.safety!, hv_warning: true } };
  const clear = { ...a, safety: { ...a.safety!, hv_warning: false } };
  expect(run(clear, warn, clear, warn, clear, warn).locked).toBe(false);
});

test("a warning stays up for 3 frames after the last frame that raised it", () => {
  const a = load("identified");
  const warn = { ...a, safety: { ...a.safety!, hv_warning: true } };
  const clear = { ...a, safety: { ...a.safety!, hv_warning: false } };
  expect(run(warn).hvHold).toBe(3);
  expect(run(warn, clear, clear).hvHold).toBe(1);
  expect(run(warn, clear, clear, clear).hvHold).toBe(0);
});

test("an unknown car ignores its untrusted closest match", () => {
  const u = load("unknown");
  const v = { ...u, prediction: { ...u.prediction!, label: "Different Closest Match 2012" } };
  expect(stabilityKey(u)).toBe(stabilityKey(v));
});

test("an ambiguous answer ignores which variant is on top", () => {
  const a = load("ambiguous");
  const flipped = { ...a, prediction: { ...a.prediction!, label: "Hyundai Sonata Sedan 2012" } };
  expect(stabilityKey(a)).toBe(stabilityKey(flipped));
});
