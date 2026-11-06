// TypeScript port of backend/app/safety.py for on-device answers. The two must agree: docs/fixtures/safety_golden.json is
// checked against both (backend/tests/test_safety_golden.py and frontend/e2e/offline-safety.spec.ts).
import type { Candidate, Powertrain, Result, Safety } from "../types";

const TWO_WORD_MAKES = ["AM General", "Aston Martin", "Land Rover"];
export const DISCLAIMER =
  "Decision-support prototype. Vehicle identification is probabilistic. Never rely on it alone for cutting, lifting or disabling a vehicle.";
const HV_ACTIONS = [
  "Do not cut orange high-voltage cables (blue on some 36 V hybrids)",
  "Immobilise the vehicle and switch off the ignition",
  "Look for HYBRID / EV badges before cutting",
  "Confirm the vehicle with VIN or plate",
];
const NO_LOCATION = { location_vehicle: null, battery: null, hv_cables: null };

export interface Hazard {
  powertrain: string;
  guide?: string | null;
  guide_title?: string | null;
  battery?: string | null;
  hv_cables?: string | null;
}

export interface CarClass {
  label: string;
  make: string;
  model: string;
  year: number;
  body: string;
  family: string;
  powertrain: Powertrain;
  hazard: Hazard | null;
}

export interface Thresholds {
  identify: number;
  family: number;
  make: number;
  hv: number;
  unknown: number;
}

export function parseLabel(label: string, hazards: Record<string, Hazard>): CarClass {
  const words = label.split(" ");
  const n = TWO_WORD_MAKES.some((m) => label.startsWith(m)) ? 2 : 1;
  const make = words.slice(0, n).join(" ");
  const modelWords = words.slice(n, -1);
  const familyWords = modelWords.length > 1 && modelWords[1] === "Series" ? modelWords.slice(0, 2) : modelWords.slice(0, 1);
  const hazard = hazards[label] ?? null;
  return {
    label, make, model: modelWords.join(" "), year: Number(words[words.length - 1]), body: words[words.length - 2],
    family: [make, ...familyWords].join(" "), powertrain: (hazard?.powertrain ?? "combustion") as Powertrain, hazard,
  };
}

const round4 = (v: number) => Math.round(v * 10000) / 10000;

function safetyFor(probs: Float32Array | number[], classes: CarClass[], unknown: boolean, t: Thresholds): Safety {
  if (unknown)
    return {
      hv_probability: null, hv_warning: true, level: "caution",
      message: "Vehicle not recognised. Powertrain unknown - treat as possibly high-voltage.",
      actions: ["Do not use this prediction for rescue decisions", "Identify the vehicle manually (VIN, plate, badges)", "Check for HYBRID / EV badges and orange cables"],
      guide: null, ...NO_LOCATION,
    };
  const hvIdx = classes.flatMap((c, i) => (c.powertrain !== "combustion" ? [i] : []));
  const pHv = hvIdx.reduce((sum, i) => sum + probs[i], 0);
  if (pHv < t.hv)
    return {
      hv_probability: round4(pHv), hv_warning: false, level: "info",
      message: "No high-voltage system expected for this vehicle.",
      actions: ["Standard airbag and seat-belt pretensioner precautions apply", "Confirm the vehicle with VIN or plate"],
      guide: null, ...NO_LOCATION,
    };
  const topHv = [...hvIdx].sort((a, b) => probs[b] - probs[a]);
  const withGuide = topHv.map((i) => classes[i]).find((c) => c.hazard?.guide);
  const top = argmax(probs);
  const certain = classes[top].powertrain !== "combustion" && probs[top] >= t.identify;
  const likeliest = classes[topHv[0]];
  return {
    hv_probability: round4(pHv), hv_warning: true, level: "danger",
    message: certain ? "Hybrid / EV identified: HIGH VOLTAGE system present." : `Hybrid possible (${(pHv * 100).toFixed(0)}%). Treat as HIGH VOLTAGE until verified.`,
    actions: HV_ACTIONS,
    guide: withGuide ? { title: withGuide.hazard?.guide_title ?? `${withGuide.label} - Emergency Response Guide`, url: withGuide.hazard!.guide! } : null,
    location_vehicle: likeliest.hazard?.battery || likeliest.hazard?.hv_cables ? likeliest.label : null,
    battery: likeliest.hazard?.battery ?? null,
    hv_cables: likeliest.hazard?.hv_cables ?? null,
  };
}

function argmax(p: Float32Array | number[]): number {
  let best = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  return best;
}

type Decision = Pick<Result, "status" | "headline" | "resolved_level" | "prediction" | "hierarchy" | "top5" | "safety" | "unknown" | "reliability" | "disclaimer">;

export function decide(probs: Float32Array | number[], unknownScore: number, classes: CarClass[], t: Thresholds): Decision {
  const order = classes.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  // brand first (same rule as safety.py): summed make, then the strongest family in it, then its likeliest vehicle
  const makeP = new Map<string, number>();
  const familyP = new Map<string, number>();
  classes.forEach((c, i) => {
    makeP.set(c.make, (makeP.get(c.make) ?? 0) + probs[i]);
    familyP.set(c.family, (familyP.get(c.family) ?? 0) + probs[i]);
  });
  let make = classes[0].make;
  for (const [m, v] of makeP) if (v > makeP.get(make)!) make = m;          // ties: first in class order
  let family = "";
  for (const c of classes) if (c.make === make && (family === "" || familyP.get(c.family)! > familyP.get(family)!)) family = c.family;
  let topI = -1;
  classes.forEach((c, i) => { if (c.family === family && (topI < 0 || probs[i] > probs[topI])) topI = i; });
  const top = classes[topI];
  const pTop = probs[topI];
  const pMake = makeP.get(make)!;
  const pFamily = familyP.get(family)!;
  const single = classes[order[0]];
  const pSingle = probs[order[0]];
  const isOod = unknownScore > t.unknown;
  const unknown = isOod || pMake < t.make;

  let status: Result["status"];
  let level: Result["resolved_level"];
  let headline: string;
  if (unknown) {
    status = "unknown";
    level = "none";
    headline = isOod ? "Not recognised - damaged, unusual or not one of the 196 supported models" : "Cannot identify this vehicle with enough confidence";
  } else if (pTop >= t.identify) {
    status = "identified";
    level = "vehicle";
    headline = top.label;
  } else if (pFamily >= t.family) {
    const years = [...new Set(order.slice(0, 5).filter((i) => classes[i].family === family && probs[i] >= 0.05).map((i) => classes[i].year))].sort((a, b) => a - b);
    status = "ambiguous";
    level = "family";
    headline = `${family} family (${years.join(" or ")}) - exact variant uncertain`;
  } else {
    status = "ambiguous";
    level = "make";
    headline = `${make} - model uncertain`;
  }

  const safety = safetyFor(probs, classes, unknown, t);
  if (status === "ambiguous" && safety.hv_warning) headline += " - hybrid cannot be ruled out";

  const second = classes[order[1]];
  const reasons: string[] = [];
  if (second.family === single.family && probs[order[1]] >= 0.15) reasons.push("Top two answers are a known look-alike pair");
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  if (single.make !== make) reasons.push(`The single top guess (${single.label}, ${pct(pSingle)}) is not a ${make}: the ${make} models together add up to ${pct(pMake)}`);
  if (status !== "identified" && pSingle < 0.3) reasons.push(`No single model stands out (top answer only ${pct(pSingle)})`);
  if (isOod) reasons.push("Looks unlike the training photos: damage, an unusual angle, or a model outside the 196");

  const top5: Candidate[] = order.slice(0, 5).map((i) => {
    const c = classes[i];
    return { label: c.label, make: c.make, model: c.model, year: c.year, prob: round4(probs[i]), powertrain: c.powertrain };
  });

  return {
    status, headline, resolved_level: level,
    prediction: { label: top.label, make: top.make, model: top.model, year: top.year, body: top.body, confidence: round4(pTop) },
    hierarchy: [
      { level: "make", name: make, prob: round4(pMake) },
      { level: "family", name: family, prob: round4(pFamily) },
      { level: "vehicle", name: top.label, prob: round4(pTop) },
    ],
    top5, safety,
    unknown: { score: round4(unknownScore), threshold: t.unknown, is_unknown: isOod },
    reliability: { score: round4(unknown ? 0 : pTop), reasons },
    disclaimer: DISCLAIMER,
  };
}
