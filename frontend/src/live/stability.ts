// Decides when a stream of per-frame answers has settled. Pure logic, so the safety rules can be tested without a camera.
import type { Result } from "../types";

export const LOCK_FRAMES = 3; // the same answer this many frames in a row locks it in
export const HV_HOLD_FRAMES = 3; // a warning stays up until this many frames in a row have not raised it

export interface ScanState {
  key: string | null;
  streak: number;
  locked: boolean;
  hvHold: number;
  frames: number;
}

export const START: ScanState = { key: null, streak: 0, locked: false, hvHold: 0, frames: 0 };

/**
 * What must stay the same for two frames to count as "the same answer".
 * - The high-voltage state is part of the key, so a car flickering between "warning" and "no warning" can never lock as safe.
 * - An unknown car's closest match is untrusted, so it isn't part of the key: "unknown" is the whole answer.
 * - An ambiguous answer only compares the level it is sure of (Sonata vs Sonata Hybrid swapping first place is still "Sonata family").
 */
export function stabilityKey(r: Result): string | null {
  if (r.status === "no_vehicle") return null;
  const hv = r.safety?.hv_warning ? "hv" : "clear";
  if (r.status === "unknown") return `unknown|${hv}`;
  if (r.status === "identified") return `identified|${r.prediction?.label ?? r.headline}|${hv}`;
  const sure = r.hierarchy.find((h) => h.level === r.resolved_level)?.name ?? r.headline;
  return `ambiguous|${r.resolved_level}|${sure}|${hv}`;
}

export function advance(s: ScanState, r: Result): ScanState {
  const key = stabilityKey(r);
  const streak = key === null ? 0 : key === s.key ? s.streak + 1 : 1;
  return {
    key,
    streak,
    locked: key !== null && streak >= LOCK_FRAMES,
    hvHold: r.safety?.hv_warning ? HV_HOLD_FRAMES : Math.max(0, s.hvHold - 1),
    frames: s.frames + 1,
  };
}
