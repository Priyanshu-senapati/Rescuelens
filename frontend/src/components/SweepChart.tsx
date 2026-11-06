import type { Status } from "../types";

export interface SweepPoint {
  level: number; // 0..1 of the slider range
  score: number; // usable confidence: 0 when the answer is unknown or no vehicle
  status: Status;
  key: string; // what the answer is, to spot when it changes
  headline: string;
  hv: boolean;
}

const COLOUR: Record<Status, string> = {
  identified: "var(--color-ok)",
  ambiguous: "var(--color-warn)",
  unknown: "var(--color-unknown)",
  no_vehicle: "var(--color-ink)",
};
const NAME: Record<Status, string> = { identified: "Identified", ambiguous: "Not certain", unknown: "Not recognised", no_vehicle: "No vehicle" };

const W = 520;
const H = 170;
const PAD = { l: 40, r: 12, t: 12, b: 30 };

/** Index of the first point whose answer differs from the undamaged photo, or -1. */
export function firstChange(points: SweepPoint[]): number {
  return points.findIndex((p) => p.key !== points[0]?.key);
}

export function sweepSummary(points: SweepPoint[], condition: string): string {
  if (points.length < 2) return "";
  const at = firstChange(points);
  const start = NAME[points[0].status];
  if (at < 0) return `Still "${start}" with the same answer, even at 100% on the ${condition} slider.`;
  return `Held "${start}" up to ${Math.round(points[at - 1].level * 100)}% on the ${condition} slider. At ${Math.round(points[at].level * 100)}% the answer changed to "${NAME[points[at].status]}"${points[at].hv && !points[0].hv ? ", and the high-voltage warning came on" : ""}.`;
}

export default function SweepChart({ points, total, condition }: { points: SweepPoint[]; total: number; condition: string }) {
  const x = (level: number) => PAD.l + level * (W - PAD.l - PAD.r);
  const y = (score: number) => PAD.t + (1 - score) * (H - PAD.t - PAD.b);
  const at = firstChange(points);
  const statuses = [...new Set(points.map((p) => p.status))];

  return (
    <figure className="rounded-lg border border-line bg-sheet p-4">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" className="tabular block w-full"
        aria-label={`Usable confidence as ${condition} increases. ${sweepSummary(points, condition)}`}>
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
            <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" fontSize="12" fill="var(--color-ink-soft)">{t * 100}%</text>
          </g>
        ))}
        <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="12" fill="var(--color-ink-soft)">{condition} slider (0 to 100%)</text>
        {at > 0 && (
          <line x1={x(points[at].level)} x2={x(points[at].level)} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-hv)" strokeWidth="2" strokeDasharray="4 4" />
        )}
        <polyline fill="none" stroke="var(--color-draft)" strokeWidth="2.5" strokeLinejoin="round"
          points={points.map((p) => `${x(p.level)},${y(p.score)}`).join(" ")} />
        {points.map((p) => (
          <circle key={p.level} cx={x(p.level)} cy={y(p.score)} r={p.hv ? 6 : 5} fill={COLOUR[p.status]} stroke={p.hv ? "var(--color-hv)" : "#fff"} strokeWidth="2" className="sweep-dot" />
        ))}
        {points.length < total && points.length > 0 && (
          <line x1={x(points[points.length - 1].level)} x2={x(points[points.length - 1].level)} y1={PAD.t} y2={H - PAD.b} stroke="var(--color-draft)" strokeOpacity="0.4" />
        )}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-soft">
        {statuses.map((s) => (
          <span key={s} className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: COLOUR[s] }} aria-hidden="true" />{NAME[s]}
          </span>
        ))}
        {points.some((p) => p.hv) && (
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full border-2 border-hv" aria-hidden="true" />High-voltage warning on
          </span>
        )}
      </figcaption>
    </figure>
  );
}
