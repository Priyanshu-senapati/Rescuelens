import type { Safety, Status } from "../types";

export const STATUS: Record<Status, { title: string; chip: string; note: string }> = {
  identified: { title: "Identified", chip: "bg-ok", note: "Confident about the exact model and year." },
  ambiguous: { title: "Not certain", chip: "bg-warn", note: "Sure about part of the answer, not all of it." },
  unknown: { title: "Not recognised", chip: "bg-unknown", note: "Don't use this answer for rescue decisions." },
  no_vehicle: { title: "No vehicle found", chip: "bg-ink", note: "Try a photo where the whole vehicle is visible." },
};

/** Where the battery and cables are, from the manufacturer's guide. Names the vehicle because the answer may be uncertain. */
export function HazardLocation({ s }: { s: Safety }) {
  if (!s.battery && !s.hv_cables) return null;
  return (
    <div className="mt-3 border-t border-current/20 pt-3">
      <p className="text-sm font-medium">If this is a {s.location_vehicle}, per the manufacturer's guide:</p>
      <dl className="mt-1.5 space-y-1.5">
        {s.battery && (
          <div>
            <dt className="text-sm font-semibold">Battery</dt>
            <dd>{s.battery}</dd>
          </div>
        )}
        {s.hv_cables && (
          <div>
            <dt className="text-sm font-semibold">Cables</dt>
            <dd>{s.hv_cables}</dd>
          </div>
        )}
      </dl>
    </div>
  );
}

export function pct(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined) return "n/a";
  const v = p * 100;
  if (v > 0 && v < 1 && digits === 0) return "<1%";
  return `${v.toFixed(digits)}%`;
}

interface BarProps {
  value: number; // 0..1
  tone?: string; // tailwind bg class
  label: string; // accessible name
}

export function Bar({ value, tone = "bg-draft", label }: BarProps) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-sm bg-draft-soft" role="meter" aria-label={label}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
      <div className={`h-full ${tone}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function SectionTitle({ title, intro }: { title: string; intro?: string }) {
  return (
    <div className="mb-4">
      <h2 className="font-display text-3xl font-bold leading-tight">{title}</h2>
      {intro && <p className="mt-1 max-w-[70ch] text-ink-soft">{intro}</p>}
    </div>
  );
}
