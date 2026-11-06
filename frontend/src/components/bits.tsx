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
