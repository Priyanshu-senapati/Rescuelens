import type { Result, Safety, Status } from "../types";
import { Bar, pct } from "./bits";

const STATUS: Record<Status, { title: string; border: string; text: string; note: string }> = {
  identified: { title: "Identified", border: "border-ok", text: "text-ok", note: "Confident about the exact model and year." },
  ambiguous: { title: "Not certain", border: "border-warn", text: "text-warn", note: "Sure about part of the answer, not all of it." },
  unknown: { title: "Not recognised", border: "border-unknown", text: "text-unknown", note: "Don't use this answer for rescue decisions." },
  no_vehicle: { title: "No vehicle found", border: "border-ink", text: "text-ink", note: "Try a photo where the whole vehicle is visible." },
};

const LEVEL_NAME = { make: "Brand", family: "Model family", vehicle: "Exact model and year" } as const;
const DEPTH = { none: 0, make: 1, family: 2, vehicle: 3 } as const;

function SafetyBanner({ s }: { s: Safety }) {
  if (s.level === "info")
    return (
      <div className="rounded-lg border border-line bg-sheet p-4">
        <p className="font-medium">{s.message}</p>
        <ul className="mt-2 list-disc pl-5 text-ink-soft">
          {s.actions.map((a) => <li key={a}>{a}</li>)}
        </ul>
      </div>
    );
  const danger = s.level === "danger";
  return (
    <div role="alert" className={`flex overflow-hidden rounded-lg border-2 ${danger ? "border-hv bg-[#fff2e9]" : "border-warn bg-[#fbf4e6]"}`}>
      {danger && <div className="hazard-stripes w-3 shrink-0 sm:w-4" aria-hidden="true" />}
      <div className="p-4 sm:p-5">
        <p className={`font-display text-3xl font-bold leading-none ${danger ? "text-hv" : "text-warn"}`}>
          {danger ? "High voltage possible" : "Powertrain unknown"}
        </p>
        <p className="mt-2 text-lg font-medium">{s.message}</p>
        <ul className="mt-3 space-y-1.5">
          {s.actions.map((a) => (
            <li key={a} className="flex gap-2">
              <span aria-hidden="true" className={`mt-2 h-2 w-2 shrink-0 rounded-full ${danger ? "bg-hv" : "bg-warn"}`} />
              <span>{a}</span>
            </li>
          ))}
        </ul>
        {s.guide && (
          <a href={s.guide.url} target="_blank" rel="noreferrer"
            className="mt-4 inline-flex min-h-11 items-center rounded-md bg-ink px-4 font-semibold text-paper hover:bg-draft">
            Open the emergency response guide
          </a>
        )}
      </div>
    </div>
  );
}

export default function ResultPanel({ result: r }: { result: Result }) {
  const st = STATUS[r.status];
  const depth = DEPTH[r.resolved_level];
  return (
    <div className="space-y-4">
      {r.mock && (
        <p className="rounded-md border border-dashed border-ink/40 px-3 py-2 text-sm text-ink-soft">
          Sample answer: the trained models aren't loaded on this server yet.
        </p>
      )}

      <div className={`rounded-lg border-l-8 bg-sheet p-5 shadow-sm ${st.border}`}>
        <p className={`font-semibold ${st.text}`}>{st.title}</p>
        <p className="mt-1 font-display text-4xl font-bold leading-[1.05]">{r.headline}</p>
        <p className="mt-2 text-ink-soft">{st.note}</p>
        {r.status === "unknown" && r.prediction && (
          <p className="mt-3 text-ink-soft">
            Closest of the 196 known models: <s>{r.prediction.label}</s> ({pct(r.prediction.confidence)}), not used.
          </p>
        )}
      </div>

      {r.safety && <SafetyBanner s={r.safety} />}

      {r.hierarchy.length > 0 && (
        <div className="rounded-lg border border-line bg-sheet p-5">
          <h3 className="font-display text-2xl font-semibold">How sure, level by level</h3>
          <dl className="mt-3 space-y-3">
            {r.hierarchy.map((h, i) => (
              <div key={h.level} className={i + 1 > depth ? "opacity-45" : ""}>
                <div className="flex items-baseline justify-between gap-3">
                  <dt className="text-sm text-ink-soft">{LEVEL_NAME[h.level]}</dt>
                  <dd className="tabular font-semibold">{pct(h.prob)}</dd>
                </div>
                <p className="truncate font-medium">{h.name}</p>
                <Bar value={h.prob} label={`${LEVEL_NAME[h.level]} confidence`} />
              </div>
            ))}
          </dl>
        </div>
      )}

      {r.top5.length > 0 && (
        <div className="rounded-lg border border-line bg-sheet p-5">
          <h3 className="font-display text-2xl font-semibold">Top 5 candidates</h3>
          <ol className="mt-3 space-y-2.5">
            {r.top5.map((c) => (
              <li key={c.label}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="font-medium">
                    {c.label}
                    {c.powertrain !== "combustion" && (
                      <span className="ml-2 rounded-sm bg-hv px-1.5 py-0.5 align-middle text-xs font-semibold text-white">
                        {c.powertrain === "ev" ? "Electric" : "Hybrid"}
                      </span>
                    )}
                  </span>
                  <span className="tabular shrink-0 text-ink-soft">{pct(c.prob, 1)}</span>
                </div>
                <Bar value={c.prob} tone={c.powertrain !== "combustion" ? "bg-hv" : "bg-draft"} label={`${c.label} probability`} />
              </li>
            ))}
          </ol>
        </div>
      )}

      {r.quality && (
        <div className="rounded-lg border border-line bg-sheet p-5">
          <h3 className="font-display text-2xl font-semibold">Why this level of confidence</h3>
          {r.reliability && r.reliability.reasons.length > 0 ? (
            <ul className="mt-2 list-disc pl-5">
              {r.reliability.reasons.map((x) => <li key={x}>{x}</li>)}
            </ul>
          ) : (
            <p className="mt-2 text-ink-soft">No problems found in the photo.</p>
          )}
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-3">
            {([["Light", r.quality.brightness], ["Contrast", r.quality.contrast], ["Sharpness", r.quality.sharpness]] as const).map(([k, v]) => (
              <div key={k}>
                <dt className="flex justify-between text-sm text-ink-soft"><span>{k}</span><span className="tabular">{v}/100</span></dt>
                <dd><Bar value={v / 100} tone={v < 30 ? "bg-warn" : "bg-draft"} label={`${k} score`} /></dd>
              </div>
            ))}
            <div>
              <dt className="text-sm text-ink-soft">Vehicle size in photo</dt>
              <dd className="tabular font-medium">{r.quality.resolution_px} px</dd>
            </div>
          </dl>
        </div>
      )}

      <p className="text-sm text-ink-soft">
        Answered in {(r.timing_ms.total / 1000).toFixed(1)} s. {r.disclaimer}
      </p>
    </div>
  );
}
