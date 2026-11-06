import type { BrandEvidence, Result, Safety } from "../types";
import RescueSheet from "./RescueSheet";
import { Bar, STATUS, pct } from "./bits";

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
    <div role="alert" className={`rise rise-2 flex overflow-hidden rounded-lg border-2 ${danger ? "border-hv bg-[#fff2e9]" : "border-warn bg-[#fbf4e6]"}`}>
      {danger && <div className="hazard-stripes w-3 shrink-0 sm:w-4" aria-hidden="true" />}
      <div className="p-4 sm:p-5">
        <p className={`font-display text-3xl font-bold leading-none ${danger ? "text-hv" : "text-warn"}`}>
          {danger ? "High voltage" : "Powertrain unknown"}
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
        <RescueSheet s={s} />
        {s.guide && (
          <a href={s.guide.url} target="_blank" rel="noreferrer"
            className="mt-4 inline-flex min-h-11 items-center rounded-md bg-ink px-4 font-semibold text-paper hover:bg-draft">
            {s.guide.title}
          </a>
        )}
      </div>
    </div>
  );
}

function BrandPanel({ e }: { e: BrandEvidence }) {
  const rows: { name: string; votes: BrandEvidence["whole_car"]; strong?: boolean }[] = [
    { name: "Whole car", votes: e.whole_car },
    { name: "Close-ups: badge, grille, lights", votes: e.details },
    { name: "Combined", votes: e.combined, strong: true },
  ];
  return (
    <div className="rounded-lg border border-line bg-sheet p-5">
      <h3 className="font-display text-2xl font-semibold">Brand evidence</h3>
      <p className="mt-1 text-sm text-ink-soft">Two experts vote on the brand: one sees the whole car, one only zoomed-in details.</p>
      {e.conflict && (
        <p className="mt-3 rounded-md border border-warn bg-[#fbf4e6] px-3 py-2 font-medium">
          The experts disagree: close-ups say {e.details[0].make}, the whole car says {e.whole_car[0].make}.
        </p>
      )}
      <dl className="mt-3 space-y-3">
        {rows.map((row) => (
          <div key={row.name}>
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-sm text-ink-soft">{row.name}</dt>
              <dd className={`tabular ${row.strong ? "font-semibold" : ""}`}>{row.votes[0].make} {pct(row.votes[0].prob)}</dd>
            </div>
            <Bar value={row.votes[0].prob} tone={row.strong ? "bg-draft" : "bg-ink-soft"} label={`${row.name}: ${row.votes[0].make}`} />
            <p className="mt-0.5 text-xs text-ink-soft">
              then {row.votes.slice(1).map((v) => `${v.make} ${pct(v.prob)}`).join(", ")}
            </p>
          </div>
        ))}
      </dl>
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

      {r.offline && (
        <p className="rounded-md border border-dashed border-ink/40 px-3 py-2 text-sm text-ink-soft">
          Answered on this device, without signal. Check it against the server when you're back online.
        </p>
      )}

      <div className="rise rounded-lg border border-line bg-sheet p-5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className={`inline-block rounded-sm px-2.5 py-1 font-semibold text-white ${st.chip}`}>{st.title}</p>
          <p className="tabular text-sm text-ink-soft">Answered in {(r.timing_ms.total / 1000).toFixed(1)} s</p>
        </div>
        <p className="mt-2 font-display text-4xl font-bold leading-[1.05]">{r.headline}</p>
        <p className="mt-2 text-ink-soft">{st.note}</p>
        {r.status === "unknown" && r.prediction && (
          <p className="mt-3 text-ink-soft">
            Closest of the 196 known models: <s>{r.prediction.label}</s> ({pct(r.prediction.confidence)}), not used.
          </p>
        )}
      </div>

      {r.safety && <SafetyBanner s={r.safety} />}

      {r.brand_evidence && <BrandPanel e={r.brand_evidence} />}

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
        {r.disclaimer}
      </p>
    </div>
  );
}
