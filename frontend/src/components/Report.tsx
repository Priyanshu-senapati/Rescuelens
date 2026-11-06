import { useEffect, useState } from "react";
import { fetchMetrics } from "../api";
import type { Metrics } from "../types";
import SafetyEvidence from "./SafetyEvidence";
import { Bar, SectionTitle, pct } from "./bits";

const CONDITION_NAME: Record<string, string> = {
  clean: "Original photos", dark: "Darkness", "dark+clahe": "Darkness, with contrast boost", haze: "Smoke or haze",
  blur: "Blur", low_res: "Low resolution", occlusion: "Partly hidden",
};
const APPROACH_NAME: Record<string, string> = { classical: "Classical only", deep: "Deep learning only", hybrid: "Hybrid (RescueLens)" };

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg border border-line bg-sheet p-4">
      <p className="tabular font-display text-4xl font-bold leading-none">{value}</p>
      <p className="mt-1 text-ink-soft">{label}</p>
    </div>
  );
}

export default function Report() {
  const [m, setM] = useState<Metrics | null | undefined>(undefined);
  useEffect(() => {
    fetchMetrics().then(setM);
  }, []);

  if (m === undefined) return <p className="py-10 text-ink-soft">Loading the evaluation report…</p>;
  if (m === null)
    return (
      <div className="space-y-10 py-10">
        <div>
          <SectionTitle title="Model report" />
          <p className="text-ink-soft">The full evaluation report isn't on this server yet. It appears once the trained models are deployed. These safety results are from the same evaluation.</p>
        </div>
        <SafetyEvidence metrics={null} />
      </div>
    );

  const c = m.classification, d = m.detection, e = m.efficiency, s = m.safety;
  const conditions = [...new Set((m.robustness ?? []).map((r) => r.condition))];

  return (
    <div className="space-y-10">
      <div>
        <SectionTitle title="Model report" intro="Measured on the official Stanford Cars test split: 8,041 photos the models never saw during training." />
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
          {c && <Figure value={pct(c.top1, 1)} label="Top-1 accuracy" />}
          {c && <Figure value={pct(c.top5, 1)} label="Top-5 accuracy" />}
          {d && <Figure value={d.mean_iou.toFixed(3)} label="Mean IoU of the vehicle box" />}
          {d && <Figure value={d.map50.toFixed(3)} label="mAP at IoU 0.5" />}
          {e && <Figure value={`${e.int8_mb.toFixed(0)} MB`} label={`Model size after INT8 (from ${e.fp32_mb.toFixed(0)} MB)`} />}
          {e && <Figure value={`${e.int8_ms.toFixed(0)} ms`} label={`Classifier on 2 CPU cores (FP32 ${e.fp32_ms.toFixed(0)} ms)`} />}
        </div>
      </div>

      {m.approaches && (
        <div>
          <SectionTitle title="Classical, deep learning and hybrid" />
          <div className="space-y-3 rounded-lg border border-line bg-sheet p-5">
            {m.approaches.map((a) => (
              <div key={a.name}>
                <div className="flex justify-between gap-3"><span className="font-medium">{APPROACH_NAME[a.name] ?? a.name}</span>
                  <span className="tabular text-ink-soft">top-1 {pct(a.top1, 1)}{a.top5 !== null && `, top-5 ${pct(a.top5, 1)}`}</span></div>
                <Bar value={a.top1 ?? 0} tone={a.name === "hybrid" ? "bg-draft" : "bg-ink-soft"} label={`${a.name} top-1`} />
              </div>
            ))}
            {m.localisation && (
              <p className="pt-2 text-ink-soft">
                Finding the vehicle: neural detector mean IoU {m.localisation.deep_mean_iou.toFixed(3)}, classical edge contours{" "}
                {m.localisation.classical_mean_iou?.toFixed(3) ?? "n/a"}.
              </p>
            )}
          </div>
        </div>
      )}

      {s && (
        <div>
          <SectionTitle title="Safety" intro="Missing a hybrid is far worse than a false alarm, so the high-voltage warning uses the summed probability of all hybrid and electric models." />
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            <Figure value={pct(s.hv_miss_rate, 1)} label="Hybrids or EVs missed by the warning" />
            {s.argmax_hv_miss_rate !== undefined && <Figure value={pct(s.argmax_hv_miss_rate, 1)} label="Missed if we trusted only the top answer" />}
            <Figure value={pct(s.hv_false_alarm_rate, 1)} label="False high-voltage alarms" />
            <Figure value={pct(s.unknown_catch_rate, 0)} label="Unseen car models correctly called unknown" />
          </div>
          {m.calibration && (
            <p className="mt-3 text-ink-soft">
              Confidence calibration error (ECE) {m.calibration.ece_before.toFixed(3)} before and {m.calibration.ece_after.toFixed(3)} after temperature scaling.
            </p>
          )}
        </div>
      )}

      <SafetyEvidence metrics={m} />

      {m.robustness && (
        <div>
          <SectionTitle title="Robustness" intro="Top-1 accuracy as conditions get worse (three severity levels)." />
          <div className="overflow-x-auto rounded-lg border border-line bg-sheet">
            <table className="tabular w-full min-w-[32rem] text-left">
              <thead className="border-b border-line text-sm text-ink-soft">
                <tr><th className="p-3 font-medium">Condition</th><th className="p-3 font-medium">Mild</th><th className="p-3 font-medium">Medium</th><th className="p-3 font-medium">Severe</th></tr>
              </thead>
              <tbody>
                {conditions.map((cond) => {
                  const rows = (m.robustness ?? []).filter((r) => r.condition === cond);
                  return (
                    <tr key={cond} className="border-b border-line last:border-0">
                      <th className="p-3 font-medium">{CONDITION_NAME[cond] ?? cond}</th>
                      {cond === "clean" ? <td className="p-3" colSpan={3}>{pct(rows[0]?.top1, 1)}</td> :
                        [1, 2, 3].map((sev) => <td key={sev} className="p-3">{pct(rows.find((r) => r.severity === sev)?.top1, 1)}</td>)}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {m.risk_coverage && (
        <div>
          <SectionTitle title="Answering only when confident" intro="If RescueLens answers only for its most confident share of photos, accuracy rises." />
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-7">
            {m.risk_coverage.map((r) => <Figure key={r.coverage} value={pct(r.accuracy, 1)} label={`Top ${Math.round(r.coverage * 100)}% of photos`} />)}
          </div>
        </div>
      )}

      {m.top_confusions && m.top_confusions.length > 0 && (
        <div>
          <SectionTitle title="Where it gets confused" intro="The most frequent mix-ups on the test set: mostly the same car in a different year, or the same vehicle sold under two brands." />
          <ul className="divide-y divide-line rounded-lg border border-line bg-sheet">
            {m.top_confusions.slice(0, 8).map((t) => (
              <li key={t.true + t.predicted} className="flex flex-wrap justify-between gap-2 p-3">
                <span><span className="font-medium">{t.true}</span> <span className="text-ink-soft">taken for</span> {t.predicted}</span>
                <span className="tabular text-ink-soft">{t.count} photos</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
