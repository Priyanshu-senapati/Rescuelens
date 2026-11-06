import measured from "../../../data/measured.json";
import type { Metrics } from "../types";
import { SectionTitle, pct } from "./bits";

/** The two results the pitch leans on. Uses the deployed metrics when they carry the table, else the committed evaluation numbers. */
export default function SafetyEvidence({ metrics }: { metrics: Metrics | null }) {
  const rows = metrics?.safety?.by_threshold ?? measured.hv_thresholds;
  const chosen = metrics?.safety?.hv_threshold ?? measured.hv_threshold;
  const argmax = metrics?.safety?.argmax_hv_miss_rate ?? measured.argmax_hv_miss_rate;
  const u = measured.unknown;
  const maxMiss = Math.max(...rows.map((r) => r.hv_miss_rate));

  return (
    <div className="space-y-10">
      <section>
        <SectionTitle
          title="Why the high-voltage warning fires at 5%"
          intro={`Raising the threshold cuts false alarms but misses more real hybrids and EVs. We chose the lowest row, because a missed high-voltage car can kill and a false alarm only costs time. Trusting only the top answer would miss ${pct(argmax, 2)}.`}
        />
        <div className="overflow-x-auto rounded-lg border border-line bg-sheet">
          <table className="tabular w-full min-w-[30rem] text-left">
            <thead className="border-b border-line text-sm text-ink-soft">
              <tr>
                <th scope="col" className="p-3 font-medium">Warn when summed hybrid/EV probability reaches</th>
                <th scope="col" className="p-3 font-medium">Hybrids and EVs missed</th>
                <th scope="col" className="p-3 font-medium">False alarms</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const inUse = r.threshold === chosen;
                return (
                  <tr key={r.threshold} className={`border-b border-line last:border-0 ${inUse ? "bg-[#fff2e9] font-semibold" : ""}`}>
                    <th scope="row" className="p-3 font-medium">
                      {r.threshold}
                      {inUse && <span className="ml-2 rounded-sm bg-hv px-1.5 py-0.5 text-xs font-semibold text-white">In use</span>}
                    </th>
                    <td className="p-3">
                      <span className="flex items-center gap-3">
                        <span className="w-14 shrink-0">{pct(r.hv_miss_rate, 2)}</span>
                        <span className="h-2 flex-1 rounded-sm bg-draft-soft" aria-hidden="true">
                          <span className="block h-full rounded-sm bg-hv" style={{ width: `${(r.hv_miss_rate / maxMiss) * 100}%` }} />
                        </span>
                      </span>
                    </td>
                    <td className="p-3">{pct(r.hv_false_alarm_rate, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-sm text-ink-soft">{measured.n_hv_test} hybrid and EV photos in the official test split. {measured.model}, temperature-calibrated.</p>
      </section>

      <section>
        <SectionTitle
          title={`${u.caught} of ${u.photos} cars it never saw were called unknown`}
          intro={`${u.photos} freely licensed photos of ${u.models} models outside the 196 classes: Indian models and EVs newer than 2012. This is the real test of "knows when it doesn't know".`}
        />
        <div className="rounded-lg border border-line bg-sheet p-5">
          <div role="img" aria-label={`${u.caught} of ${u.photos} photos correctly answered unknown, ${u.photos - u.caught} wrongly identified`}
            className="grid max-w-xl grid-cols-[repeat(20,minmax(0,1fr))] gap-1">
            {Array.from({ length: u.photos }, (_, i) => (
              <span key={i} className={`aspect-square rounded-[2px] ${i < u.caught ? "bg-draft" : "border-2 border-hv"}`} />
            ))}
          </div>
          <dl className="mt-5 grid gap-x-8 gap-y-4 md:grid-cols-3">
            <div>
              <dt className="text-sm text-ink-soft">The {u.photos - u.caught} it got wrong</dt>
              <dd>{u.misses.join("; ")}.</dd>
            </div>
            <div>
              <dt className="text-sm text-ink-soft">What that costs</dt>
              <dd><span className="tabular font-semibold">{pct(u.known_flagged_rate, 1)}</span> of cars it does know get flagged unknown too.</dd>
            </div>
            <div>
              <dt className="text-sm text-ink-soft">Earlier model</dt>
              <dd>{u.earlier_model.name} caught <span className="tabular font-semibold">{pct(u.earlier_model.catch_rate, 1)}</span> on the same photos, one reason we switched.</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  );
}
