import type { Result } from "../types";
import { Bar, SectionTitle, pct } from "./bits";

type Comparison = NonNullable<Result["comparison"]>;

const COLUMNS: { key: keyof Comparison; title: string; how: string }[] = [
  { key: "classical", title: "Classical only", how: "Hand-crafted shape (HOG), colour and texture features with a linear classifier. No neural network." },
  { key: "deep", title: "Deep learning only", how: "The ConvNeXt network on the whole photo, without locating the vehicle first." },
  { key: "hybrid", title: "Hybrid (what you see above)", how: "Detector crop, contrast boost when dark, calibrated network, then the safety rules." },
];

export default function Compare({ comparison }: { comparison: Comparison }) {
  return (
    <section>
      <SectionTitle title="Three approaches on this photo" intro="The same photo through each approach the problem statement asks us to compare. Full test-set numbers are in the model report." />
      <div className="grid gap-4 md:grid-cols-3">
        {COLUMNS.map((c) => {
          const pick = comparison[c.key];
          return (
            <div key={c.key} className={`rounded-lg border bg-sheet p-5 ${c.key === "hybrid" ? "border-2 border-draft" : "border-line"}`}>
              <h3 className="font-display text-2xl font-semibold">{c.title}</h3>
              <p className="mt-1 text-sm text-ink-soft">{c.how}</p>
              {pick ? (
                <>
                  <p className="mt-4 font-medium">{pick.label}</p>
                  <div className="mt-1 flex items-center gap-3">
                    <Bar value={pick.prob} label={`${c.title} confidence`} />
                    <span className="tabular shrink-0 text-ink-soft">{pct(pick.prob)}</span>
                  </div>
                </>
              ) : (
                <p className="mt-4 text-ink-soft">Not available on this server.</p>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
