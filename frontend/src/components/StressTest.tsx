import { useEffect, useRef, useState } from "react";
import { ApiError, analyze } from "../api";
import { NO_DEGRADE, degradeImage, type Degrade, type Prepared } from "../image";
import type { Result } from "../types";
import { SectionTitle, pct } from "./bits";

const SLIDERS: { key: keyof Degrade; label: string; max: number }[] = [
  { key: "darkness", label: "Darkness", max: 0.9 },
  { key: "haze", label: "Smoke or haze", max: 0.7 },
  { key: "blur", label: "Blur and low resolution", max: 1 },
  { key: "occlusion", label: "Vehicle partly hidden", max: 0.5 },
];

const STATUS_TEXT = { identified: "Identified", ambiguous: "Not certain", unknown: "Not recognised", no_vehicle: "No vehicle found" };

function Outcome({ title, r }: { title: string; r: Result }) {
  const hv = r.safety?.hv_warning;
  return (
    <div className="rounded-lg border border-line bg-sheet p-4">
      <p className="text-sm text-ink-soft">{title}</p>
      <p className="font-semibold">{STATUS_TEXT[r.status]}</p>
      <p className="font-display text-xl font-semibold leading-tight">{r.headline}</p>
      {r.prediction && <p className="tabular text-ink-soft">Top answer {pct(r.prediction.confidence)} confident</p>}
      {hv && <p className={`mt-1 font-semibold ${r.safety?.level === "danger" ? "text-hv" : "text-warn"}`}>High-voltage warning on</p>}
    </div>
  );
}

export default function StressTest({ photo, baseline }: { photo: Prepared; baseline: Result }) {
  const [d, setD] = useState<Degrade>(NO_DEGRADE);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const previewRef = useRef<string | null>(null);
  const touched = Object.values(d).some((v) => v > 0);

  function showPreview(url: string | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = url;
    setPreview(url);
  }

  useEffect(() => () => showPreview(null), []); // free the last preview image on unmount

  useEffect(() => {
    if (!touched) {
      showPreview(null);
      setResult(null);
      setBusy(false);
      return;
    }
    const ctrl = new AbortController(); // aborted when the sliders move again, on reset, or on unmount
    const timer = setTimeout(async () => {
      setBusy(true);
      setError(null);
      try {
        const img = await degradeImage(photo.url, d, baseline.detection?.box ?? null);
        if (ctrl.signal.aborted) return URL.revokeObjectURL(img.url);
        showPreview(img.url);
        const r = await analyze(img.blob, img.name, ctrl.signal);
        if (!ctrl.signal.aborted) setResult(r);
      } catch (err: unknown) {
        if (!ctrl.signal.aborted) setError(err instanceof ApiError ? err.message : "Couldn't run the stress test.");
      } finally {
        if (!ctrl.signal.aborted) setBusy(false);
      }
    }, 650);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [d, touched, photo.url, baseline.detection]);

  return (
    <section>
      <SectionTitle
        title="Stress test"
        intro="Crash scenes are dark, smoky and cluttered. Push the photo towards those conditions and watch how the answer, and the safety advice, change."
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-4 rounded-lg border border-line bg-sheet p-5">
          {SLIDERS.map((s) => (
            <label key={s.key} className="block">
              <span className="flex justify-between font-medium">
                <span>{s.label}</span>
                <span className="tabular text-ink-soft">{Math.round((d[s.key] / s.max) * 100)}%</span>
              </span>
              <input type="range" min={0} max={s.max} step={s.max / 20} value={d[s.key]}
                onChange={(e) => setD({ ...d, [s.key]: +e.target.value })} className="mt-1 w-full accent-draft" />
            </label>
          ))}
          <button onClick={() => setD(NO_DEGRADE)} disabled={!touched}
            className="min-h-11 rounded-md border border-ink/30 px-4 font-medium enabled:hover:border-ink disabled:opacity-40">
            Reset
          </button>
        </div>
        <div className="space-y-3">
          <div className="relative overflow-hidden rounded-lg border border-line bg-ink">
            <img src={preview ?? photo.url} alt="Photo with simulated conditions" className={`block w-full ${busy ? "opacity-70" : ""}`} />
            {busy && <p className="absolute left-3 top-3 rounded bg-paper px-2 py-1 text-sm font-medium">Re-analysing…</p>}
          </div>
          {error && <p role="alert" className="rounded-md border border-ink px-3 py-2">{error}</p>}
          <div className="grid gap-3 sm:grid-cols-2" aria-live="polite">
            <Outcome title="Original photo" r={baseline} />
            {result ? <Outcome title="With these conditions" r={result} /> : (
              <div className="rounded-lg border border-dashed border-line p-4 text-ink-soft">Move a slider to see the new answer.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
