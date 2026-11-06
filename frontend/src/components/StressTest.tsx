import { useEffect, useRef, useState } from "react";
import { ApiError, analyze } from "../api";
import { NO_DEGRADE, degradeImage, type Degrade, type Prepared } from "../image";
import { stabilityKey } from "../live/stability";
import type { Result } from "../types";
import SweepChart, { sweepSummary, type SweepPoint } from "./SweepChart";
import { SectionTitle, pct } from "./bits";

const SLIDERS: { key: keyof Degrade; label: string; short: string; max: number }[] = [
  { key: "darkness", label: "Darkness", short: "darkness", max: 0.9 },
  { key: "haze", label: "Smoke or haze", short: "smoke", max: 0.7 },
  { key: "blur", label: "Blur and low resolution", short: "blur", max: 1 },
  { key: "occlusion", label: "Vehicle partly hidden", short: "occlusion", max: 0.5 },
];
const SWEEP_STEPS = 8;

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

function toPoint(level: number, r: Result): SweepPoint {
  return {
    level, status: r.status, headline: r.headline, hv: !!r.safety?.hv_warning,
    score: r.reliability?.score ?? 0,
    key: stabilityKey(r) ?? "no_vehicle",
  };
}

export default function StressTest({ photo, baseline }: { photo: Prepared; baseline: Result }) {
  const [d, setD] = useState<Degrade>(NO_DEGRADE);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sweep, setSweep] = useState<{ cond: (typeof SLIDERS)[number]; points: SweepPoint[]; running: boolean } | null>(null);
  const previewRef = useRef<string | null>(null);
  const sweepAbort = useRef<AbortController | null>(null);
  const touched = Object.values(d).some((v) => v > 0);

  function showPreview(url: string | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = url;
    setPreview(url);
  }

  useEffect(() => () => {
    sweepAbort.current?.abort();
    showPreview(null);
  }, []); // free the last preview image and stop any sweep on unmount

  useEffect(() => {
    if (sweepAbort.current) return; // a sweep is driving the sliders and asking the model itself
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

  async function runSweep(cond: (typeof SLIDERS)[number]) {
    sweepAbort.current?.abort();
    const ctrl = new AbortController();
    sweepAbort.current = ctrl;
    const points: SweepPoint[] = [toPoint(0, baseline)];
    setSweep({ cond, points: [...points], running: true });
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      for (let i = 1; i <= SWEEP_STEPS; i++) {
        const level = i / SWEEP_STEPS;
        const next = { ...NO_DEGRADE, [cond.key]: cond.max * level };
        setD(next);
        const img = await degradeImage(photo.url, next, baseline.detection?.box ?? null);
        if (ctrl.signal.aborted) return URL.revokeObjectURL(img.url);
        showPreview(img.url);
        const r = await analyze(img.blob, img.name, ctrl.signal);
        if (ctrl.signal.aborted) return;
        setResult(r);
        points.push(toPoint(level, r));
        setSweep({ cond, points: [...points], running: i < SWEEP_STEPS });
      }
    } catch (err: unknown) {
      if (!ctrl.signal.aborted) {
        setError(err instanceof ApiError ? err.message : "Couldn't finish the sweep.");
        setSweep((s) => s && { ...s, running: false });
      }
    } finally {
      if (sweepAbort.current === ctrl) {
        sweepAbort.current = null;
        setBusy(false);
      }
    }
  }

  function reset() {
    sweepAbort.current?.abort();
    sweepAbort.current = null;
    setSweep(null);
    setBusy(false);
    setD(NO_DEGRADE);
  }

  const running = !!sweep?.running;
  const summary = sweep && !sweep.running ? sweepSummary(sweep.points, sweep.cond.short) : "";

  return (
    <section>
      <SectionTitle
        title="Stress test"
        intro="Crash scenes are dark, smoky and cluttered. Push this photo towards those conditions and watch how the answer, and the safety advice, change."
      />
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-5 rounded-lg border border-line bg-sheet p-5">
          <div>
            <h3 className="font-display text-2xl font-semibold leading-tight">Automatic sweep</h3>
            <p className="mt-1 text-ink-soft">Runs this photo through the real model at {SWEEP_STEPS} rising levels and plots where the answer breaks.</p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {SLIDERS.map((s) => (
                <button key={s.key} onClick={() => runSweep(s)} disabled={running}
                  className="min-h-11 rounded-md bg-ink px-3 font-semibold capitalize text-paper enabled:hover:bg-draft disabled:opacity-45">
                  Sweep {s.short}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-4 border-t border-line pt-4">
            <h3 className="font-display text-2xl font-semibold leading-tight">Or set it by hand</h3>
            {SLIDERS.map((s) => (
              <label key={s.key} className="block">
                <span className="flex justify-between font-medium">
                  <span>{s.label}</span>
                  <span className="tabular text-ink-soft">{Math.round((d[s.key] / s.max) * 100)}%</span>
                </span>
                <input type="range" min={0} max={s.max} step={s.max / 20} value={d[s.key]} disabled={running}
                  onChange={(e) => { setSweep(null); setD({ ...d, [s.key]: +e.target.value }); }} className="mt-1 h-6 w-full accent-draft disabled:opacity-60" />
              </label>
            ))}
            <button onClick={reset} disabled={!touched && !sweep}
              className="min-h-11 rounded-md border border-ink/30 px-4 font-medium enabled:hover:border-ink disabled:opacity-40">
              {running ? "Stop" : "Reset"}
            </button>
          </div>
        </div>
        <div className="space-y-3">
          <div className="relative overflow-hidden rounded-lg border border-line bg-ink">
            <img src={preview ?? photo.url} alt="Photo with simulated conditions" className={`block w-full ${busy ? "opacity-80" : ""}`} />
            {busy && (
              <p className="tabular absolute left-3 top-3 rounded bg-paper px-2 py-1 text-sm font-medium">
                {sweep ? `Sweep ${sweep.points.length - 1} of ${SWEEP_STEPS}…` : "Re-analysing…"}
              </p>
            )}
          </div>
          {error && <p role="alert" className="rounded-md border border-ink px-3 py-2">{error}</p>}
          {sweep && <SweepChart points={sweep.points} total={SWEEP_STEPS + 1} condition={sweep.cond.short} />}
          {summary && <p className="font-display text-2xl font-semibold leading-tight" role="status">{summary}</p>}
          <div className="grid gap-3 sm:grid-cols-2" aria-live="polite">
            <Outcome title="Original photo" r={baseline} />
            {result ? <Outcome title="With these conditions" r={result} /> : (
              <div className="rounded-lg border border-dashed border-line p-4 text-ink-soft">Run a sweep or move a slider to see the new answer.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
