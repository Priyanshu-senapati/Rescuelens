import { useEffect, useRef, useState } from "react";
import { ApiError, analyze } from "./api";
import Compare from "./components/Compare";
import PhotoBox from "./components/PhotoBox";
import Pipeline from "./components/Pipeline";
import Report from "./components/Report";
import ResultPanel from "./components/ResultPanel";
import StressTest from "./components/StressTest";
import Uploader from "./components/Uploader";
import { prepareImage, type Prepared } from "./image";
import type { Result } from "./types";

type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: Result }
  | { kind: "error"; message: string };

type View = "identify" | "report";

const STAGES = ["Detecting the vehicle", "Identifying make, model and year", "Checking what is safe to conclude"];

function useElapsed(active: boolean): number {
  const [s, setS] = useState(0);
  useEffect(() => {
    if (!active) return setS(0);
    const t0 = Date.now();
    const id = setInterval(() => setS((Date.now() - t0) / 1000), 250);
    return () => clearInterval(id);
  }, [active]);
  return s;
}

export default function App() {
  const [view, setView] = useState<View>("identify");
  const [photo, setPhoto] = useState<Prepared | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const abort = useRef<AbortController | null>(null);
  const elapsed = useElapsed(phase.kind === "loading");

  async function run(p: Prepared) {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setPhase({ kind: "loading" });
    try {
      const result = await analyze(p.blob, p.name, ctrl.signal);
      setPhase({ kind: "done", result });
    } catch (err: unknown) {
      if (ctrl.signal.aborted) return;
      setPhase({ kind: "error", message: err instanceof ApiError ? err.message : "Something went wrong. Try again." });
    }
  }

  async function pick(file: Blob, name: string) {
    const p = await prepareImage(file, name);
    if (photo) URL.revokeObjectURL(photo.url);
    setPhoto(p);
    void run(p);
  }

  function reset() {
    abort.current?.abort();
    if (photo) URL.revokeObjectURL(photo.url);
    setPhoto(null);
    setPhase({ kind: "idle" });
  }

  const result = phase.kind === "done" ? phase.result : null;
  const stage = STAGES[Math.min(STAGES.length - 1, Math.floor(elapsed / 1.2))];

  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-ink text-paper">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
          <button onClick={reset} className="flex items-center gap-3 rounded text-left" aria-label="RescueLens, start again">
            <svg width="36" height="36" viewBox="0 0 32 32" aria-hidden="true">
              <rect width="32" height="32" rx="6" fill="#26384a" />
              <path d="M6 20h20l-3.5-7.5H9.5z" fill="none" stroke="#e8651a" strokeWidth="2.4" strokeLinejoin="round" />
              <circle cx="11" cy="22" r="2.2" fill="#e9edf0" />
              <circle cx="21" cy="22" r="2.2" fill="#e9edf0" />
            </svg>
            <span>
              <span className="block font-display text-2xl font-bold leading-none tracking-wide">RescueLens</span>
              <span className="block text-sm text-paper/70">Vehicle identification for crash response</span>
            </span>
          </button>
          <nav className="ml-auto flex rounded-md bg-white/10 p-1" aria-label="Sections">
            {(["identify", "report"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setView(v)}
                aria-current={view === v ? "page" : undefined}
                className={`min-h-11 rounded px-4 text-sm font-medium ${view === v ? "bg-paper text-ink" : "text-paper/80 hover:text-paper"}`}
              >
                {v === "identify" ? "Identify" : "Model report"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16 pt-6 sm:px-6">
        {view === "report" ? (
          <Report />
        ) : !photo ? (
          <Uploader onPick={pick} />
        ) : (
          <>
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
              <div className="order-2 space-y-3 lg:order-1 lg:sticky lg:top-4 lg:self-start">
                <PhotoBox url={photo.url} result={result} />
                <div className="flex flex-wrap gap-2">
                  <button onClick={reset} className="min-h-11 rounded-md border border-ink/30 bg-sheet px-4 font-medium hover:border-ink">
                    Check another vehicle
                  </button>
                  {phase.kind === "error" && (
                    <button onClick={() => run(photo)} className="min-h-11 rounded-md bg-ink px-4 font-medium text-paper">
                      Try again
                    </button>
                  )}
                </div>
              </div>
              {/* on phones the verdict comes first: a responder needs the answer before the picture */}
              <section className="order-1 lg:order-2" aria-live="polite" aria-busy={phase.kind === "loading"}>
                {phase.kind === "loading" && (
                  <div className="rounded-lg border border-line bg-sheet p-6">
                    <p className="font-display text-2xl font-semibold">{stage}…</p>
                    <div className="mt-4 h-1.5 overflow-hidden rounded bg-draft-soft">
                      <div className="h-full w-1/3 animate-pulse rounded bg-draft" />
                    </div>
                    {elapsed > 6 && (
                      <p className="mt-4 text-ink-soft">
                        The server may be waking up after a quiet period. The first answer can take up to half a minute.
                      </p>
                    )}
                  </div>
                )}
                {phase.kind === "error" && (
                  <div role="alert" className="rounded-lg border-2 border-ink bg-sheet p-6">
                    <p className="font-display text-2xl font-semibold">Couldn't analyse this photo</p>
                    <p className="mt-2 text-ink-soft">{phase.message}</p>
                  </div>
                )}
                {result && <ResultPanel result={result} />}
              </section>
            </div>
            {result && result.status !== "no_vehicle" && (
              <div className="mt-10 space-y-10">
                <Pipeline steps={result.steps} />
                <StressTest photo={photo} baseline={result} />
                {result.comparison && <Compare comparison={result.comparison} />}
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
