import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, analyze } from "./api";
import Compare from "./components/Compare";
import LiveCamera from "./components/LiveCamera";
import PhotoBox from "./components/PhotoBox";
import Pipeline from "./components/Pipeline";
import Report from "./components/Report";
import ResultPanel from "./components/ResultPanel";
import StressTest from "./components/StressTest";
import Uploader from "./components/Uploader";
import { prepareImage, type Prepared } from "./image";
import type { Result, ViewSummary } from "./types";

type Phase =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "done"; result: Result }
  | { kind: "error"; message: string };

type View = "identify" | "report";

/** What one photo said before the photos were combined. */
function alone(v: ViewSummary, threshold: number): string {
  if (!v.vehicle_found || !v.top) return "no vehicle found";
  return `${v.top} ${Math.round((v.prob ?? 0) * 100)}%${(v.unknown_score ?? 0) > threshold ? ", not recognised" : ""}`;
}

const MAX_EXTRA = 2; // server takes 3 photos of one vehicle
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
  const [live, setLive] = useState(false);
  const [extras, setExtras] = useState<Prepared[]>([]); // other sides of the same vehicle (rear, undamaged side)
  const sideInput = useRef<HTMLInputElement>(null);
  const extrasNow = useRef(extras); // addSide awaits a decode: read the photos as they are then, not at click time
  extrasNow.current = extras;
  const abort = useRef<AbortController | null>(null);
  const pickId = useRef(0); // discards a slow photo decode that finishes after a newer pick or a reset
  const elapsed = useElapsed(phase.kind === "loading");

  async function run(p: Prepared, more: Prepared[] = extras) {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setPhase({ kind: "loading" });
    try {
      const result = await analyze(p.blob, p.name, ctrl.signal, more);
      if (!ctrl.signal.aborted) setPhase({ kind: "done", result });
    } catch (err: unknown) {
      if (ctrl.signal.aborted) return;
      setPhase({ kind: "error", message: err instanceof ApiError ? err.message : "Something went wrong. Try again." });
    }
  }

  function swapPhoto(next: Prepared | null) {
    setPhoto((old) => {
      if (old) URL.revokeObjectURL(old.url);
      return next;
    });
  }

  function swapExtras(next: Prepared[]) {
    setExtras((old) => {
      old.filter((o) => !next.includes(o)).forEach((o) => URL.revokeObjectURL(o.url));
      return next;
    });
  }

  async function pick(file: Blob, name: string) {
    const id = ++pickId.current;
    const p = await prepareImage(file, name);
    if (id !== pickId.current) return URL.revokeObjectURL(p.url);
    swapPhoto(p);
    swapExtras([]);
    void run(p, []);
  }

  async function addSide(file: Blob, name: string) {
    const id = pickId.current; // a reset or a new vehicle meanwhile discards this photo
    const p = await prepareImage(file, name);
    const now = extrasNow.current;
    if (id !== pickId.current || !photo || now.length >= MAX_EXTRA) return URL.revokeObjectURL(p.url);
    const next = [...now, p];
    swapExtras(next);
    void run(photo, next);
  }

  function removeSide(x: Prepared) {
    const next = extras.filter((o) => o !== x);
    swapExtras(next);
    if (photo) void run(photo, next);
  }

  function reset() {
    pickId.current++;
    abort.current?.abort();
    swapPhoto(null);
    swapExtras([]);
    setPhase({ kind: "idle" });
  }

  function acceptLive(p: Prepared, r: Result) {
    pickId.current++;
    abort.current?.abort();
    swapPhoto(p);
    swapExtras([]);
    setPhase({ kind: "done", result: r });
    setLive(false);
    window.scrollTo(0, 0);
  }

  const closeLive = useCallback(() => setLive(false), []);

  const result = phase.kind === "done" ? phase.result : null;
  const canAddSide = phase.kind !== "loading" && extras.length < MAX_EXTRA && !result?.offline;
  const clash = result?.headline.startsWith("These photos seem to show different vehicles"); // remove one, don't add
  const needsSide = result && !clash && (result.status === "unknown" || result.status === "ambiguous");
  const stageIndex = Math.min(STAGES.length - 1, Math.floor(elapsed / 1.2));

  return (
    <div className="min-h-dvh">
      <header className="border-b border-line bg-ink text-paper">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3 sm:px-6">
          <button onClick={reset} className="flex items-center gap-3 rounded text-left" aria-label="RescueLens, start again">
            <img src="/logo-mark.png" width="48" height="48" alt="" className="rounded-md" />
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
          <Uploader onPick={pick} onLive={() => setLive(true)} />
        ) : (
          <>
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
              <div className="order-2 space-y-3 lg:order-1 lg:sticky lg:top-4 lg:self-start">
                <PhotoBox url={photo.url} width={photo.width} height={photo.height} result={result} loading={phase.kind === "loading"} />
                {extras.length > 0 && result?.views?.[0] && (
                  <p className="text-sm text-ink-soft">
                    Photo 1 alone: {alone(result.views[0], result.unknown?.threshold ?? 1)}
                  </p>
                )}
                {extras.length > 0 && (
                  <ul className="flex flex-wrap gap-3" aria-label="Other sides of this vehicle">
                    {extras.map((x, i) => {
                      const v = result?.views?.[i + 1];
                      return (
                        <li key={x.url} className="relative w-32">
                          <img src={x.url} alt={`Photo ${i + 2} of this vehicle`} className="h-20 w-32 rounded-md border border-line object-cover" />
                          <button
                            onClick={() => removeSide(x)}
                            aria-label={`Remove photo ${i + 2}`}
                            className="absolute -right-2 -top-2 grid h-8 w-8 place-items-center rounded-full bg-ink text-lg leading-none text-paper hover:bg-draft"
                          >
                            ×
                          </button>
                          {v && (
                            <p className="mt-1 text-xs leading-snug text-ink-soft">
                              Photo {i + 2} alone: {alone(v, result?.unknown?.threshold ?? 1)}
                            </p>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                )}
                {extras.length > 0 && result?.offline && (
                  <p className="text-sm text-ink-soft">Answered offline from photo 1 only: the other photos need the server.</p>
                )}
                <input
                  ref={sideInput}
                  type="file"
                  accept="image/*,.heic,.heif"
                  hidden
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = "";
                    if (f) void addSide(f, f.name);
                  }}
                />
                <div className="flex flex-wrap gap-2">
                  {canAddSide && !needsSide && result && (
                    <button onClick={() => sideInput.current?.click()} className="min-h-11 rounded-md border border-ink/30 bg-sheet px-4 font-medium hover:border-ink">
                      Add another side
                    </button>
                  )}
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
                    <ol className="space-y-2">
                      {STAGES.map((s, i) => (
                        <li key={s} className={`flex items-center gap-3 font-display text-2xl font-semibold ${i > stageIndex ? "opacity-35" : ""}`}>
                          <span aria-hidden="true" className={`h-3 w-3 shrink-0 rounded-full ${i < stageIndex ? "bg-ok" : i === stageIndex ? "animate-pulse bg-draft" : "bg-line"}`} />
                          {s}
                        </li>
                      ))}
                    </ol>
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
                {needsSide && canAddSide && (
                  <div className="mt-4 rounded-lg border-2 border-dashed border-ink/30 bg-sheet p-5">
                    <p className="font-display text-xl font-semibold">Front crushed or badge hidden?</p>
                    <p className="mt-1 text-ink-soft">
                      Photograph another side of the same vehicle, such as the rear or an undamaged side. RescueLens combines
                      the photos into one answer.
                    </p>
                    <button onClick={() => sideInput.current?.click()} className="mt-3 min-h-11 rounded-md bg-ink px-4 font-medium text-paper hover:bg-draft">
                      Add another side
                    </button>
                  </div>
                )}
              </section>
            </div>
            {result && result.status !== "no_vehicle" && (
              <div className="mt-10 space-y-10">
                {result.steps.length > 0 && <Pipeline steps={result.steps} />}
                {extras.length === 0 && <StressTest photo={photo} baseline={result} />}
                {result.comparison && <Compare comparison={result.comparison} />}
              </div>
            )}
          </>
        )}
      </main>
      {live && <LiveCamera onClose={closeLive} onUsePhoto={closeLive} onAccept={acceptLive} />}
    </div>
  );
}
