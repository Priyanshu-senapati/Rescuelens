import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, analyze } from "../api";
import { captureFrame, type Prepared } from "../image";
import { EdgeRenderer, type Box } from "../live/edges";
import { LOCK_FRAMES, START, advance, type ScanState } from "../live/stability";
import type { Result } from "../types";
import { HazardLocation, STATUS, pct } from "./bits";

interface Props {
  onClose: () => void;
  onUsePhoto: () => void;
  onAccept: (photo: Prepared, result: Result) => void;
}

type Phase =
  | { kind: "starting" }
  | { kind: "scanning" }
  | { kind: "locked"; result: Result; frame: Blob; url: string }
  | { kind: "error"; title: string; detail: string; retry: boolean };

const SCAN_MS = 1500; // one frame every 1.5 s, and never two requests at once
const MAX_FAILURES = 3;
const IDLE_BOX: Box = [0.16, 0.26, 0.84, 0.74]; // where the ticks wait until a vehicle is found

const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function cameraError(err: unknown): Extract<Phase, { kind: "error" }> {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError" || name === "SecurityError")
    return { kind: "error", title: "Camera access is off", detail: "Allow the camera for this site in your browser settings, or take a single photo instead.", retry: true };
  if (name === "NotFoundError" || name === "OverconstrainedError")
    return { kind: "error", title: "No camera found", detail: "This device has no camera RescueLens can use. Choose a photo instead.", retry: false };
  if (name === "NotReadableError" || name === "AbortError")
    return { kind: "error", title: "The camera is busy", detail: "Another app is using the camera. Close it and try again, or take a single photo instead.", retry: true };
  return { kind: "error", title: "Live camera isn't available here", detail: "This browser only opens the camera on a secure (https) page. Take a single photo instead.", retry: false };
}

function normalise(r: Result): Box | null {
  if (!r.detection || !r.image.width || !r.image.height) return null;
  const [a, b, c, d] = r.detection.box;
  return [a / r.image.width, b / r.image.height, c / r.image.width, d / r.image.height];
}

function ticksPath([x1, y1, x2, y2]: Box, W: number, H: number): string {
  const [l, t, r, b] = [x1 * W, y1 * H, x2 * W, y2 * H];
  const n = Math.min(r - l, b - t) * 0.16;
  return `M${l},${t + n}V${t}H${l + n}M${r - n},${t}H${r}V${t + n}M${r},${b - n}V${b}H${r - n}M${l + n},${b}H${l}V${b - n}`;
}

function isDanger(r: Result | null) {
  return r?.safety?.hv_warning && r.safety.level === "danger";
}

export default function LiveCamera({ onClose, onUsePhoto, onAccept }: Props) {
  const [attempt, setAttempt] = useState(0);
  const [phase, setPhase] = useState<Phase>({ kind: "starting" });
  const [scan, setScan] = useState<ScanState>(START);
  const [last, setLast] = useState<Result | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [frameSize, setFrameSize] = useState({ w: 16, h: 9 });

  const video = useRef<HTMLVideoElement>(null);
  const edges = useRef<HTMLCanvasElement>(null);
  const ticks = useRef<SVGPathElement>(null);
  const frozen = useRef<HTMLImageElement>(null);
  const target = useRef<Box | null>(null);
  const hvTint = useRef(false);
  const handedOff = useRef(false);
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  // Camera: opened for each attempt, always released on the way out.
  useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    setPhase({ kind: "starting" });
    setScan(START);
    setLast(null);
    setLatency(null);
    target.current = null;

    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("no mediaDevices");
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        const v = video.current!;
        v.srcObject = stream;
        await v.play();
        if (cancelled) return;
        setFrameSize({ w: v.videoWidth || 16, h: v.videoHeight || 9 });
        setPhase({ kind: "scanning" });
      } catch (err) {
        if (!cancelled) setPhase(cameraError(err));
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [attempt]);

  // Frames: capture, ask the API, update the streak, repeat until it locks.
  useEffect(() => {
    if (phase.kind !== "scanning") return;
    let alive = true;
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    (async () => {
      let s = START;
      let failures = 0;
      while (alive) {
        const t0 = performance.now();
        const v = video.current;
        if (v && v.videoWidth && !document.hidden) {
          try {
            const frame = await captureFrame(v);
            const r = await analyze(frame, "camera-frame.jpg", ctrl.signal);
            if (!alive) return;
            failures = 0;
            s = advance(s, r);
            target.current = normalise(r);
            hvTint.current = !!isDanger(r);
            setLast(r);
            setScan(s);
            setLatency(Math.round(performance.now() - t0));
            if (s.locked) {
              navigator.vibrate?.([40, 30, 90]);
              setPhase({ kind: "locked", result: r, frame, url: URL.createObjectURL(frame) });
              return;
            }
          } catch (err) {
            if (!alive) return;
            if (++failures >= MAX_FAILURES) {
              const detail = err instanceof ApiError ? err.message : "The last few frames couldn't be analysed.";
              setPhase({ kind: "error", title: "Lost contact with the server", detail, retry: true });
              return;
            }
          }
        }
        const wait = Math.max(0, SCAN_MS - (performance.now() - t0));
        await new Promise((resolve) => (timer = setTimeout(resolve, wait)));
      }
    })();

    return () => {
      alive = false;
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [phase.kind]);

  // Drawing: the edge map at about 12 fps and the sprung ticks at display rate, outside React.
  useEffect(() => {
    const canvas = edges.current;
    if (!canvas || phase.kind !== "scanning") return;
    const still = reducedMotion();
    const renderer = new EdgeRenderer(canvas);
    let raf = 0;
    let lastEdges = 0;
    let lastTick = performance.now();
    const cur: Box = [...IDLE_BOX];

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop);
      const v = video.current;
      if (!v || document.hidden) return;
      if (!still && now - lastEdges > 80) {
        lastEdges = now;
        renderer.draw(v, target.current, hvTint.current);
      }
      const goal = target.current ?? IDLE_BOX;
      const k = still ? 1 : 1 - Math.exp(-(now - lastTick) / 110);
      lastTick = now;
      for (let i = 0; i < 4; i++) cur[i] += (goal[i] - cur[i]) * k;
      ticks.current?.setAttribute("d", ticksPath(cur, frameSize.w, frameSize.h));
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [phase.kind, frameSize]);

  // Lock: draw the final blueprint from the exact frame the answer came from, then snap the ticks onto it.
  const onFrozenLoad = useCallback(() => {
    const canvas = edges.current;
    const img = frozen.current;
    const p = phaseRef.current;
    if (!canvas || !img || p.kind !== "locked") return;
    const box = normalise(p.result);
    new EdgeRenderer(canvas).draw(img, box, !!isDanger(p.result), 1.5);
    ticks.current?.setAttribute("d", ticksPath(box ?? IDLE_BOX, frameSize.w, frameSize.h));
  }, [frameSize]);

  useEffect(() => {
    if (phase.kind !== "locked") return;
    video.current?.pause();
    (video.current?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
    const { url } = phase;
    return () => {
      if (!handedOff.current) URL.revokeObjectURL(url);
    };
  }, [phase]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [onClose]);

  const locked = phase.kind === "locked" ? phase : null;
  const reading = locked?.result ?? last;
  const danger = !!isDanger(reading);
  const showHv = !locked && scan.hvHold > 0;
  const colour = danger ? "#ff8a3d" : "#7cc0ff";
  const hint =
    phase.kind === "starting" ? "Opening the camera…"
    : !last ? "Point the camera at the vehicle"
    : last.status === "no_vehicle" ? "No vehicle in view yet"
    : scan.streak >= 2 ? "Hold steady, nearly locked"
    : "Hold steady";

  return (
    <div role="dialog" aria-modal="true" aria-label="Live camera scan" className="fixed inset-0 z-50 overflow-hidden bg-[#0d141b] text-paper">
      <video ref={video} muted playsInline autoPlay aria-hidden="true" className="absolute inset-0 h-full w-full object-cover"
        onResize={(e) => setFrameSize({ w: e.currentTarget.videoWidth || 16, h: e.currentTarget.videoHeight || 9 })} />
      {locked && <img ref={frozen} src={locked.url} alt="The frame the answer locked on" onLoad={onFrozenLoad} className="absolute inset-0 h-full w-full object-cover" />}
      <canvas ref={edges} aria-hidden="true" className={`absolute inset-0 h-full w-full object-cover ${locked ? "opacity-100" : "opacity-90"}`} />

      <svg viewBox={`0 0 ${frameSize.w} ${frameSize.h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true" className="absolute inset-0 h-full w-full">
        <path ref={ticks} fill="none" stroke={colour} strokeWidth={locked ? 6 : 4} strokeLinecap="square" vectorEffect="non-scaling-stroke"
          className={locked ? "ticks-snap" : target.current ? "" : "opacity-60"} />
      </svg>

      {locked && <div key="flash" aria-hidden="true" className="lock-flash absolute inset-0 bg-white" />}

      {/* Top bar: the way out is always in the same place. */}
      <div className="absolute inset-x-0 top-0 flex items-start justify-between gap-3 bg-gradient-to-b from-black/70 to-transparent p-4 pb-10">
        <p className="font-display text-2xl font-semibold leading-tight" aria-hidden={!!locked}>{locked ? "Locked" : "Live scan"}</p>
        <button onClick={onClose} aria-label="Close the camera" className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md bg-black/50 hover:bg-black/80">
          <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M5 5l12 12M17 5L5 17" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" /></svg>
        </button>
      </div>

      {/* The warning never waits for the lock: it shows on the first frame that raises it. */}
      {showHv && last?.safety && (
        <div role="alert" className="absolute inset-x-0 top-[4.5rem] mx-4 overflow-hidden rounded-md bg-black/80 sm:mx-auto sm:max-w-xl">
          <div className={danger ? "hazard-stripes h-2" : "h-2 bg-warn"} aria-hidden="true" />
          <p className="px-4 py-3 text-lg font-semibold leading-snug">{last.safety.message}</p>
        </div>
      )}

      {!locked && phase.kind !== "error" && (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 via-black/60 to-transparent px-4 pb-[max(1.25rem,env(safe-area-inset-bottom))] pt-16">
          <div className="mx-auto max-w-xl">
            <p className="text-lg font-medium" role="status">{hint}</p>
            {last && last.status !== "no_vehicle" && (
              <p className="mt-1 line-clamp-2 font-display text-3xl font-bold leading-[1.05] sm:text-4xl">{last.headline}</p>
            )}
            <div className="mt-4 flex items-center gap-2" role="meter" aria-label="Frames agreeing" aria-valuemin={0} aria-valuemax={LOCK_FRAMES} aria-valuenow={Math.min(scan.streak, LOCK_FRAMES)}>
              {Array.from({ length: LOCK_FRAMES }, (_, i) => (
                <span key={i} className={`h-2 flex-1 rounded-sm transition-colors duration-200 ${i < scan.streak ? (danger ? "bg-hv" : "bg-[#7cc0ff]") : "bg-white/25"}`} />
              ))}
            </div>
            <p className="tabular mt-2 text-sm text-paper/70">
              {scan.frames > 0 ? `Frame ${scan.frames}${latency !== null ? `, answered in ${latency} ms` : ""}` : "Locks when 3 frames in a row agree"}
              {last?.mock ? ". Sample answers: the trained models aren't loaded." : ""}
            </p>
          </div>
        </div>
      )}

      {locked && (
        <div className="plate-rise absolute inset-x-0 bottom-0 max-h-[78dvh] overflow-y-auto rounded-t-xl bg-sheet text-ink shadow-[0_-12px_40px_rgb(0_0_0/0.45)] sm:inset-x-auto sm:bottom-6 sm:left-6 sm:max-w-lg sm:rounded-xl">
          {locked.result.safety?.hv_warning && <div className={`stripe-wipe h-3 ${danger ? "hazard-stripes" : "bg-warn"}`} aria-hidden="true" />}
          <div className="p-5 sm:p-6" role="status">
            <p className={`inline-block rounded-sm px-2.5 py-1 font-semibold text-white ${STATUS[locked.result.status].chip}`}>{STATUS[locked.result.status].title}</p>
            <p className="mt-2 font-display text-4xl font-bold leading-[1.05] text-balance">{locked.result.headline}</p>
            {locked.result.prediction && locked.result.status !== "unknown" && (
              <p className="tabular mt-1 text-ink-soft">Top answer {pct(locked.result.prediction.confidence)} confident, held for {LOCK_FRAMES} frames in a row</p>
            )}
            {locked.result.safety && locked.result.safety.level !== "info" && (
              <div className="mt-3">
                <p className={`font-display text-2xl font-bold leading-tight ${danger ? "text-hv" : "text-warn"}`}>{danger ? "High voltage" : "Powertrain unknown"}</p>
                <p className="mt-0.5 font-medium">{locked.result.safety.message}</p>
                <HazardLocation s={locked.result.safety} />
              </div>
            )}
            <div className="mt-5 flex flex-wrap gap-2">
              <button
                onClick={() => {
                  handedOff.current = true;
                  onAccept({ blob: locked.frame, url: locked.url, name: "camera-frame.jpg" }, locked.result);
                }}
                className="min-h-12 flex-1 rounded-md bg-ink px-5 text-lg font-semibold text-paper hover:bg-draft"
              >
                See the full answer
              </button>
              <button onClick={() => setAttempt((a) => a + 1)} className="min-h-12 rounded-md border-2 border-ink px-5 text-lg font-semibold hover:bg-paper">
                Scan again
              </button>
            </div>
            <p className="mt-3 text-sm text-ink-soft">{locked.result.disclaimer}</p>
          </div>
        </div>
      )}

      {phase.kind === "error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#0d141b]/90 p-4">
          <div role="alert" className="w-full max-w-md rounded-lg bg-sheet p-6 text-ink">
            <h2 className="font-display text-4xl font-bold leading-none text-balance">{phase.title}</h2>
            <p className="mt-3 text-lg leading-relaxed text-ink-soft">{phase.detail}</p>
            <div className="mt-5 flex flex-wrap gap-2">
              <button onClick={onUsePhoto} className="min-h-12 flex-1 rounded-md bg-ink px-5 text-lg font-semibold text-paper hover:bg-draft">Use a photo instead</button>
              {phase.retry && (
                <button onClick={() => setAttempt((a) => a + 1)} className="min-h-12 rounded-md border-2 border-ink px-5 text-lg font-semibold hover:bg-paper">Try again</button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
