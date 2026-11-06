import { useEffect, useState } from "react";
import type { Step } from "../types";
import { SectionTitle } from "./bits";

export default function Pipeline({ steps }: { steps: Step[] }) {
  const [active, setActive] = useState(0);
  const [split, setSplit] = useState(50);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setActive((a) => {
        if (a + 1 >= steps.length) {
          setPlaying(false);
          return a;
        }
        return a + 1;
      });
    }, 1100);
    return () => clearInterval(id);
  }, [playing, steps.length]);

  if (steps.length === 0) return null;
  const original = steps.find((s) => s.key === "original");
  const enhanced = steps.find((s) => s.key === "enhanced");
  const current = steps[active];

  return (
    <section>
      <SectionTitle
        title="How it saw the photo"
        intro="The classical computer-vision stages that run alongside the neural network: they find the vehicle's outline, wheels and colours, and measure whether the photo is good enough to trust."
      />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <figure className="self-start overflow-hidden rounded-lg border border-line bg-ink">
          <img src={current.image} alt={`${current.title}: ${current.caption}`} className="mx-auto block max-h-[26rem] w-auto" />
          <figcaption className="flex flex-wrap items-center justify-between gap-2 bg-sheet px-4 py-3">
            <span><span className="font-semibold">{active + 1}. {current.title}.</span> <span className="text-ink-soft">{current.caption}</span></span>
            <button onClick={() => { if (!playing) setActive(0); setPlaying(!playing); }} className="min-h-11 shrink-0 rounded-md bg-ink px-4 font-semibold text-paper hover:bg-draft">
              {playing ? "Pause" : "Play every step"}
            </button>
          </figcaption>
        </figure>

        {original && enhanced && (
          <div>
            <h3 className="font-display text-2xl font-semibold">Before and after contrast boost</h3>
            <p className="mb-3 text-ink-soft">Drag to compare. The boost reveals detail for people and is used to retry detection in dark photos.</p>
            <div className="relative overflow-hidden rounded-lg border border-line">
              <img src={enhanced.image} alt="Photo after contrast boost" className="block w-full" />
              <img src={original.image} alt="Photo before contrast boost" className="absolute inset-0 block h-full w-full"
                style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }} />
              <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-paper" style={{ left: `${split}%` }} />
            </div>
            <label className="mt-2 block">
              <span className="sr-only">Comparison position</span>
              <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(+e.target.value)} className="h-6 w-full accent-draft" />
            </label>
          </div>
        )}
      </div>

      <ol className="mt-4 flex gap-3 overflow-x-auto pb-2">
        {steps.map((s, i) => (
          <li key={s.key} className="shrink-0">
            <button onClick={() => { setPlaying(false); setActive(i); }} aria-pressed={i === active}
              className={`w-36 overflow-hidden rounded-md border-2 bg-sheet text-left ${i === active ? "border-draft" : "border-transparent hover:border-line"}`}>
              <img src={s.image} alt="" className="aspect-[4/3] w-full bg-ink object-cover" />
              <span className="block px-2 py-1.5 text-sm"><span className="font-semibold">{i + 1}.</span> {s.title}</span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}
