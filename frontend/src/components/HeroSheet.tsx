import { useEffect, useState } from "react";
import { HAZARDS, RescueDiagram } from "./RescueSheet";

const CARS: { label: string; short: string }[] = [
  { label: "Tesla Model S Sedan 2012", short: "Tesla" },
  { label: "Chevrolet Malibu Hybrid Sedan 2010", short: "Malibu" },
  { label: "Fisker Karma Sedan 2012", short: "Karma" },
  { label: "Chevrolet Tahoe Hybrid SUV 2012", short: "Tahoe" },
  { label: "BMW ActiveHybrid 5 Sedan 2012", short: "BMW" },
  { label: "Nissan Leaf Hatchback 2012", short: "Leaf" },
  { label: "Chevrolet Silverado 1500 Hybrid Crew Cab 2012", short: "Silverado" },
  { label: "Hyundai Sonata Hybrid Sedan 2012", short: "Sonata" },
];

/** The idea in one look: similar bodies, with the danger in a different place in each. Taken from the official guides. */
export default function HeroSheet() {
  const [index, setIndex] = useState(0);
  const [auto, setAuto] = useState(() => !window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => setIndex((i) => (i + 1) % CARS.length), 4200);
    return () => clearInterval(id);
  }, [auto]);

  const car = CARS[index];
  const hazard = HAZARDS[car.label];
  return (
    <section aria-label="Where the high voltage is on different models" className="rounded-xl bg-ink p-5 text-paper sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4">
        <h2 className="font-display text-3xl font-bold leading-tight">Same shape. Different danger.</h2>
        {auto && (
          <button onClick={() => setAuto(false)} className="inline-flex min-h-6 items-center text-sm text-paper/75 underline underline-offset-2 hover:text-paper">Pause</button>
        )}
      </div>
      {hazard.schematic && <RescueDiagram vehicle={car.label} schematic={hazard.schematic} className="mt-2 text-paper" />}
      <p className="min-h-[3.2em] leading-snug"><span className="font-semibold">{car.label}:</span> <span className="text-paper/80">{hazard.battery}</span></p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {CARS.map((c, i) => (
          <button key={c.label} onClick={() => { setAuto(false); setIndex(i); }} aria-pressed={i === index}
            className={`min-h-8 rounded px-2.5 text-sm font-medium ${i === index ? "bg-paper text-ink" : "bg-white/10 text-paper/85 hover:bg-white/20"}`}>
            {c.short}
          </button>
        ))}
      </div>
    </section>
  );
}
