import hazardFile from "../../../data/hazards.json";
import type { Safety } from "../types";

// A rescue-sheet style side view. Where the battery zone sits comes from "schematic" in data/hazards.json, which was read from
// each manufacturer's guide (docs/hazard-sources.md). Positions are schematic: this shows the region, never a place to cut.

type Body = "sedan" | "hatch" | "suv" | "pickup";
type Zone = "underfloor" | "under_rear_seat" | "behind_rear_seat" | "boot" | "tunnel";

export interface Schematic {
  body: Body;
  zone: Zone;
  cable: "none" | "underbody";
  cable_colour?: "orange" | "blue";
}

interface Hazard {
  battery?: string | null;
  hv_cables?: string | null;
  schematic?: Schematic;
}

export const HAZARDS = hazardFile.high_voltage as unknown as Record<string, Hazard>;

const BODIES: Record<Body, { outline: string; glass: string[]; wheels: [number, number] }> = {
  sedan: {
    outline: "M40 150C40 130 60 122 110 116L175 108C215 78 265 62 330 60C395 60 440 72 480 100L560 110C590 114 604 128 604 150V160H526A36 36 0 0 0 454 160H176A36 36 0 0 0 104 160H40Z",
    glass: ["M192 107C226 84 268 72 322 71V107Z", "M334 71C385 72 424 84 460 106L334 107Z"],
    wheels: [140, 490],
  },
  hatch: {
    outline: "M40 150C42 128 70 118 120 110L185 98C220 70 270 56 340 54C410 54 470 66 520 98C548 114 572 122 580 150V160H516A36 36 0 0 0 444 160H176A36 36 0 0 0 104 160H40Z",
    glass: ["M202 97C232 76 272 66 326 65V97Z", "M338 65C392 66 440 76 490 97L338 97Z"],
    wheels: [140, 480],
  },
  suv: {
    outline: "M36 150C38 124 60 112 110 106L165 98C190 62 225 46 270 44H510C545 46 565 80 590 110C596 124 598 140 598 160H526A36 36 0 0 0 454 160H176A36 36 0 0 0 104 160H36Z",
    glass: ["M182 97C202 70 232 57 274 56H344V97Z", "M356 56H440V97H356Z", "M452 56H506C526 58 544 78 560 97H452Z"],
    wheels: [140, 490],
  },
  pickup: {
    outline: "M36 150C38 124 60 114 110 108L170 100C195 62 225 48 260 46H395C405 46 412 56 412 70V104H600L604 160H536A36 36 0 0 0 464 160H176A36 36 0 0 0 104 160H36Z",
    glass: ["M186 99C206 72 232 59 264 58H304V99Z", "M316 58H394C399 58 400 62 400 70V99H316Z"],
    wheels: [140, 500],
  },
};

/** Battery rectangles per zone: x, y, width, height. */
function zoneRects(zone: Zone, body: Body): [number, number, number, number][] {
  const rearSeat = body === "pickup" ? 290 : 320;
  switch (zone) {
    case "underfloor": return [[178, 148, 274, 20]];
    case "under_rear_seat": return [[rearSeat, 138, 100, 30]];
    case "behind_rear_seat": return [[420, 104, 52, 56]];
    case "boot": return [[458, 126, 96, 30]];
    case "tunnel": return [[186, 142, 250, 24], [424, 108, 44, 58]];
  }
}

const BLUE = "#3f9be0";

export function RescueDiagram({ vehicle, schematic, className = "" }: { vehicle: string; schematic: Schematic; className?: string }) {
  const b = BODIES[schematic.body];
  const rects = zoneRects(schematic.zone, schematic.body);
  const [zx, zy, zw] = rects[rects.length - 1];
  const hasCable = schematic.cable === "underbody";
  const cable = schematic.cable_colour === "blue" ? BLUE : "var(--color-hv)";
  const patternId = `hazard-${schematic.body}-${schematic.zone}`;
  const where = schematic.zone.replaceAll("_", " ");

  return (
    <svg key={vehicle} viewBox="0 0 640 214" role="img" aria-label={`Side view of a ${vehicle} with the high-voltage battery marked: ${where}`} className={`block w-full ${className}`}>
      <defs>
        <pattern id={patternId} width="11" height="11" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="11" height="11" fill="var(--color-hv)" />
          <rect width="5" height="11" fill="#1c2733" />
        </pattern>
      </defs>
      <path d="M14 197H626" stroke="currentColor" strokeOpacity="0.35" strokeWidth="1.5" className="sheet-draw" pathLength={1} />
      {[0, 1].map((i) => (
        <g key={i} className="sheet-draw-late">
          <circle cx={b.wheels[i]} cy={165} r={31} fill="none" stroke="currentColor" strokeWidth="2.5" />
          <circle cx={b.wheels[i]} cy={165} r={12} fill="none" stroke="currentColor" strokeOpacity="0.6" strokeWidth="1.5" />
        </g>
      ))}
      <path d={b.outline} fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" className="sheet-draw" pathLength={1} />
      {b.glass.map((d) => <path key={d} d={d} fill="none" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1.5" className="sheet-draw-late" />)}

      {hasCable && (
        <path d={`M${zx + zw / 2} ${zy + 20}V176H188V120H112`} fill="none" stroke={cable} strokeWidth="4" strokeLinejoin="round" className="sheet-cable" />
      )}
      <g className="sheet-zone">
        {rects.map(([x, y, w, h]) => (
          <rect key={`${x}${y}`} x={x} y={y} width={w} height={h} rx="2" fill={`url(#${patternId})`} stroke="var(--color-hv)" strokeWidth="2" />
        ))}
      </g>
      <g className="sheet-zone">
        <circle cx={zx + zw / 2} cy={zy - 16} r="13" fill="var(--color-hv)" />
        <text x={zx + zw / 2} y={zy - 11} textAnchor="middle" fontSize="16" fontWeight="700" fill="#fff">1</text>
        <path d={`M${zx + zw / 2} ${zy - 3}V${zy}`} stroke="var(--color-hv)" strokeWidth="2" />
        {hasCable && (
          <>
            <circle cx={300} cy={190} r="13" fill={cable} />
            <text x={300} y={195} textAnchor="middle" fontSize="16" fontWeight="700" fill="#fff">2</text>
          </>
        )}
      </g>
    </svg>
  );
}

/** The diagram plus what the guide says, naming the vehicle because the answer may be uncertain. */
export default function RescueSheet({ s }: { s: Safety }) {
  if (!s.battery && !s.hv_cables) return null;
  const schematic = s.location_vehicle ? HAZARDS[s.location_vehicle]?.schematic : undefined;
  return (
    <div className="mt-3 border-t border-current/20 pt-3">
      <p className="text-sm font-medium">If this is a {s.location_vehicle}, per the manufacturer's guide:</p>
      {schematic && s.location_vehicle && <RescueDiagram vehicle={s.location_vehicle} schematic={schematic} className="my-1 max-h-44" />}
      <dl className="mt-1.5 space-y-1.5">
        {s.battery && (
          <div>
            <dt className="text-sm font-semibold">{schematic && "1. "}Battery</dt>
            <dd>{s.battery}</dd>
          </div>
        )}
        {s.hv_cables && (
          <div>
            <dt className="text-sm font-semibold">{schematic?.cable === "underbody" && "2. "}Cables</dt>
            <dd>{s.hv_cables}</dd>
          </div>
        )}
      </dl>
      {schematic && <p className="mt-2 text-sm opacity-75">Schematic: shows the region, not to scale, and not a guide for where to cut.</p>}
    </div>
  );
}
