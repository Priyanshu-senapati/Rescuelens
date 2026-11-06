import type { Result } from "../types";

interface Props {
  url: string;
  result: Result | null;
}

/** The photo with the detected vehicle outlined like a drafting overlay (SVG in original pixel coordinates). */
export default function PhotoBox({ url, result }: Props) {
  const det = result?.detection;
  const W = result?.image.width ?? 1;
  const H = result?.image.height ?? 1;
  const hv = result?.safety?.hv_warning && result.safety.level === "danger";
  const colour = hv ? "#e8651a" : "#7cc0ff";

  let ticks = "";
  if (det) {
    const [x1, y1, x2, y2] = det.box;
    const t = Math.min(x2 - x1, y2 - y1) * 0.14;
    ticks = [
      `M${x1},${y1 + t}V${y1}H${x1 + t}`,
      `M${x2 - t},${y1}H${x2}V${y1 + t}`,
      `M${x2},${y2 - t}V${y2}H${x2 - t}`,
      `M${x1 + t},${y2}H${x1}V${y2 - t}`,
    ].join(" ");
  }

  return (
    <figure className="relative overflow-hidden rounded-lg border border-line bg-ink">
      <img src={url} alt="The photo being analysed" className="block h-auto w-full" />
      {det && (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden="true">
            {result?.other_boxes.map(([a, b, c, d], i) => (
              <rect key={i} x={a} y={b} width={c - a} height={d - b} fill="none" stroke="#e9edf0" strokeOpacity="0.6"
                strokeDasharray="6 6" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            ))}
            <rect x={det.box[0]} y={det.box[1]} width={det.box[2] - det.box[0]} height={det.box[3] - det.box[1]}
              fill="none" stroke={colour} strokeOpacity="0.7" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            <path d={ticks} fill="none" stroke={colour} strokeWidth="4" vectorEffect="non-scaling-stroke" />
          </svg>
          <figcaption
            className="tabular absolute rounded-sm px-2 py-0.5 text-sm font-semibold text-ink"
            style={{
              left: `${(det.box[0] / W) * 100}%`,
              top: `max(0px, calc(${(det.box[1] / H) * 100}% - 1.75rem))`,
              background: colour,
            }}
          >
            Vehicle {Math.round(det.score * 100)}%
          </figcaption>
        </>
      )}
    </figure>
  );
}
