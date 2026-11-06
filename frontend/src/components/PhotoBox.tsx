import { useEffect, useRef } from "react";
import { EdgeRenderer, type Box } from "../live/edges";
import type { Result } from "../types";

interface Props {
  url: string;
  result: Result | null;
  loading?: boolean;
  width?: number;
  height?: number;
}

/**
 * The photo with the detected vehicle outlined like a drafting overlay (SVG in original pixel coordinates).
 * While the model works, the photo's real edge map runs over it; when the answer lands, the drawing narrows onto the vehicle and fades.
 */
export default function PhotoBox({ url, result, loading = false, width, height }: Props) {
  const img = useRef<HTMLImageElement>(null);
  const edges = useRef<HTMLCanvasElement>(null);

  const det = result?.detection;
  const W = result?.image.width ?? 1;
  const H = result?.image.height ?? 1;
  const hv = result?.safety?.hv_warning && result.safety.level === "danger";
  const colour = hv ? "#e8651a" : "#7cc0ff";

  useEffect(() => {
    const canvas = edges.current;
    const image = img.current;
    if (!canvas || !image || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const box: Box | null = det ? [det.box[0] / W, det.box[1] / H, det.box[2] / W, det.box[3] / H] : null;
    const paint = () => new EdgeRenderer(canvas).draw(image, box, !!hv, box ? 1.5 : 1.7);
    if (image.complete) paint();
    else {
      image.addEventListener("load", paint, { once: true });
      return () => image.removeEventListener("load", paint);
    }
  }, [url, det, W, H, hv]);

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

  const showEdges = loading || !!det;
  return (
    <figure className="relative overflow-hidden rounded-lg border border-line bg-ink">
      <img ref={img} src={url} width={width} height={height} alt="The photo being analysed" className="block h-auto w-full" />
      {showEdges && (
        <canvas key={det ? "answer" : "scan"} ref={edges} aria-hidden="true" className={`absolute inset-0 h-full w-full bg-[#0d141b]/70 ${det ? "edges-fade" : "edges-on"}`} />
      )}
      {loading && <div aria-hidden="true" className="scan-sweep absolute inset-x-0 h-24" />}
      {det && (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-full" aria-hidden="true">
            {result?.other_boxes.map(([a, b, c, d], i) => (
              <rect key={i} x={a} y={b} width={c - a} height={d - b} fill="none" stroke="#e9edf0" strokeOpacity="0.6"
                strokeDasharray="6 6" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            ))}
            <rect x={det.box[0]} y={det.box[1]} width={det.box[2] - det.box[0]} height={det.box[3] - det.box[1]}
              fill="none" stroke={colour} strokeOpacity="0.7" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            <path d={ticks} fill="none" stroke={colour} strokeWidth="4" vectorEffect="non-scaling-stroke" className="ticks-snap" />
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
