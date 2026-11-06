// Live "technical drawing" overlay: thin edges of the camera feed (Sobel + non-maximum suppression, the core of Canny),
// brightest inside the detected vehicle. Runs on a small buffer about 12 times a second to stay cheap on mid-range phones.

export type Box = [number, number, number, number]; // normalised 0..1

const WIDTH = 420;
const LOW = 90; // gradient strength where a line starts to show
const RANGE = 170; // and how quickly it reaches full strength
const DRAFT = [124, 192, 255];
const HV = [255, 138, 61];

export class EdgeRenderer {
  private src = document.createElement("canvas");
  private srcCtx = this.src.getContext("2d", { willReadFrequently: true });
  private out: ImageData | null = null;
  private gray = new Float32Array(0);
  private mag = new Float32Array(0);
  private dir = new Uint8Array(0);

  constructor(private canvas: HTMLCanvasElement) {}

  /** Returns false when the source has no picture yet. */
  draw(source: HTMLVideoElement | HTMLImageElement, box: Box | null, hv: boolean, boost = 1): boolean {
    const sw = source instanceof HTMLVideoElement ? source.videoWidth : source.naturalWidth;
    const sh = source instanceof HTMLVideoElement ? source.videoHeight : source.naturalHeight;
    const ctx = this.canvas.getContext("2d");
    if (!sw || !sh || !this.srcCtx || !ctx) return false;

    const w = Math.min(WIDTH, sw);
    const h = Math.max(3, Math.round((w * sh) / sw));
    if (this.src.width !== w || this.src.height !== h || !this.out) {
      this.src.width = this.canvas.width = w;
      this.src.height = this.canvas.height = h;
      this.out = ctx.createImageData(w, h);
      this.gray = new Float32Array(w * h);
      this.mag = new Float32Array(w * h);
      this.dir = new Uint8Array(w * h);
    }
    this.srcCtx.drawImage(source, 0, 0, w, h);
    const px = this.srcCtx.getImageData(0, 0, w, h).data;
    const { gray, mag, dir } = this;
    const dst = this.out.data;

    for (let i = 0, p = 0; i < gray.length; i++, p += 4) gray[i] = 0.299 * px[p] + 0.587 * px[p + 1] + 0.114 * px[p + 2];

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1] - gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1];
        const gy = gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1] - gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1];
        const ax = Math.abs(gx);
        const ay = Math.abs(gy);
        mag[i] = ax + ay;
        // gradient direction in 4 bins: which two neighbours this pixel must beat to count as the line's centre
        dir[i] = ay * 2.4 < ax ? 0 : ax * 2.4 < ay ? 1 : gx * gy > 0 ? 2 : 3;
      }
    }

    const STEP = [1, w, w + 1, w - 1];
    const x1 = box ? box[0] * w : -1;
    const y1 = box ? box[1] * h : -1;
    const x2 = box ? box[2] * w : -1;
    const y2 = box ? box[3] * h : -1;
    const tint = hv ? HV : DRAFT;

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const o = i * 4;
        const m = mag[i];
        const step = STEP[dir[i]];
        let strength = 0;
        if (m > LOW && m >= mag[i - step] && m >= mag[i + step]) {
          const inside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
          strength = Math.min(1, (m - LOW) / RANGE) * (box ? (inside ? 1 : 0.22) : 0.55) * boost;
        }
        dst[o] = tint[0];
        dst[o + 1] = tint[1];
        dst[o + 2] = tint[2];
        dst[o + 3] = Math.min(255, strength * 255);
      }
    }
    ctx.putImageData(this.out, 0, 0);
    return true;
  }
}
