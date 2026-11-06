// Client-side image handling: shrink before upload (fast on phones) and the stress-test degradations.
const MAX_SIDE = 1600;
export const MAX_BYTES = 10 * 1024 * 1024;

export interface Prepared {
  blob: Blob;
  url: string; // object URL of exactly what was uploaded, so boxes line up
  name: string;
  width?: number; // known up front, so the page doesn't jump while the image decodes
  height?: number;
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("encode failed"))), "image/jpeg", 0.9),
  );
}

export async function prepareImage(file: Blob, name: string): Promise<Prepared> {
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const s = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * s);
    canvas.height = Math.round(bmp.height * s);
    canvas.getContext("2d")!.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    bmp.close();
    const blob = await canvasToBlob(canvas);
    return { blob, url: URL.createObjectURL(blob), name: name.replace(/\.\w+$/, "") + ".jpg", width: canvas.width, height: canvas.height };
  } catch {
    // browser can't decode it (e.g. HEIC outside Safari): send the original, the server converts it
    return { blob: file, url: URL.createObjectURL(file), name };
  }
}

/** One JPEG from the live video, small enough to upload every 1.5 s on mobile data. */
export function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const s = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * s);
  canvas.height = Math.round(video.videoHeight * s);
  canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvasToBlob(canvas);
}

export interface Degrade {
  darkness: number; // 0..0.9
  haze: number; // 0..0.7
  blur: number; // 0..1
  occlusion: number; // 0..0.5 of the vehicle box
}

export const NO_DEGRADE: Degrade = { darkness: 0, haze: 0, blur: 0, occlusion: 0 };

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

/** Re-renders the uploaded photo with simulated crash-scene conditions (works in every browser, no ctx.filter). */
export async function degradeImage(url: string, d: Degrade, box: number[] | null): Promise<Prepared> {
  const img = await loadImage(url);
  const { naturalWidth: w, naturalHeight: h } = img;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  if (d.blur > 0) {
    const s = 1 - 0.92 * d.blur; // downscale + upscale = blur / low resolution
    const small = document.createElement("canvas");
    small.width = Math.max(8, Math.round(w * s));
    small.height = Math.max(8, Math.round(h * s));
    small.getContext("2d")!.drawImage(img, 0, 0, small.width, small.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(small, 0, 0, w, h);
  } else {
    ctx.drawImage(img, 0, 0);
  }
  if (d.occlusion > 0) {
    const [x1, y1, x2, y2] = box ?? [0, 0, w, h];
    ctx.fillStyle = "rgb(120,120,120)";
    ctx.fillRect(x1, y1, (x2 - x1) * Math.sqrt(d.occlusion) * 1.2, (y2 - y1) * Math.min(1, Math.sqrt(d.occlusion) * 1.2));
  }
  if (d.darkness > 0) {
    ctx.fillStyle = `rgba(0,0,0,${d.darkness})`;
    ctx.fillRect(0, 0, w, h);
  }
  if (d.haze > 0) {
    ctx.fillStyle = `rgba(205,205,205,${d.haze})`;
    ctx.fillRect(0, 0, w, h);
  }
  const blob = await canvasToBlob(canvas);
  return { blob, url: URL.createObjectURL(blob), name: "stress-test.jpg" };
}
