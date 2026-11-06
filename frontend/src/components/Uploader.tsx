import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { MAX_BYTES } from "../image";

interface Props {
  onPick: (file: Blob, name: string) => void;
  onLive: () => void;
}

const SAMPLES = [
  { file: "clear.jpg", label: "Clear daylight photo" },
  { file: "sonata-hybrid.jpg", label: "Hybrid that looks like a petrol car" },
  { file: "night.jpg", label: "Taken at night" },
  { file: "hidden.jpg", label: "Partly hidden" },
  { file: "electric.jpg", label: "Electric car" },
];

export default function Uploader({ onPick, onLive }: Props) {
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const camera = useRef<HTMLInputElement>(null);
  const files = useRef<HTMLInputElement>(null);

  function accept(f: File | undefined) {
    if (!f) return;
    if (!f.type.startsWith("image/") && !/\.(heic|heif)$/i.test(f.name))
      return setError("That file isn't a photo. Choose a JPEG, PNG or HEIC image.");
    if (f.size > MAX_BYTES) return setError("That photo is over 10 MB. Choose a smaller one.");
    setError(null);
    onPick(f, f.name);
  }

  async function pickSample(file: string) {
    const res = await fetch(`/samples/${file}`).catch(() => null);
    if (!res || !res.ok) return setError("Couldn't load that sample. Try uploading a photo instead.");
    onPick(await res.blob(), file);
  }

  function onInput(e: ChangeEvent<HTMLInputElement>) {
    accept(e.target.files?.[0]);
    e.target.value = ""; // so choosing the same photo again still triggers a change
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files[0]);
  }

  return (
    <section className="grid items-start gap-8 pt-4 lg:grid-cols-[1.1fr_1fr] lg:pt-10">
      <div>
        <h1 className="max-w-[13ch] font-display text-5xl font-bold leading-[0.95] sm:text-7xl">
          Which vehicle is this, and is it safe to cut?
        </h1>
        <p className="mt-5 max-w-[60ch] text-lg leading-relaxed text-ink-soft">
          Photograph the vehicle. RescueLens finds it, names the make, model and year among 196 models, and tells you
          when the answer can't be trusted. Hybrids and electric cars trigger a high-voltage warning even when they are
          only a possibility.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`rounded-xl border-2 border-dashed bg-sheet p-6 sm:p-8 ${dragging ? "border-draft bg-draft-soft" : "border-line"}`}
      >
        <button
          onClick={onLive}
          className="mb-3 flex min-h-16 w-full items-center justify-center gap-3 rounded-md bg-ink px-5 text-paper hover:bg-draft"
        >
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="#7cc0ff" strokeWidth="2.4" aria-hidden="true">
            <path d="M3 9V3h6M19 3h6v6M25 19v6h-6M9 25H3v-6" />
            <circle cx="14" cy="14" r="4" />
          </svg>
          <span className="text-left">
            <span className="block text-xl font-semibold leading-tight">Point the camera</span>
            <span className="block text-sm text-paper/75">The answer locks in once it holds steady</span>
          </span>
        </button>
        <div className="flex flex-col gap-3 sm:flex-row">
          <button
            onClick={() => camera.current?.click()}
            className="min-h-12 flex-1 rounded-md border-2 border-ink px-5 text-lg font-semibold hover:bg-paper"
          >
            Take a photo
          </button>
          <button
            onClick={() => files.current?.click()}
            className="min-h-12 flex-1 rounded-md border-2 border-ink px-5 text-lg font-semibold hover:bg-paper"
          >
            Choose a photo
          </button>
        </div>
        <p className="mt-4 text-center text-ink-soft">or drop a photo here. JPEG, PNG or HEIC, up to 10 MB.</p>
        <input ref={camera} type="file" accept="image/*" capture="environment" hidden onChange={onInput} />
        <input ref={files} type="file" accept="image/*,.heic,.heif" hidden onChange={onInput} />
        {error && (
          <p role="alert" className="mt-4 rounded-md border border-ink bg-paper px-3 py-2 font-medium">
            {error}
          </p>
        )}
      </div>

      <div className="lg:col-span-2">
        <h2 className="font-display text-2xl font-semibold">No photo to hand? Try one of these</h2>
        <ul className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          {SAMPLES.map((s) => (
            <li key={s.file}>
              <button
                onClick={() => pickSample(s.file)}
                className="group block w-full overflow-hidden rounded-lg border border-line bg-sheet text-left hover:border-draft"
              >
                <img src={`/samples/${s.file}`} alt="" loading="lazy" className="aspect-[4/3] w-full object-cover" />
                <span className="block px-3 py-2 text-sm font-medium group-hover:text-draft">{s.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
