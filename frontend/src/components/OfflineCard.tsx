import { useEffect, useState } from "react";
import { offlineAvailable, offlineReady, prepareOffline, totalDownloadBytes } from "../offline/store";

type State =
  | { kind: "hidden" }
  | { kind: "idle"; bytes: number | null }
  | { kind: "downloading"; fraction: number | null }
  | { kind: "ready" }
  | { kind: "error"; message: string };

/** Offered only where the server actually has the model files, and never downloads without being asked: it's large. */
export default function OfflineCard() {
  const [state, setState] = useState<State>({ kind: "hidden" });

  useEffect(() => {
    let live = true;
    (async () => {
      if (await offlineReady()) return live && setState({ kind: "ready" });
      if (!(await offlineAvailable())) return;
      const bytes = await totalDownloadBytes();
      if (live) setState({ kind: "idle", bytes });
    })();
    return () => {
      live = false;
    };
  }, []);

  async function download() {
    setState({ kind: "downloading", fraction: 0 });
    try {
      await prepareOffline((fraction) => setState({ kind: "downloading", fraction }));
      setState({ kind: "ready" });
    } catch (err) {
      console.error("Offline download failed", err);
      setState({ kind: "error", message: "The download didn't finish. Check your connection and storage space, then try again." });
    }
  }

  if (state.kind === "hidden") return null;
  return (
    <div className="mt-6 border-t border-line pt-5">
      {state.kind === "ready" ? (
        <p className="font-medium" role="status">
          Ready without signal. If the connection drops, this device answers by itself.
        </p>
      ) : state.kind === "downloading" ? (
        <div role="status">
          <p className="font-medium">{state.fraction === null ? "Preparing the models on this device…" : `Downloading the models: ${Math.round(state.fraction * 100)}%`}</p>
          <div className="mt-2 h-2 overflow-hidden rounded-sm bg-draft-soft">
            <div className={`h-full bg-draft ${state.fraction === null ? "w-full animate-pulse" : ""}`} style={state.fraction === null ? undefined : { width: `${state.fraction * 100}%` }} />
          </div>
        </div>
      ) : (
        <>
          <p className="font-medium">Crash sites often have no signal</p>
          <p className="mt-1 text-ink-soft">
            Store the models on this device to keep working without a connection
            {state.kind === "idle" && state.bytes ? ` (${Math.round(state.bytes / 2 ** 20)} MB, once)` : ""}. Use Wi-Fi if you can.
          </p>
          {state.kind === "error" && <p role="alert" className="mt-2 font-medium">{state.message}</p>}
          <button onClick={download} className="mt-3 min-h-11 rounded-md border-2 border-ink px-4 font-semibold hover:bg-paper">
            {state.kind === "error" ? "Try the download again" : "Make available offline"}
          </button>
        </>
      )}
    </div>
  );
}
