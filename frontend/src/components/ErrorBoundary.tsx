import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  failed: boolean;
  attempt: number; // bumped on restart so the whole app remounts with fresh state
}

/**
 * Last line of defence: a render crash must never leave a responder looking at a blank page.
 * The fallback tells them what to do without the app, because at a real scene that matters more than the bug.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { failed: false, attempt: 0 };

  static getDerivedStateFromError(): Partial<State> {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("RescueLens UI crashed:", error, info.componentStack);
  }

  private restart = () => {
    window.scrollTo(0, 0);
    this.setState((s) => ({ failed: false, attempt: s.attempt + 1 }));
  };

  render() {
    if (!this.state.failed) return <div key={this.state.attempt}>{this.props.children}</div>;
    return (
      <main className="mx-auto flex min-h-dvh max-w-2xl flex-col justify-center px-4 py-10 sm:px-6">
        <div role="alert" className="overflow-hidden rounded-lg border-2 border-ink bg-sheet">
          <div className="hazard-stripes h-3" aria-hidden="true" />
          <div className="p-6 sm:p-8">
            <h1 className="font-display text-5xl font-bold leading-none text-balance">RescueLens hit a problem</h1>
            <p className="mt-4 text-lg leading-relaxed text-ink-soft">
              The last answer can't be shown, so don't rely on it. Treat the vehicle as possibly high-voltage and
              identify it by VIN, plate and badges until you have a new answer.
            </p>
            <button
              onClick={this.restart}
              className="mt-6 min-h-12 rounded-md bg-ink px-6 text-lg font-semibold text-paper hover:bg-draft"
            >
              Start again
            </button>
          </div>
        </div>
      </main>
    );
  }
}
