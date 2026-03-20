import { Component, type ReactNode } from "react";
import { AlertTriangle, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
  label?: string;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="h-full min-h-[120px] flex flex-col items-center justify-center gap-3 border border-dashed border-[var(--border-color)] rounded-lg">
          <AlertTriangle size={20} className="text-status-amber opacity-60" />
          <p className="text-xs text-zinc-500">
            {this.props.label ?? "Chart"} failed to render
          </p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-[var(--text-primary)] border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-1.5 transition-colors"
          >
            <RotateCcw size={11} />
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
