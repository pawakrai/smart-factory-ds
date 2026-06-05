import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { Batch } from "@/types";

// Status → dot color. Pending=zinc, in_progress=amber, completed=green.
const STATUS_COLORS: Record<string, string> = {
  pending: "#52525B",
  in_progress: "#F59E0B",
  completed: "#22C55E",
};

interface Props {
  batches: Batch[];
  selected: Batch | null;
  onSelect: (b: Batch) => void;
}

/** Dropdown for picking which charge the operator will execute next. The
 *  consumer (OperatorExecutionPage) feeds this only the "still pending"
 *  filtered list so completed charges don't reappear. */
export default function ChargeSelector({ batches, selected, onSelect }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between bg-bg-elevated border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] transition-colors"
      >
        <span className="flex items-center gap-2">
          {selected ? (
            <>
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: STATUS_COLORS[selected.status] }}
              />
              <span className="font-mono text-xs">Charge #{selected.batch_number}</span>
              <span className="text-zinc-500 text-xs">— {selected.furnace ?? "?"} furnace</span>
            </>
          ) : (
            <span className="text-zinc-500">Select a charge…</span>
          )}
        </span>
        <ChevronDown size={14} className="text-zinc-400" />
      </button>

      {open && (
        <div className="absolute z-10 mt-1 w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg shadow-xl overflow-hidden">
          {batches.length === 0 ? (
            <p className="px-3 py-2 text-xs text-zinc-500">No pending charges</p>
          ) : (
            batches.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => { onSelect(b); setOpen(false); }}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-bg-card transition-colors text-left"
              >
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: STATUS_COLORS[b.status] }}
                />
                <span className="font-mono text-[var(--text-primary)]">Charge #{b.batch_number}</span>
                <span className="text-zinc-500">Furnace {b.furnace ?? "?"}</span>
                <span className="ml-auto text-zinc-600 capitalize">{b.status.replace("_", " ")}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export { STATUS_COLORS };
