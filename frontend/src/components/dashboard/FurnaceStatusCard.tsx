export type FurnaceStatus = "running" | "warning" | "fault" | "offline";

const STATUS_CONFIG: Record<FurnaceStatus, { color: string; label: string }> = {
  running: { color: "#22C55E", label: "Running" },
  warning: { color: "#F59E0B", label: "Warning" },
  fault:   { color: "#E3000F", label: "Fault" },
  offline: { color: "#52525B", label: "Offline" },
};

export interface FurnaceData {
  name: string;
  temp: string;
  power: string;
  status: FurnaceStatus;
}

interface Props {
  furnace: FurnaceData;
}

export default function FurnaceStatusCard({ furnace }: Props) {
  const st = STATUS_CONFIG[furnace.status];

  return (
    <div className="flex items-center justify-between bg-bg-elevated rounded-lg px-3 py-2.5">
      <div className="flex items-center gap-2.5">
        {/* Pulsing dot for running state */}
        <span className="relative flex h-2 w-2 shrink-0">
          {furnace.status === "running" && (
            <span
              className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-60"
              style={{ backgroundColor: st.color }}
            />
          )}
          <span
            className="relative inline-flex rounded-full h-2 w-2"
            style={{ backgroundColor: st.color }}
          />
        </span>

        <div>
          <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">{furnace.name}</p>
          <p className="text-[10px] text-zinc-500">{st.label}</p>
        </div>
      </div>

      <div className="text-right">
        <p className="font-kpi text-xs text-[var(--text-primary)]">{furnace.temp}</p>
        <p className="font-kpi text-[10px] text-zinc-400">{furnace.power}</p>
      </div>
    </div>
  );
}
