import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import type { EnergyLog } from "@/types";

interface TrendPoint {
  i: number;
  actual: number;
  sim: number;
}

function buildTrend(logs: EnergyLog[]): TrendPoint[] {
  return [...logs]
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp))
    .slice(-12)
    .map((log, i) => ({
      i,
      actual: Math.round(log.actual_kw ?? 0),
      sim: Math.round(log.sim_kw ?? 0),
    }));
}

interface Metric {
  label: string;
  value: string;
  delta: string;
  up: boolean;
}

function computeMetrics(logs: EnergyLog[]): Metric[] {
  if (logs.length < 2) return [];

  const sorted = [...logs].sort((a, b) =>
    a.timestamp.localeCompare(b.timestamp)
  );
  const half = Math.max(1, Math.floor(sorted.length / 2));
  const recent = sorted.slice(-half);
  const prev = sorted.slice(-half * 2, -half);

  const avg = (arr: EnergyLog[], key: keyof EnergyLog) =>
    arr.reduce((s, l) => s + ((l[key] as number) ?? 0), 0) / arr.length;

  const recentActual = avg(recent, "actual_kw");
  const prevActual = avg(prev, "actual_kw");
  const recentSim = avg(recent, "sim_kw");
  const delta = prevActual > 0 ? ((recentActual - prevActual) / prevActual) * 100 : 0;
  const accuracy =
    recentSim > 0 ? 100 - Math.abs(((recentActual - recentSim) / recentSim) * 100) : 0;

  return [
    {
      label: "Avg Power",
      value: `${Math.round(recentActual)} kW`,
      delta: `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`,
      up: delta <= 0, // lower power is better → green when going down
    },
    {
      label: "Forecast Accuracy",
      value: `${accuracy.toFixed(1)}%`,
      delta: accuracy >= 92 ? "On target" : "Off target",
      up: accuracy >= 92,
    },
    {
      label: "Peak vs Off-Peak",
      value: (() => {
        const peak = sorted.filter((l) => {
          const h = new Date(l.timestamp).getHours();
          return h >= 9 && h < 22;
        });
        const off = sorted.filter((l) => {
          const h = new Date(l.timestamp).getHours();
          return h < 9 || h >= 22;
        });
        const peakAvg = peak.length ? avg(peak, "actual_kw") : 0;
        const offAvg = off.length ? avg(off, "actual_kw") : 0;
        const ratio = offAvg > 0 ? peakAvg / offAvg : 1;
        return `${ratio.toFixed(1)}×`;
      })(),
      delta: "peak/off-peak",
      up: true,
    },
  ];
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-[11px]">
      <div className="flex gap-3">
        {payload.map((p: any) => (
          <span key={p.dataKey} style={{ color: p.color }}>
            {p.name}: {p.value} kW
          </span>
        ))}
      </div>
    </div>
  );
}

interface Props {
  logs: EnergyLog[];
}

export default function TrendSummaryWidget({ logs }: Props) {
  const trend = buildTrend(logs);
  const metrics = computeMetrics(logs);

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Sparkline */}
      <div className="flex-1 min-h-[80px]">
        {trend.length === 0 ? (
          <div className="h-20 flex items-center justify-center text-zinc-600 text-xs">
            No data
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={90}>
            <AreaChart
              data={trend}
              margin={{ top: 4, right: 4, bottom: 0, left: -28 }}
            >
              <defs>
                <linearGradient id="gradSim" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#E3000F" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#E3000F" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="i" hide />
              <YAxis tick={{ fill: "var(--text-dim)" as string, fontSize: 9 }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="sim"
                name="Sim"
                stroke="#E3000F"
                strokeWidth={1.5}
                fill="url(#gradSim)"
                dot={false}
              />
              <Area
                type="monotone"
                dataKey="actual"
                name="Actual"
                stroke="#3B82F6"
                strokeWidth={1.5}
                fill="url(#gradActual)"
                dot={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Metric badges */}
      {metrics.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          {metrics.map((m) => (
            <div key={m.label} className="bg-bg-elevated rounded-lg px-3 py-2">
              <p className="text-[10px] text-zinc-500 mb-0.5 uppercase tracking-wide">
                {m.label}
              </p>
              <p className="font-kpi text-sm text-[var(--text-primary)] leading-tight">{m.value}</p>
              <p
                className="text-[10px] font-medium mt-0.5"
                style={{ color: m.up ? "#22C55E" : "#E3000F" }}
              >
                {m.delta}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
