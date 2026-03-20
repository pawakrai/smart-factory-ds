import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { EnergyLog } from "@/types";

interface ChartPoint {
  hour: string;
  sim_kw: number;
  actual_kw: number;
}

function aggregateByHour(logs: EnergyLog[]): ChartPoint[] {
  const map = new Map<string, { sim: number[]; actual: number[] }>();

  for (const log of logs) {
    const d = new Date(log.timestamp);
    const hour = `${String(d.getHours()).padStart(2, "0")}:00`;
    if (!map.has(hour)) map.set(hour, { sim: [], actual: [] });
    const entry = map.get(hour)!;
    if (log.sim_kw != null) entry.sim.push(log.sim_kw);
    if (log.actual_kw != null) entry.actual.push(log.actual_kw);
  }

  return Array.from(map.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([hour, { sim, actual }]) => ({
      hour,
      sim_kw: sim.length
        ? Math.round(sim.reduce((a, b) => a + b, 0) / sim.length)
        : 0,
      actual_kw: actual.length
        ? Math.round(actual.reduce((a, b) => a + b, 0) / actual.length)
        : 0,
    }));
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-[var(--border-color)] rounded-lg p-3 text-xs shadow-xl">
      <p className="text-zinc-400 mb-2 font-medium">{label}</p>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-0.5">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: p.fill }}
          />
          <span className="text-zinc-400">{p.name}:</span>
          <span className="font-kpi text-[var(--text-primary)]">{p.value} kW</span>
        </div>
      ))}
    </div>
  );
}

interface Props {
  logs: EnergyLog[];
}

export default function EnergyBarChart({ logs }: Props) {
  const data = aggregateByHour(logs);

  if (data.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center text-zinc-600 text-sm">
        No energy data — seed mock data to preview
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={224}>
      <BarChart
        data={data}
        barGap={2}
        margin={{ top: 4, right: 8, bottom: 0, left: -8 }}
      >
        <CartesianGrid
          vertical={false}
          stroke="var(--border-color)"
          strokeDasharray="3 3"
        />
        <XAxis
          dataKey="hour"
          tick={{ fill: "var(--text-dim)" as string, fontSize: 10, fontFamily: "Inter" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: "var(--text-dim)" as string, fontSize: 10, fontFamily: "Inter" }}
          axisLine={false}
          tickLine={false}
          unit=" kW"
          width={58}
        />
        <Tooltip
          content={<CustomTooltip />}
          cursor={{ fill: "var(--bg-elevated)", opacity: 0.4 }}
        />
        <Bar
          dataKey="sim_kw"
          name="Simulated"
          fill="#E3000F"
          radius={[2, 2, 0, 0]}
          maxBarSize={14}
        />
        <Bar
          dataKey="actual_kw"
          name="Actual"
          fill="#3B82F6"
          radius={[2, 2, 0, 0]}
          maxBarSize={14}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
