import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { PowerProfilePoint } from "@/api/batches";

interface Props {
  data: PowerProfilePoint[];
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-zinc-400 mb-1">t = {label} min</p>
      <p className="font-mono text-brand-red font-semibold">
        {payload[0].value.toFixed(0)} kW
      </p>
    </div>
  );
}

export default function PowerProfileChart({ data }: Props) {
  if (!data.length) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-sm">
        No profile data
      </div>
    );
  }

  const maxKw = Math.max(...data.map((d) => d.power_kw));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
        <XAxis
          dataKey="time_min"
          tick={{ fill: "var(--text-muted)" as string, fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={{ stroke: "var(--border-color)" as string }}
          tickFormatter={(v) => `${v}m`}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: "var(--text-muted)" as string, fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}`}
          domain={[0, Math.ceil(maxKw / 100) * 100 + 50]}
          width={44}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: "var(--border-color)" as string, strokeWidth: 1 }} />
        <ReferenceLine y={400} stroke="#F59E0B" strokeDasharray="4 4" strokeWidth={1} label={{ value: "Max 400 kW", fill: "#F59E0B", fontSize: 10, position: "insideTopRight" }} />
        <Line
          type="monotone"
          dataKey="power_kw"
          stroke="#E3000F"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#E3000F", stroke: "#09090B", strokeWidth: 2 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
