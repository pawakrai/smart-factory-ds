import {
  AreaChart,
  Area,
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
  durationMin?: number;
  elapsedMin?: number;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const isSiFePause = Number(label) >= 35 && Number(label) < 36 && payload[0].value === 0;
  return (
    <div className="bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-zinc-400 mb-1">t = {Number(label).toFixed(0)} min</p>
      <p className="font-mono text-brand-red font-semibold">
        {payload[0].value.toFixed(0)} kW
      </p>
      {isSiFePause && (
        <p className="text-amber-400 mt-1">Si + Fe addition</p>
      )}
    </div>
  );
}

export default function PowerProfileChart({ data, durationMin, elapsedMin }: Props) {
  if (!data.length) {
    return (
      <div className="h-full flex items-center justify-center text-zinc-600 text-sm">
        No profile data
      </div>
    );
  }

  const maxTime = data[data.length - 1]?.time_min ?? 90;
  const tickStep = 5;
  const ticks = Array.from(
    { length: Math.floor(maxTime / tickStep) + 1 },
    (_, i) => i * tickStep
  );

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 20, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="powerFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#71717A" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#71717A" stopOpacity={0.03} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
        <XAxis
          type="number"
          dataKey="time_min"
          domain={[0, maxTime]}
          ticks={ticks}
          tick={{ fill: "var(--text-muted)" as string, fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={{ stroke: "var(--border-color)" as string }}
          tickFormatter={(v) => `${v}m`}
        />
        <YAxis
          tick={{ fill: "var(--text-muted)" as string, fontSize: 11, fontFamily: "JetBrains Mono" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}`}
          domain={[0, 500]}
          width={44}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: "var(--border-color)" as string, strokeWidth: 1 }} />

        {/* Max power reference */}
        <ReferenceLine
          y={450}
          stroke="#F59E0B"
          strokeDasharray="4 4"
          strokeWidth={1}
          label={{ value: "Max 450 kW", fill: "#F59E0B", fontSize: 10, position: "insideTopRight" }}
        />

        {/* Si+Fe addition pause */}
        <ReferenceLine
          x={35}
          stroke="#3B82F6"
          strokeDasharray="3 3"
          strokeWidth={1}
          label={{ value: "Si+Fe", fill: "#3B82F6", fontSize: 9, position: "insideTopRight" }}
        />

        {/* Batch end */}
        {durationMin != null && (
          <ReferenceLine
            x={durationMin}
            stroke="#F59E0B"
            strokeDasharray="4 4"
            strokeWidth={1}
            label={{ value: "End", fill: "#F59E0B", fontSize: 10, position: "insideTopRight" }}
          />
        )}

        {/* Live elapsed-time line */}
        {elapsedMin != null && elapsedMin > 0 && (
          <ReferenceLine
            x={elapsedMin}
            stroke="#E3000F"
            strokeWidth={2}
            label={{ value: `${Math.floor(elapsedMin)}m`, fill: "#E3000F", fontSize: 10, position: "insideTopLeft" }}
          />
        )}

        <Area
          type="stepAfter"
          dataKey="power_kw"
          stroke="#71717A"
          strokeWidth={2}
          fill="url(#powerFill)"
          dot={false}
          activeDot={{ r: 4, fill: "#71717A", stroke: "#09090B", strokeWidth: 2 }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
