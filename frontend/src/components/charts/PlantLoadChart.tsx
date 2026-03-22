import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
} from "recharts";
import type { ScheduleData } from "@/types";

interface Props {
  data: ScheduleData;
}

function buildChartData(data: ScheduleData) {
  const shiftStart = new Date(data.shift_start_iso);
  const step = data.sample_interval_min;

  return data.total_plant_kw.map((_, i) => {
    const minuteOffset = i * step;
    const t = new Date(shiftStart.getTime() + minuteOffset * 60_000);
    const hh = String(t.getHours()).padStart(2, "0");
    const mm = String(t.getMinutes()).padStart(2, "0");
    return {
      label: `${hh}:${mm}`,
      minuteOffset,
      baseline: data.baseline_kw[i] ?? null,
      ifkw: data.if_kw[i] ?? null,
      total: data.total_plant_kw[i] ?? null,
    };
  });
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-[var(--text-muted)] font-mono mb-1">{label}</p>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex justify-between gap-4">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="text-[var(--text-primary)] font-mono">
            {p.value != null ? `${p.value.toFixed(0)} kW` : "—"}
          </span>
        </div>
      ))}
    </div>
  );
};

export default function PlantLoadChart({ data }: Props) {
  const chartData = buildChartData(data);

  // Solar window reference area (in minuteOffset units)
  const solarStart = data.solar_window_start_min;
  const solarEnd = data.solar_window_end_min;
  const hasSolar =
    solarStart != null && solarEnd != null && solarEnd > solarStart;

  // X-axis ticks: every 60 min
  const tickIndices = chartData
    .map((d, i) => ({ i, offset: d.minuteOffset }))
    .filter(({ offset }) => offset % 60 === 0)
    .map(({ offset }) => offset);

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-4">
      <p className="text-sm font-semibold text-[var(--text-primary)] mb-3">Plant Load</p>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />

          <XAxis
            dataKey="minuteOffset"
            type="number"
            scale="linear"
            domain={["dataMin", "dataMax"]}
            ticks={tickIndices}
            tickFormatter={(v) => {
              const idx = chartData.findIndex((d) => d.minuteOffset === v);
              return idx >= 0 ? chartData[idx].label : "";
            }}
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={{ stroke: "var(--border-color)" as string }}
            tickLine={false}
          />

          <YAxis
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={{ stroke: "var(--border-color)" as string }}
            tickLine={false}
            label={{ value: "kW", angle: -90, position: "insideLeft", fill: "var(--text-dim)", fontSize: 10, dy: 20 }}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }}
          />

          {hasSolar && (
            <ReferenceArea
              x1={solarStart}
              x2={solarEnd}
              fill="#F59E0B"
              fillOpacity={0.1}
              strokeOpacity={0}
            />
          )}

          <ReferenceLine
            y={data.contract_demand_kw}
            stroke="#52525B"
            strokeDasharray="6 3"
            strokeWidth={1.5}
            label={{ value: `Contract ${data.contract_demand_kw.toFixed(0)} kW`, fill: "#71717A", fontSize: 9, position: "insideTopRight" }}
          />

          <Line
            dataKey="baseline"
            name="Baseline"
            stroke="#71717A"
            strokeWidth={1.5}
            strokeDasharray="4 2"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="ifkw"
            name="IF kW"
            stroke="#3B82F6"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="total"
            name="Total Plant"
            stroke="#E3000F"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
