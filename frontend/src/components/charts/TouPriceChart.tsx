import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
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
  const n = data.tou_effective_price.length;

  return Array.from({ length: n }, (_, i) => {
    const minuteOffset = i * step;
    const t = new Date(shiftStart.getTime() + minuteOffset * 60_000);
    const hh = String(t.getHours()).padStart(2, "0");
    const mm = String(t.getMinutes()).padStart(2, "0");
    const ifActive = (data.if_kw[i] ?? 0) > 0.1;
    return {
      label: `${hh}:${mm}`,
      minuteOffset,
      rawPrice: data.tou_raw_price[i] ?? null,
      effectivePrice: data.tou_effective_price[i] ?? null,
      // For IF active area: show a thin band just below min price
      ifBand: ifActive ? (data.tou_effective_price[i] ?? 0) * 0.04 : 0,
    };
  });
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const raw = payload.find((p: any) => p.dataKey === "rawPrice");
  const eff = payload.find((p: any) => p.dataKey === "effectivePrice");
  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-lg px-3 py-2 text-xs space-y-1">
      <p className="text-[var(--text-muted)] font-mono mb-1">{label}</p>
      {raw && (
        <div className="flex justify-between gap-4">
          <span className="text-[var(--text-muted)]">TOU Raw</span>
          <span className="text-[var(--text-primary)] font-mono">
            {raw.value != null ? `${raw.value.toFixed(4)} ฿/kWh` : "—"}
          </span>
        </div>
      )}
      {eff && (
        <div className="flex justify-between gap-4">
          <span style={{ color: "#A855F7" }}>Effective</span>
          <span className="text-[var(--text-primary)] font-mono">
            {eff.value != null ? `${eff.value.toFixed(4)} ฿/kWh` : "—"}
          </span>
        </div>
      )}
    </div>
  );
};

export default function TouPriceChart({ data }: Props) {
  const chartData = buildChartData(data);

  const allPrices = [
    ...data.tou_raw_price,
    ...data.tou_effective_price,
  ].filter((v) => v != null && isFinite(v));
  const priceMin = allPrices.length ? Math.min(...allPrices) - 0.3 : 0;
  const priceMax = allPrices.length ? Math.max(...allPrices) + 0.3 : 5;

  const solarStart = data.solar_window_start_min;
  const solarEnd = data.solar_window_end_min;
  const hasSolar =
    solarStart != null && solarEnd != null && solarEnd > solarStart;

  const tickIndices = chartData
    .map((d) => d.minuteOffset)
    .filter((offset) => offset % 60 === 0);

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-4">
      <p className="text-sm font-semibold text-[var(--text-primary)] mb-3">TOU Raw vs Effective Price</p>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />

          <XAxis
            dataKey="minuteOffset"
            type="number"
            scale="linear"
            domain={["dataMin", "dataMax"]}
            ticks={tickIndices}
            tickFormatter={(v) => {
              const item = chartData.find((d) => d.minuteOffset === v);
              return item?.label ?? "";
            }}
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={{ stroke: "var(--border-color)" as string }}
            tickLine={false}
          />

          <YAxis
            domain={[priceMin, priceMax]}
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "JetBrains Mono" }}
            axisLine={{ stroke: "var(--border-color)" as string }}
            tickLine={false}
            tickFormatter={(v) => v.toFixed(2)}
            label={{ value: "฿/kWh", angle: -90, position: "insideLeft", fill: "var(--text-dim)", fontSize: 10, dy: 30 }}
          />

          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: "var(--text-muted)" }} />

          {hasSolar && (
            <ReferenceArea
              x1={solarStart}
              x2={solarEnd}
              fill="#F59E0B"
              fillOpacity={0.12}
              strokeOpacity={0}
            />
          )}

          {/* IF active window — thin steelblue area at bottom of chart */}
          <Area
            dataKey="ifBand"
            name="IF Active"
            fill="#4F86C6"
            fillOpacity={0.4}
            stroke="transparent"
            dot={false}
            isAnimationActive={false}
            legendType="square"
            baseValue={priceMin}
          />

          <Line
            dataKey="rawPrice"
            name="TOU Raw"
            stroke="#A1A1AA"
            strokeWidth={1.8}
            dot={false}
            isAnimationActive={false}
            type="stepAfter"
          />
          <Line
            dataKey="effectivePrice"
            name="TOU Effective (after solar)"
            stroke="#A855F7"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            type="stepAfter"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
