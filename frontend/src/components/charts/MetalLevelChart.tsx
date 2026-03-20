import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { ScheduleData } from "@/types";

interface LevelPoint {
  label: string;
  mhA: number;
  mhB: number;
}

function minuteToHHMM(shiftStartIso: string, offsetMin: number): string {
  const base = new Date(shiftStartIso).getTime();
  const t = new Date(base + offsetMin * 60_000);
  return `${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-[var(--border-color)] rounded-lg p-2.5 text-xs shadow-xl">
      <p className="text-zinc-400 mb-1.5 font-medium">{label}</p>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center gap-2 mb-0.5">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: p.color }} />
          <span className="text-zinc-400">{p.name}:</span>
          <span className="font-kpi text-[var(--text-primary)]">{p.value} kg</span>
        </div>
      ))}
    </div>
  );
}

interface Props {
  scheduleData: ScheduleData;
}

export default function MetalLevelChart({ scheduleData }: Props) {
  const {
    mh_a_levels_kg,
    mh_b_levels_kg,
    mh_a_min_level_kg,
    mh_b_min_level_kg,
    sample_interval_min,
    shift_start_iso,
  } = scheduleData;

  const data: LevelPoint[] = mh_a_levels_kg.map((val, i) => ({
    label: minuteToHHMM(shift_start_iso, i * sample_interval_min),
    mhA: Math.round(val),
    mhB: Math.round(mh_b_levels_kg[i] ?? 0),
  }));

  if (data.length === 0) {
    return (
      <div className="h-40 flex items-center justify-center text-zinc-600 text-sm">
        No schedule data
      </div>
    );
  }

  const maxY = Math.max(
    ...mh_a_levels_kg,
    ...mh_b_levels_kg,
    mh_a_min_level_kg,
    mh_b_min_level_kg,
  ) + 30;

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
        <CartesianGrid stroke="var(--border-color)" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fill: "var(--text-dim)" as string, fontSize: 9, fontFamily: "JetBrains Mono, monospace" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: "var(--text-dim)" as string, fontSize: 10, fontFamily: "Inter, sans-serif" }}
          axisLine={false}
          tickLine={false}
          unit=" kg"
          width={52}
          domain={[0, maxY]}
        />
        <Tooltip content={<CustomTooltip />} />
        {/* Min operational level reference lines */}
        <ReferenceLine
          y={mh_a_min_level_kg}
          stroke="#E3000F"
          strokeDasharray="4 4"
          strokeOpacity={0.5}
          label={{ value: `Min A (${mh_a_min_level_kg}kg)`, fill: "#E3000F", fontSize: 8, position: "insideTopLeft" }}
        />
        <ReferenceLine
          y={mh_b_min_level_kg}
          stroke="#3B82F6"
          strokeDasharray="4 4"
          strokeOpacity={0.5}
          label={{ value: `Min B (${mh_b_min_level_kg}kg)`, fill: "#3B82F6", fontSize: 8, position: "insideTopRight" }}
        />
        <Line
          type="monotone"
          dataKey="mhA"
          name="M&H A"
          stroke="#E3000F"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="mhB"
          name="M&H B"
          stroke="#3B82F6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
