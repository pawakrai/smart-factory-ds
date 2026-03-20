import type { Batch, Plan } from "@/types";

// Layout constants
const ROW_HEIGHT = 36;
const ROW_GAP = 10;
const AXIS_HEIGHT = 28;
const LEGEND_HEIGHT = 20;
const PADDING = { top: 12, left: 72, right: 16, bottom: 8 };

// Phase colors
const COLOR_MELT = "#71717A";     // zinc-500 — melting phase
const COLOR_HOLD = "#F59E0B";     // amber   — holding/queue phase
const COLOR_POUR = "#22C55E";     // green   — pour complete marker

function formatHHMM(date: Date): string {
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function msToMin(ms: number) {
  return ms / 60_000;
}

interface Props {
  plan: Plan;
  batches: Batch[];
}

export default function GanttChart({ plan, batches }: Props) {
  if (batches.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-zinc-600 text-sm">
        No schedule data
      </div>
    );
  }

  const shiftStart = new Date(plan.shift_start);

  // Build time windows per batch (all in minutes from shiftStart)
  const batchWindows = batches
    .filter((b) => b.expected_start != null)
    .map((b) => {
      const startMin = msToMin(new Date(b.expected_start!).getTime() - shiftStart.getTime());
      const dur = b.duration_min ?? 88;

      // Melt phase end
      const meltFinishMin = b.melt_finish_at
        ? msToMin(new Date(b.melt_finish_at).getTime() - shiftStart.getTime())
        : startMin + dur;

      // Pour time
      const pourMin = b.pour_at
        ? msToMin(new Date(b.pour_at).getTime() - shiftStart.getTime())
        : meltFinishMin;

      return { ...b, startMin, meltFinishMin, pourMin };
    });

  if (batchWindows.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-zinc-600 text-sm">
        No schedule data
      </div>
    );
  }

  const minTime = Math.min(0, Math.min(...batchWindows.map((b) => b.startMin)));
  const maxTime = Math.max(...batchWindows.map((b) => b.pourMin)) + 30;
  const totalMin = maxTime - minTime;

  // Tick marks every 60 min
  const ticks: number[] = [];
  for (let t = Math.ceil(minTime / 60) * 60; t <= maxTime; t += 60) {
    ticks.push(t);
  }

  const furnaces = ["A", "B"];
  const svgWidth = 600;
  const chartW = svgWidth - PADDING.left - PADDING.right;
  const chartH = furnaces.length * (ROW_HEIGHT + ROW_GAP) - ROW_GAP;
  const svgH = PADDING.top + chartH + AXIS_HEIGHT + LEGEND_HEIGHT + PADDING.bottom;

  const toX = (min: number) => ((min - minTime) / totalMin) * chartW;
  const toY = (furnace: string) =>
    PADDING.top + furnaces.indexOf(furnace) * (ROW_HEIGHT + ROW_GAP);

  return (
    <div className="w-full overflow-x-auto">
      <svg
        width="100%"
        viewBox={`0 0 ${svgWidth} ${svgH}`}
        preserveAspectRatio="xMinYMid meet"
        style={{ display: "block" }}
      >
        {/* Furnace labels */}
        {furnaces.map((f) => (
          <text
            key={f}
            x={PADDING.left - 8}
            y={toY(f) + ROW_HEIGHT / 2 + 4}
            textAnchor="end"
            fontSize={10}
            fill="var(--text-muted)"
            fontFamily="Inter, sans-serif"
          >
            IF {f}
          </text>
        ))}

        {/* Row backgrounds */}
        {furnaces.map((f) => (
          <rect
            key={f}
            x={PADDING.left}
            y={toY(f)}
            width={chartW}
            height={ROW_HEIGHT}
            fill="var(--bg-elevated)"
            rx={4}
          />
        ))}

        {/* Time grid lines */}
        {ticks.map((t) => {
          const x = PADDING.left + toX(t);
          const absTime = new Date(shiftStart.getTime() + t * 60_000);
          return (
            <g key={t}>
              <line
                x1={x} y1={PADDING.top}
                x2={x} y2={PADDING.top + chartH}
                stroke="var(--border-color)" strokeWidth={1} strokeDasharray="3 3"
              />
              <text
                x={x}
                y={PADDING.top + chartH + AXIS_HEIGHT - 8}
                textAnchor="middle"
                fontSize={9}
                fill="var(--text-dim)"
                fontFamily="JetBrains Mono, monospace"
              >
                {formatHHMM(absTime)}
              </text>
            </g>
          );
        })}

        {/* Batch bars — 3 phases */}
        {batchWindows.map((b) => {
          const furnace = b.furnace ?? "A";
          const y = toY(furnace);
          const barY = y + 4;
          const barH = ROW_HEIGHT - 8;

          const meltX = PADDING.left + toX(b.startMin);
          const meltW = Math.max(4, toX(b.meltFinishMin) - toX(b.startMin));
          const holdW = Math.max(0, toX(b.pourMin) - toX(b.meltFinishMin));
          const holdX = meltX + meltW;
          const pourX = holdX + holdW;
          const powerLabel = b.power_kw != null ? `${b.power_kw}kW` : "";

          return (
            <g key={b.id}>
              {/* Melt phase (gray) */}
              <rect x={meltX} y={barY} width={meltW} height={barH}
                fill={COLOR_MELT} rx={3} opacity={0.9} />

              {/* Batch number inside melt bar */}
              {meltW > 22 && (
                <text
                  x={meltX + Math.min(meltW / 2, 20)}
                  y={barY + barH / 2 + 4}
                  textAnchor="middle" fontSize={9} fill="white"
                  fontFamily="JetBrains Mono, monospace" fontWeight="600"
                >
                  #{b.batch_number}
                </text>
              )}

              {/* Power label inside melt bar (if wide enough) */}
              {meltW > 40 && powerLabel && (
                <text
                  x={meltX + meltW - 4}
                  y={barY + barH / 2 + 4}
                  textAnchor="end" fontSize={8} fill="#D4D4D8"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {powerLabel}
                </text>
              )}

              {/* Cold start badge */}
              {b.is_cold_start && (
                <text
                  x={meltX + 3} y={barY + 9}
                  fontSize={8} fill="#93C5FD"
                  fontFamily="Inter, sans-serif"
                >
                  ❄
                </text>
              )}

              {/* Hold/queue phase (amber) */}
              {holdW > 0 && (
                <rect x={holdX} y={barY} width={holdW} height={barH}
                  fill={COLOR_HOLD} rx={0} opacity={0.9} />
              )}

              {/* Pour marker (green tick) */}
              <rect
                x={Math.max(pourX - 2, meltX + meltW - 2)}
                y={barY}
                width={4} height={barH}
                fill={COLOR_POUR} rx={1}
              />
            </g>
          );
        })}

        {/* Legend */}
        {[
          { color: COLOR_MELT, label: "Melt" },
          { color: COLOR_HOLD, label: "Hold" },
          { color: COLOR_POUR, label: "Pour" },
        ].map(({ color, label }, i) => (
          <g key={label} transform={`translate(${PADDING.left + i * 70}, ${svgH - LEGEND_HEIGHT + 4})`}>
            <rect width={10} height={10} fill={color} rx={2} />
            <text x={14} y={9} fontSize={9} fill="var(--text-muted)" fontFamily="Inter, sans-serif">
              {label}
            </text>
          </g>
        ))}
        <g transform={`translate(${PADDING.left + 3 * 70}, ${svgH - LEGEND_HEIGHT + 4})`}>
          <text x={0} y={9} fontSize={9} fill="#93C5FD" fontFamily="Inter, sans-serif">
            ❄ Cold Start
          </text>
        </g>
      </svg>
    </div>
  );
}
