/**
 * SyncedScheduleChart — Gantt (IF A/B) + M&H levels on a shared time axis.
 *
 * Mirrors the matplotlib reference: Gantt rows on top, M&H level lines below,
 * solar window as a semi-transparent yellow band across both sections.
 */
import { useRef, useState } from "react";
import type { Batch, Plan, ScheduleData } from "@/types";

// ── Layout ────────────────────────────────────────────────────────────────
const PAD = { top: 16, left: 68, right: 24, bottom: 8 };
const GANTT_ROW_H = 32;
const GANTT_GAP = 8;
const SECTION_GAP = 18;
const MH_H = 120;
const AXIS_H = 26;
const LEGEND_H = 24;
const SVG_W = 660;

// ── Colors ────────────────────────────────────────────────────────────────
const C_MELT  = "#71717A";
const C_HOLD  = "#F59E0B";
const C_POUR  = "#22C55E";
const C_COLD  = "#93C5FD";
const C_SOLAR = "#FEF08A";
const C_MH_A  = "#E3000F";
const C_MH_B  = "#3B82F6";
const C_GRID  = "var(--border-color)";
const C_ROW_BG   = "var(--bg-elevated)";
const C_MH_BG    = "var(--bg-card)";
const C_TEXT_MUTED = "var(--text-muted)";
const C_TEXT_DIM   = "var(--text-dim)";
const C_NOW        = "var(--text-primary)";

// ── Derived layout ────────────────────────────────────────────────────────
const FURNACES = ["A", "B"] as const;
const GANTT_H  = FURNACES.length * (GANTT_ROW_H + GANTT_GAP) - GANTT_GAP;
const MH_Y     = PAD.top + GANTT_H + SECTION_GAP;
const AXIS_Y   = MH_Y + MH_H;
const SVG_H    = AXIS_Y + AXIS_H + LEGEND_H + PAD.bottom;
const CHART_W  = SVG_W - PAD.left - PAD.right;

function fmtHHMM(date: Date) {
  return `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

interface Props {
  plan: Plan;
  batches: Batch[];
  scheduleData: ScheduleData;
}

interface TooltipState {
  svgX: number;
  timeMin: number;
  mhA: number;
  mhB: number;
}

export default function SyncedScheduleChart({ plan, batches, scheduleData }: Props) {
  const shiftStart = new Date(plan.shift_start);
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // useState initializer runs once — safe place for Date.now()
  const [nowTime] = useState<number>(() => Date.now());

  // ── Batch windows (minutes from shift start) ───────────────────────────
  const batchWindows = batches
    .filter((b) => b.expected_start != null)
    .map((b) => {
      const ms0 = shiftStart.getTime();
      const startMin      = (new Date(b.expected_start!).getTime() - ms0) / 60_000;
      const meltFinishMin = b.melt_finish_at
        ? (new Date(b.melt_finish_at).getTime() - ms0) / 60_000
        : startMin + (b.duration_min ?? 88);
      const pourMin = b.pour_at
        ? (new Date(b.pour_at).getTime() - ms0) / 60_000
        : meltFinishMin;
      return { ...b, startMin, meltFinishMin, pourMin };
    });

  // ── Time domain ────────────────────────────────────────────────────────
  const tMax = batchWindows.length > 0
    ? Math.max(...batchWindows.map((b) => b.pourMin)) + 30
    : scheduleData.duration_minutes;
  const tSpan = tMax;

  // ── M&H data ───────────────────────────────────────────────────────────
  const {
    mh_a_levels_kg, mh_b_levels_kg,
    mh_a_min_level_kg, mh_b_min_level_kg,
    sample_interval_min,
  } = scheduleData;

  const maxMhY = Math.max(
    ...mh_a_levels_kg, ...mh_b_levels_kg,
    mh_a_min_level_kg * 1.4,
  ) * 1.08;

  // ── Coordinate helpers ─────────────────────────────────────────────────
  const toX    = (min: number) => PAD.left + (min / tSpan) * CHART_W;
  const toMhY  = (kg: number) => MH_Y + MH_H - (kg / maxMhY) * MH_H;

  // ── Time ticks (every 120 min = 2 hours) ──────────────────────────────
  const ticks: number[] = [];
  for (let t = 0; t <= tMax; t += 120) ticks.push(t);

  // ── Current time ───────────────────────────────────────────────────────
  const nowMin = (nowTime - shiftStart.getTime()) / 60_000;
  const showNow = nowMin >= 0 && nowMin <= tMax;
  const nowX = toX(nowMin);

  // ── Solar window ───────────────────────────────────────────────────────
  const swStart = scheduleData.solar_window_start_min;
  const swEnd   = scheduleData.solar_window_end_min;
  const solarVisible = swStart != null && swEnd != null && swStart < tMax && swEnd > 0;
  const solarX1 = toX(Math.max(0, swStart ?? 0));
  const solarX2 = toX(Math.min(tMax, swEnd ?? 0));

  function mhPolyline(levels: number[]): string {
    return levels
      .map((v, i) => {
        const min = i * sample_interval_min;
        if (min > tMax + sample_interval_min) return null;
        return `${toX(min).toFixed(1)},${toMhY(v).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");
  }

  const mhYTicks = [
    0,
    Math.round(maxMhY * 0.25),
    Math.round(maxMhY * 0.5),
    Math.round(maxMhY * 0.75),
    Math.round(maxMhY),
  ];

  // ── Tooltip mouse handler ──────────────────────────────────────────────
  function handleMouseMove(e: React.MouseEvent<SVGRectElement>) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const svgX = ((e.clientX - rect.left) / rect.width) * SVG_W;
    const min = ((svgX - PAD.left) / CHART_W) * tSpan;
    if (min < 0 || min > tMax) {
      setTooltip(null);
      return;
    }
    const idx = Math.min(
      Math.max(0, Math.round(min / sample_interval_min)),
      mh_a_levels_kg.length - 1,
    );
    setTooltip({
      svgX: Math.max(PAD.left, Math.min(svgX, PAD.left + CHART_W)),
      timeMin: min,
      mhA: mh_a_levels_kg[idx] ?? 0,
      mhB: mh_b_levels_kg[idx] ?? 0,
    });
  }

  // ── Tooltip rendering ──────────────────────────────────────────────────
  const TOOLTIP_W = 118;
  const TOOLTIP_H = 58;

  function renderTooltip() {
    if (!tooltip) return null;
    const { svgX, timeMin, mhA, mhB } = tooltip;
    const absTime = new Date(shiftStart.getTime() + timeMin * 60_000);
    const boxX = svgX + 8 + TOOLTIP_W > SVG_W - PAD.right
      ? svgX - TOOLTIP_W - 8
      : svgX + 8;
    const boxY = MH_Y + 6;
    const dotAY = toMhY(mhA);
    const dotBY = toMhY(mhB);
    return (
      <g pointerEvents="none">
        {/* Crosshair */}
        <line
          x1={svgX} y1={PAD.top} x2={svgX} y2={MH_Y + MH_H}
          stroke="white" strokeWidth={0.6} strokeDasharray="3 2" opacity={0.35}
        />
        {/* Dots on lines */}
        <circle cx={svgX} cy={dotAY} r={3.5} fill={C_MH_A} stroke="#18181B" strokeWidth={1.5} />
        <circle cx={svgX} cy={dotBY} r={3.5} fill={C_MH_B} stroke="#18181B" strokeWidth={1.5} />
        {/* Box background */}
        <rect x={boxX} y={boxY} width={TOOLTIP_W} height={TOOLTIP_H}
          fill="#18181B" stroke="#3F3F46" strokeWidth={1} rx={5} />
        {/* Time label */}
        <text x={boxX + 9} y={boxY + 15} fontSize={9} fill="#A1A1AA"
          fontFamily="JetBrains Mono, monospace">
          {fmtHHMM(absTime)}
        </text>
        {/* Divider */}
        <line x1={boxX + 9} y1={boxY + 20} x2={boxX + TOOLTIP_W - 9} y2={boxY + 20}
          stroke="#3F3F46" strokeWidth={0.5} />
        {/* M&H A row */}
        <rect x={boxX + 9} y={boxY + 26} width={8} height={8} fill={C_MH_A} rx={2} />
        <text x={boxX + 21} y={boxY + 34} fontSize={9} fill="#FAFAFA"
          fontFamily="JetBrains Mono, monospace">
          A: {mhA.toFixed(0)} kg
        </text>
        {/* M&H B row */}
        <rect x={boxX + 9} y={boxY + 40} width={8} height={8} fill={C_MH_B} rx={2} />
        <text x={boxX + 21} y={boxY + 48} fontSize={9} fill="#FAFAFA"
          fontFamily="JetBrains Mono, monospace">
          B: {mhB.toFixed(0)} kg
        </text>
      </g>
    );
  }

  return (
    <div className="w-full overflow-x-auto">
      <svg
        ref={svgRef}
        width="100%"
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        preserveAspectRatio="xMinYMid meet"
        style={{ display: "block" }}
      >
        {/* ── GANTT SECTION ────────────────────────────────────────────── */}

        {/* Row backgrounds */}
        {FURNACES.map((f, fi) => {
          const y = PAD.top + fi * (GANTT_ROW_H + GANTT_GAP);
          return <rect key={f} x={PAD.left} y={y} width={CHART_W} height={GANTT_ROW_H} fill={C_ROW_BG} rx={4} />;
        })}

        {/* Furnace labels */}
        {FURNACES.map((f, fi) => {
          const y = PAD.top + fi * (GANTT_ROW_H + GANTT_GAP);
          return (
            <text key={f} x={PAD.left - 6} y={y + GANTT_ROW_H / 2 + 4}
              textAnchor="end" fontSize={10} fill={C_TEXT_MUTED} fontFamily="Inter, sans-serif">
              IF {f}
            </text>
          );
        })}

        {/* Batch bars */}
        {batchWindows.map((b) => {
          const fi = FURNACES.indexOf((b.furnace ?? "A") as typeof FURNACES[number]);
          if (fi < 0) return null;
          const y    = PAD.top + fi * (GANTT_ROW_H + GANTT_GAP);
          const barY = y + 4;
          const barH = GANTT_ROW_H - 8;

          const meltX = toX(b.startMin);
          const meltW = Math.max(4, toX(b.meltFinishMin) - toX(b.startMin));
          const holdW = Math.max(0, toX(b.pourMin) - toX(b.meltFinishMin));
          const holdX = meltX + meltW;
          const pourX = holdX + holdW;

          return (
            <g key={b.id}>
              {/* Melt */}
              <rect x={meltX} y={barY} width={meltW} height={barH} fill={C_MELT} rx={3} opacity={0.9} />
              {meltW > 20 && (
                <text x={meltX + Math.min(meltW / 2, 18)} y={barY + barH / 2 + 4}
                  textAnchor="middle" fontSize={9} fill="white"
                  fontFamily="JetBrains Mono, monospace" fontWeight="600">
                  #{b.batch_number}
                </text>
              )}
              {meltW > 44 && b.power_kw != null && (
                <text x={meltX + meltW - 3} y={barY + barH / 2 + 4}
                  textAnchor="end" fontSize={8} fill="#D4D4D8"
                  fontFamily="JetBrains Mono, monospace">
                  {b.power_kw}kW
                </text>
              )}
              {b.is_cold_start && (
                <text x={meltX + 3} y={barY + 9} fontSize={8} fill={C_COLD} fontFamily="Inter, sans-serif">
                  ❄
                </text>
              )}
              {/* Hold */}
              {holdW > 0 && (
                <rect x={holdX} y={barY} width={holdW} height={barH} fill={C_HOLD} rx={0} opacity={0.9} />
              )}
              {/* Pour marker */}
              <rect
                x={Math.max(pourX - 2, meltX + meltW - 2)} y={barY}
                width={4} height={barH} fill={C_POUR} rx={1}
              />
            </g>
          );
        })}

        {/* ── SECTION DIVIDER ── */}
        <line
          x1={PAD.left} y1={MH_Y - SECTION_GAP / 2}
          x2={PAD.left + CHART_W} y2={MH_Y - SECTION_GAP / 2}
          stroke={C_GRID} strokeWidth={1} opacity={0.4}
        />

        {/* ── M&H SECTION ──────────────────────────────────────────────── */}

        {/* Background */}
        <rect x={PAD.left} y={MH_Y} width={CHART_W} height={MH_H} fill={C_MH_BG} rx={4} />

        {/* ── Solar window band (rendered after all backgrounds) ── */}
        {solarVisible && (
          <rect
            x={solarX1} y={PAD.top}
            width={solarX2 - solarX1}
            height={MH_Y + MH_H - PAD.top}
            fill={C_SOLAR} opacity={0.10}
          />
        )}

        {/* Y-axis ticks + horizontal grid */}
        {mhYTicks.map((kg) => {
          const y = toMhY(kg);
          return (
            <g key={kg}>
              <line
                x1={PAD.left} y1={y} x2={PAD.left + CHART_W} y2={y}
                stroke={C_GRID} strokeWidth={1} strokeDasharray="3 3" opacity={0.5}
              />
              <text x={PAD.left - 4} y={y + 3} textAnchor="end" fontSize={9}
                fill={C_TEXT_DIM} fontFamily="JetBrains Mono, monospace">
                {kg}
              </text>
            </g>
          );
        })}

        {/* kg unit */}
        <text
          x={10} y={MH_Y + MH_H / 2} textAnchor="middle" fontSize={8}
          fill={C_TEXT_DIM} fontFamily="Inter, sans-serif"
          transform={`rotate(-90, 10, ${MH_Y + MH_H / 2})`}>
          kg
        </text>

        {/* Min level reference lines */}
        <line
          x1={PAD.left} y1={toMhY(mh_a_min_level_kg)}
          x2={PAD.left + CHART_W} y2={toMhY(mh_a_min_level_kg)}
          stroke={C_MH_A} strokeWidth={1} strokeDasharray="5 3" opacity={0.5}
        />
        <text x={PAD.left + 4} y={toMhY(mh_a_min_level_kg) - 2} fontSize={8}
          fill={C_MH_A} fontFamily="Inter, sans-serif" opacity={0.8}>
          Min A ({mh_a_min_level_kg}kg)
        </text>

        <line
          x1={PAD.left} y1={toMhY(mh_b_min_level_kg)}
          x2={PAD.left + CHART_W} y2={toMhY(mh_b_min_level_kg)}
          stroke={C_MH_B} strokeWidth={1} strokeDasharray="5 3" opacity={0.5}
        />
        <text x={PAD.left + CHART_W - 4} y={toMhY(mh_b_min_level_kg) - 2}
          textAnchor="end" fontSize={8} fill={C_MH_B} fontFamily="Inter, sans-serif" opacity={0.8}>
          Min B ({mh_b_min_level_kg}kg)
        </text>

        {/* M&H level lines */}
        <polyline points={mhPolyline(mh_a_levels_kg)} fill="none" stroke={C_MH_A} strokeWidth={1.5} />
        <polyline points={mhPolyline(mh_b_levels_kg)} fill="none" stroke={C_MH_B} strokeWidth={1.5} />

        {/* ── SHARED TIME AXIS ──────────────────────────────────────────── */}
        {ticks.map((t) => {
          const x = toX(t);
          const absTime = new Date(shiftStart.getTime() + t * 60_000);
          return (
            <g key={t}>
              <line
                x1={x} y1={PAD.top} x2={x} y2={AXIS_Y}
                stroke={C_GRID} strokeWidth={1} strokeDasharray="3 3" opacity={0.35}
              />
              <text x={x} y={AXIS_Y + AXIS_H - 8} textAnchor="middle" fontSize={9}
                fill={C_TEXT_DIM} fontFamily="JetBrains Mono, monospace">
                {fmtHHMM(absTime)}
              </text>
            </g>
          );
        })}

        {/* ── Current time line ── */}
        {showNow && (
          <>
            <line
              x1={nowX} y1={PAD.top} x2={nowX} y2={MH_Y + MH_H}
              stroke={C_NOW} strokeWidth={0.8} opacity={0.8}
            />
            <text x={nowX + 3} y={PAD.top + 10} fontSize={8} fill={C_NOW} opacity={0.7}
              fontFamily="JetBrains Mono, monospace">
              Now
            </text>
          </>
        )}

        {/* ── Solar window borders + label ── */}
        {solarVisible && (
          <>
            <line x1={solarX1} y1={PAD.top} x2={solarX1} y2={MH_Y + MH_H}
              stroke="#CA8A04" strokeWidth={0.8} strokeDasharray="4 2" opacity={0.6} />
            <line x1={solarX2} y1={PAD.top} x2={solarX2} y2={MH_Y + MH_H}
              stroke="#CA8A04" strokeWidth={0.8} strokeDasharray="4 2" opacity={0.6} />
            <text x={(solarX1 + solarX2) / 2} y={PAD.top + 9} textAnchor="middle"
              fontSize={8} fill="#CA8A04" fontFamily="Inter, sans-serif" opacity={0.9}>
              ☀ Solar
            </text>
          </>
        )}

        {/* ── Tooltip overlay ───────────────────────────────────────────── */}
        {renderTooltip()}

        {/* ── Transparent mouse-capture rect ── */}
        <rect
          x={PAD.left} y={PAD.top}
          width={CHART_W} height={MH_Y + MH_H - PAD.top}
          fill="transparent"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
          style={{ cursor: "crosshair" }}
        />

        {/* ── LEGEND ────────────────────────────────────────────────────── */}
        {[
          { color: C_MELT, label: "Melt" },
          { color: C_HOLD, label: "Hold" },
          { color: C_POUR, label: "Pour" },
        ].map(({ color, label }, i) => (
          <g key={label} transform={`translate(${PAD.left + i * 56}, ${SVG_H - LEGEND_H + 4})`}>
            <rect width={10} height={10} fill={color} rx={2} />
            <text x={13} y={9} fontSize={9} fill={C_TEXT_MUTED} fontFamily="Inter, sans-serif">{label}</text>
          </g>
        ))}
        <g transform={`translate(${PAD.left + 3 * 56}, ${SVG_H - LEGEND_H + 4})`}>
          <text x={0} y={9} fontSize={9} fill={C_COLD} fontFamily="Inter, sans-serif">❄ Cold</text>
        </g>
        {solarVisible && (
          <g transform={`translate(${PAD.left + 4 * 56}, ${SVG_H - LEGEND_H + 4})`}>
            <rect width={10} height={10} fill={C_SOLAR} rx={2} opacity={0.6} />
            <text x={13} y={9} fontSize={9} fill="#CA8A04" fontFamily="Inter, sans-serif">Solar</text>
          </g>
        )}
        {/* M&H legend (right side) */}
        <g transform={`translate(${SVG_W - PAD.right - 112}, ${SVG_H - LEGEND_H + 4})`}>
          <line x1={0} y1={5} x2={12} y2={5} stroke={C_MH_A} strokeWidth={2} />
          <text x={16} y={9} fontSize={9} fill={C_TEXT_MUTED} fontFamily="Inter, sans-serif">M&amp;H A</text>
        </g>
        <g transform={`translate(${SVG_W - PAD.right - 56}, ${SVG_H - LEGEND_H + 4})`}>
          <line x1={0} y1={5} x2={12} y2={5} stroke={C_MH_B} strokeWidth={2} />
          <text x={16} y={9} fontSize={9} fill={C_TEXT_MUTED} fontFamily="Inter, sans-serif">M&amp;H B</text>
        </g>
      </svg>
    </div>
  );
}
