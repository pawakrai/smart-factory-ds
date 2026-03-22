import { useState } from "react";
import { Activity, Zap, Factory, TrendingUp, RefreshCw } from "lucide-react";
import client from "@/api/client";
import PageWrapper from "@/components/layout/PageWrapper";
import EnergyBarChart from "@/components/charts/EnergyBarChart";
import TrendSummaryWidget from "@/components/charts/TrendSummaryWidget";
import FurnaceStatusCard from "@/components/dashboard/FurnaceStatusCard";
import type { FurnaceData } from "@/components/dashboard/FurnaceStatusCard";
import { useEnergyLogs } from "@/hooks/useEnergyLogs";
import { ChartSkeleton, KpiCardSkeleton } from "@/components/ui/Skeleton";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import type { EnergyLog } from "@/types";

// Realistic 24-hour mock data (peak 09:00–22:00 ~420 kW, off-peak ~190 kW)
const today = new Date();
today.setMinutes(0, 0, 0);
const MOCK_ENERGY_LOGS: EnergyLog[] = Array.from({ length: 24 }, (_, h) => {
  const ts = new Date(today);
  ts.setHours(h);
  const isPeak = h >= 9 && h < 22;
  const base = isPeak ? 420 : 190;
  const noise = () => (Math.random() - 0.4) * 60;
  const sim_kw = Math.round(base + noise());
  const actual_kw = Math.round(sim_kw * (1 + (Math.random() - 0.45) * 0.12));
  return {
    id: `mock-${h}`,
    batch_id: null,
    timestamp: ts.toISOString(),
    sim_kw,
    actual_kw,
  };
});

const FURNACES: FurnaceData[] = [
  { name: "Induction Furnace A", temp: "720°C", power: "380 kW", status: "running" },
  { name: "Induction Furnace B", temp: "680°C", power: "340 kW", status: "running" },
  { name: "M&H Furnace A",       temp: "750°C", power: "450 kW", status: "running" },
  { name: "M&H Furnace B",       temp: "—",     power: "0 kW",   status: "offline" },
];

export default function DashboardPage() {
  const [seeding, setSeeding] = useState(false);
  const { data: logs = [], isLoading, refetch } = useEnergyLogs({ limit: 200 });

  // Use live data when available, fall back to mock data
  const displayLogs = logs.length > 0 ? logs : MOCK_ENERGY_LOGS;

  // Derive KPIs from live energy data
  const avgActual =
    displayLogs.reduce((s, l) => s + (l.actual_kw ?? 0), 0) / displayLogs.length;
  const avgSim =
    displayLogs.reduce((s, l) => s + (l.sim_kw ?? 0), 0) / displayLogs.length;
  const efficiencyPct = avgSim > 0 ? Math.min(100, (avgSim / avgActual) * 100) : null;

  const kpiCards = [
    {
      label: "Total Batches Today",
      value: "12",
      unit: "batches",
      icon: Factory,
      delta: "+2 vs yesterday",
    },
    {
      label: "Avg Power (Actual)",
      value: Math.round(avgActual).toString(),
      unit: "kW",
      icon: Zap,
      delta: `${Math.round(avgSim)} kW simulated`,
    },
    {
      label: "Active Furnaces",
      value: "3",
      unit: "/ 4",
      icon: Activity,
      delta: "1 offline",
    },
    {
      label: "Shift Efficiency",
      value: efficiencyPct != null ? efficiencyPct.toFixed(1) : "—",
      unit: "%",
      icon: TrendingUp,
      delta: efficiencyPct != null && efficiencyPct >= 90 ? "On target" : "Check schedule",
    },

  ];

  async function handleSeedMock() {
    setSeeding(true);
    try {
      await client.post("/energy-logs/seed-mock");
      refetch();
    } finally {
      setSeeding(false);
    }
  }

  return (
    <PageWrapper
      title="Dashboard"
      subtitle="Factory overview · Real-time monitoring"
      actions={
        <button
          onClick={handleSeedMock}
          disabled={seeding}
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-[var(--text-primary)] border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
        >
          <RefreshCw size={12} className={seeding ? "animate-spin" : ""} />
          {seeding ? "Seeding…" : "Seed mock data"}
        </button>
      }
    >
      {/* KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => <KpiCardSkeleton key={i} />)
          : kpiCards.map((kpi) => (
            <div
              key={kpi.label}
              className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-zinc-400 font-medium uppercase tracking-wide">
                  {kpi.label}
                </span>
                <kpi.icon size={16} className="text-zinc-600" />
              </div>
              <div className="flex items-end gap-2">
                <span className="font-kpi text-2xl text-[var(--text-primary)]">{kpi.value}</span>
                <span className="text-sm text-zinc-500 mb-0.5">{kpi.unit}</span>
              </div>
              <div className="mt-2 text-xs text-zinc-500">{kpi.delta}</div>
            </div>
          ))}
      </div>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Energy Bar Chart — 2/3 width */}
        <div className="lg:col-span-2 bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">
              Energy Consumption — Simulated vs Actual
            </h2>
            <div className="flex items-center gap-4 text-xs text-zinc-400">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-brand-red inline-block" />
                Simulated
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-sm bg-blue-500 inline-block" />
                Actual
              </span>
            </div>
          </div>

          {isLoading ? (
            <div className="h-56"><ChartSkeleton /></div>
          ) : (
            <ErrorBoundary label="Energy Chart">
              <EnergyBarChart logs={displayLogs} />
            </ErrorBoundary>
          )}
        </div>

        {/* Furnace status list — 1/3 width */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4">Furnace Status</h2>
          <div className="space-y-2.5">
            {FURNACES.map((f) => (
              <FurnaceStatusCard key={f.name} furnace={f} />
            ))}
          </div>
        </div>
      </div>

      {/* Bottom row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-4">
        {/* Operation Status */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Operation Status</h2>
          <div className="space-y-2">
            {[
              { label: "Production",   dot: "#22C55E" },
              { label: "Shipping",     dot: "#52525B" },
              { label: "Maintenance",  dot: "#E3000F" },
            ].map((s) => (
              <div key={s.label} className="flex items-center justify-between py-1">
                <span className="text-xs text-zinc-400">{s.label}</span>
                <span
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: s.dot }}
                />
              </div>
            ))}
          </div>

          {/* Capacity utilisation donut (simplified) */}
          <div className="mt-4 pt-4 border-t border-[var(--border-color)]">
            <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-2">
              Furnace Utilisation
            </p>
            <div className="flex items-center gap-3">
              {/* Simple SVG donut */}
              <svg width={48} height={48} viewBox="0 0 36 36">
                <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--bg-elevated)" strokeWidth="3.8" />
                {/* 75% utilisation = 3/4 of 100 units circumference */}
                <circle
                  cx="18" cy="18" r="15.9"
                  fill="none"
                  stroke="#E3000F"
                  strokeWidth="3.8"
                  strokeDasharray="75 25"
                  strokeDashoffset="25"
                  strokeLinecap="round"
                />
              </svg>
              <div>
                <p className="font-kpi text-lg text-[var(--text-primary)]">75%</p>
                <p className="text-[10px] text-zinc-500">3 of 4 active</p>
              </div>
            </div>
          </div>
        </div>

        {/* Trend Summary — 2/3 width */}
        <div className="lg:col-span-2 bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Trend Summary</h2>
          {isLoading ? (
            <div className="h-28"><ChartSkeleton /></div>
          ) : (
            <ErrorBoundary label="Trend Chart">
              <TrendSummaryWidget logs={displayLogs} />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </PageWrapper>
  );
}
