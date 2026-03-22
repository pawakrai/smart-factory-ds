import { useState, useCallback, useEffect, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  CalendarClock, Loader2, ChevronRight, AlertCircle, Zap, Factory,
  Trash2, Flame, ChevronDown, ChevronUp, CheckCircle, Calendar, Filter,
} from "lucide-react";

import PageWrapper from "@/components/layout/PageWrapper";
import SyncedScheduleChart from "@/components/charts/SyncedScheduleChart";
import PlantLoadChart from "@/components/charts/PlantLoadChart";
import TouPriceChart from "@/components/charts/TouPriceChart";
import {
  usePlans, usePlanBatches, usePlanScheduleData,
  useCreatePlan, useDeletePlan, usePlanPolling, useActivatePlan,
} from "@/hooks/usePlans";
import { useAppStore } from "@/store/appStore";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import type { Plan, ScheduleMetrics } from "@/types";

// ── Form schema ────────────────────────────────────────────────────────────
const schema = z.object({
  target_batches: z.number().int().min(1, "Min 1").max(20, "Max 20"),
  shift_start: z.string().min(1, "Required"),
  opt_mode: z.enum(["energy", "service"]),
  if_a_enabled: z.boolean(),
  if_b_enabled: z.boolean(),
  mh_a_consumption_rate: z.number().min(0.1, "Min 0.1").max(10, "Max 10"),
  mh_b_consumption_rate: z.number().min(0.1, "Min 0.1").max(10, "Max 10"),
}).refine((d) => d.if_a_enabled || d.if_b_enabled, {
  message: "At least one furnace must be enabled",
  path: ["if_a_enabled"],
});
type FormValues = z.infer<typeof schema>;

// ── Helpers ────────────────────────────────────────────────────────────────
function statusBadge(status: string) {
  const map: Record<string, string> = {
    pending:   "bg-amber-500/15 text-amber-600",
    draft:     "bg-[var(--bg-elevated)] text-[var(--text-muted)]",
    active:    "bg-green-500/15 text-green-600",
    completed: "bg-blue-500/15 text-blue-600",
  };
  return map[status] ?? "bg-[var(--bg-elevated)] text-[var(--text-muted)]";
}

function fmtBaht(n: number) {
  return `฿${n.toLocaleString("en", { maximumFractionDigits: 0 })}`;
}

function fmtKwh(n: number) {
  return `${n.toLocaleString("en", { maximumFractionDigits: 0 })} kWh`;
}

function isSameDay(a: Date, b: Date) {
  return a.getFullYear() === b.getFullYear()
    && a.getMonth() === b.getMonth()
    && a.getDate() === b.getDate();
}

// ── Plan Form ──────────────────────────────────────────────────────────────
function PlanForm({ onSuccess }: { onSuccess: (plan: Plan) => void }) {
  const { mutate, isPending, error } = useCreatePlan();

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      target_batches: 12,
      opt_mode: "energy",
      if_a_enabled: true,
      if_b_enabled: true,
      mh_a_consumption_rate: 2.20,
      mh_b_consumption_rate: 2.30,
      shift_start: (() => {
        const d = new Date();
        d.setHours(8, 0, 0, 0);
        const pad = (n: number) => String(n).padStart(2, "0");
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
      })(),
    },
  });

  const optMode = watch("opt_mode");
  const ifAEnabled = watch("if_a_enabled");
  const ifBEnabled = watch("if_b_enabled");

  function onSubmit(values: FormValues) {
    mutate(
      {
        target_batches: values.target_batches,
        shift_start: values.shift_start.length === 16
          ? values.shift_start + ":00"
          : values.shift_start,
        opt_mode: values.opt_mode,
        if_a_enabled: values.if_a_enabled,
        if_b_enabled: values.if_b_enabled,
        mh_a_consumption_rate: values.mh_a_consumption_rate,
        mh_b_consumption_rate: values.mh_b_consumption_rate,
      },
      { onSuccess }
    );
  }

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      {/* Target batches */}
      <div>
        <label className="block text-xs text-zinc-400 mb-1.5">Target Batches</label>
        <input
          type="number"
          {...register("target_batches", { valueAsNumber: true })}
          className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red"
          placeholder="e.g. 8"
        />
        {errors.target_batches && (
          <p className="text-xs text-brand-red mt-1">{errors.target_batches.message}</p>
        )}
      </div>

      {/* Shift start */}
      <div>
        <label className="block text-xs text-zinc-400 mb-1.5">Shift Start</label>
        <input
          type="datetime-local"
          {...register("shift_start")}
          className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
        />
        {errors.shift_start && (
          <p className="text-xs text-brand-red mt-1">{errors.shift_start.message}</p>
        )}
      </div>

      {/* IF Furnace toggles */}
      <div>
        <div className="flex items-center gap-1.5 mb-1.5">
          <Flame size={11} className="text-zinc-400" />
          <label className="text-xs text-zinc-400">Induction Furnaces</label>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setValue("if_a_enabled", !ifAEnabled)}
            className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium border transition-colors ${
              ifAEnabled
                ? "bg-brand-red border-brand-red text-white"
                : "bg-bg-elevated border-[var(--border-color)] text-zinc-500 hover:border-zinc-500"
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${ifAEnabled ? "bg-white" : "bg-zinc-600"}`} />
            IF-A
          </button>
          <button
            type="button"
            onClick={() => setValue("if_b_enabled", !ifBEnabled)}
            className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium border transition-colors ${
              ifBEnabled
                ? "bg-brand-red border-brand-red text-white"
                : "bg-bg-elevated border-[var(--border-color)] text-zinc-500 hover:border-zinc-500"
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${ifBEnabled ? "bg-white" : "bg-zinc-600"}`} />
            IF-B
          </button>
        </div>
        {errors.if_a_enabled && (
          <p className="text-xs text-brand-red mt-1">{errors.if_a_enabled.message}</p>
        )}
      </div>

      {/* M&H Consumption Rate */}
      <div>
        <label className="block text-xs text-zinc-400 mb-1.5">M&H Consumption Rate (kg/min)</label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-[10px] text-zinc-600 mb-1">MH-A</label>
            <input
              type="number"
              step="0.01"
              {...register("mh_a_consumption_rate", { valueAsNumber: true })}
              className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
            />
            {errors.mh_a_consumption_rate && (
              <p className="text-[10px] text-brand-red mt-0.5">{errors.mh_a_consumption_rate.message}</p>
            )}
          </div>
          <div>
            <label className="block text-[10px] text-zinc-600 mb-1">MH-B</label>
            <input
              type="number"
              step="0.01"
              {...register("mh_b_consumption_rate", { valueAsNumber: true })}
              className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
            />
            {errors.mh_b_consumption_rate && (
              <p className="text-[10px] text-brand-red mt-0.5">{errors.mh_b_consumption_rate.message}</p>
            )}
          </div>
        </div>
      </div>

      {/* Opt mode toggle */}
      <div>
        <label className="block text-xs text-zinc-400 mb-1.5">Optimization Goal</label>
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => setValue("opt_mode", "energy")}
            className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium border transition-colors ${
              optMode === "energy"
                ? "bg-brand-red border-brand-red text-white"
                : "bg-bg-elevated border-[var(--border-color)] text-zinc-400 hover:border-zinc-500"
            }`}
          >
            <Zap size={12} />
            Energy Cost
          </button>
          <button
            type="button"
            onClick={() => setValue("opt_mode", "service")}
            className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium border transition-colors ${
              optMode === "service"
                ? "bg-brand-red border-brand-red text-white"
                : "bg-bg-elevated border-[var(--border-color)] text-zinc-400 hover:border-zinc-500"
            }`}
          >
            <Factory size={12} />
            Max Output
          </button>
        </div>
        <p className="text-[10px] text-zinc-600 mt-1">
          {optMode === "energy"
            ? "Minimize electricity cost using TOU pricing"
            : "Maximize batches poured, minimize M&H empty time"}
        </p>
      </div>

      {/* Error from API */}
      {error && (
        <div className="flex items-center gap-2 text-xs text-brand-red bg-brand-red/5 border border-brand-red/20 rounded-lg px-3 py-2">
          <AlertCircle size={12} />
          Failed to create plan. Please try again.
        </div>
      )}

      <button
        type="submit"
        disabled={isPending}
        className="w-full flex items-center justify-center gap-2 bg-brand-red hover:bg-brand-red-dark disabled:opacity-60 text-white text-sm font-semibold py-2.5 rounded-lg transition-colors mt-2"
      >
        {isPending ? (
          <>
            <Loader2 size={14} className="animate-spin" />
            Running GA…
          </>
        ) : (
          "Create Plan (Run GA)"
        )}
      </button>

      {isPending && (
        <p className="text-[11px] text-zinc-500 text-center">
          Submitting…
        </p>
      )}
    </form>
  );
}

// ── Plan list row ──────────────────────────────────────────────────────────
function PlanRow({
  plan, active, onClick, onContextMenu,
}: {
  plan: Plan;
  active: boolean;
  onClick: () => void;
  onContextMenu: (e: React.MouseEvent) => void;
}) {
  const isPending = plan.status === "pending";
  return (
    <div
      className={`flex items-center gap-1 rounded-lg border transition-colors cursor-pointer ${
        active ? "bg-brand-red/10 border-brand-red/30" : "bg-bg-elevated hover:bg-[var(--border-color)]/40 border-transparent"
      } ${isPending ? "opacity-70" : ""}`}
      onContextMenu={onContextMenu}
    >
      <button
        onClick={onClick}
        disabled={isPending}
        className="flex-1 flex items-center justify-between px-3 py-2.5 text-left min-w-0"
      >
        <div className="min-w-0">
          <p className="text-xs font-medium text-[var(--text-primary)] truncate">
            {plan.target_batches} batches · {plan.opt_mode === "energy" ? "Energy" : "Service"} mode
          </p>
          <p className="text-[10px] text-zinc-500 mt-0.5">
            {new Date(plan.shift_start).toLocaleString("en-GB", {
              day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
            })}
          </p>
        </div>
        <div className="flex items-center gap-2 ml-2 shrink-0">
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${statusBadge(plan.status)}`}>
            {isPending ? (
              <span className="flex items-center gap-1">
                <Loader2 size={8} className="animate-spin" />
                running…
              </span>
            ) : plan.status}
          </span>
          {!isPending && <ChevronRight size={12} className="text-zinc-600" />}
        </div>
      </button>
    </div>
  );
}

// ── KPI card ──────────────────────────────────────────────────────────────
function KpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-bg-elevated rounded-lg px-3 py-2.5">
      <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-0.5">{label}</p>
      <p className="font-kpi text-sm text-[var(--text-primary)]">{value}</p>
      {sub && <p className="text-[10px] text-zinc-600 mt-0.5">{sub}</p>}
    </div>
  );
}

// ── Context menu ───────────────────────────────────────────────────────────
type ContextMenuState = { planId: string; x: number; y: number } | null;

// ── Main page ──────────────────────────────────────────────────────────────
export default function ProductionPlanningPage() {
  const selectedPlanId = useAppStore((s) => s.selectedPlanId);
  const setSelectedPlanId = useAppStore((s) => s.setSelectedPlanId);
  const setAppActivePlan = useAppStore((s) => s.setActivePlan);

  const [pollingPlanId, setPollingPlanId] = useState<string | null>(null);
  const [deletingPlanId, setDeletingPlanId] = useState<string | null>(null);
  const [showAdvancedCharts, setShowAdvancedCharts] = useState(false);
  const [contextMenu, setContextMenu] = useState<ContextMenuState>(null);
  const [filterDate, setFilterDate] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  const contextMenuRef = useRef<HTMLDivElement>(null);

  const { data: plans = [] } = usePlans();
  const { data: batches = [], isLoading: batchesLoading } = usePlanBatches(selectedPlanId);
  const { data: scheduleData, isLoading: scheduleLoading } = usePlanScheduleData(selectedPlanId);
  const { mutate: deletePlan } = useDeletePlan();
  const { mutate: activatePlan } = useActivatePlan();

  // Auto-select: on mount or when plans change, pick the best plan to view
  useEffect(() => {
    if (!plans.length) return;
    if (selectedPlanId && plans.find((p) => p.id === selectedPlanId)) return;

    const today = new Date();
    const todayActive = plans.find(
      (p) => p.status === "active" && isSameDay(new Date(p.shift_start), today)
    );
    if (todayActive) {
      setSelectedPlanId(todayActive.id);
      return;
    }
    setSelectedPlanId(plans[0].id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plans]);

  // Close context menu on outside click
  useEffect(() => {
    function close(e: MouseEvent) {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenu(null);
      }
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const onGaComplete = useCallback((planId: string) => {
    setSelectedPlanId(planId);
    setPollingPlanId(null);
  }, [setSelectedPlanId]);

  function confirmDelete() {
    if (!deletingPlanId) return;
    const planId = deletingPlanId;
    setDeletingPlanId(null);
    deletePlan(planId, {
      onSuccess: () => {
        if (selectedPlanId === planId) setSelectedPlanId(null);
        if (pollingPlanId === planId) setPollingPlanId(null);
        // If we deleted the appStore activePlan, clear it
        setAppActivePlan(null);
      },
    });
  }

  function handleActivatePlan(planId: string) {
    setContextMenu(null);
    activatePlan(planId, {
      onSuccess: (updatedPlan) => {
        setSelectedPlanId(updatedPlan.id);
        setAppActivePlan(updatedPlan);
      },
    });
  }

  function handleContextMenu(e: React.MouseEvent, planId: string) {
    e.preventDefault();
    const x = Math.min(e.clientX, window.innerWidth - 180);
    const y = Math.min(e.clientY, window.innerHeight - 90);
    setContextMenu({ planId, x, y });
  }

  usePlanPolling(pollingPlanId, onGaComplete);

  const selectedPlan = plans.find((p) => p.id === selectedPlanId) ?? null;
  const isGaRunning = !!pollingPlanId;

  function handlePlanCreated(plan: Plan) {
    setPollingPlanId(plan.id);
    setSelectedPlanId(plan.id);
  }

  const metrics: ScheduleMetrics | null = (() => {
    if (selectedPlan?.schedule_metrics) {
      try { return JSON.parse(selectedPlan.schedule_metrics); } catch { return null; }
    }
    return null;
  })();

  const batchCount = batches.length;
  const furnaceA = batches.filter((b) => b.furnace === "A").length;
  const furnaceB = batches.filter((b) => b.furnace === "B").length;

  // Today's plans
  const today = new Date();
  const todayPlans = plans.filter((p) => isSameDay(new Date(p.shift_start), today));

  // Filtered Recent Plans (excludes today's plans, applies filters)
  const filteredPlans = plans.filter((p) => {
    const matchDate = !filterDate
      || new Date(p.shift_start).toLocaleDateString("en-CA") === filterDate;
    const matchStatus = filterStatus === "all" || p.status === filterStatus;
    return matchDate && matchStatus;
  });

  // Context menu's plan (for checking status)
  const contextPlan = plans.find((p) => p.id === contextMenu?.planId);
  const canActivate = contextPlan
    && contextPlan.status !== "pending"
    && contextPlan.status !== "completed";

  return (
    <PageWrapper
      title="Production Planning"
      subtitle="Configure shift parameters and generate GA-optimised schedule"
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* ── Left panel: form + plan lists ─────────────────────────── */}
        <div className="space-y-4">
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <CalendarClock size={16} className="text-brand-red" />
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Plan Parameters</h2>
            </div>
            <PlanForm onSuccess={handlePlanCreated} />
          </div>

          {/* Today's Plans box */}
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Calendar size={14} className="text-brand-red" />
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Today's Plans</h2>
              <span className="ml-auto text-[10px] text-zinc-500">
                {new Date().toLocaleDateString("en-GB", { day: "2-digit", month: "short" })}
              </span>
            </div>
            {todayPlans.length === 0 ? (
              <p className="text-xs text-zinc-600 py-2">No plans for today</p>
            ) : (
              <div className="space-y-1.5">
                {todayPlans.map((plan) => (
                  <PlanRow
                    key={plan.id}
                    plan={plan}
                    active={plan.id === selectedPlanId}
                    onClick={() => setSelectedPlanId(plan.id)}
                    onContextMenu={(e) => handleContextMenu(e, plan.id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Recent Plans with filters */}
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Recent Plans</h2>

            {/* Filters */}
            <div className="flex items-center gap-2 mb-3">
              <div className="flex items-center gap-1 text-zinc-500">
                <Filter size={11} />
              </div>
              <input
                type="date"
                value={filterDate}
                onChange={(e) => setFilterDate(e.target.value)}
                className="flex-1 bg-bg-elevated border border-[var(--border-color)] rounded px-2 py-1 text-[10px] text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
              />
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="bg-bg-elevated border border-[var(--border-color)] rounded px-2 py-1 text-[10px] text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
              >
                <option value="all">All</option>
                <option value="pending">Pending</option>
                <option value="draft">Draft</option>
                <option value="active">Active</option>
                <option value="completed">Completed</option>
              </select>
            </div>

            {filteredPlans.length === 0 ? (
              <p className="text-xs text-zinc-600 py-2">No plans match the filters</p>
            ) : (
              <div className="space-y-1.5 max-h-52 overflow-y-auto">
                {filteredPlans.slice(0, 20).map((plan) => (
                  <PlanRow
                    key={plan.id}
                    plan={plan}
                    active={plan.id === selectedPlanId}
                    onClick={() => setSelectedPlanId(plan.id)}
                    onContextMenu={(e) => handleContextMenu(e, plan.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Right panel: charts ───────────────────────────────────── */}
        <div className="lg:col-span-3 space-y-4">
          {/* Combined synced chart */}
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Production Schedule</h2>
              {selectedPlan && batchCount > 0 && (
                <div className="flex items-center gap-3 text-[10px] text-zinc-400">
                  <span>IF-A: {furnaceA}</span>
                  <span>IF-B: {furnaceB}</span>
                </div>
              )}
            </div>

            {!selectedPlan ? (
              <div className="h-64 flex flex-col items-center justify-center gap-2 text-zinc-600 text-sm border border-dashed border-[var(--border-color)] rounded-lg">
                <CalendarClock size={24} className="opacity-40" />
                Create a plan to see the schedule
              </div>
            ) : isGaRunning ? (
              <div className="h-64 flex flex-col items-center justify-center gap-2 text-[var(--text-muted)] text-sm border border-dashed border-[var(--border-color)] rounded-lg">
                <Loader2 size={24} className="animate-spin text-brand-red" />
                <p>GA optimising schedule…</p>
                <p className="text-[11px] text-zinc-600">This may take 20–60 s</p>
              </div>
            ) : batchesLoading || scheduleLoading ? (
              <div className="h-64 flex items-center justify-center">
                <Loader2 size={20} className="animate-spin text-brand-red" />
              </div>
            ) : batchCount === 0 ? (
              <div className="h-64 flex items-center justify-center text-zinc-600 text-sm">
                No batches — GA may have failed
              </div>
            ) : !scheduleData ? (
              <div className="h-64 flex items-center justify-center text-zinc-600 text-sm">
                Schedule data unavailable
              </div>
            ) : (
              <ErrorBoundary label="Schedule Chart">
                <SyncedScheduleChart
                  plan={selectedPlan}
                  batches={batches}
                  scheduleData={scheduleData}
                />
              </ErrorBoundary>
            )}
          </div>

          {/* Advanced charts toggle */}
          {scheduleData && (
            <div>
              <button
                onClick={() => setShowAdvancedCharts((v) => !v)}
                className="flex items-center gap-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors border border-[var(--border-color)] hover:border-[var(--text-dim)] rounded-lg px-3 py-2"
              >
                <Zap size={12} />
                Plant Load &amp; TOU
                {showAdvancedCharts ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              </button>
              {showAdvancedCharts && (
                <div className="mt-3 space-y-4">
                  <ErrorBoundary label="Plant Load Chart">
                    <PlantLoadChart data={scheduleData} />
                  </ErrorBoundary>
                  <ErrorBoundary label="TOU Price Chart">
                    <TouPriceChart data={scheduleData} />
                  </ErrorBoundary>
                </div>
              )}
            </div>
          )}

          {/* Schedule summary — 6 KPI cards */}
          {selectedPlan && batchCount > 0 && (
            <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
              <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Schedule Summary</h2>
              <div className="grid grid-cols-3 gap-3">
                <KpiCard
                  label="Batches Poured"
                  value={`${metrics?.poured_batches_count ?? batchCount} / ${selectedPlan.target_batches}`}
                  sub={metrics?.missing_batches != null ? `${metrics.missing_batches} missing` : undefined}
                />
                <KpiCard
                  label="IF-A / IF-B"
                  value={`${furnaceA} / ${furnaceB}`}
                  sub="starts"
                />
                <KpiCard
                  label="Total Energy"
                  value={metrics ? fmtKwh(metrics.total_if_kwh) : "—"}
                />
                <KpiCard
                  label="Energy Cost"
                  value={metrics ? fmtBaht(metrics.total_energy_cost_day) : "—"}
                  sub="est. day"
                />
                <KpiCard
                  label="Peak Demand"
                  value={metrics ? `${metrics.peak_kw.toFixed(0)} kW` : "—"}
                />
                <KpiCard
                  label="Solar Savings"
                  value={metrics ? fmtBaht(metrics.solar_cost_saving) : "—"}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Delete confirmation dialog ─────────────────────────────── */}
      <AlertDialog open={!!deletingPlanId} onOpenChange={(open) => !open && setDeletingPlanId(null)}>
        <AlertDialogContent className="bg-bg-card border border-[var(--border-color)] text-[var(--text-primary)]">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[var(--text-primary)]">Delete Plan</AlertDialogTitle>
            <AlertDialogDescription className="text-zinc-400">
              Plan นี้จะถูกลบพร้อมกับ batch ทั้งหมด ไม่สามารถกู้คืนได้
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-bg-elevated border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-[var(--bg-elevated)] hover:text-[var(--text-primary)]">
              Cancel
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={confirmDelete}
              className="bg-brand-red hover:bg-brand-red-dark text-white border-0"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* ── Right-click context menu ──────────────────────────────── */}
      {contextMenu && (
        <div
          ref={contextMenuRef}
          className="fixed z-50 bg-bg-card border border-[var(--border-color)] rounded-lg shadow-xl py-1 min-w-[168px]"
          style={{ top: contextMenu.y, left: contextMenu.x }}
        >
          <button
            disabled={!canActivate}
            onClick={() => handleActivatePlan(contextMenu.planId)}
            className={`w-full px-3 py-2 text-left text-xs flex items-center gap-2 transition-colors ${
              canActivate
                ? "text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] cursor-pointer"
                : "text-[var(--text-dim)] cursor-not-allowed"
            }`}
          >
            <CheckCircle size={12} className={canActivate ? "text-green-400" : "text-zinc-600"} />
            Set as Active Plan
          </button>
          <div className="border-t border-[var(--border-color)] my-1" />
          <button
            onClick={() => {
              setDeletingPlanId(contextMenu.planId);
              setContextMenu(null);
            }}
            className="w-full px-3 py-2 text-left text-xs text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] flex items-center gap-2 transition-colors"
          >
            <Trash2 size={12} className="text-red-400" />
            Delete
          </button>
        </div>
      )}
    </PageWrapper>
  );
}
