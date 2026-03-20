import { useState, useCallback } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { CalendarClock, Loader2, ChevronRight, AlertCircle, Zap, Factory, Trash2 } from "lucide-react";

import PageWrapper from "@/components/layout/PageWrapper";
import SyncedScheduleChart from "@/components/charts/SyncedScheduleChart";
import { usePlans, usePlanBatches, usePlanScheduleData, useCreatePlan, useDeletePlan, usePlanPolling } from "@/hooks/usePlans";
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
});
type FormValues = z.infer<typeof schema>;

// ── Helpers ────────────────────────────────────────────────────────────────
function statusBadge(status: string) {
  const map: Record<string, string> = {
    pending:   "bg-amber-900/60 text-amber-400",
    draft:     "bg-zinc-700 text-[var(--text-secondary)]",
    active:    "bg-green-900/60 text-green-400",
    completed: "bg-blue-900/60 text-blue-400",
  };
  return map[status] ?? "bg-zinc-700 text-zinc-400";
}

function fmtBaht(n: number) {
  return `฿${n.toLocaleString("en", { maximumFractionDigits: 0 })}`;
}

function fmtKwh(n: number) {
  return `${n.toLocaleString("en", { maximumFractionDigits: 0 })} kWh`;
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
      target_batches: 8,
      opt_mode: "energy",
      shift_start: (() => {
        const d = new Date();
        d.setHours(8, 0, 0, 0);
        const pad = (n: number) => String(n).padStart(2, "0");
        return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
      })(),
    },
  });

  const optMode = watch("opt_mode");

  function onSubmit(values: FormValues) {
    mutate(
      {
        target_batches: values.target_batches,
        shift_start: values.shift_start.length === 16
          ? values.shift_start + ":00"
          : values.shift_start,
        opt_mode: values.opt_mode,
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
        <div className="flex items-center gap-2 text-xs text-brand-red bg-red-950/30 border border-red-900/40 rounded-lg px-3 py-2">
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
  plan, active, onClick, onDelete,
}: {
  plan: Plan; active: boolean; onClick: () => void; onDelete: () => void;
}) {
  const isPending = plan.status === "pending";
  return (
    <div className={`group flex items-center gap-1 rounded-lg border transition-colors ${
      active ? "bg-brand-red/10 border-brand-red/30" : "bg-bg-elevated hover:bg-zinc-700/40 border-transparent"
    } ${isPending ? "opacity-70" : ""}`}>
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
      {!isPending && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="p-2 mr-1 text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
          title="Delete plan"
        >
          <Trash2 size={13} />
        </button>
      )}
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

// ── Main page ──────────────────────────────────────────────────────────────
export default function ProductionPlanningPage() {
  const [activePlanId, setActivePlanId] = useState<string | null>(null);
  const [pollingPlanId, setPollingPlanId] = useState<string | null>(null);
  const [deletingPlanId, setDeletingPlanId] = useState<string | null>(null);

  const { data: plans = [] } = usePlans();
  const { data: batches = [], isLoading: batchesLoading } = usePlanBatches(activePlanId);
  const { data: scheduleData, isLoading: scheduleLoading } = usePlanScheduleData(activePlanId);
  const { mutate: deletePlan } = useDeletePlan();

  const onGaComplete = useCallback((planId: string) => {
    setActivePlanId(planId);
    setPollingPlanId(null);
  }, []);

  function confirmDelete() {
    if (!deletingPlanId) return;
    const planId = deletingPlanId;
    setDeletingPlanId(null);
    deletePlan(planId, {
      onSuccess: () => {
        if (activePlanId === planId) setActivePlanId(null);
        if (pollingPlanId === planId) setPollingPlanId(null);
      },
    });
  }

  usePlanPolling(pollingPlanId, onGaComplete);

  const activePlan = plans.find((p) => p.id === activePlanId) ?? null;
  const isGaRunning = !!pollingPlanId;

  function handlePlanCreated(plan: Plan) {
    setPollingPlanId(plan.id);
    setActivePlanId(plan.id);
  }

  const metrics: ScheduleMetrics | null = (() => {
    if (activePlan?.schedule_metrics) {
      try { return JSON.parse(activePlan.schedule_metrics); } catch { return null; }
    }
    return null;
  })();

  const batchCount = batches.length;
  const furnaceA = batches.filter((b) => b.furnace === "A").length;
  const furnaceB = batches.filter((b) => b.furnace === "B").length;

  return (
    <PageWrapper
      title="Production Planning"
      subtitle="Configure shift parameters and generate GA-optimised schedule"
    >
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* ── Left panel: form + plan list ─────────────────────────── */}
        <div className="space-y-4">
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <div className="flex items-center gap-2 mb-4">
              <CalendarClock size={16} className="text-brand-red" />
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Plan Parameters</h2>
            </div>
            <PlanForm onSuccess={handlePlanCreated} />
          </div>

          {plans.length > 0 && (
            <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
              <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Recent Plans</h2>
              <div className="space-y-1.5 max-h-52 overflow-y-auto">
                {plans.slice(0, 8).map((plan) => (
                  <PlanRow
                    key={plan.id}
                    plan={plan}
                    active={plan.id === activePlanId}
                    onClick={() => setActivePlanId(plan.id)}
                    onDelete={() => setDeletingPlanId(plan.id)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ── Right panel: charts ───────────────────────────────────── */}
        <div className="lg:col-span-3 space-y-4">
          {/* Combined synced chart */}
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-sm font-semibold text-[var(--text-primary)]">Production Schedule</h2>
              {activePlan && batchCount > 0 && (
                <div className="flex items-center gap-3 text-[10px] text-zinc-400">
                  <span>IF-A: {furnaceA}</span>
                  <span>IF-B: {furnaceB}</span>
                </div>
              )}
            </div>

            {!activePlan ? (
              <div className="h-64 flex flex-col items-center justify-center gap-2 text-zinc-600 text-sm border border-dashed border-[var(--border-color)] rounded-lg">
                <CalendarClock size={24} className="opacity-40" />
                Create a plan to see the schedule
              </div>
            ) : isGaRunning ? (
              <div className="h-64 flex flex-col items-center justify-center gap-2 text-zinc-400 text-sm border border-dashed border-zinc-700 rounded-lg">
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
                  plan={activePlan}
                  batches={batches}
                  scheduleData={scheduleData}
                />
              </ErrorBoundary>
            )}
          </div>

          {/* Schedule summary — 6 KPI cards */}
          {activePlan && batchCount > 0 && (
            <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
              <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-3">Schedule Summary</h2>
              <div className="grid grid-cols-3 gap-3">
                <KpiCard
                  label="Batches Poured"
                  value={`${metrics?.poured_batches_count ?? batchCount} / ${activePlan.target_batches}`}
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
      <AlertDialog open={!!deletingPlanId} onOpenChange={(open) => !open && setDeletingPlanId(null)}>
        <AlertDialogContent className="bg-bg-card border border-[var(--border-color)] text-[var(--text-primary)]">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-[var(--text-primary)]">Delete Plan</AlertDialogTitle>
            <AlertDialogDescription className="text-zinc-400">
              Plan นี้จะถูกลบพร้อมกับ batch ทั้งหมด ไม่สามารถกู้คืนได้
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-bg-elevated border-[var(--border-color)] text-[var(--text-secondary)] hover:bg-zinc-700 hover:text-[var(--text-primary)]">
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
    </PageWrapper>
  );
}
