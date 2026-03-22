import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import {
  ClipboardCheck,
  ChevronDown,
  CheckCircle2,
  AlertCircle,
  Play,
  Pencil,
  Check,
  Maximize2,
  Minimize2,
  ChevronRight,
  Save,
  X,
  CalendarDays,
} from "lucide-react";
import PageWrapper from "@/components/layout/PageWrapper";
import PowerProfileChart from "@/components/charts/PowerProfileChart";
import { useBatches, usePowerProfile, useUpdateBatch } from "@/hooks/useBatches";
import { usePlans } from "@/hooks/usePlans";
import { useAppStore } from "@/store/appStore";
import { plansApi } from "@/api/plans";
import { useQueryClient } from "@tanstack/react-query";
import type { Batch } from "@/types";
import type { PowerProfilePoint } from "@/api/batches";

const schema = z.object({
  ingot_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  fe_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  si_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  scrap_kg: z.coerce.number().min(0, "Must be ≥ 0"),
});
type FormData = z.infer<typeof schema>;

const STATUS_COLORS: Record<string, string> = {
  pending: "#52525B",
  in_progress: "#F59E0B",
  completed: "#22C55E",
};

function BatchSelector({
  batches,
  selected,
  onSelect,
}: {
  batches: Batch[];
  selected: Batch | null;
  onSelect: (b: Batch) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between bg-bg-elevated border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] transition-colors"
      >
        <span className="flex items-center gap-2">
          {selected ? (
            <>
              <span
                className="w-2 h-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: STATUS_COLORS[selected.status] }}
              />
              <span className="font-mono text-xs">Batch #{selected.batch_number}</span>
              <span className="text-zinc-500 text-xs">— {selected.furnace ?? "?"} furnace</span>
            </>
          ) : (
            <span className="text-zinc-500">Select a batch…</span>
          )}
        </span>
        <ChevronDown size={14} className="text-zinc-400" />
      </button>

      {open && (
        <div className="absolute z-10 mt-1 w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg shadow-xl overflow-hidden">
          {batches.length === 0 ? (
            <p className="px-3 py-2 text-xs text-zinc-500">No pending batches</p>
          ) : (
            batches.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => { onSelect(b); setOpen(false); }}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-bg-card transition-colors text-left"
              >
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: STATUS_COLORS[b.status] }}
                />
                <span className="font-mono text-[var(--text-primary)]">Batch #{b.batch_number}</span>
                <span className="text-zinc-500">Furnace {b.furnace ?? "?"}</span>
                <span className="ml-auto text-zinc-600 capitalize">{b.status.replace("_", " ")}</span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function getPowerAtTime(profile: PowerProfilePoint[], t: number): number {
  if (!profile.length) return 0;
  for (let i = profile.length - 1; i >= 0; i--) {
    if (profile[i].time_min <= t) return profile[i].power_kw;
  }
  return profile[0].power_kw;
}

function toDatetimeLocal(iso: string) {
  // Force UTC interpretation if no timezone info (naive datetime from backend)
  const hasTimezone = iso.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(iso);
  const d = new Date(hasTimezone ? iso : iso + "Z");
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function nowLocalISO(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function fmtTime(iso: string): string {
  const hasTimezone = iso.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(iso);
  const d = new Date(hasTimezone ? iso : iso + "Z");
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function fmtDateTime(iso: string): string {
  const hasTimezone = iso.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(iso);
  const d = new Date(hasTimezone ? iso : iso + "Z");
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// ── Inline row editor state type ──
type RowEditState = {
  actual_start: string;
  ingot_kg: string;
  fe_kg: string;
  si_kg: string;
  scrap_kg: string;
};

function initRowEdit(b: Batch): RowEditState {
  return {
    actual_start: b.actual_start ? toDatetimeLocal(b.actual_start) : "",
    ingot_kg: b.ingot_kg != null ? String(b.ingot_kg) : "",
    fe_kg: b.fe_kg != null ? String(b.fe_kg) : "",
    si_kg: b.si_kg != null ? String(b.si_kg) : "",
    scrap_kg: b.scrap_kg != null ? String(b.scrap_kg) : "",
  };
}

export default function OperatorExecutionPage() {
  const storedActivePlan = useAppStore((s) => s.activePlan);
  const qc = useQueryClient();

  // ── Plan selection (active plans only) ──
  const { data: allPlans = [] } = usePlans();
  const activePlans = allPlans.filter((p) => p.status === "active");
  const [selectedPlanId, setSelectedPlanId] = useState<string | null>(
    storedActivePlan?.status === "active" ? storedActivePlan.id : null,
  );

  // Auto-select when plans load
  useEffect(() => {
    if (!selectedPlanId) {
      if (storedActivePlan?.status === "active") {
        setSelectedPlanId(storedActivePlan.id);
      } else if (activePlans.length === 1) {
        setSelectedPlanId(activePlans[0].id);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activePlans.length, storedActivePlan?.id]);

  const selectedPlan = activePlans.find((p) => p.id === selectedPlanId) ?? null;

  // ── Date filter ──
  const [dateFilter, setDateFilter] = useState<string>("");

  // ── Batch data ──
  const [selectedBatch, setSelectedBatch] = useState<Batch | null>(null);
  const [toast, setToast] = useState<{ type: "success" | "error"; msg: string } | null>(null);
  const [batchStarted, setBatchStarted] = useState(false);
  const [startedAt, setStartedAt] = useState<string | null>(null);
  const [editingStart, setEditingStart] = useState(false);
  const [editStartValue, setEditStartValue] = useState("");
  const [elapsedMin, setElapsedMin] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);

  // ── Expandable row edit ──
  const [expandedBatchId, setExpandedBatchId] = useState<string | null>(null);
  const [rowEdit, setRowEdit] = useState<RowEditState | null>(null);
  const [rowSaving, setRowSaving] = useState(false);

  const { data: batches = [] } = useBatches(selectedPlanId ?? undefined);
  const { data: profile = [] } = usePowerProfile(selectedBatch?.id ?? null);
  const updateBatch = useUpdateBatch();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({ resolver: zodResolver(schema) });

  // ── Auto-select in_progress batch when batches load ──
  useEffect(() => {
    if (selectedBatch) return; // already have a selection
    const inProgress = batches.find((b) => b.status === "in_progress");
    if (inProgress) {
      setSelectedBatch(inProgress);
      setBatchStarted(true);
      setStartedAt(inProgress.actual_start ?? null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [batches]);

  // ── Timer ──
  useEffect(() => {
    if (!batchStarted || !startedAt) { setElapsedMin(0); return; }
    const tick = () => {
      setElapsedMin(Math.max(0, (Date.now() - new Date(startedAt).getTime()) / 60000));
    };
    tick();
    const id = setInterval(tick, 10_000);
    return () => clearInterval(id);
  }, [batchStarted, startedAt]);

  const currentPower = batchStarted && profile.length > 0
    ? getPowerAtTime(profile, elapsedMin)
    : null;

  // ── Filtered batches ──
  const filteredBatches = batches.filter((b) => {
    if (!dateFilter) return true;
    const batchDate = b.expected_start
      ? new Date(b.expected_start).toISOString().slice(0, 10)
      : null;
    return batchDate === dateFilter;
  });

  const pendingBatches = filteredBatches.filter((b) => b.status !== "completed");

  function showToast(type: "success" | "error", msg: string) {
    setToast({ type, msg });
    setTimeout(() => setToast(null), 4000);
  }

  function handleSelectBatch(b: Batch) {
    setSelectedBatch(b);
    setBatchStarted(b.status === "in_progress" || b.status === "completed");
    setStartedAt(b.actual_start ?? null);
    setEditingStart(false);
    setEditStartValue("");
    setElapsedMin(0);
    reset();
  }

  async function handleStartBatch() {
    if (!selectedBatch) return;
    const now = nowLocalISO();
    try {
      await updateBatch.mutateAsync({ id: selectedBatch.id, data: { actual_start: now, status: "in_progress" } });
      setBatchStarted(true);
      setStartedAt(now);
      showToast("success", `Batch #${selectedBatch.batch_number} started`);
    } catch {
      showToast("error", "Failed to start batch. Check your connection.");
    }
  }

  async function handleSaveEditedStart() {
    if (!selectedBatch || !editStartValue) return;
    try {
      const iso = new Date(editStartValue).toISOString();
      await updateBatch.mutateAsync({ id: selectedBatch.id, data: { actual_start: iso } });
      setStartedAt(iso);
      setEditingStart(false);
    } catch {
      showToast("error", "Failed to update start time.");
    }
  }

  async function onSubmit(values: FormData) {
    if (!selectedBatch) return;
    try {
      await updateBatch.mutateAsync({ id: selectedBatch.id, data: { ...values, status: "completed" } });
      showToast("success", `Batch #${selectedBatch.batch_number} marked completed`);

      const allDone = batches
        .map((b) => (b.id === selectedBatch.id ? "completed" : b.status))
        .every((s) => s === "completed");
      if (allDone && selectedPlanId) {
        try {
          await plansApi.updateStatus(selectedPlanId, "completed");
          qc.invalidateQueries({ queryKey: ["plans"] });
        } catch { /* non-critical */ }
      }

      setSelectedBatch(null);
      setBatchStarted(false);
      setStartedAt(null);
      setElapsedMin(0);
      setFullscreen(false);
      reset();
    } catch {
      showToast("error", "Failed to update batch. Check your connection.");
    }
  }

  // ── Row expand/edit handlers ──
  function handleToggleExpand(b: Batch) {
    if (expandedBatchId === b.id) {
      setExpandedBatchId(null);
      setRowEdit(null);
    } else {
      setExpandedBatchId(b.id);
      setRowEdit(initRowEdit(b));
    }
  }

  async function handleSaveRowEdit(batchId: string) {
    if (!rowEdit) return;
    setRowSaving(true);
    try {
      await updateBatch.mutateAsync({
        id: batchId,
        data: {
          actual_start: rowEdit.actual_start ? new Date(rowEdit.actual_start).toISOString() : undefined,
          ingot_kg: rowEdit.ingot_kg !== "" ? parseFloat(rowEdit.ingot_kg) : undefined,
          fe_kg: rowEdit.fe_kg !== "" ? parseFloat(rowEdit.fe_kg) : undefined,
          si_kg: rowEdit.si_kg !== "" ? parseFloat(rowEdit.si_kg) : undefined,
          scrap_kg: rowEdit.scrap_kg !== "" ? parseFloat(rowEdit.scrap_kg) : undefined,
        },
      });
      showToast("success", "Batch updated");
      setExpandedBatchId(null);
      setRowEdit(null);
    } catch {
      showToast("error", "Failed to save changes.");
    } finally {
      setRowSaving(false);
    }
  }

  const furnaceLabel = selectedBatch?.furnace ? `IF_${selectedBatch.furnace}` : null;
  const peakKw = profile.length ? Math.max(...profile.map((p) => p.power_kw)) : 0;
  const avgKw = profile.length ? profile.reduce((s, p) => s + p.power_kw, 0) / profile.length : 0;

  return (
    <PageWrapper
      title="Operator Execution"
      subtitle="Follow the power profile guideline and log actual batch data"
    >
      {/* Toast */}
      {toast && (
        <div
          className={`fixed top-4 right-4 z-50 flex items-center gap-2 px-4 py-3 rounded-lg border shadow-xl text-sm font-medium transition-all
            ${toast.type === "success"
              ? "bg-green-500/10 border-green-500/30 text-green-700 dark:text-green-300"
              : "bg-brand-red/10 border-brand-red/30 text-red-700 dark:text-red-300"
            }`}
        >
          {toast.type === "success"
            ? <CheckCircle2 size={16} className="text-green-400" />
            : <AlertCircle size={16} className="text-red-400" />}
          {toast.msg}
        </div>
      )}

      {/* ── Full-screen overlay ── */}
      {fullscreen && selectedBatch && (
        <div className="fixed inset-0 z-[100] bg-bg-base flex flex-col p-6">
          <div className="flex items-center justify-between mb-5 flex-shrink-0">
            <div className="flex items-center gap-8">
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Furnace</p>
                <p className="text-3xl font-mono font-bold text-[var(--text-primary)]">{furnaceLabel ?? "IF_?"}</p>
              </div>
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Current Power</p>
                <p className="font-mono font-bold text-brand-red leading-none">
                  <span className="text-6xl">{currentPower?.toFixed(0) ?? "—"}</span>
                  <span className="text-2xl text-zinc-400 ml-2">kW</span>
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Elapsed</p>
                <p className="text-3xl font-mono font-bold text-[var(--text-primary)]">
                  {Math.floor(elapsedMin)}<span className="text-xl text-zinc-400"> min</span>
                </p>
              </div>
              <div>
                <p className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Batch</p>
                <p className="text-3xl font-mono font-bold text-[var(--text-primary)]">#{selectedBatch.batch_number}</p>
              </div>
            </div>
            <button onClick={() => setFullscreen(false)} className="text-zinc-400 hover:text-white p-2 rounded-lg hover:bg-zinc-800 transition-colors">
              <Minimize2 size={20} />
            </button>
          </div>
          <div className="flex-1 min-h-0 bg-bg-card rounded-xl border border-[var(--border-color)] p-4">
            <PowerProfileChart data={profile} durationMin={selectedBatch.duration_min ?? undefined} elapsedMin={batchStarted ? elapsedMin : undefined} />
          </div>
          <div className="mt-4 flex gap-8 text-sm text-zinc-500 flex-shrink-0">
            <span>Peak: <span className="font-mono text-white">{peakKw.toFixed(0)} kW</span></span>
            <span>Duration: <span className="font-mono text-white">{selectedBatch.duration_min ?? "—"} min</span></span>
            <span>Avg: <span className="font-mono text-white">{avgKw.toFixed(0)} kW</span></span>
          </div>
        </div>
      )}

      {/* ── Plan + Date filter bar ── */}
      <div className="mb-4 flex flex-wrap items-end gap-4">
        {/* Active plan selector */}
        <div>
          <label className="block text-xs text-zinc-400 font-medium mb-1.5">Active Plan</label>
          {activePlans.length === 0 ? (
            <span className="text-xs text-amber-500">No active plans — activate a plan in Production Planning first</span>
          ) : (
            <div className="relative inline-block">
              <select
                value={selectedPlanId ?? ""}
                onChange={(e) => {
                  setSelectedPlanId(e.target.value || null);
                  setSelectedBatch(null);
                  setBatchStarted(false);
                  setStartedAt(null);
                  reset();
                }}
                className="appearance-none bg-bg-elevated border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-2 pr-8 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red cursor-pointer transition-colors"
              >
                <option value="">Select plan…</option>
                {activePlans.map((p) => (
                  <option key={p.id} value={p.id}>
                    Plan — {fmtDateTime(p.shift_start)}
                  </option>
                ))}
              </select>
              <ChevronDown size={14} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
            </div>
          )}
        </div>

        {/* Date filter */}
        {selectedPlanId && (
          <div>
            <label className="flex items-center gap-1 text-xs text-zinc-400 font-medium mb-1.5">
              <CalendarDays size={11} />
              Filter by Date
            </label>
            <div className="flex items-center gap-1.5">
              <input
                type="date"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className="bg-bg-elevated border border-[var(--border-color)] hover:border-zinc-500 rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red transition-colors font-mono"
              />
              {dateFilter && (
                <button
                  type="button"
                  onClick={() => setDateFilter("")}
                  className="text-zinc-500 hover:text-zinc-300 p-1.5 rounded transition-colors"
                  title="Clear date filter"
                >
                  <X size={13} />
                </button>
              )}
            </div>
          </div>
        )}

        {/* Active plan info badge */}
        {selectedPlan && (
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            <span className="text-xs text-zinc-400">
              {filteredBatches.length} batch{filteredBatches.length !== 1 ? "es" : ""}
              {dateFilter ? " on this date" : ""}
              {" · "}
              {pendingBatches.length} pending
            </span>
          </div>
        )}
      </div>

      {/* ── Batch selector + start controls ── */}
      {selectedPlanId && (
        <div className="mb-4">
          <div className="flex items-center gap-1.5 mb-1.5">
            <label className="text-xs text-zinc-400 font-medium">Select Batch to Execute</label>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="w-64">
              <BatchSelector batches={pendingBatches} selected={selectedBatch} onSelect={handleSelectBatch} />
            </div>

            {selectedBatch && !batchStarted && (
              <button
                type="button"
                onClick={handleStartBatch}
                disabled={updateBatch.isPending}
                className="flex items-center gap-1.5 bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 text-white text-xs font-semibold px-3 py-2 rounded-lg transition-colors"
              >
                <Play size={12} fill="white" />
                Start Batch
              </button>
            )}

            {selectedBatch && batchStarted && startedAt && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500">Started:</span>
                {editingStart ? (
                  <div className="flex items-center gap-1.5">
                    <input
                      type="datetime-local"
                      defaultValue={toDatetimeLocal(startedAt)}
                      onChange={(e) => setEditStartValue(e.target.value)}
                      className="bg-bg-elevated border border-[var(--border-color)] rounded px-2 py-1 text-xs text-[var(--text-primary)] focus:outline-none focus:border-brand-red font-mono"
                    />
                    <button type="button" onClick={handleSaveEditedStart} className="text-green-400 hover:text-green-300 p-1">
                      <Check size={12} />
                    </button>
                    <button type="button" onClick={() => setEditingStart(false)} className="text-zinc-500 hover:text-zinc-300 p-1">
                      <X size={12} />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-mono text-[var(--text-primary)]">
                      {fmtTime(startedAt)}
                    </span>
                    <button
                      type="button"
                      onClick={() => { setEditStartValue(toDatetimeLocal(startedAt)); setEditingStart(true); }}
                      className="text-[var(--text-muted)] hover:text-[var(--text-primary)] p-1"
                    >
                      <Pencil size={11} />
                    </button>
                  </div>
                )}
              </div>
            )}

            {batchStarted && currentPower !== null && !fullscreen && (
              <div className="ml-2 flex items-baseline gap-1">
                <span className="text-2xl font-mono font-bold text-brand-red">{currentPower.toFixed(0)}</span>
                <span className="text-xs text-zinc-400">kW</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Main grid ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4" style={{ minHeight: "360px" }}>
        {/* Power profile chart */}
        <div className="lg:col-span-2 bg-bg-card border border-[var(--border-color)] rounded-xl p-5 flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">Power Profile — Guideline</h2>
            <div className="flex items-center gap-2">
              {furnaceLabel ? (
                <span className="text-xs bg-[var(--bg-elevated)] text-[var(--text-secondary)] border border-[var(--border-color)] rounded px-2 py-0.5 font-mono">
                  {furnaceLabel}
                </span>
              ) : (
                <span className="text-xs bg-brand-red/10 text-brand-red border border-brand-red/20 rounded px-2 py-0.5">
                  Select a batch
                </span>
              )}
              {selectedBatch && (
                <button
                  type="button"
                  onClick={() => setFullscreen((v) => !v)}
                  className="text-[var(--text-muted)] hover:text-[var(--text-primary)] p-1 rounded transition-colors"
                  title={fullscreen ? "Exit full screen" : "Full screen"}
                >
                  {fullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 min-h-0">
            {selectedBatch ? (
              <PowerProfileChart
                data={profile}
                durationMin={selectedBatch.duration_min ?? undefined}
                elapsedMin={batchStarted ? elapsedMin : undefined}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-zinc-600 text-sm border border-dashed border-[var(--border-color)] rounded-lg">
                Select a batch to view its power profile
              </div>
            )}
          </div>

          {profile.length > 0 && (
            <div className="mt-3 flex gap-4 text-xs text-zinc-500 flex-shrink-0">
              <span>Peak: <span className="font-mono text-[var(--text-primary)]">{peakKw.toFixed(0)} kW</span></span>
              <span>Duration: <span className="font-mono text-[var(--text-primary)]">{selectedBatch?.duration_min ?? profile[profile.length - 1]?.time_min ?? 0} min</span></span>
              <span>Avg: <span className="font-mono text-[var(--text-primary)]">{avgKw.toFixed(0)} kW</span></span>
            </div>
          )}
        </div>

        {/* Batch Entry Form */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5 flex flex-col">
          <div className="flex items-center gap-2 mb-4">
            <ClipboardCheck size={16} className="text-brand-red" />
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">Actual Data Entry</h2>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-3 flex flex-col flex-1">
            {(["ingot_kg", "fe_kg", "si_kg", "scrap_kg"] as const).map((field) => {
              const labels: Record<string, string> = {
                ingot_kg: "Ingot (kg)",
                fe_kg: "Fe (kg)",
                si_kg: "Si (kg)",
                scrap_kg: "Scrap (kg)",
              };
              return (
                <div key={field}>
                  <label className="block text-xs text-zinc-400 mb-1.5">{labels[field]}</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="0.0"
                    {...register(field)}
                    className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red"
                  />
                  {errors[field] && (
                    <p className="text-xs text-red-400 mt-1">{errors[field]?.message}</p>
                  )}
                </div>
              );
            })}

            <div className="flex-1" />

            <button
              type="submit"
              disabled={!selectedBatch || !batchStarted || isSubmitting || updateBatch.isPending}
              className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold py-2.5 rounded-lg transition-colors mt-1"
            >
              {updateBatch.isPending ? "Saving…" : "Complete Batch"}
            </button>

            {!selectedBatch && (
              <p className="text-xs text-zinc-600 text-center">Select a batch above first</p>
            )}
            {selectedBatch && !batchStarted && (
              <p className="text-xs text-zinc-600 text-center">Press Start Batch to begin</p>
            )}
          </form>
        </div>
      </div>

      {/* ── Batch Progress table ── */}
      {filteredBatches.length > 0 && (
        <div className="mt-4 bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
          <div className="px-5 py-3 border-b border-[var(--border-color)] flex items-center justify-between">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">Batch Progress</h3>
            <span className="text-xs text-zinc-600">Click a row to view &amp; edit details</span>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border-color)]">
                <th className="w-8 px-2 py-2" />
                {["#", "Furnace", "Expected Start", "Actual Start", "Status"].map((h) => (
                  <th key={h} className="px-4 py-2 text-left text-zinc-500 font-medium uppercase tracking-wide">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredBatches.map((b, i) => {
                const isExpanded = expandedBatchId === b.id;
                return (
                  <>
                    <tr
                      key={b.id}
                      onClick={() => handleToggleExpand(b)}
                      className={`border-b border-[var(--border-color)] cursor-pointer transition-colors
                        ${i % 2 === 0 ? "" : "bg-bg-elevated/20"}
                        ${selectedBatch?.id === b.id ? "bg-brand-red/5" : ""}
                        ${isExpanded ? "bg-bg-elevated/40" : "hover:bg-bg-elevated/30"}
                      `}
                    >
                      <td className="px-2 py-2 text-zinc-600">
                        <ChevronRight
                          size={13}
                          className={`transition-transform ${isExpanded ? "rotate-90" : ""}`}
                        />
                      </td>
                      <td className="px-4 py-2 font-mono text-[var(--text-primary)]">#{b.batch_number}</td>
                      <td className="px-4 py-2 text-zinc-400 font-mono">{b.furnace ? `IF_${b.furnace}` : "—"}</td>
                      <td className="px-4 py-2 font-mono text-zinc-400">
                        {b.expected_start ? fmtTime(b.expected_start) : "—"}
                      </td>
                      <td className="px-4 py-2 font-mono text-zinc-400">
                        {b.actual_start ? fmtTime(b.actual_start) : "—"}
                      </td>
                      <td className="px-4 py-2">
                        <span className="flex items-center gap-1.5" style={{ color: STATUS_COLORS[b.status] }}>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: STATUS_COLORS[b.status] }} />
                          {b.status.replace("_", " ")}
                        </span>
                      </td>
                    </tr>

                    {/* Expanded detail / edit row */}
                    {isExpanded && rowEdit && (
                      <tr key={`${b.id}-edit`} className="border-b border-[var(--border-color)] bg-bg-elevated/20">
                        <td colSpan={6} className="px-5 py-4">
                          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 items-end">
                            {/* Actual Start */}
                            <div className="md:col-span-2">
                              <label className="block text-xs text-zinc-500 mb-1">Actual Start Time</label>
                              <input
                                type="datetime-local"
                                value={rowEdit.actual_start}
                                onChange={(e) => setRowEdit((v) => v ? { ...v, actual_start: e.target.value } : v)}
                                className="w-full bg-bg-card border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-primary)] focus:outline-none focus:border-brand-red font-mono transition-colors"
                              />
                            </div>

                            {/* Ingot */}
                            <div>
                              <label className="block text-xs text-zinc-500 mb-1">Ingot (kg)</label>
                              <input
                                type="number"
                                step="0.1"
                                placeholder="—"
                                value={rowEdit.ingot_kg}
                                onChange={(e) => setRowEdit((v) => v ? { ...v, ingot_kg: e.target.value } : v)}
                                className="w-full bg-bg-card border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red transition-colors"
                              />
                            </div>

                            {/* Fe */}
                            <div>
                              <label className="block text-xs text-zinc-500 mb-1">Fe (kg)</label>
                              <input
                                type="number"
                                step="0.1"
                                placeholder="—"
                                value={rowEdit.fe_kg}
                                onChange={(e) => setRowEdit((v) => v ? { ...v, fe_kg: e.target.value } : v)}
                                className="w-full bg-bg-card border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red transition-colors"
                              />
                            </div>

                            {/* Si */}
                            <div>
                              <label className="block text-xs text-zinc-500 mb-1">Si (kg)</label>
                              <input
                                type="number"
                                step="0.1"
                                placeholder="—"
                                value={rowEdit.si_kg}
                                onChange={(e) => setRowEdit((v) => v ? { ...v, si_kg: e.target.value } : v)}
                                className="w-full bg-bg-card border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red transition-colors"
                              />
                            </div>

                            {/* Scrap */}
                            <div>
                              <label className="block text-xs text-zinc-500 mb-1">Scrap (kg)</label>
                              <input
                                type="number"
                                step="0.1"
                                placeholder="—"
                                value={rowEdit.scrap_kg}
                                onChange={(e) => setRowEdit((v) => v ? { ...v, scrap_kg: e.target.value } : v)}
                                className="w-full bg-bg-card border border-[var(--border-color)] rounded-lg px-2.5 py-1.5 text-xs text-[var(--text-primary)] placeholder-zinc-600 focus:outline-none focus:border-brand-red transition-colors"
                              />
                            </div>

                            {/* Save / Cancel */}
                            <div className="flex gap-2 items-end">
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); handleSaveRowEdit(b.id); }}
                                disabled={rowSaving}
                                className="flex items-center gap-1 bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
                              >
                                <Save size={11} />
                                {rowSaving ? "Saving…" : "Save"}
                              </button>
                              <button
                                type="button"
                                onClick={(e) => { e.stopPropagation(); setExpandedBatchId(null); setRowEdit(null); }}
                                className="flex items-center gap-1 bg-bg-card hover:bg-bg-elevated border border-[var(--border-color)] text-zinc-400 hover:text-[var(--text-primary)] text-xs px-3 py-1.5 rounded-lg transition-colors"
                              >
                                <X size={11} />
                                Cancel
                              </button>
                            </div>
                          </div>

                          {/* Read-only info */}
                          <div className="mt-3 pt-3 border-t border-[var(--border-color)]/50 flex flex-wrap gap-4 text-xs text-zinc-600">
                            {b.expected_start && (
                              <span>Expected Start: <span className="font-mono text-zinc-400">{fmtDateTime(b.expected_start)}</span></span>
                            )}
                            {b.duration_min != null && (
                              <span>Duration: <span className="font-mono text-zinc-400">{b.duration_min} min</span></span>
                            )}
                            {b.power_kw != null && (
                              <span>Power: <span className="font-mono text-zinc-400">{b.power_kw} kW</span></span>
                            )}
                            {b.energy_kwh != null && (
                              <span>Energy: <span className="font-mono text-zinc-400">{b.energy_kwh.toFixed(1)} kWh</span></span>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Empty state */}
      {selectedPlanId && filteredBatches.length === 0 && (
        <div className="mt-4 py-10 text-center text-zinc-600 text-sm border border-dashed border-[var(--border-color)] rounded-xl">
          {dateFilter ? `No batches on ${dateFilter}` : "No batches found for this plan"}
        </div>
      )}
    </PageWrapper>
  );
}
