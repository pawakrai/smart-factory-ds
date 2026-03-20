import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { ClipboardCheck, ChevronDown, CheckCircle2, AlertCircle } from "lucide-react";
import PageWrapper from "@/components/layout/PageWrapper";
import PowerProfileChart from "@/components/charts/PowerProfileChart";
import { useBatches, usePowerProfile, useUpdateBatch } from "@/hooks/useBatches";
import { useAppStore } from "@/store/appStore";
import type { Batch } from "@/types";

const schema = z.object({
  actual_start: z.string().min(1, "Required"),
  ingot_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  fe_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  si_kg: z.coerce.number().min(0, "Must be ≥ 0"),
  scrap_kg: z.coerce.number().min(0, "Must be ≥ 0"),
});
type FormData = z.infer<typeof schema>;

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
  const statusColors: Record<string, string> = {
    pending: "#52525B",
    in_progress: "#F59E0B",
    completed: "#22C55E",
  };

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
                style={{ backgroundColor: statusColors[selected.status] }}
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
            <p className="px-3 py-2 text-xs text-zinc-500">No batches — create a plan first</p>
          ) : (
            batches.map((b) => (
              <button
                key={b.id}
                type="button"
                onClick={() => {
                  onSelect(b);
                  setOpen(false);
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-bg-card transition-colors text-left"
              >
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: statusColors[b.status] }}
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

export default function OperatorExecutionPage() {
  const activePlan = useAppStore((s) => s.activePlan);
  const [selectedBatch, setSelectedBatch] = useState<Batch | null>(null);
  const [toast, setToast] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  const { data: batches = [] } = useBatches(activePlan?.id);
  const { data: profile = [] } = usePowerProfile(selectedBatch?.id ?? null);
  const updateBatch = useUpdateBatch();

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isSubmitting },
  } = useForm<FormData>({ resolver: zodResolver(schema) });

  function showToast(type: "success" | "error", msg: string) {
    setToast({ type, msg });
    setTimeout(() => setToast(null), 4000);
  }

  async function onSubmit(values: FormData) {
    if (!selectedBatch) return;
    try {
      await updateBatch.mutateAsync({
        id: selectedBatch.id,
        data: { ...values, status: "completed" },
      });
      showToast("success", `Batch #${selectedBatch.batch_number} marked completed`);
      setSelectedBatch(null);
      reset();
    } catch {
      showToast("error", "Failed to update batch. Check your connection.");
    }
  }

  const pendingBatches = batches.filter((b) => b.status !== "completed");

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
              ? "bg-green-950 border-green-800 text-green-300"
              : "bg-red-950 border-red-900 text-red-300"
            }`}
        >
          {toast.type === "success" ? (
            <CheckCircle2 size={16} className="text-green-400" />
          ) : (
            <AlertCircle size={16} className="text-red-400" />
          )}
          {toast.msg}
        </div>
      )}

      {/* Batch selector */}
      <div className="mb-4">
        <div className="flex items-center gap-3 mb-2">
          <label className="text-xs text-zinc-400 font-medium">Active Batch</label>
          {!activePlan && (
            <span className="text-xs text-amber-500">No active plan — go to Production Planning first</span>
          )}
        </div>
        <div className="max-w-xs">
          <BatchSelector batches={pendingBatches} selected={selectedBatch} onSelect={setSelectedBatch} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Power profile chart */}
        <div className="lg:col-span-2 bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">Power Profile — Guideline</h2>
            <span className="text-xs bg-brand-red/10 text-brand-red border border-brand-red/20 rounded px-2 py-0.5">
              RL Model Output
            </span>
          </div>
          <div className="h-56">
            {selectedBatch ? (
              <PowerProfileChart data={profile} />
            ) : (
              <div className="h-full flex items-center justify-center text-zinc-600 text-sm border border-dashed border-[var(--border-color)] rounded-lg">
                Select a batch to view its power profile
              </div>
            )}
          </div>

          {/* Profile legend */}
          {profile.length > 0 && (
            <div className="mt-3 flex gap-4 text-xs text-zinc-500">
              <span>
                Peak:{" "}
                <span className="font-mono text-[var(--text-primary)]">
                  {Math.max(...profile.map((p) => p.power_kw)).toFixed(0)} kW
                </span>
              </span>
              <span>
                Duration:{" "}
                <span className="font-mono text-[var(--text-primary)]">
                  {profile[profile.length - 1]?.time_min ?? 0} min
                </span>
              </span>
              <span>
                Avg:{" "}
                <span className="font-mono text-[var(--text-primary)]">
                  {(profile.reduce((s, p) => s + p.power_kw, 0) / profile.length).toFixed(0)} kW
                </span>
              </span>
            </div>
          )}
        </div>

        {/* Batch Entry Form */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <div className="flex items-center gap-2 mb-4">
            <ClipboardCheck size={16} className="text-brand-red" />
            <h2 className="text-sm font-semibold text-[var(--text-primary)]">Actual Data Entry</h2>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-3">
            <div>
              <label className="block text-xs text-zinc-400 mb-1.5">Actual Start Time</label>
              <input
                type="datetime-local"
                {...register("actual_start")}
                className="w-full bg-bg-elevated border border-[var(--border-color)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-brand-red"
              />
              {errors.actual_start && (
                <p className="text-xs text-red-400 mt-1">{errors.actual_start.message}</p>
              )}
            </div>

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

            <button
              type="submit"
              disabled={!selectedBatch || isSubmitting || updateBatch.isPending}
              className="w-full bg-brand-red hover:bg-brand-red-dark disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold py-2.5 rounded-lg transition-colors mt-1"
            >
              {updateBatch.isPending ? "Saving…" : "Complete Batch"}
            </button>

            {!selectedBatch && (
              <p className="text-xs text-zinc-600 text-center">Select a batch above first</p>
            )}
          </form>
        </div>
      </div>

      {/* All batches summary table */}
      {batches.length > 0 && (
        <div className="mt-4 bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
          <div className="px-5 py-3 border-b border-[var(--border-color)]">
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide">
              Batch Progress
            </h3>
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-[var(--border-color)]">
                {["#", "Furnace", "Expected Start", "Actual Start", "Status"].map((h) => (
                  <th key={h} className="px-4 py-2 text-left text-zinc-500 font-medium uppercase tracking-wide">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {batches.map((b, i) => {
                const statusColors: Record<string, string> = {
                  pending: "#52525B",
                  in_progress: "#F59E0B",
                  completed: "#22C55E",
                };
                return (
                  <tr
                    key={b.id}
                    className={`border-b border-[var(--border-color)] ${i % 2 === 0 ? "" : "bg-bg-elevated/20"} ${
                      selectedBatch?.id === b.id ? "bg-brand-red/5" : ""
                    }`}
                  >
                    <td className="px-4 py-2 font-mono text-[var(--text-primary)]">#{b.batch_number}</td>
                    <td className="px-4 py-2 text-zinc-400">{b.furnace ?? "—"}</td>
                    <td className="px-4 py-2 font-mono text-zinc-400">
                      {b.expected_start
                        ? new Date(b.expected_start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                        : "—"}
                    </td>
                    <td className="px-4 py-2 font-mono text-zinc-400">
                      {b.actual_start
                        ? new Date(b.actual_start).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                        : "—"}
                    </td>
                    <td className="px-4 py-2">
                      <span className="flex items-center gap-1.5" style={{ color: statusColors[b.status] }}>
                        <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: statusColors[b.status] }} />
                        {b.status.replace("_", " ")}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </PageWrapper>
  );
}
