import { useState, useMemo } from "react";
import { Download, ChevronUp, ChevronDown, ChevronsUpDown, Search } from "lucide-react";
import PageWrapper from "@/components/layout/PageWrapper";
import { useBatches } from "@/hooks/useBatches";
import type { Batch } from "@/types";

type SortField = "batch_number" | "expected_start" | "actual_start" | "ingot_kg" | "status";
type SortDir = "asc" | "desc";

const statusColors: Record<string, string> = {
  completed: "#22C55E",
  in_progress: "#F59E0B",
  pending: "#52525B",
};

function fmtDatetime(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function SortIcon({ field, sortField, sortDir }: { field: string; sortField: string; sortDir: SortDir }) {
  if (field !== sortField) return <ChevronsUpDown size={12} className="text-zinc-600" />;
  return sortDir === "asc"
    ? <ChevronUp size={12} className="text-brand-red" />
    : <ChevronDown size={12} className="text-brand-red" />;
}

function exportCSV(batches: Batch[]) {
  const headers = [
    "batch_number", "plan_id", "furnace", "expected_start", "actual_start",
    "ingot_kg", "fe_kg", "si_kg", "scrap_kg", "status", "created_at",
  ];
  const rows = batches.map((b) =>
    headers.map((h) => {
      const val = b[h as keyof Batch];
      if (val === null || val === undefined) return "";
      const s = String(val);
      return s.includes(",") ? `"${s}"` : s;
    }).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `batches-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ReportsPage() {
  const { data: batches = [], isLoading } = useBatches();
  const [sortField, setSortField] = useState<SortField>("batch_number");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  function handleSort(field: SortField) {
    if (field === sortField) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  }

  const filtered = useMemo(() => {
    let result = [...batches];
    if (statusFilter !== "all") result = result.filter((b) => b.status === statusFilter);
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (b) =>
          String(b.batch_number).includes(q) ||
          (b.furnace ?? "").toLowerCase().includes(q) ||
          b.plan_id.toLowerCase().includes(q)
      );
    }
    result.sort((a, b) => {
      let av: any = a[sortField];
      let bv: any = b[sortField];
      if (av === null || av === undefined) av = "";
      if (bv === null || bv === undefined) bv = "";
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return result;
  }, [batches, sortField, sortDir, statusFilter, search]);

  const cols: { label: string; field: SortField; className?: string }[] = [
    { label: "Batch #", field: "batch_number" },
    { label: "Expected Start", field: "expected_start" },
    { label: "Actual Start", field: "actual_start" },
    { label: "Ingot (kg)", field: "ingot_kg", className: "text-right" },
    { label: "Status", field: "status" },
  ];

  return (
    <PageWrapper
      title="Reports & Logs"
      subtitle="Historical batch records and energy data"
      actions={
        <div className="flex items-center gap-2">
          {/* Status filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-bg-elevated border border-[var(--border-color)] text-xs text-[var(--text-primary)] rounded-lg px-3 py-2 focus:outline-none focus:border-zinc-500"
          >
            <option value="all">All statuses</option>
            <option value="pending">Pending</option>
            <option value="in_progress">In Progress</option>
            <option value="completed">Completed</option>
          </select>

          {/* Search */}
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              placeholder="Search…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-bg-elevated border border-[var(--border-color)] text-xs text-[var(--text-primary)] rounded-lg pl-8 pr-3 py-2 focus:outline-none focus:border-zinc-500 w-36"
            />
          </div>

          {/* Export */}
          <button
            onClick={() => exportCSV(filtered)}
            disabled={filtered.length === 0}
            className="flex items-center gap-2 bg-bg-elevated border border-[var(--border-color)] hover:border-zinc-500 disabled:opacity-40 text-[var(--text-primary)] text-xs font-medium px-3 py-2 rounded-lg transition-colors"
          >
            <Download size={14} />
            Export CSV
          </button>
        </div>
      }
    >
      <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--border-color)]">
              {cols.map((col) => (
                <th key={col.field} className={`px-4 py-3 text-left ${col.className ?? ""}`}>
                  <button
                    onClick={() => handleSort(col.field)}
                    className="flex items-center gap-1 text-xs font-medium text-zinc-400 uppercase tracking-wide hover:text-[var(--text-primary)] transition-colors"
                  >
                    {col.label}
                    <SortIcon field={col.field} sortField={sortField} sortDir={sortDir} />
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 4 }).map((_, i) => (
                <tr key={i} className="border-b border-[var(--border-color)]">
                  {cols.map((c) => (
                    <td key={c.field} className="px-4 py-3">
                      <div className="h-3 rounded bg-[var(--bg-elevated)] animate-pulse w-24" />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={cols.length} className="px-4 py-10 text-center text-xs text-zinc-600">
                  {batches.length === 0
                    ? "No batches yet — create a production plan to get started"
                    : "No batches match the current filter"}
                </td>
              </tr>
            ) : (
              filtered.map((b, i) => (
                <tr
                  key={b.id}
                  className={`border-b border-[var(--border-color)] hover:bg-bg-elevated/30 transition-colors ${
                    i % 2 === 0 ? "bg-bg-card" : "bg-bg-elevated/20"
                  }`}
                >
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)]">
                    #{b.batch_number}
                    {b.furnace && (
                      <span
                        className="ml-2 text-zinc-500"
                        style={{ color: b.furnace === "A" ? "#E3000F" : "#3B82F6" }}
                      >
                        {b.furnace}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-400">
                    {fmtDatetime(b.expected_start)}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-zinc-400">
                    {fmtDatetime(b.actual_start)}
                  </td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-secondary)] text-right">
                    {b.ingot_kg != null ? b.ingot_kg.toFixed(1) : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className="inline-flex items-center gap-1.5 text-xs font-medium capitalize"
                      style={{ color: statusColors[b.status] }}
                    >
                      <span
                        className="w-1.5 h-1.5 rounded-full"
                        style={{ backgroundColor: statusColors[b.status] }}
                      />
                      {b.status.replace("_", " ")}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>

        <div className="px-4 py-3 border-t border-[var(--border-color)] flex items-center justify-between text-xs text-zinc-500">
          <span>
            {filtered.length} of {batches.length} record{batches.length !== 1 ? "s" : ""}
          </span>
          {batches.length > 0 && (
            <span>
              {batches.filter((b) => b.status === "completed").length} completed ·{" "}
              {batches.filter((b) => b.status === "in_progress").length} in progress ·{" "}
              {batches.filter((b) => b.status === "pending").length} pending
            </span>
          )}
        </div>
      </div>
    </PageWrapper>
  );
}
