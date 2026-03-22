import { useState, useMemo } from "react";
import { Download, ChevronUp, ChevronDown, ChevronsUpDown, Search, CalendarDays, X } from "lucide-react";
import PageWrapper from "@/components/layout/PageWrapper";
import { useBatches } from "@/hooks/useBatches";
import type { Batch } from "@/types";

type SortField =
  | "batch_number"
  | "expected_start"
  | "actual_start"
  | "ingot_kg"
  | "fe_kg"
  | "si_kg"
  | "scrap_kg"
  | "power_kw"
  | "energy_kwh"
  | "status";
type SortDir = "asc" | "desc";

function fmtDatetime(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fmtNum(v: number | null | undefined, decimals = 1) {
  return v != null ? v.toFixed(decimals) : "—";
}

function SortIcon({ field, sortField, sortDir }: { field: string; sortField: string; sortDir: SortDir }) {
  if (field !== sortField) return <ChevronsUpDown size={12} className="text-zinc-600" />;
  return sortDir === "asc"
    ? <ChevronUp size={12} className="text-brand-red" />
    : <ChevronDown size={12} className="text-brand-red" />;
}

function exportCSV(batches: Batch[]) {
  const headers = [
    "batch_number", "furnace", "expected_start", "actual_start",
    "ingot_kg", "fe_kg", "si_kg", "scrap_kg", "power_kw", "energy_kwh", "duration_min", "status",
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
  a.download = `completed-batches-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

type ColDef = { label: string; field: SortField; className?: string };

const COLS: ColDef[] = [
  { label: "Batch #",       field: "batch_number" },
  { label: "Expected Start", field: "expected_start" },
  { label: "Actual Start",   field: "actual_start" },
  { label: "Ingot (kg)",    field: "ingot_kg",   className: "text-right" },
  { label: "Fe (kg)",       field: "fe_kg",      className: "text-right" },
  { label: "Si (kg)",       field: "si_kg",      className: "text-right" },
  { label: "Scrap (kg)",    field: "scrap_kg",   className: "text-right" },
  { label: "Power (kW)",    field: "power_kw",   className: "text-right" },
  { label: "Energy (kWh)",  field: "energy_kwh", className: "text-right" },
];

export default function ReportsPage() {
  // Fetch all batches (no plan filter — show all completed across plans)
  const { data: batches = [], isLoading } = useBatches();

  const [sortField, setSortField] = useState<SortField>("actual_start");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [search, setSearch] = useState("");
  const [dateFilter, setDateFilter] = useState("");

  // Only completed batches — this page is a completed-batch log
  const completedBatches = useMemo(
    () => batches.filter((b) => b.status === "completed"),
    [batches],
  );

  function handleSort(field: SortField) {
    if (field === sortField) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortField(field); setSortDir("asc"); }
  }

  const filtered = useMemo(() => {
    let result = [...completedBatches];

    // Date filter on actual_start (fallback to expected_start)
    if (dateFilter) {
      result = result.filter((b) => {
        const ref = b.actual_start ?? b.expected_start;
        return ref ? new Date(ref).toISOString().slice(0, 10) === dateFilter : false;
      });
    }

    // Search by batch number or furnace
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (b) =>
          String(b.batch_number).includes(q) ||
          (b.furnace ?? "").toLowerCase().includes(q),
      );
    }

    result.sort((a, b) => {
      let av: unknown = a[sortField];
      let bv: unknown = b[sortField];
      if (av === null || av === undefined) av = "";
      if (bv === null || bv === undefined) bv = "";
      if (av < bv) return sortDir === "asc" ? -1 : 1;
      if (av > bv) return sortDir === "asc" ? 1 : -1;
      return 0;
    });

    return result;
  }, [completedBatches, sortField, sortDir, search, dateFilter]);

  // Summary stats for filtered set
  const totalIngot   = filtered.reduce((s, b) => s + (b.ingot_kg ?? 0), 0);
  const totalEnergy  = filtered.reduce((s, b) => s + (b.energy_kwh ?? 0), 0);
  const avgPower     = filtered.length
    ? filtered.reduce((s, b) => s + (b.power_kw ?? 0), 0) / filtered.filter((b) => b.power_kw != null).length
    : 0;

  return (
    <PageWrapper
      title="Reports & Logs"
      subtitle="Completed batch records"
      actions={
        <div className="flex items-center gap-2 flex-wrap">
          {/* Date filter */}
          <div className="flex items-center gap-1 bg-bg-elevated border border-[var(--border-color)] rounded-lg px-2.5 py-1.5">
            <CalendarDays size={13} className="text-zinc-500 flex-shrink-0" />
            <input
              type="date"
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="bg-transparent text-xs text-[var(--text-primary)] focus:outline-none font-mono"
            />
            {dateFilter && (
              <button type="button" onClick={() => setDateFilter("")} className="text-zinc-500 hover:text-zinc-300 ml-0.5">
                <X size={12} />
              </button>
            )}
          </div>

          {/* Search */}
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-500" />
            <input
              type="text"
              placeholder="Batch # or furnace…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-bg-elevated border border-[var(--border-color)] text-xs text-[var(--text-primary)] rounded-lg pl-8 pr-3 py-2 focus:outline-none focus:border-zinc-500 w-44"
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
      {/* Summary KPI bar */}
      {filtered.length > 0 && (
        <div className="mb-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Completed Batches", value: String(filtered.length), unit: "" },
            { label: "Total Ingot",       value: totalIngot.toFixed(0),   unit: "kg" },
            { label: "Total Energy",      value: totalEnergy.toFixed(0),  unit: "kWh" },
            { label: "Avg Power",         value: isNaN(avgPower) ? "—" : avgPower.toFixed(0), unit: "kW" },
          ].map((kpi) => (
            <div key={kpi.label} className="bg-bg-card border border-[var(--border-color)] rounded-xl px-4 py-3">
              <p className="text-xs text-zinc-500 mb-1">{kpi.label}</p>
              <p className="font-mono text-lg font-bold text-[var(--text-primary)]">
                {kpi.value}
                {kpi.unit && <span className="text-sm font-normal text-zinc-400 ml-1">{kpi.unit}</span>}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Table */}
      <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-x-auto">
        <table className="w-full text-sm min-w-[900px]">
          <thead>
            <tr className="border-b border-[var(--border-color)]">
              {COLS.map((col) => (
                <th key={col.field} className={`px-4 py-3 text-left ${col.className ?? ""}`}>
                  <button
                    onClick={() => handleSort(col.field)}
                    className="flex items-center gap-1 text-xs font-medium text-zinc-400 uppercase tracking-wide hover:text-[var(--text-primary)] transition-colors ml-auto"
                    style={col.className?.includes("text-right") ? { marginLeft: "auto" } : {}}
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
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="border-b border-[var(--border-color)]">
                  {COLS.map((c) => (
                    <td key={c.field} className="px-4 py-3">
                      <div className="h-3 rounded bg-[var(--bg-elevated)] animate-pulse w-20" />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={COLS.length} className="px-4 py-12 text-center text-xs text-zinc-600">
                  {completedBatches.length === 0
                    ? "No completed batches yet — complete batches in Operator Execution"
                    : dateFilter || search
                    ? "No completed batches match the current filter"
                    : "No data"}
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
                  {/* Batch # + furnace */}
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)] whitespace-nowrap">
                    #{b.batch_number}
                    {b.furnace && (
                      <span
                        className="ml-2 font-semibold"
                        style={{ color: b.furnace === "A" ? "#E3000F" : "#3B82F6" }}
                      >
                        {b.furnace}
                      </span>
                    )}
                  </td>

                  {/* Expected Start */}
                  <td className="px-4 py-3 font-mono text-xs text-zinc-400 whitespace-nowrap">
                    {fmtDatetime(b.expected_start)}
                  </td>

                  {/* Actual Start */}
                  <td className="px-4 py-3 font-mono text-xs text-zinc-400 whitespace-nowrap">
                    {fmtDatetime(b.actual_start)}
                  </td>

                  {/* Ingot */}
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)] text-right">
                    {fmtNum(b.ingot_kg)}
                  </td>

                  {/* Fe */}
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)] text-right">
                    {fmtNum(b.fe_kg)}
                  </td>

                  {/* Si */}
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)] text-right">
                    {fmtNum(b.si_kg)}
                  </td>

                  {/* Scrap */}
                  <td className="px-4 py-3 font-mono text-xs text-[var(--text-primary)] text-right">
                    {fmtNum(b.scrap_kg)}
                  </td>

                  {/* Power kW */}
                  <td className="px-4 py-3 font-mono text-xs text-right">
                    <span className={b.power_kw != null ? "text-amber-400" : "text-zinc-600"}>
                      {fmtNum(b.power_kw, 0)}
                    </span>
                  </td>

                  {/* Energy kWh */}
                  <td className="px-4 py-3 font-mono text-xs text-right">
                    <span className={b.energy_kwh != null ? "text-blue-400" : "text-zinc-600"}>
                      {fmtNum(b.energy_kwh, 1)}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>

          {/* Totals footer */}
          {filtered.length > 1 && (
            <tfoot>
              <tr className="border-t-2 border-[var(--border-color)] bg-bg-elevated/30">
                <td colSpan={3} className="px-4 py-2.5 text-xs text-zinc-500 font-medium uppercase tracking-wide">
                  Total ({filtered.length} batches)
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-[var(--text-primary)] text-right font-semibold">
                  {totalIngot.toFixed(1)}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-[var(--text-primary)] text-right font-semibold">
                  {filtered.reduce((s, b) => s + (b.fe_kg ?? 0), 0).toFixed(1)}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-[var(--text-primary)] text-right font-semibold">
                  {filtered.reduce((s, b) => s + (b.si_kg ?? 0), 0).toFixed(1)}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-[var(--text-primary)] text-right font-semibold">
                  {filtered.reduce((s, b) => s + (b.scrap_kg ?? 0), 0).toFixed(1)}
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-amber-400 text-right font-semibold">
                  {isNaN(avgPower) ? "—" : avgPower.toFixed(0)} <span className="text-zinc-600 font-normal">avg</span>
                </td>
                <td className="px-4 py-2.5 font-mono text-xs text-blue-400 text-right font-semibold">
                  {totalEnergy.toFixed(1)}
                </td>
              </tr>
            </tfoot>
          )}
        </table>

        <div className="px-4 py-3 border-t border-[var(--border-color)] flex items-center justify-between text-xs text-zinc-500">
          <span>
            {filtered.length} of {completedBatches.length} completed batch{completedBatches.length !== 1 ? "es" : ""}
            {dateFilter && ` · ${dateFilter}`}
          </span>
          {batches.length > 0 && (
            <span>
              All plans: {batches.filter((b) => b.status === "completed").length} completed ·{" "}
              {batches.filter((b) => b.status === "in_progress").length} in progress ·{" "}
              {batches.filter((b) => b.status === "pending").length} pending
            </span>
          )}
        </div>
      </div>
    </PageWrapper>
  );
}
