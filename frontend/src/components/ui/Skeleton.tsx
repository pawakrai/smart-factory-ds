import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded bg-[var(--bg-elevated)]", className)} />;
}

/** Placeholder for a bar/line chart while loading */
export function ChartSkeleton({ className }: { className?: string }) {
  const bars = [65, 45, 80, 55, 70, 40, 75, 60, 85, 50, 65, 45];
  return (
    <div className={cn("w-full h-full flex items-end gap-1.5 px-2 pb-2", className)}>
      {bars.map((h, i) => (
        <div key={i} className="flex-1 flex flex-col items-center justify-end" style={{ height: "100%" }}>
          <div
            className="w-full rounded-sm animate-pulse bg-[var(--bg-elevated)]"
            style={{ height: `${h}%` }}
          />
        </div>
      ))}
    </div>
  );
}

/** Skeleton for a single KPI card */
export function KpiCardSkeleton() {
  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <Skeleton className="h-3 w-28" />
        <Skeleton className="h-4 w-4 rounded" />
      </div>
      <Skeleton className="h-8 w-20 mb-2" />
      <Skeleton className="h-3 w-24" />
    </div>
  );
}

/** Skeleton for a table row */
export function TableRowSkeleton({ cols = 5 }: { cols?: number }) {
  return (
    <tr className="border-b border-[var(--border-color)]">
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <Skeleton className="h-3 w-full max-w-[120px]" />
        </td>
      ))}
    </tr>
  );
}
