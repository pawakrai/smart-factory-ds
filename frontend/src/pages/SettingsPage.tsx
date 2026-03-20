import { useState } from "react";
import {
  Settings2, Save, RotateCcw, Check, AlertCircle,
  Flame, Container, Zap, Cpu, Clock, ChevronDown, ChevronRight,
} from "lucide-react";
import PageWrapper from "@/components/layout/PageWrapper";
import { Skeleton } from "@/components/ui/Skeleton";
import { useSettings, useUpdateSetting } from "@/hooks/useSettings";
import { useToast } from "@/store/toastStore";
import type { Setting } from "@/types";

// ── Settings groups ────────────────────────────────────────────────────────
const GROUPS: { label: string; icon: React.ReactNode; prefixes: string[] }[] = [
  {
    label: "IF Furnace",
    icon: <Flame size={14} />,
    prefixes: ["if_", "cold_start_", "post_pour_"],
  },
  {
    label: "M&H Furnace",
    icon: <Container size={14} />,
    prefixes: ["mh_"],
  },
  {
    label: "Energy & Tariff",
    icon: <Zap size={14} />,
    prefixes: ["tou_", "ft_", "demand_", "contract_", "peak_hours_", "solar_"],
  },
  {
    label: "GA Optimization",
    icon: <Cpu size={14} />,
    prefixes: ["ga_"],
  },
  {
    label: "Shift Configuration",
    icon: <Clock size={14} />,
    prefixes: ["shift_", "target_batches_default"],
  },
];

function matchesGroup(key: string, prefixes: string[]): boolean {
  return prefixes.some((p) => key.startsWith(p));
}

function groupSettings(settings: Setting[]): { group: typeof GROUPS[0]; items: Setting[] }[] {
  const assigned = new Set<string>();
  const result = GROUPS.map((group) => {
    const items = settings.filter((s) => {
      if (assigned.has(s.config_key)) return false;
      if (matchesGroup(s.config_key, group.prefixes)) {
        assigned.add(s.config_key);
        return true;
      }
      return false;
    });
    return { group, items };
  });
  // Any remaining ungrouped settings
  const ungrouped = settings.filter((s) => !assigned.has(s.config_key));
  if (ungrouped.length > 0) {
    result.push({
      group: { label: "Other", icon: <Settings2 size={14} />, prefixes: [] },
      items: ungrouped,
    });
  }
  return result;
}

// ── SettingRow ─────────────────────────────────────────────────────────────
function SettingRow({ setting }: { setting: Setting }) {
  const [value, setValue] = useState(setting.config_value);
  const [saved, setSaved] = useState(false);
  const isDirty = value !== setting.config_value;
  const updateSetting = useUpdateSetting();
  const toast = useToast();

  async function handleSave() {
    try {
      await updateSetting.mutateAsync({ key: setting.config_key, value });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      toast.success("Saved", setting.config_key);
    } catch {
      toast.error("Save failed", `Could not update ${setting.config_key}`);
    }
  }

  return (
    <div className="flex items-center justify-between px-5 py-3.5 gap-4">
      <div className="min-w-0 flex-1">
        <p className="text-xs font-mono text-[var(--text-secondary)] leading-tight">{setting.config_key}</p>
        {setting.description && (
          <p className="text-[11px] text-zinc-500 mt-0.5">{setting.description}</p>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0">
        <input
          type="text"
          value={value}
          onChange={(e) => { setValue(e.target.value); setSaved(false); }}
          onKeyDown={(e) => e.key === "Enter" && isDirty && handleSave()}
          className={`w-36 bg-bg-elevated border rounded-lg px-3 py-1.5 text-sm font-mono text-[var(--text-primary)] text-right focus:outline-none transition-colors
            ${isDirty ? "border-brand-red/60 focus:border-brand-red" : "border-[var(--border-color)] focus:border-zinc-500"}`}
        />
        {isDirty && (
          <button
            onClick={() => setValue(setting.config_value)}
            title="Reset"
            className="text-zinc-600 hover:text-[var(--text-secondary)] transition-colors"
          >
            <RotateCcw size={13} />
          </button>
        )}
        {saved ? (
          <span className="flex items-center gap-1 text-xs text-green-400 w-16">
            <Check size={13} /> Saved
          </span>
        ) : (
          <button
            onClick={handleSave}
            disabled={!isDirty || updateSetting.isPending}
            className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg w-16 justify-center transition-colors
              ${isDirty ? "bg-brand-red hover:bg-brand-red-dark text-white" : "text-zinc-700 cursor-default"}`}
          >
            <Save size={12} />
            Save
          </button>
        )}
      </div>
    </div>
  );
}

function SettingRowSkeleton() {
  return (
    <div className="flex items-center justify-between px-5 py-3.5 gap-4">
      <div className="flex-1 space-y-1.5">
        <Skeleton className="h-3 w-56" />
        <Skeleton className="h-2.5 w-40" />
      </div>
      <Skeleton className="h-8 w-36 rounded-lg" />
    </div>
  );
}

// ── Collapsible settings group card ──────────────────────────────────────
function SettingsGroupCard({
  label,
  icon,
  items,
  defaultOpen = true,
}: {
  label: string;
  icon: React.ReactNode;
  items: Setting[];
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-4 border-b border-[var(--border-color)] hover:bg-[var(--bg-elevated)]/50 transition-colors"
      >
        <div className="flex items-center gap-2 text-brand-red">
          {icon}
          <span className="text-sm font-semibold text-[var(--text-primary)]">{label}</span>
          <span className="text-[10px] text-zinc-500 font-normal">({items.length})</span>
        </div>
        {open ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
      </button>

      {open && (
        <div className="divide-y divide-[var(--border-color)]">
          {items.map((s) => (
            <SettingRow key={s.config_key} setting={s} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────
export default function SettingsPage() {
  const { data: settings, isLoading, isError } = useSettings();

  const grouped = settings ? groupSettings(settings) : [];

  return (
    <PageWrapper
      title="Master Settings"
      subtitle="Furnace parameters, energy tariff, GA optimization, and shift configuration"
    >
      <div className="space-y-4">
        {isLoading ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-[var(--border-color)] flex items-center gap-2">
              <Settings2 size={16} className="text-brand-red" />
              <span className="text-sm font-semibold text-[var(--text-primary)]">Loading…</span>
            </div>
            <div className="divide-y divide-[var(--border-color)]">
              {Array.from({ length: 8 }).map((_, i) => <SettingRowSkeleton key={i} />)}
            </div>
          </div>
        ) : isError ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl px-5 py-12 flex flex-col items-center gap-2 text-zinc-500">
            <AlertCircle size={20} className="text-brand-red/60" />
            <p className="text-sm">Failed to load settings</p>
            <p className="text-xs">Check that the backend is running</p>
          </div>
        ) : settings?.length === 0 ? (
          <div className="bg-bg-card border border-[var(--border-color)] rounded-xl px-5 py-12 text-center text-xs text-zinc-600">
            No settings found — check that the database is seeded
          </div>
        ) : (
          grouped.map(({ group, items }) =>
            items.length === 0 ? null : (
              <SettingsGroupCard
                key={group.label}
                label={group.label}
                icon={group.icon}
                items={items}
                defaultOpen={true}
              />
            )
          )
        )}

        {/* System info */}
        <div className="bg-bg-card border border-[var(--border-color)] rounded-xl p-5">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide mb-4">
            System Info
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Backend", value: "FastAPI 0.115", status: "ok" },
              { label: "Database", value: "Supabase / PostgreSQL", status: "ok" },
              { label: "GA Engine", value: "src/app_v9.py", status: "ok" },
              { label: "RL Model", value: "DQN (mock fallback)", status: "warn" },
            ].map((info) => (
              <div key={info.label} className="bg-bg-elevated rounded-lg px-3 py-2.5">
                <p className="text-[10px] text-zinc-500 uppercase tracking-wide mb-1">{info.label}</p>
                <p className="text-xs text-[var(--text-primary)] font-mono truncate">{info.value}</p>
                <span
                  className="inline-block mt-1 text-[10px] font-medium"
                  style={{ color: info.status === "ok" ? "#22C55E" : "#F59E0B" }}
                >
                  {info.status === "ok" ? "● Online" : "● Warning"}
                </span>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[10px] text-zinc-600 font-mono">
            PUT /api/settings/{"{"}<span className="text-zinc-500">key</span>{"}"} · values stored as text · press Enter or click Save per row
          </p>
        </div>
      </div>
    </PageWrapper>
  );
}
