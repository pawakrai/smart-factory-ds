import React from "react";
import { NavLink } from "react-router-dom";
import { Sun, Moon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/appStore";

const tabs = [
  { label: "Dashboard", path: "/" },
  { label: "Production Planning", path: "/planning" },
  { label: "Operator Execution", path: "/operator" },
  { label: "Reports & Logs", path: "/reports" },
  { label: "Master Settings", path: "/settings" },
];

export default function TopNav() {
  const { theme, toggleTheme } = useAppStore();

  return (
    <header className="sticky top-0 z-50 w-full bg-bg-card border-b border-[var(--border-color)] transition-colors duration-200">
      <div className="max-w-screen-2xl mx-auto px-6 flex items-center gap-8 h-14">
        {/* Logo */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center">
            <span className="text-brand-red font-bold text-xl leading-none">S</span>
            <span className="text-[var(--brand-gray)] font-bold text-xl leading-none">HARP</span>
          </div>
          <span className="text-xs text-[var(--text-muted)] border-l border-[var(--border-color)] pl-2 ml-1 leading-tight">
            FurnaceFlow
          </span>
        </div>

        {/* Tabs */}
        <nav className="flex items-center gap-1">
          {tabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end={tab.path === "/"}
              className={({ isActive }) =>
                cn(
                  "px-4 py-1 text-sm font-medium transition-colors border-b-2 h-14 flex items-center",
                  isActive
                    ? "border-brand-red text-[var(--text-primary)]"
                    : "border-transparent text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                )
              }
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>

        {/* Right side: theme toggle + live clock */}
        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={toggleTheme}
            aria-label="Toggle theme"
            className={cn(
              "w-8 h-8 flex items-center justify-center rounded-lg transition-colors",
              "text-[var(--text-muted)] hover:text-[var(--text-primary)]",
              "bg-transparent hover:bg-[var(--bg-elevated)]"
            )}
          >
            {theme === "dark" ? (
              <Sun size={16} strokeWidth={2} />
            ) : (
              <Moon size={16} strokeWidth={2} />
            )}
          </button>
          <span className="font-kpi text-xs text-[var(--text-muted)]">
            <LiveClock />
          </span>
        </div>
      </div>
    </header>
  );
}

function LiveClock() {
  const [time, setTime] = React.useState(() => new Date().toLocaleTimeString());

  React.useEffect(() => {
    const id = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(id);
  }, []);

  return <span>{time}</span>;
}
