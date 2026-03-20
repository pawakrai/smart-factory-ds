import { create } from "zustand";
import type { Plan, Batch } from "@/types";

type Theme = "dark" | "light";

interface AppState {
  activePlan: Plan | null;
  activeBatches: Batch[];
  theme: Theme;
  setActivePlan: (plan: Plan | null) => void;
  setActiveBatches: (batches: Batch[]) => void;
  toggleTheme: () => void;
}

const savedTheme = (localStorage.getItem("theme") as Theme) ?? "dark";

export const useAppStore = create<AppState>((set) => ({
  activePlan: null,
  activeBatches: [],
  theme: savedTheme,
  setActivePlan: (plan) => set({ activePlan: plan }),
  setActiveBatches: (batches) => set({ activeBatches: batches }),
  toggleTheme: () =>
    set((state) => {
      const next: Theme = state.theme === "dark" ? "light" : "dark";
      localStorage.setItem("theme", next);
      return { theme: next };
    }),
}));
