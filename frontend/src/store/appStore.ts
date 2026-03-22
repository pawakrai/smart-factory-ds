import { create } from "zustand";
import type { Plan, Batch } from "@/types";

type Theme = "dark" | "light";

interface AppState {
  activePlan: Plan | null;
  activeBatches: Batch[];
  theme: Theme;
  selectedPlanId: string | null;
  setActivePlan: (plan: Plan | null) => void;
  setActiveBatches: (batches: Batch[]) => void;
  toggleTheme: () => void;
  setSelectedPlanId: (id: string | null) => void;
}

const savedTheme = (localStorage.getItem("theme") as Theme) ?? "dark";
const savedSelectedPlanId = localStorage.getItem("selectedPlanId") ?? null;

export const useAppStore = create<AppState>((set) => ({
  activePlan: null,
  activeBatches: [],
  theme: savedTheme,
  selectedPlanId: savedSelectedPlanId,
  setActivePlan: (plan) => set({ activePlan: plan }),
  setActiveBatches: (batches) => set({ activeBatches: batches }),
  toggleTheme: () =>
    set((state) => {
      const next: Theme = state.theme === "dark" ? "light" : "dark";
      localStorage.setItem("theme", next);
      return { theme: next };
    }),
  setSelectedPlanId: (id) => {
    if (id) localStorage.setItem("selectedPlanId", id);
    else localStorage.removeItem("selectedPlanId");
    set({ selectedPlanId: id });
  },
}));
