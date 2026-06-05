import client from "./client";
import type { Plan, Batch, ScheduleData } from "@/types";

export type PlanCreateInput = {
  target_batches: number;
  shift_start: string;
  opt_mode: string;
  if_a_enabled: boolean;
  if_b_enabled: boolean;
  mh_a_consumption_rate: number;
  mh_b_consumption_rate: number;
  mh_a_initial_level_kg: number;
  mh_b_initial_level_kg: number;
  consider_tou_price: boolean;
  consider_plant_load: boolean;
  preferred_start_furnace: "A" | "B";
};

// All fields optional — server only updates what's provided. Editing any
// GA-input field triggers a re-run (plan transitions back to "pending").
export type PlanUpdateInput = Partial<PlanCreateInput> & { status?: string };

export const plansApi = {
  list: () => client.get<Plan[]>("/plans").then((r) => r.data),
  get: (id: string) => client.get<Plan>(`/plans/${id}`).then((r) => r.data),
  create: (data: PlanCreateInput) =>
    client.post<Plan>("/plans", data).then((r) => r.data),
  update: (id: string, data: PlanUpdateInput) =>
    client.patch<Plan>(`/plans/${id}`, data).then((r) => r.data),
  getBatches: (id: string) =>
    client.get<Batch[]>(`/plans/${id}/batches`).then((r) => r.data),
  getScheduleData: (id: string) =>
    client.get<ScheduleData>(`/plans/${id}/schedule-data`).then((r) => r.data),
  delete: (id: string) => client.delete(`/plans/${id}`),
  activate: (id: string) =>
    client.post<Plan>(`/plans/${id}/activate`).then((r) => r.data),
  updateStatus: (id: string, status: string) =>
    client.patch<Plan>(`/plans/${id}`, { status }).then((r) => r.data),
};
