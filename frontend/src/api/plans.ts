import client from "./client";
import type { Plan, Batch, ScheduleData } from "@/types";

export const plansApi = {
  list: () => client.get<Plan[]>("/plans").then((r) => r.data),
  get: (id: string) => client.get<Plan>(`/plans/${id}`).then((r) => r.data),
  create: (data: { target_batches: number; shift_start: string; opt_mode: string }) =>
    client.post<Plan>("/plans", data).then((r) => r.data),
  getBatches: (id: string) =>
    client.get<Batch[]>(`/plans/${id}/batches`).then((r) => r.data),
  getScheduleData: (id: string) =>
    client.get<ScheduleData>(`/plans/${id}/schedule-data`).then((r) => r.data),
  delete: (id: string) => client.delete(`/plans/${id}`),
};
