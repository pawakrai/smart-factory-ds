import client from "./client";
import type { Plan, Batch, ScheduleData } from "@/types";

export const plansApi = {
  list: () => client.get<Plan[]>("/plans").then((r) => r.data),
  get: (id: string) => client.get<Plan>(`/plans/${id}`).then((r) => r.data),
  create: (data: {
    target_batches: number;
    shift_start: string;
    opt_mode: string;
    if_a_enabled: boolean;
    if_b_enabled: boolean;
    mh_a_consumption_rate: number;
    mh_b_consumption_rate: number;
  }) => client.post<Plan>("/plans", data).then((r) => r.data),
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
