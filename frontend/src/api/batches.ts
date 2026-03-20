import client from "./client";
import type { Batch } from "@/types";

export interface PowerProfilePoint {
  time_min: number;
  power_kw: number;
}

export const batchesApi = {
  list: (params?: { plan_id?: string }) =>
    client.get<Batch[]>("/batches", { params }).then((r) => r.data),

  get: (id: string) => client.get<Batch>(`/batches/${id}`).then((r) => r.data),

  update: (
    id: string,
    data: Partial<Pick<Batch, "actual_start" | "ingot_kg" | "fe_kg" | "si_kg" | "scrap_kg" | "status">>
  ) => client.patch<Batch>(`/batches/${id}`, data).then((r) => r.data),

  getPowerProfile: (id: string) =>
    client.get<PowerProfilePoint[]>(`/batches/${id}/power-profile`).then((r) => r.data),
};
