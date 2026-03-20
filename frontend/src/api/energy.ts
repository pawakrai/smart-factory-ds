import client from "./client";
import type { EnergyLog } from "@/types";

export const energyApi = {
  list: (params?: { shift_start?: string; limit?: number }) =>
    client.get<EnergyLog[]>("/energy-logs", { params }).then((r) => r.data),
};
