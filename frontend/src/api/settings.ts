import client from "./client";
import type { Setting } from "@/types";

export const settingsApi = {
  list: () => client.get<Setting[]>("/settings").then((r) => r.data),
  update: (key: string, value: string) =>
    client.put<Setting>(`/settings/${key}`, { config_value: value }).then((r) => r.data),
};
