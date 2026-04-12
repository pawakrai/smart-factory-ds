import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { settingsApi } from "@/api/settings";

export function useSettings() {
  return useQuery({
    queryKey: ["settings"],
    queryFn: settingsApi.list,
  });
}

export function useSettingDefaults() {
  return useQuery({
    queryKey: ["settings-defaults"],
    queryFn: settingsApi.defaults,
    staleTime: Infinity, // defaults never change at runtime
  });
}

export function useUpdateSetting() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ key, value }: { key: string; value: string }) =>
      settingsApi.update(key, value),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["settings"] }),
  });
}
