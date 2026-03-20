import { useQuery } from "@tanstack/react-query";
import { energyApi } from "@/api/energy";

export function useEnergyLogs(params?: { shift_start?: string; limit?: number }) {
  return useQuery({
    queryKey: ["energy-logs", params],
    queryFn: () => energyApi.list(params),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}
