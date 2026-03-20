import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { batchesApi } from "@/api/batches";
import type { Batch } from "@/types";

export function useBatches(planId?: string) {
  return useQuery({
    queryKey: ["batches", planId],
    queryFn: () => batchesApi.list(planId ? { plan_id: planId } : {}),
    enabled: true,
  });
}

export function usePowerProfile(batchId: string | null) {
  return useQuery({
    queryKey: ["power-profile", batchId],
    queryFn: () => batchesApi.getPowerProfile(batchId!),
    enabled: !!batchId,
  });
}

export function useUpdateBatch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Pick<Batch, "actual_start" | "ingot_kg" | "fe_kg" | "si_kg" | "scrap_kg" | "status">> }) =>
      batchesApi.update(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["batches"] });
    },
  });
}
