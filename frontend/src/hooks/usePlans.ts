import { useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { plansApi, type PlanUpdateInput } from "@/api/plans";

export function usePlans() {
  return useQuery({
    queryKey: ["plans"],
    queryFn: plansApi.list,
    staleTime: 10_000,
  });
}

export function usePlanBatches(planId: string | null) {
  return useQuery({
    queryKey: ["plan-batches", planId],
    queryFn: () => plansApi.getBatches(planId!),
    enabled: !!planId,
    staleTime: 30_000,
  });
}

// Schedule data is rebuilt on the server whenever GA-input fields change. Use a
// short stale window so an edit-driven regeneration is picked up automatically
// once `usePlanPolling` invalidates this query.
export function usePlanScheduleData(planId: string | null) {
  return useQuery({
    queryKey: ["plan-schedule-data", planId],
    queryFn: () => plansApi.getScheduleData(planId!),
    enabled: !!planId,
    staleTime: 30_000,
  });
}

/** Poll a plan every 2 s while its status is "pending". Invalidates related
 *  queries automatically when the plan transitions to active/draft. */
export function usePlanPolling(
  planId: string | null,
  onComplete?: (planId: string) => void,
) {
  const qc = useQueryClient();
  const query = useQuery({
    queryKey: ["plan-poll", planId],
    queryFn: () => plansApi.get(planId!),
    enabled: !!planId,
    refetchInterval: (q) =>
      q.state.data?.status === "pending" ? 2_000 : false,
  });

  useEffect(() => {
    if (!planId || !query.data) return;
    if (query.data.status !== "pending") {
      qc.invalidateQueries({ queryKey: ["plans"] });
      qc.invalidateQueries({ queryKey: ["plan-batches", planId] });
      qc.invalidateQueries({ queryKey: ["plan-schedule-data", planId] });
      onComplete?.(planId);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query.data?.status]);

  return query;
}

export function useCreatePlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: plansApi.create,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["plans"] }),
  });
}

/** Edit an existing plan. Editing any GA-input field forces a server-side
 *  re-run; caller should pass the returned plan id to `usePlanPolling` to
 *  refetch schedule data once status leaves "pending". */
export function useUpdatePlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: PlanUpdateInput }) =>
      plansApi.update(id, data),
    onSuccess: (plan) => {
      qc.invalidateQueries({ queryKey: ["plans"] });
      qc.invalidateQueries({ queryKey: ["plan-batches", plan.id] });
      qc.invalidateQueries({ queryKey: ["plan-schedule-data", plan.id] });
    },
  });
}

export function useDeletePlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: plansApi.delete,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["plans"] }),
  });
}

export function useActivatePlan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => plansApi.activate(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["plans"] }),
  });
}
