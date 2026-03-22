export type PlanStatus = "pending" | "draft" | "active" | "completed";
export type BatchStatus = "pending" | "in_progress" | "completed";
export type OptMode = "energy" | "service";

export interface Plan {
  id: string;
  target_batches: number;
  shift_start: string;
  opt_mode: OptMode;
  status: PlanStatus;
  schedule_metrics: string | null;
  created_at: string;
}

export interface Batch {
  id: string;
  plan_id: string;
  batch_number: number;
  expected_start: string | null;
  actual_start: string | null;
  furnace: "A" | "B" | null;
  duration_min: number | null;
  melt_finish_at: string | null;
  pour_at: string | null;
  power_kw: number | null;
  is_cold_start: boolean;
  energy_kwh: number | null;
  ingot_kg: number | null;
  fe_kg: number | null;
  si_kg: number | null;
  scrap_kg: number | null;
  status: BatchStatus;
  created_at: string;
}

export interface EnergyLog {
  id: string;
  batch_id: string | null;
  timestamp: string;
  sim_kw: number | null;
  actual_kw: number | null;
}

export interface Setting {
  id: string;
  config_key: string;
  config_value: string;
  description: string | null;
  updated_at: string;
}

export interface ScheduleMetrics {
  poured_batches_count: number;
  missing_batches: number;
  total_if_kwh: number;
  total_energy_cost_day: number;
  demand_charge_day_equiv: number;
  peak_kw: number;
  makespan_minutes: number;
  mh_empty_minutes_a: number;
  mh_empty_minutes_b: number;
  solar_cost_saving: number;
  if_use_count_a: number;
  if_use_count_b: number;
}

export interface ScheduleData {
  plan_id: string;
  duration_minutes: number;
  shift_start_iso: string;
  sample_interval_min: number;
  mh_a_levels_kg: number[];
  mh_b_levels_kg: number[];
  mh_a_min_level_kg: number;
  mh_b_min_level_kg: number;
  if_kw: number[];
  baseline_kw: number[];
  total_plant_kw: number[];
  tou_effective_price: number[];
  tou_raw_price: number[];
  contract_demand_kw: number;
  solar_window_start_min?: number;
  solar_window_end_min?: number;
}

export interface PlanCreateResponse {
  plan: Plan;
  metrics: ScheduleMetrics | null;
  batch_count: number;
}
