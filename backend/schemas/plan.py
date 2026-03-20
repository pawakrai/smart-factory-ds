from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class PlanCreate(BaseModel):
    target_batches: int
    shift_start: datetime
    opt_mode: str = "energy"  # "energy" | "service"


class PlanUpdate(BaseModel):
    status: str | None = None


class ScheduleMetrics(BaseModel):
    poured_batches_count: int
    missing_batches: int
    total_if_kwh: float
    total_energy_cost_day: float          # Baht
    demand_charge_day_equiv: float        # Baht
    peak_kw: float
    makespan_minutes: float
    mh_empty_minutes_a: float
    mh_empty_minutes_b: float
    solar_cost_saving: float
    if_use_count_a: int
    if_use_count_b: int


class ScheduleData(BaseModel):
    plan_id: str
    duration_minutes: int
    shift_start_iso: str
    sample_interval_min: int              # e.g., 5 minutes
    mh_a_levels_kg: list[float]
    mh_b_levels_kg: list[float]
    mh_a_min_level_kg: float
    mh_b_min_level_kg: float
    if_kw: list[float]
    baseline_kw: list[float]
    total_plant_kw: list[float]
    tou_effective_price: list[float]
    # Solar window in minutes relative to shift start (None if outside shift)
    solar_window_start_min: Optional[int] = None
    solar_window_end_min: Optional[int] = None


class PlanRead(BaseModel):
    id: str
    target_batches: int
    shift_start: datetime
    opt_mode: str
    status: str
    schedule_metrics: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PlanCreateResponse(BaseModel):
    plan: PlanRead
    metrics: Optional[ScheduleMetrics] = None
    batch_count: int
