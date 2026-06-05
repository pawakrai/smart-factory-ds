from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class PlanCreate(BaseModel):
    target_batches: int
    shift_start: datetime
    opt_mode: str = "energy"  # "energy" | "service"
    if_a_enabled: bool = True
    if_b_enabled: bool = True
    mh_a_consumption_rate: Optional[float] = None   # None = use settings default
    mh_b_consumption_rate: Optional[float] = None
    mh_a_initial_level_kg: Optional[float] = None   # None = use settings default
    mh_b_initial_level_kg: Optional[float] = None
    consider_tou_price: bool = True
    consider_plant_load: bool = True
    preferred_start_furnace: str = "A"  # "A" or "B" — which IF starts the first charge


class PlanUpdate(BaseModel):
    """Mutable plan fields. Editing any GA_RECOMPUTE_FIELDS triggers a re-run."""
    status: Optional[str] = None
    target_batches: Optional[int] = None
    shift_start: Optional[datetime] = None
    opt_mode: Optional[str] = None
    if_a_enabled: Optional[bool] = None
    if_b_enabled: Optional[bool] = None
    mh_a_consumption_rate: Optional[float] = None
    mh_b_consumption_rate: Optional[float] = None
    mh_a_initial_level_kg: Optional[float] = None
    mh_b_initial_level_kg: Optional[float] = None
    consider_tou_price: Optional[bool] = None
    consider_plant_load: Optional[bool] = None
    preferred_start_furnace: Optional[str] = None


# Fields whose change requires re-running the GA so schedule_data stays consistent
# with the plan inputs. Anything outside this set (status, etc.) is a metadata edit.
GA_RECOMPUTE_FIELDS = frozenset({
    "target_batches",
    "shift_start",
    "opt_mode",
    "if_a_enabled",
    "if_b_enabled",
    "mh_a_consumption_rate",
    "mh_b_consumption_rate",
    "mh_a_initial_level_kg",
    "mh_b_initial_level_kg",
    "consider_tou_price",
    "consider_plant_load",
    "preferred_start_furnace",
})


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
    mh_a_max_capacity_kg: float = 800.0
    mh_b_max_capacity_kg: float = 1100.0
    if_kw: list[float]
    baseline_kw: list[float]
    total_plant_kw: list[float]
    tou_effective_price: list[float]
    tou_raw_price: list[float] = []
    contract_demand_kw: float = 1600.0
    # Solar window in minutes relative to shift start (None if outside shift)
    solar_window_start_min: Optional[int] = None
    solar_window_end_min: Optional[int] = None


class PlanRead(BaseModel):
    id: str
    target_batches: int
    shift_start: datetime
    opt_mode: str
    status: str
    if_a_enabled: Optional[bool] = None
    if_b_enabled: Optional[bool] = None
    mh_a_consumption_rate: Optional[float] = None
    mh_b_consumption_rate: Optional[float] = None
    mh_a_initial_level_kg: Optional[float] = None
    mh_b_initial_level_kg: Optional[float] = None
    consider_tou_price: Optional[bool] = None
    consider_plant_load: Optional[bool] = None
    preferred_start_furnace: Optional[str] = None
    schedule_metrics: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PlanCreateResponse(BaseModel):
    plan: PlanRead
    metrics: Optional[ScheduleMetrics] = None
    batch_count: int
