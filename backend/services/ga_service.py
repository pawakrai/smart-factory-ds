"""
GA Service — wraps src/app_v9.py (read-only) to generate a production schedule.

Returns a GaResult dataclass containing:
  - schedule_items: list of batch dicts with timing, furnace, power, cold-start info
  - metrics: ScheduleMetrics (KPI summary)
  - schedule_data: ScheduleData (time-series arrays for charts, sampled every 5 min)

Falls back to a simple sequential schedule on any error.

Threading note: app_v9 mutates module-level globals (MH_MAX_CAPACITY_KG etc.) when
settings are applied. _GA_LOCK prevents concurrent runs from corrupting each other.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.plan import Plan

from ..schemas.plan import ScheduleMetrics, ScheduleData

logger = logging.getLogger(__name__)

# One GA run at a time — protects module-global mutation in app_v9
_GA_LOCK = threading.Lock()


@dataclass
class GaResult:
    schedule_items: list[dict[str, Any]] = field(default_factory=list)
    metrics: Optional[ScheduleMetrics] = None
    schedule_data: Optional[ScheduleData] = None


def generate_schedule(plan: "Plan", settings_dict: dict[str, str]) -> GaResult:
    """Run app_v9 GA and return GaResult (falls back to sequential on error)."""
    try:
        return _run_app_v9_ga(plan, settings_dict)
    except Exception as exc:
        logger.warning("GA failed (%s), using sequential fallback.", exc, exc_info=True)
        return _sequential_fallback(plan)


# ---------------------------------------------------------------------------
# app_v9 GA integration
# ---------------------------------------------------------------------------

def _run_app_v9_ga(plan: "Plan", settings_dict: dict[str, str]) -> GaResult:
    import sys, os, importlib
    import numpy as np
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PolynomialMutation
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    # Ensure project root is importable
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

    app_v9 = importlib.import_module("src.app_v9")

    pop_size = int(settings_dict.get("ga_pop_size", "80"))
    n_gen = int(settings_dict.get("ga_n_generations", "100"))
    patience = int(settings_dict.get("ga_early_stop_patience", "20"))
    seed = int(settings_dict.get("ga_random_seed", "42"))

    with _GA_LOCK:
        # Apply settings overrides to app_v9 module globals
        _apply_settings_overrides(app_v9, settings_dict, plan)

        problem = app_v9.PolicyProblem()
        algorithm = GA(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.92, eta=8),
            mutation=PolynomialMutation(prob=0.45, eta=8),
            eliminate_duplicates=app_v9.PolicyDuplicateElimination(),
        )
        termination = get_termination("n_gen", n_gen)
        callback = app_v9.StagnationEarlyStopCallback(
            patience_gens=patience,
            delta_obj=app_v9.EARLY_STOP_DELTA_OBJ1,
        )

        logger.info("Starting GA optimization (pop=%d, n_gen=%d, opt_mode=%s)...",
                    pop_size, n_gen, plan.opt_mode)
        result = minimize(
            problem, algorithm, termination,
            seed=seed, verbose=False, save_history=False, callback=callback,
        )
        logger.info("GA optimization finished.")

        # Extract best solution
        best_x = _extract_best_x(result, algorithm)
        if best_x is None:
            raise RuntimeError("GA returned no valid solution")

        policy = app_v9._decode_policy_vector(best_x)
        sim = app_v9.simulate_policy_day(policy)

    return _build_ga_result(plan, sim, settings_dict)


def _extract_best_x(result, algorithm):
    """Get best_x from GA result, falling back to least-infeasible population member."""
    import numpy as np
    if result is not None and result.X is not None:
        return np.asarray(result.X, dtype=float).ravel()

    pop = None
    if result is not None and getattr(result, "pop", None) is not None:
        pop = result.pop
    elif getattr(algorithm, "pop", None) is not None:
        pop = algorithm.pop

    if pop is None:
        return None

    pop_X = pop.get("X")
    pop_F = pop.get("F")
    pop_G = pop.get("G")
    if pop_X is None or len(pop_X) == 0:
        return None

    if pop_G is not None:
        cv = np.sum(np.maximum(0.0, pop_G), axis=1)
    else:
        cv = np.zeros(len(pop_X), dtype=float)

    if pop_F is not None:
        f_vals = pop_F.ravel() if pop_F.ndim > 1 else pop_F
    else:
        f_vals = np.zeros(len(pop_X), dtype=float)

    best_idx = int(np.lexsort((f_vals, cv))[0])
    return np.asarray(pop_X[best_idx], dtype=float).ravel()


def _apply_settings_overrides(app_v9, settings_dict: dict[str, str], plan: "Plan"):
    """Patch app_v9 module globals from settings dict + plan inputs."""
    def f(key, default):
        return float(settings_dict.get(key, str(default)))

    def i(key, default):
        return int(settings_dict.get(key, str(default)))

    # Plan-level overrides
    app_v9.NUM_BATCHES_RUN_OVERRIDE = plan.target_batches
    app_v9.OPT_MODE = plan.opt_mode
    shift_dt = plan.shift_start
    app_v9.SHIFT_START = shift_dt.hour * 60 + shift_dt.minute

    # IF Furnace
    app_v9.IF_POWER_OPTIONS = [
        f("if_power_option_low_kw", 450),
        f("if_power_option_mid_kw", 475),
        f("if_power_option_high_kw", 500),
    ]
    # Update POWER_PROFILE to match new power options — keep the existing entries
    # (durations/energies are fixed physics, not user-configurable)
    app_v9.IF_BATCH_OUTPUT_KG = f("if_batch_output_kg", 500)
    app_v9.IF_FURNACE_EFFICIENCY_FACTOR = {
        0: f("if_efficiency_factor_a", 0.99),
        1: f("if_efficiency_factor_b", 1.03),
    }
    app_v9.COLD_START_GAP_THRESHOLD_MIN = f("cold_start_gap_threshold_min", 180)
    app_v9.COLD_START_EXTRA_DURATION_MIN = f("cold_start_extra_duration_min", 8)
    app_v9.COLD_START_EXTRA_ENERGY_KWH = f("cold_start_extra_energy_kwh", 30)
    app_v9.POST_POUR_DOWNTIME_MIN = i("post_pour_downtime_min", 10)

    # M&H Furnace
    app_v9.MH_MAX_CAPACITY_KG = {
        "A": f("mh_a_capacity_kg", 400),
        "B": f("mh_b_capacity_kg", 250),
    }
    app_v9.MH_INITIAL_LEVEL_KG = {
        "A": f("mh_a_initial_level_kg", 400),
        "B": f("mh_b_initial_level_kg", 230),
    }
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN = {
        "A": f("mh_a_consumption_rate_kg_per_min", 2.20),
        "B": f("mh_b_consumption_rate_kg_per_min", 2.30),
    }
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG = {
        "A": f("mh_a_min_operational_level_kg", 200),
        "B": f("mh_b_min_operational_level_kg", 125),
    }

    # Energy & Tariff
    onpeak_base = f("tou_onpeak_baht_per_kwh", 4.1839)
    offpeak_base = f("tou_offpeak_baht_per_kwh", 2.6037)
    ft = f("ft_baht_per_kwh", 0.0972)
    app_v9.TOU_ONPEAK_BAHT_PER_KWH_BASE = onpeak_base
    app_v9.TOU_OFFPEAK_BAHT_PER_KWH_BASE = offpeak_base
    app_v9.FT_BAHT_PER_KWH = ft
    app_v9.TOU_ONPEAK_BAHT_PER_KWH = onpeak_base + ft
    app_v9.TOU_OFFPEAK_BAHT_PER_KWH = offpeak_base + ft
    app_v9.DEMAND_CHARGE_BAHT_PER_KW_MONTH = f("demand_charge_baht_per_kw_month", 132.93)
    app_v9.CONTRACT_DEMAND_KW = f("contract_demand_kw", 1600)

    # Solar window (parse HH:MM strings)
    solar_start_str = settings_dict.get("solar_window_start", "12:00")
    solar_end_str = settings_dict.get("solar_window_end", "13:00")
    app_v9.SOLAR_START = _hhmm_to_min(solar_start_str)
    app_v9.SOLAR_END = _hhmm_to_min(solar_end_str)
    app_v9.SOLAR_EFFECTIVE_PRICE_FACTOR = f("solar_price_factor", 0.35)

    # Clear evaluation cache so new settings take effect
    app_v9._EVAL_CACHE.clear()


def _hhmm_to_min(hhmm: str) -> int:
    """Parse 'HH:MM' string to minutes from midnight."""
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def _build_ga_result(plan: "Plan", sim: dict, settings_dict: dict[str, str]) -> GaResult:
    """Convert simulate_policy_day() output into GaResult."""
    import numpy as np

    schedule = sim.get("schedule", [])
    m = sim.get("metrics", {})
    shift_start: datetime = plan.shift_start

    # --- Schedule items (one per batch) ---
    schedule_items: list[dict[str, Any]] = []
    for b in schedule:
        start_min = b.get("start_min")
        melt_finish_min = b.get("melt_finish_min")
        pour_min = b.get("pour_min")
        furnace_idx = b.get("if_furnace", 0)
        furnace = "A" if furnace_idx == 0 else "B"
        power_kw = b.get("power_kw")
        is_cold = bool(b.get("is_cold_start", False))

        # Estimate energy for this batch from POWER_PROFILE
        energy_kwh = None
        if power_kw is not None:
            try:
                from src.app_v9 import POWER_PROFILE, COLD_START_EXTRA_ENERGY_KWH
                profile = POWER_PROFILE.get(float(power_kw), {})
                energy_kwh = profile.get("energy_kwh_hot", 0.0)
                if is_cold:
                    energy_kwh += COLD_START_EXTRA_ENERGY_KWH
            except Exception:
                pass

        schedule_items.append({
            "batch_id": b.get("batch_id", 1),  # already 1-based from app_v9
            "expected_start": shift_start + timedelta(minutes=start_min) if start_min is not None else None,
            "melt_finish_at": shift_start + timedelta(minutes=melt_finish_min) if melt_finish_min is not None else None,
            "pour_at": shift_start + timedelta(minutes=pour_min) if pour_min is not None else None,
            "furnace": furnace,
            "duration_min": int(melt_finish_min - start_min) if (melt_finish_min and start_min is not None) else 90,
            "power_kw": float(power_kw) if power_kw is not None else None,
            "is_cold_start": is_cold,
            "energy_kwh": energy_kwh,
        })

    # --- Metrics ---
    metrics = ScheduleMetrics(
        poured_batches_count=int(m.get("poured_batches_count", 0)),
        missing_batches=int(m.get("missing_batches", 0)),
        total_if_kwh=float(m.get("total_if_kwh", 0.0)),
        total_energy_cost_day=float(m.get("total_energy_cost_day", 0.0)),
        demand_charge_day_equiv=float(m.get("demand_charge_day_equiv", 0.0)),
        peak_kw=float(m.get("peak_kw", 0.0)),
        makespan_minutes=float(m.get("makespan_minutes", 0.0)),
        mh_empty_minutes_a=float((m.get("mh_empty_minutes") or {}).get("A", 0.0)),
        mh_empty_minutes_b=float((m.get("mh_empty_minutes") or {}).get("B", 0.0)),
        solar_cost_saving=float(m.get("solar_cost_saving", 0.0)),
        if_use_count_a=int(m.get("if_use_count_A", 0)),
        if_use_count_b=int(m.get("if_use_count_B", 0)),
    )

    # --- ScheduleData (time-series, downsampled every SAMPLE_INTERVAL minutes) ---
    SAMPLE_INTERVAL = 5
    mh_levels = sim.get("mh_levels", {})
    if_kw_raw = sim.get("if_kw", np.zeros(1440))
    baseline_raw = sim.get("baseline_kw", np.zeros(1440))
    total_kw_raw = sim.get("total_plant_kw", np.zeros(1440))
    tou_raw = sim.get("tou_effective_price", np.zeros(1440))

    mh_a_raw = mh_levels.get("A", np.zeros(1440)) if mh_levels else np.zeros(1440)
    mh_b_raw = mh_levels.get("B", np.zeros(1440)) if mh_levels else np.zeros(1440)

    duration_min = len(np.asarray(if_kw_raw))
    indices = list(range(0, duration_min, SAMPLE_INTERVAL))

    def _sample(arr):
        a = np.asarray(arr, dtype=float)
        return [round(float(a[i]), 2) for i in indices if i < len(a)]

    from src.app_v9 import MH_MIN_OPERATIONAL_LEVEL_KG

    # Solar window relative to shift start (minutes)
    solar_start_abs = _hhmm_to_min(settings_dict.get("solar_window_start", "12:00"))
    solar_end_abs = _hhmm_to_min(settings_dict.get("solar_window_end", "13:00"))
    shift_start_abs = shift_start.hour * 60 + shift_start.minute
    solar_start_rel = solar_start_abs - shift_start_abs
    solar_end_rel = solar_end_abs - shift_start_abs

    schedule_data = ScheduleData(
        plan_id=plan.id,
        duration_minutes=duration_min,
        shift_start_iso=shift_start.isoformat(),
        sample_interval_min=SAMPLE_INTERVAL,
        mh_a_levels_kg=_sample(mh_a_raw),
        mh_b_levels_kg=_sample(mh_b_raw),
        mh_a_min_level_kg=float(MH_MIN_OPERATIONAL_LEVEL_KG.get("A", 200)),
        mh_b_min_level_kg=float(MH_MIN_OPERATIONAL_LEVEL_KG.get("B", 125)),
        if_kw=_sample(if_kw_raw),
        baseline_kw=_sample(baseline_raw),
        total_plant_kw=_sample(total_kw_raw),
        tou_effective_price=_sample(tou_raw),
        solar_window_start_min=solar_start_rel,
        solar_window_end_min=solar_end_rel,
    )

    return GaResult(
        schedule_items=schedule_items,
        metrics=metrics,
        schedule_data=schedule_data,
    )


# ---------------------------------------------------------------------------
# Fallback — evenly-spaced schedule when GA is unavailable
# ---------------------------------------------------------------------------

def _sequential_fallback(plan: "Plan") -> GaResult:
    """Simple 90-minute interval schedule, alternating furnaces A/B."""
    shift_start: datetime = plan.shift_start
    interval = timedelta(minutes=90)
    melt_duration = timedelta(minutes=88)  # default hot melt duration

    items = []
    for i in range(plan.target_batches):
        expected_start = shift_start + i * interval
        melt_finish_at = expected_start + melt_duration
        pour_at = melt_finish_at  # no hold in fallback
        items.append({
            "batch_id": i + 1,
            "expected_start": expected_start,
            "melt_finish_at": melt_finish_at,
            "pour_at": pour_at,
            "furnace": "A" if i % 2 == 0 else "B",
            "duration_min": 88,
            "power_kw": 475.0,
            "is_cold_start": False,
            "energy_kwh": 582.5,
        })

    metrics = ScheduleMetrics(
        poured_batches_count=plan.target_batches,
        missing_batches=0,
        total_if_kwh=582.5 * plan.target_batches,
        total_energy_cost_day=0.0,
        demand_charge_day_equiv=0.0,
        peak_kw=475.0,
        makespan_minutes=90.0 * plan.target_batches,
        mh_empty_minutes_a=0.0,
        mh_empty_minutes_b=0.0,
        solar_cost_saving=0.0,
        if_use_count_a=sum(1 for x in items if x["furnace"] == "A"),
        if_use_count_b=sum(1 for x in items if x["furnace"] == "B"),
    )

    return GaResult(schedule_items=items, metrics=metrics, schedule_data=None)
