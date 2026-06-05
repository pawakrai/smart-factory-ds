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

        # Force "start at shift_start" when operator turned off both cost flags.
        # The chromosome encodes `start_delay_min` (0-60 min) and `start_score_bias`
        # (−2..+2) which together control how quickly the simulator dispatches
        # the first IF. The optimal solution under cost-aware objectives usually
        # *delays* the start (so the first pour lands the moment MH levels have
        # drained enough to accept 600 kg — zero hold). With flags off we want
        # the opposite: dispatch at shift_start and tolerate ~28 min hold before
        # the first pour fits, so MH peaks return to MAX after refill.
        if (not bool(getattr(plan, "consider_tou_price", True))
                and not bool(getattr(plan, "consider_plant_load", True))):
            # CAP start_delay_min at 35 min instead of forcing 0. The GA's
            # chromosome can still encode a *small* delay so the first IF's
            # melt-finish aligns with the moment MH has 600 kg free — that
            # eliminates the ~28 min hold + reheat cost without pushing the
            # finish-time out (last-pour timing is unchanged). Forcing 0 wastes
            # one hold-cycle of reheat energy; capping lets the GA pick the
            # JIT-optimal value (≈ 28-30 min for typical rates).
            policy["start_delay_min"] = min(float(policy.get("start_delay_min", 0.0)), 35.0)
            # Force min_start_gap to the *steady-state* cycle time so each
            # subsequent IF starts exactly when its melt-finish will align
            # with the next pour-ready moment. Eliminates the 28-min hold per
            # cycle for charges #2-N (the chart's biggest hold contributor).
            # cycle = IF_BATCH_OUTPUT_KG / (rate_A + rate_B). Using app_v9
            # constants which were just set by _apply_settings_overrides.
            rate_total = (
                app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN["A"]
                + app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN["B"]
            )
            if rate_total > 1e-9:
                steady_cycle = app_v9.IF_BATCH_OUTPUT_KG / rate_total
                # Subtract 1 min for float safety — better slight overlap than gap.
                policy["min_start_gap_min"] = max(0.0, steady_cycle - 1.0)
            else:
                policy["min_start_gap_min"] = 0.0
            # start_score is computed every minute and must be ≥ 0 for dispatch
            # to fire. Push bias positive so an idle furnace dispatches once
            # start_delay_min has elapsed (even when both MHs are full and
            # depletion_urgency = 0).
            policy["start_score_bias"] = max(float(policy.get("start_score_bias", 0.0)), 1.0)
            policy["tou_weight"] = 0.0  # eliminate TOU penalty from start_score
            policy["peak_avoid_weight"] = 0.0  # eliminate peak penalty from start_score
            logger.info(
                "Flags-off policy override: start_delay_min=%.1f (cap 35), "
                "min_start_gap_min=0, start_score_bias=%s, tou=peak=0",
                policy["start_delay_min"], policy["start_score_bias"],
            )

        sim = app_v9.simulate_policy_day(policy)

    return _build_ga_result(plan, sim, settings_dict, app_v9)


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

    # Furnace enable flags from plan
    app_v9.USE_FURNACE_A = bool(getattr(plan, "if_a_enabled", True))
    app_v9.USE_FURNACE_B = bool(getattr(plan, "if_b_enabled", True))

    # First-charge preference: forces the GA dispatcher to start on A or B.
    # Only applies to the first batch; subsequent batches use scoring as before.
    preferred = getattr(plan, "preferred_start_furnace", "A")
    app_v9.PREFERRED_START_FURNACE = preferred if preferred in ("A", "B") else None

    # M&H consumption rates: plan-level values take precedence over settings
    mh_a_rate = getattr(plan, "mh_a_consumption_rate", None)
    mh_b_rate = getattr(plan, "mh_b_consumption_rate", None)
    if mh_a_rate is None:
        mh_a_rate = f("mh_a_consumption_rate_kg_per_min", 2.20)
    if mh_b_rate is None:
        mh_b_rate = f("mh_b_consumption_rate_kg_per_min", 2.30)

    # IF Furnace
    app_v9.IF_POWER_OPTIONS = [
        f("if_power_option_low_kw", 450),
        f("if_power_option_mid_kw", 475),
        f("if_power_option_high_kw", 500),
    ]
    # Update POWER_PROFILE to match new power options — keep the existing entries
    # (durations/energies are fixed physics, not user-configurable)
    app_v9.IF_BATCH_OUTPUT_KG = f("if_batch_output_kg", 600)
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
    # MH initial level: plan override > settings default
    mh_a_init = getattr(plan, "mh_a_initial_level_kg", None)
    mh_b_init = getattr(plan, "mh_b_initial_level_kg", None)
    app_v9.MH_INITIAL_LEVEL_KG = {
        "A": float(mh_a_init) if mh_a_init is not None else f("mh_a_initial_level_kg", 400),
        "B": float(mh_b_init) if mh_b_init is not None else f("mh_b_initial_level_kg", 230),
    }
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN = {
        "A": float(mh_a_rate),
        "B": float(mh_b_rate),
    }
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG = {
        "A": f("mh_a_min_operational_level_kg", 200),
        "B": f("mh_b_min_operational_level_kg", 125),
    }
    app_v9.MH_EMPTY_PENALTY_PER_MIN = f("mh_empty_penalty_per_min", 150)
    app_v9.MH_LOW_LEVEL_MINUTE_PENALTY = f("mh_low_level_minute_penalty", 40)
    app_v9.MH_LOW_LEVEL_PENALTY_RATE = f("mh_low_level_penalty_rate", 200)
    app_v9.LOW_LEVEL_NONLINEAR_FACTOR = f("mh_low_level_nonlinear_factor", 3.0)
    app_v9.MAX_EMPTY_MIN_ALLOW = f("mh_max_empty_min_allow", 120)
    app_v9.MAX_LOW_LEVEL_MIN_ALLOW = f("mh_max_low_level_min_allow", 240)

    # Objective weights for MH level components
    app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] = f("ga_obj_weight_empty_penalty", 1.10)
    app_v9.OBJ1_COMPONENT_WEIGHTS["low_level_min_penalty"] = f("ga_obj_weight_low_level_min", 0.80)
    app_v9.OBJ1_COMPONENT_WEIGHTS["low_level_shape_penalty"] = f("ga_obj_weight_low_level_shape", 0.90)

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

    # Reset service-mode tuning to module defaults so an earlier service+flags-off
    # run doesn't leak its boosted weights into the next plan.
    # (PREFERRED_MH_FURNACE_TO_FILL_FIRST is no longer read by the simulator —
    # the pour is split proportionally to free space — so we don't bother
    # resetting it. See _distribute_pour_proportional in src/app_v9.py.)
    if hasattr(app_v9, "SERVICE_W_MAKESPAN"):
        app_v9.SERVICE_W_MAKESPAN = 0.0
    if hasattr(app_v9, "SERVICE_W_HOLDING_MINUTES"):
        app_v9.SERVICE_W_HOLDING_MINUTES = 1e1
    if hasattr(app_v9, "SERVICE_W_REHEAT_KWH"):
        app_v9.SERVICE_W_REHEAT_KWH = 1e3

    # Per-plan cost-consideration flags
    if not bool(getattr(plan, "consider_tou_price", True)):
        # Flat off-peak price 24h: collapse onpeak to offpeak and zero the high-TOU penalty weight
        app_v9.TOU_ONPEAK_BAHT_PER_KWH_BASE = app_v9.TOU_OFFPEAK_BAHT_PER_KWH_BASE
        app_v9.TOU_ONPEAK_BAHT_PER_KWH = app_v9.TOU_OFFPEAK_BAHT_PER_KWH
        if hasattr(app_v9, "ENERGY_MODE_HIGH_TOU_GAP_WEIGHT"):
            app_v9.ENERGY_MODE_HIGH_TOU_GAP_WEIGHT = 0.0
    if not bool(getattr(plan, "consider_plant_load", True)):
        # Drop demand-charge term so GA stops avoiding peak demand
        app_v9.DEMAND_CHARGE_BAHT_PER_KW_MONTH = 0.0

    # Both-flags-off = "ignore cost, melt continuously without MH violations".
    # Cost penalties go away; *safety* penalties (MH low-level / empty) stay strong
    # — they're operator safety constraints, not cost trade-offs.
    both_flags_off = (
        not bool(getattr(plan, "consider_tou_price", True))
        and not bool(getattr(plan, "consider_plant_load", True))
    )
    if both_flags_off:
        # Unified "continuous melt" mode for BOTH energy and service opt_mode.
        # The operator's intent is the same regardless of nominal mode: ignore
        # cost optimisation, keep MH safe, finish as fast as the physics allow.
        # We differentiate only in how the makespan/hold REWARD is wired into
        # each mode's objective (energy → ENERGY_*_COST_PER_MIN; service →
        # SERVICE_W_*).

        # ── Safety: keep quadratic low-level penalty + boost OBJ1 weights ──
        # Boosted hard in Fix 12: operator reported that even with proportional
        # pour the GA tolerated MH-A dipping to ~Min. Tripling these weights
        # forces the optimiser to schedule pours BEFORE Min is reached.
        if hasattr(app_v9, "OBJ1_COMPONENT_WEIGHTS"):
            w = app_v9.OBJ1_COMPONENT_WEIGHTS
            w["empty_penalty"]           = max(w.get("empty_penalty", 1.10), 30.0)
            w["low_level_min_penalty"]   = max(w.get("low_level_min_penalty", 0.80), 30.0)
            w["low_level_shape_penalty"] = max(w.get("low_level_shape_penalty", 0.90), 15.0)
            # holding_penalty is an energy-mode hold-time term in OBJ1; we
            # drive holding via mode-specific terms below instead so this
            # doesn't double-count.
            w["holding_penalty"] = 0.0
        # Hard MH-min budget: 5 min cumulative below Min (default 240) — near
        # zero tolerance; any sustained dip blows the CV vector and the GA
        # picks a feasible neighbour instead.
        app_v9.MAX_LOW_LEVEL_MIN_ALLOW = min(
            getattr(app_v9, "MAX_LOW_LEVEL_MIN_ALLOW", 240.0), 5.0
        )
        # Hard empty budget: 15 min (default 120) — MH-empty is unsafe.
        if hasattr(app_v9, "MAX_EMPTY_MIN_ALLOW"):
            app_v9.MAX_EMPTY_MIN_ALLOW = min(app_v9.MAX_EMPTY_MIN_ALLOW, 15.0)

        # ── Continuous-melt geometry: zero start-gap + kill peak-risk gap ──
        if hasattr(app_v9, "POLICY_OVERRIDE_MIN_START_GAP_MAX"):
            app_v9.POLICY_OVERRIDE_MIN_START_GAP_MAX = 0.0
        # _dynamic_min_start_gap injects 50 min of gap whenever projected
        # baseline + IF kW exceeds CONTRACT_DEMAND_KW. With contract → ∞,
        # margin_ratio stays positive → peak_risk = 0 → no auto-gap.
        app_v9.CONTRACT_DEMAND_KW = 1e9

        # (Pour distribution is now proportional to per-furnace free space at
        # the simulator level — see _distribute_pour_proportional in app_v9.py.
        # No per-plan override needed; the algorithm self-balances.)

        # ── Remove price/peak-shifting incentives that fight continuous melt ──
        if hasattr(app_v9, "ENERGY_MODE_MIN_JIT_SLACK_MIN"):
            app_v9.ENERGY_MODE_MIN_JIT_SLACK_MIN = 0.0
        if hasattr(app_v9, "ENERGY_IDLE_GAP_SUPERLINEAR_COEFF"):
            app_v9.ENERGY_IDLE_GAP_SUPERLINEAR_COEFF = 0.0

        # ── Mode-specific makespan/hold REWARD wiring ──
        # The previous design *zeroed* these for energy mode under the
        # mistaken assumption that "ignore cost" = "ignore all penalties".
        # The result was a flat objective with no incentive to compress the
        # schedule → GA wandered (3-hour delays, MH-A empty 8+ hours).
        # Instead: KEEP and BOOST the throughput-related cost terms; only
        # the TOU/demand-related terms (already zeroed above) go away.
        # Fix 12 trade-off: operator wants "start at shift_start, tolerate some
        # hold, finish ASAP, keep MH high". The hold/makespan ratios below tip
        # the GA toward an early start with a short (~26 min) hold before the
        # first pour fits — instead of the old "delay start 85 min to skip
        # holding entirely" choice which left A.peak at 660 / makespan +90 min.
        if plan.opt_mode == "service":
            app_v9.SERVICE_W_MAKESPAN = 1000.0       # 20× original — finish-fast must dominate everything else
            # Bumped back up in Fix 12.1: hold visible on the chart was operator
            # feedback. With start_delay_min capped at 35 (above) the GA can
            # JIT-align melt-finish to pour-ready, so hold should be 0 at the
            # optimum — making hold expensive nudges the GA there.
            app_v9.SERVICE_W_HOLDING_MINUTES = 30.0
            app_v9.IF_HOLDING_PENALTY_PER_MIN = 0.0  # avoid double-count vs SERVICE_W_HOLDING_MINUTES
            # Service objective normally charges SERVICE_W_REHEAT_KWH (1e3) per
            # reheat-kWh, which dominates makespan and locks the GA into "late
            # start, zero reheat" choices. With flags off the operator does
            # not care about reheat energy cost — zero this so the GA only
            # cares about makespan + MH safety.
            if hasattr(app_v9, "SERVICE_W_REHEAT_KWH"):
                app_v9.SERVICE_W_REHEAT_KWH = 0.0
            if hasattr(app_v9, "ENERGY_MAKESPAN_COST_PER_MIN"):
                app_v9.ENERGY_MAKESPAN_COST_PER_MIN = 0.0
            if hasattr(app_v9, "ENERGY_IDLE_GAP_COST_PER_MIN"):
                app_v9.ENERGY_IDLE_GAP_COST_PER_MIN = 0.0
        else:
            # energy mode: drive continuous melt through energy-mode terms.
            # ENERGY_MAKESPAN_COST_PER_MIN boosted to 100 (default 0.6) so
            # makespan dominates; IF_HOLDING_PENALTY_PER_MIN dropped to 1 so
            # GA tolerates early-start hold without lockout.
            if hasattr(app_v9, "ENERGY_MAKESPAN_COST_PER_MIN"):
                app_v9.ENERGY_MAKESPAN_COST_PER_MIN = 100.0
            # Idle-gap penalty matches makespan weight to discourage gaps.
            if hasattr(app_v9, "ENERGY_IDLE_GAP_COST_PER_MIN"):
                app_v9.ENERGY_IDLE_GAP_COST_PER_MIN = 100.0
            # IF holding cost — bumped to 10 in Fix 12.1 so GA prefers JIT
            # dispatch (start late enough that melt-finish = pour-ready) over
            # "start at t=0 then hold 28 min" — same makespan, no reheat.
            app_v9.IF_HOLDING_PENALTY_PER_MIN = 10.0
            # Service-only globals stay at module defaults (reset at fn top).

    # Solar window (parse HH:MM strings)
    solar_start_str = settings_dict.get("solar_window_start", "12:00")
    solar_end_str = settings_dict.get("solar_window_end", "13:00")
    app_v9.SOLAR_START = _hhmm_to_min(solar_start_str)
    app_v9.SOLAR_END = _hhmm_to_min(solar_end_str)
    app_v9.SOLAR_EFFECTIVE_PRICE_FACTOR = f("solar_price_factor", 0.35)

    # Disable solar window discount when both flags are off — must come AFTER the
    # unconditional settings load above (which would otherwise overwrite this).
    if (not bool(getattr(plan, "consider_tou_price", True))
            and not bool(getattr(plan, "consider_plant_load", True))):
        app_v9.SOLAR_EFFECTIVE_PRICE_FACTOR = 1.0

    # Clear evaluation cache so new settings take effect
    app_v9._EVAL_CACHE.clear()


def _hhmm_to_min(hhmm: str) -> int:
    """Parse 'HH:MM' string to minutes from midnight."""
    try:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return 0


def _build_visual_if_kw(sim: dict, kw_max: float, app_v9) -> "np.ndarray":
    """Recompute IF kW series using build_visual_if_profile for realistic chart shape."""
    import numpy as np
    duration = len(sim.get("if_kw", []))
    if duration == 0:
        return np.zeros(1440, dtype=float)
    result = np.zeros(duration, dtype=float)
    for b in sim.get("schedule", []):
        try:
            start = int(b.get("start_min", 0))
            end = int(b.get("melt_finish_min", start))
            start = max(0, min(duration, start))
            end = max(start, min(duration, end))
            dur = end - start
            if dur <= 0:
                continue
            p_kw = float(b.get("power_kw", 0.0))
            target_kwh = float(b.get("energy_kwh_profile", p_kw * (dur / 60.0)))
            vis = np.asarray(
                app_v9.build_visual_if_profile(dur, target_kwh=target_kwh, kw_max=kw_max),
                dtype=float,
            )
            if len(vis) == dur:
                result[start:end] += vis
        except Exception:
            continue
    return np.clip(result, 0.0, None)


def _build_ga_result(plan: "Plan", sim: dict, settings_dict: dict[str, str], app_v9=None) -> GaResult:
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
    baseline_raw = sim.get("baseline_kw", np.zeros(1440))
    tou_raw = sim.get("tou_effective_price", np.zeros(1440))

    # Use visual IF profile for chart display (realistic ramp shape, not flat accounting kW)
    if app_v9 is not None:
        visual_kw_max = float(settings_dict.get("if_visual_kw_max", "450"))
        if_kw_raw = _build_visual_if_kw(sim, visual_kw_max, app_v9)
        total_kw_raw = np.asarray(baseline_raw, dtype=float) + if_kw_raw
    else:
        if_kw_raw = sim.get("if_kw", np.zeros(1440))
        total_kw_raw = sim.get("total_plant_kw", np.zeros(1440))

    tou_raw_price_raw = sim.get("tou_raw_price", np.zeros(1440))
    mh_a_raw = mh_levels.get("A", np.zeros(1440)) if mh_levels else np.zeros(1440)
    mh_b_raw = mh_levels.get("B", np.zeros(1440)) if mh_levels else np.zeros(1440)

    # Per-plan display overrides — flatten arrays for the charts when the
    # corresponding consideration was disabled at plan time.
    if not bool(getattr(plan, "consider_tou_price", True)):
        offpeak_eff = float(getattr(app_v9, "TOU_OFFPEAK_BAHT_PER_KWH", 0.0)) if app_v9 is not None else 0.0
        offpeak_base = float(getattr(app_v9, "TOU_OFFPEAK_BAHT_PER_KWH_BASE", offpeak_eff)) if app_v9 is not None else 0.0
        tou_raw = np.full_like(np.asarray(tou_raw, dtype=float), offpeak_eff)
        tou_raw_price_raw = np.full_like(np.asarray(tou_raw_price_raw, dtype=float), offpeak_base)
    if not bool(getattr(plan, "consider_plant_load", True)):
        baseline_arr = np.asarray(baseline_raw, dtype=float)
        baseline_mean = float(baseline_arr.mean()) if baseline_arr.size else 0.0
        baseline_raw = np.full_like(baseline_arr, baseline_mean)
        total_kw_raw = baseline_raw + np.asarray(if_kw_raw, dtype=float)

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
        mh_a_max_capacity_kg=float(app_v9.MH_MAX_CAPACITY_KG.get("A", 800)) if app_v9 is not None else 800.0,
        mh_b_max_capacity_kg=float(app_v9.MH_MAX_CAPACITY_KG.get("B", 1100)) if app_v9 is not None else 1100.0,
        if_kw=_sample(if_kw_raw),
        baseline_kw=_sample(baseline_raw),
        total_plant_kw=_sample(total_kw_raw),
        tou_effective_price=_sample(tou_raw),
        tou_raw_price=_sample(tou_raw_price_raw),
        contract_demand_kw=float(settings_dict.get("contract_demand_kw", "1600")),
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
