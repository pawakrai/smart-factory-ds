import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.callback import Callback
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2


# =============== CONFIG ==================
HOURS_A_DAY = 24 * 60
SIM_DURATION_MIN = HOURS_A_DAY
NUM_BATCHES = 12

USE_FURNACE_A = True
USE_FURNACE_B = True
ALLOW_PARALLEL_IF = False

IF_BATCH_OUTPUT_KG = 500.0
POST_POUR_DOWNTIME_MIN = 10
PREFERRED_MH_FURNACE_TO_FILL_FIRST = "B"

IF_POWER_OPTIONS = [450.0, 475.0, 500.0]
POWER_PROFILE = {
    450.0: {"duration_min_hot": 88.0, "energy_kwh_hot": 565.8},
    475.0: {"duration_min_hot": 86.0, "energy_kwh_hot": 582.5},
    500.0: {"duration_min_hot": 84.0, "energy_kwh_hot": 587.4},
}

# Slight per-furnace efficiency difference gives policy leverage when choosing A vs B.
IF_FURNACE_EFFICIENCY_FACTOR = {0: 0.99, 1: 1.03}

COLD_START_GAP_THRESHOLD_MIN = 180
COLD_START_EXTRA_DURATION_MIN = 8.0
COLD_START_EXTRA_ENERGY_KWH = 30.0

MH_MAX_CAPACITY_KG = {"A": 400.0, "B": 250.0}
MH_INITIAL_LEVEL_KG = {"A": 400.0, "B": 230.0}
MH_CONSUMPTION_RATE_KG_PER_MIN = {"A": 2.10, "B": 2.20}
MH_EMPTY_THRESHOLD_KG = 0.0
MH_MIN_OPERATIONAL_LEVEL_KG = {"A": 200.0, "B": 125.0}
MH_LOW_LEVEL_PENALTY_RATE = 200.0
LOW_LEVEL_NONLINEAR_FACTOR = 3.0

# TOU price ($/kWh equivalent unit)
TOU_PRICE_BY_HOUR = np.array(
    [
        1.8,  # 00:00
        1.8,  # 01:00
        1.8,  # 02:00
        1.8,  # 03:00
        1.8,  # 04:00
        1.8,  # 05:00
        1.9,  # 06:00
        2.0,  # 07:00
        2.1,  # 08:00
        3.2,  # 09:00
        3.3,  # 10:00
        3.4,  # 11:00
        3.5,  # 12:00
        3.6,  # 13:00
        3.7,  # 14:00
        3.8,  # 15:00
        3.9,  # 16:00
        4.1,  # 17:00
        4.3,  # 18:00
        4.2,  # 19:00
        4.0,  # 20:00
        3.6,  # 21:00
        2.0,  # 22:00
        1.9,  # 23:00
    ],
    dtype=float,
)

CONTRACT_DEMAND_KW = 1200.0
DEMAND_CHARGE_RATE_PER_KW = 1800.0
DEMAND_SOFT_ZONE_RATIO = 0.90

# Service penalties (raw units before normalization). Tuned so raw components are
# closer in order of magnitude and do not completely dominate energy cost.
IF_HOLDING_PENALTY_PER_MIN = 8.0
MH_EMPTY_PENALTY_PER_MIN = 150.0
MH_LOW_LEVEL_MINUTE_PENALTY = 40.0
OVERFLOW_PENALTY_PER_KG = 5000.0
UNPOURED_BATCH_PENALTY = 250000.0

# NEW/CHANGED: Prep-time model for ingot loading.
# Alternating furnaces allows prep for the other furnace during melting.
PREP_LOAD_TIME_MIN = 15
ENABLE_PREP_MODEL = True
ENFORCE_ALTERNATION_FLAG = False  # If True, enforce A/B alternation in force mode when feasible.
SWITCH_PENALTY_PER_REPEAT = 8.0
PREP_WAIT_COST_PER_MIN = 25.0

SHIFT_START = 8 * 60
FURNACE_COLORS = {"A": "blue", "B": "green"}
MH_FURNACE_COLORS = {"A": "red", "B": "orange"}
furnace_y = {0: 10, 1: 25}
height = 8

DEBUG = True
_EVAL_CACHE = {}
DETERMINISTIC_SIMULATION = True

# Objective aggregation mode:
# - "raw": obj1 = sum(raw components)
# - "normalized_weighted": obj1 = scale * sum(weight_i * component_i / ref_i)
OBJ1_AGGREGATION_MODE = "normalized_weighted"
OBJ1_NORMALIZATION_SCALE = 100000.0
OBJ1_COMPONENT_REFS = {
    "energy_cost": 30000.0,
    "demand_penalty": 150000.0,
    "holding_penalty": 50000.0,
    "empty_penalty": 200000.0,
    "low_level_min_penalty": 80000.0,
    "low_level_shape_penalty": 250000.0,
    "overflow_penalty": 120000.0,
    "unpoured_penalty": 300000.0,
    "prep_wait_penalty": 50000.0,  # NEW/CHANGED
    "switch_penalty": 30000.0,  # NEW/CHANGED
}
OBJ1_COMPONENT_WEIGHTS = {
    "energy_cost": 1.00,
    "demand_penalty": 0.90,
    "holding_penalty": 0.50,
    "empty_penalty": 1.10,
    "low_level_min_penalty": 0.80,
    "low_level_shape_penalty": 0.90,
    "overflow_penalty": 1.20,
    "unpoured_penalty": 1.30,
    "prep_wait_penalty": 0.50,  # NEW/CHANGED
    "switch_penalty": 0.40,  # NEW/CHANGED
}

# Optional constraint-handling mode to prevent "all-penalty" domination.
USE_CONSTRAINT_HANDLING = False
OBJ1_EXCLUDE_SERVICE_PENALTIES_WHEN_CONSTRAINED = (
    True  # CHANGED (Step D): keep Obj1 focused on cost when using constraints.
)
MAX_OVERFLOW_KG_ALLOW = 1e-6
MAX_EMPTY_MIN_ALLOW = 120.0
MAX_LOW_LEVEL_MIN_ALLOW = 240.0
MAX_SHORTFALL_KG_ALLOW = 1e-6  # CHANGED (Step B): allow near-zero daily shortfall only.
HARD_FORBID_OVERFLOW = True
SIMPLE_POLICY_MODE = True  # NEW/CHANGED: use reduced 10-variable policy for smoother Pareto fronts.

# 3-objective model:
# Obj0: weighted violation severity (dimensionless score)
# Obj1: real operating cost proxy (energy + demand + non-hard service penalties)
# Obj2: makespan (min)
OBJ0_WEIGHTS = {
    "overflow_kg": 1000.0,
    "shortfall_kg": 120.0,  # CHANGED (Step B): main production-feasibility violation weight.
    "empty_min": 3.0,
    "low_level_min": 1.0,
    "parallel_peak_min": 4.0,
}
MAX_PARALLEL_PEAK_MIN_ALLOW = 0.0

# Optional two-stage optimization:
# Stage-1 focuses on reducing violations, stage-2 optimizes objective.
USE_TWO_STAGE_OPTIMIZATION = False
STAGE1_GENS = 30

# Quantization for policy-aware duplicate elimination / cache keys.
# This follows decision granularity in the simulator to avoid wasting evaluations
# on policies that decode to effectively the same control behavior.
POLICY_KEY_STEPS = {
    # NEW/CHANGED: steps for SIMPLE policy mode keys.
    "trig_a_frac": 0.01,
    "trig_b_frac": 0.01,
    "lookahead_min": 1.0,
    "demand_headroom_kw": 5.0,
    "tou_weight": 0.05,
    "peak_avoid_weight": 0.01,
    "min_start_gap_min": 1.0,
    "alternation_bias": 0.02,
    "force_urgency_threshold": 0.01,
    "power_aggressiveness": 0.01,
    # Legacy/full-mode keys retained for compatibility.
    "trigger_a_frac": 0.01,
    "trigger_b_frac": 0.01,
    "level_crit_a_frac": 0.01,
    "level_crit_b_frac": 0.01,
    "level_mid_frac": 0.01,
    "demand_headroom_kw": 5.0,
    "peak_avoid_weight": 0.01,
    "furnace_bias": 0.02,
    "wait_vs_rush": 0.01,
    "start_score_bias": 0.05,
    "start_w_depletion": 0.05,
    "start_w_queue": 0.05,
    "start_w_peak": 0.05,
    "min_start_gap_min": 1.0,
    "gap_peak_coeff": 0.05,
    "parallel_open_margin_kw": 5.0,
    "parallel_score_bias": 0.05,
    "coldstart_weight": 0.05,
    "tou_weight": 0.05,
    "balance_weight": 0.05,
}

EARLY_STOP_PATIENCE_GENS = 20
EARLY_STOP_DELTA_OBJ0 = 1.0
EARLY_STOP_DELTA_OBJ1 = 50.0
EARLY_STOP_DELTA_OBJ2 = 0.5


def _build_baseline_load_kw(duration_min=SIM_DURATION_MIN):
    t = np.arange(duration_min, dtype=float)
    baseline = 310.0 + 70.0 * np.sin(2.0 * np.pi * (t / 1440.0 - 0.15))
    baseline += 45.0 * np.exp(-((t - 13 * 60.0) ** 2) / (2.0 * (140.0**2)))
    baseline += 30.0 * np.exp(-((t - 20 * 60.0) ** 2) / (2.0 * (120.0**2)))
    return np.clip(baseline, 220.0, None)


def _hour_price(minute_idx):
    hour = int(np.clip(minute_idx // 60, 0, 23))
    return TOU_PRICE_BY_HOUR[hour]


def _compute_batch_profile(power_kw, is_cold_start, if_furnace):
    profile = POWER_PROFILE.get(power_kw, POWER_PROFILE[450.0])
    duration_min = profile["duration_min_hot"]
    energy_kwh = profile["energy_kwh_hot"]
    if is_cold_start:
        duration_min += COLD_START_EXTRA_DURATION_MIN
        energy_kwh += COLD_START_EXTRA_ENERGY_KWH

    eff = IF_FURNACE_EFFICIENCY_FACTOR.get(if_furnace, 1.0)
    energy_kwh *= eff
    return {
        "power_kw": power_kw,
        "duration_min": max(1.0, duration_min),
        "energy_kwh": energy_kwh,
        "is_cold_start": is_cold_start,
    }


def _decode_policy_vector(x):
    # NEW/CHANGED: simple interpretable policy mode (10 vars) for smoother trade-offs.
    if SIMPLE_POLICY_MODE:
        x = np.asarray(x, dtype=float)
        lookahead_min = int(np.clip(round(np.clip(x[2], 0.0, 1.0) * 180.0), 0, 180))
        return {
            "trig_a_frac": float(np.clip(x[0], 0.0, 1.0)),
            "trig_b_frac": float(np.clip(x[1], 0.0, 1.0)),
            "L_trigger_A": float(np.clip(x[0], 0.0, 1.0) * MH_MAX_CAPACITY_KG["A"]),
            "L_trigger_B": float(np.clip(x[1], 0.0, 1.0) * MH_MAX_CAPACITY_KG["B"]),
            "lookahead_min": lookahead_min,
            "demand_headroom_kw": float(np.clip(x[3], 0.0, 1.0) * 500.0),
            "tou_weight": float(np.clip(x[4], 0.0, 2.0)),
            "peak_avoid_weight": float(np.clip(x[5], 0.0, 1.0)),
            "min_start_gap_min": float(np.clip(x[6], 0.0, 1.0) * 120.0),
            "alternation_bias": float(np.clip(x[7], -1.0, 1.0)),
            "force_urgency_threshold": float(np.clip(x[8], 0.70, 0.98)),
            "power_aggressiveness": float(np.clip(x[9], 0.0, 1.0)),
            # Keep these for compatibility with existing helper logic.
            "wait_vs_rush": float(np.clip(x[9], 0.0, 1.0)),
            "start_score_bias": 0.0,
            "start_w_depletion": 2.0,
            "start_w_queue": 1.2,
            "start_w_peak": 1.2,
            "gap_peak_coeff": 1.2,
            "parallel_open_margin_kw": 120.0,
            "parallel_score_bias": 0.0,
            "coldstart_weight": 0.6,
            "balance_weight": 0.4,
            "furnace_bias": 0.0,
            "level_crit_A": MH_MIN_OPERATIONAL_LEVEL_KG["A"],
            "level_crit_B": MH_MIN_OPERATIONAL_LEVEL_KG["B"],
            "level_mid_frac": 0.5,
        }

    # x layout:
    # [0..1] normalized controls to reduce hard clipping and preserve sensitivity.
    # [trigA_frac, trigB_frac, lookahead_norm, critA_frac, critB_frac, mid_frac,
    #  headroom_norm, peak_avoid, furnace_bias, wait_vs_rush,
    #  start_bias, w_depletion, w_queue, w_peak, min_gap_norm, gap_peak_coeff,
    #  parallel_margin_norm, parallel_bias, coldstart_weight, tou_weight, balance_weight]
    x = np.asarray(x, dtype=float)
    return {
        "trigger_a_frac": float(np.clip(x[0], 0.0, 1.0)),
        "trigger_b_frac": float(np.clip(x[1], 0.0, 1.0)),
        "L_trigger_A": float(np.clip(x[0], 0.0, 1.0) * MH_MAX_CAPACITY_KG["A"]),
        "L_trigger_B": float(np.clip(x[1], 0.0, 1.0) * MH_MAX_CAPACITY_KG["B"]),
        "lookahead_min": int(np.clip(round(np.clip(x[2], 0.0, 1.0) * 180.0), 0, 180)),
        "level_crit_a_frac": float(np.clip(x[3], 0.0, 1.0)),
        "level_crit_b_frac": float(np.clip(x[4], 0.0, 1.0)),
        "level_crit_A": float(
            MH_MIN_OPERATIONAL_LEVEL_KG["A"]
            + np.clip(x[3], 0.0, 1.0)
            * (MH_MAX_CAPACITY_KG["A"] - MH_MIN_OPERATIONAL_LEVEL_KG["A"])
        ),
        "level_crit_B": float(
            MH_MIN_OPERATIONAL_LEVEL_KG["B"]
            + np.clip(x[4], 0.0, 1.0)
            * (MH_MAX_CAPACITY_KG["B"] - MH_MIN_OPERATIONAL_LEVEL_KG["B"])
        ),
        "level_mid_frac": float(np.clip(x[5], 0.0, 1.0)),
        "demand_headroom_kw": float(np.clip(x[6], 0.0, 1.0) * 500.0),
        "peak_avoid_weight": float(np.clip(x[7], 0.0, 1.0)),
        "furnace_bias": float(np.clip(x[8], -1.0, 1.0)),
        "wait_vs_rush": float(np.clip(x[9], 0.0, 1.0)),
        "start_score_bias": float(np.clip(x[10], -2.0, 2.0)),
        "start_w_depletion": float(np.clip(x[11], 0.0, 4.0)),
        "start_w_queue": float(np.clip(x[12], 0.0, 4.0)),
        "start_w_peak": float(np.clip(x[13], 0.0, 4.0)),
        "min_start_gap_min": float(np.clip(x[14], 0.0, 1.0) * 120.0),
        "gap_peak_coeff": float(np.clip(x[15], 0.0, 3.0)),
        "parallel_open_margin_kw": float(np.clip(x[16], 0.0, 1.0) * 500.0),
        "parallel_score_bias": float(np.clip(x[17], -2.0, 2.0)),
        "coldstart_weight": float(np.clip(x[18], 0.0, 2.0)),
        "tou_weight": float(np.clip(x[19], 0.0, 2.0)),
        "balance_weight": float(np.clip(x[20], 0.0, 2.0)),
    }


def _quantize_to_step(value, step):
    if step <= 0:
        return float(value)
    return float(np.round(value / step) * step)


def _policy_cache_key(x):
    p = _decode_policy_vector(np.asarray(x, dtype=float))
    if SIMPLE_POLICY_MODE:
        return (
            _quantize_to_step(p["trig_a_frac"], POLICY_KEY_STEPS["trig_a_frac"]),
            _quantize_to_step(p["trig_b_frac"], POLICY_KEY_STEPS["trig_b_frac"]),
            int(p["lookahead_min"]),
            _quantize_to_step(
                p["demand_headroom_kw"], POLICY_KEY_STEPS["demand_headroom_kw"]
            ),
            _quantize_to_step(p["tou_weight"], POLICY_KEY_STEPS["tou_weight"]),
            _quantize_to_step(
                p["peak_avoid_weight"], POLICY_KEY_STEPS["peak_avoid_weight"]
            ),
            _quantize_to_step(
                p["min_start_gap_min"], POLICY_KEY_STEPS["min_start_gap_min"]
            ),
            _quantize_to_step(
                p["alternation_bias"], POLICY_KEY_STEPS["alternation_bias"]
            ),
            _quantize_to_step(
                p["force_urgency_threshold"],
                POLICY_KEY_STEPS["force_urgency_threshold"],
            ),
            _quantize_to_step(
                p["power_aggressiveness"], POLICY_KEY_STEPS["power_aggressiveness"]
            ),
        )
    return (
        _quantize_to_step(p["trigger_a_frac"], POLICY_KEY_STEPS["trigger_a_frac"]),
        _quantize_to_step(p["trigger_b_frac"], POLICY_KEY_STEPS["trigger_b_frac"]),
        int(p["lookahead_min"]),
        _quantize_to_step(
            p["level_crit_a_frac"], POLICY_KEY_STEPS["level_crit_a_frac"]
        ),
        _quantize_to_step(
            p["level_crit_b_frac"], POLICY_KEY_STEPS["level_crit_b_frac"]
        ),
        _quantize_to_step(p["level_mid_frac"], POLICY_KEY_STEPS["level_mid_frac"]),
        _quantize_to_step(
            p["demand_headroom_kw"], POLICY_KEY_STEPS["demand_headroom_kw"]
        ),
        _quantize_to_step(
            p["peak_avoid_weight"], POLICY_KEY_STEPS["peak_avoid_weight"]
        ),
        _quantize_to_step(p["furnace_bias"], POLICY_KEY_STEPS["furnace_bias"]),
        _quantize_to_step(p["wait_vs_rush"], POLICY_KEY_STEPS["wait_vs_rush"]),
        _quantize_to_step(p["start_score_bias"], POLICY_KEY_STEPS["start_score_bias"]),
        _quantize_to_step(
            p["start_w_depletion"], POLICY_KEY_STEPS["start_w_depletion"]
        ),
        _quantize_to_step(p["start_w_queue"], POLICY_KEY_STEPS["start_w_queue"]),
        _quantize_to_step(p["start_w_peak"], POLICY_KEY_STEPS["start_w_peak"]),
        _quantize_to_step(
            p["min_start_gap_min"], POLICY_KEY_STEPS["min_start_gap_min"]
        ),
        _quantize_to_step(p["gap_peak_coeff"], POLICY_KEY_STEPS["gap_peak_coeff"]),
        _quantize_to_step(
            p["parallel_open_margin_kw"], POLICY_KEY_STEPS["parallel_open_margin_kw"]
        ),
        _quantize_to_step(
            p["parallel_score_bias"], POLICY_KEY_STEPS["parallel_score_bias"]
        ),
        _quantize_to_step(p["coldstart_weight"], POLICY_KEY_STEPS["coldstart_weight"]),
        _quantize_to_step(p["tou_weight"], POLICY_KEY_STEPS["tou_weight"]),
        _quantize_to_step(p["balance_weight"], POLICY_KEY_STEPS["balance_weight"]),
    )


def _compute_obj1_components(m):
    empty_minutes_total = m["mh_empty_minutes"]["A"] + m["mh_empty_minutes"]["B"]
    low_minutes_total = m["mh_low_level_minutes"]["A"] + m["mh_low_level_minutes"]["B"]
    comp = {
        "energy_cost": float(m["total_energy_cost"]),
        "demand_penalty": float(m["demand_penalty"]),
        "holding_penalty": float(
            m["holding_minutes_total"] * IF_HOLDING_PENALTY_PER_MIN
        ),
        "empty_penalty": float(empty_minutes_total * MH_EMPTY_PENALTY_PER_MIN),
        "low_level_min_penalty": float(low_minutes_total * MH_LOW_LEVEL_MINUTE_PENALTY),
        "low_level_shape_penalty": float(m["mh_low_level_penalty"]),
        "overflow_penalty": float(m["overflow_kg_total"] * OVERFLOW_PENALTY_PER_KG),
        "unpoured_penalty": float(m["unpoured_batches_count"] * UNPOURED_BATCH_PENALTY),
        "prep_wait_penalty": float(
            m.get("prep_wait_minutes", 0.0) * PREP_WAIT_COST_PER_MIN
        ),  # NEW/CHANGED
        "switch_penalty": float(
            m.get("non_alternation_count", 0.0) * SWITCH_PENALTY_PER_REPEAT
        ),  # NEW/CHANGED
    }
    return comp


def _aggregate_obj1(components):
    if OBJ1_AGGREGATION_MODE == "raw":
        return float(sum(components.values()))

    # normalized_weighted
    weighted_sum = 0.0
    for k, v in components.items():
        ref = max(1e-9, float(OBJ1_COMPONENT_REFS.get(k, 1.0)))
        w = float(OBJ1_COMPONENT_WEIGHTS.get(k, 1.0))
        weighted_sum += w * (float(v) / ref)
    return float(OBJ1_NORMALIZATION_SCALE * weighted_sum)


def _violation_vector(m):
    empty_total = float(m["mh_empty_minutes"]["A"] + m["mh_empty_minutes"]["B"])
    low_total = float(m["mh_low_level_minutes"]["A"] + m["mh_low_level_minutes"]["B"])
    overflow_total = float(m["overflow_kg_total"])
    shortfall_total = float(m.get("shortfall_kg", 0.0))  # CHANGED (Step B)
    parallel_peak_total = float(m.get("parallel_peak_minutes", 0.0))  # CHANGED (Step E)
    # Guard against NaN/Inf/negative in any upstream metric.
    if not np.isfinite(empty_total) or empty_total < 0.0:
        empty_total = 0.0
    if not np.isfinite(low_total) or low_total < 0.0:
        low_total = 0.0
    if not np.isfinite(overflow_total) or overflow_total < 0.0:
        overflow_total = 0.0
    if not np.isfinite(shortfall_total) or shortfall_total < 0.0:  # CHANGED (Step B)
        shortfall_total = 0.0
    if (
        not np.isfinite(parallel_peak_total) or parallel_peak_total < 0.0
    ):  # CHANGED (Step E)
        parallel_peak_total = 0.0
    return np.array(
        [
            overflow_total - MAX_OVERFLOW_KG_ALLOW,
            shortfall_total - MAX_SHORTFALL_KG_ALLOW,  # CHANGED (Step B)
            empty_total - MAX_EMPTY_MIN_ALLOW,
            low_total - MAX_LOW_LEVEL_MIN_ALLOW,
            parallel_peak_total - MAX_PARALLEL_PEAK_MIN_ALLOW,  # CHANGED (Step E)
        ],
        dtype=float,
    )


class PolicyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        xa = a.get("X") if hasattr(a, "get") else a.X
        xb = b.get("X") if hasattr(b, "get") else b.X
        return _policy_cache_key(xa) == _policy_cache_key(xb)


class StagnationEarlyStopCallback(Callback):
    def __init__(self, patience_gens, delta_obj0=0.0, delta_obj1=0.0, delta_obj2=0.0):
        super().__init__()
        self.patience_gens = int(patience_gens)
        self.delta_obj0 = float(delta_obj0)
        self.delta_obj1 = float(delta_obj1)
        self.delta_obj2 = float(delta_obj2)
        self.best_obj0 = None
        self.best_obj1 = None
        self.best_obj2 = None
        self.stagnant_gens = 0

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        cur_obj0 = float(np.min(F[:, 0]))
        cur_obj1 = float(np.min(F[:, 1]))
        cur_obj2 = float(np.min(F[:, 2]))

        if self.best_obj0 is None:
            self.best_obj0 = cur_obj0
            self.best_obj1 = cur_obj1
            self.best_obj2 = cur_obj2
            self.stagnant_gens = 0
            return

        improved_obj0 = cur_obj0 < (self.best_obj0 - self.delta_obj0)
        improved_obj1 = cur_obj1 < (self.best_obj1 - self.delta_obj1)
        improved_obj2 = cur_obj2 < (self.best_obj2 - self.delta_obj2)

        if improved_obj0 or improved_obj1 or improved_obj2:
            self.best_obj0 = min(self.best_obj0, cur_obj0)
            self.best_obj1 = min(self.best_obj1, cur_obj1)
            self.best_obj2 = min(self.best_obj2, cur_obj2)
            self.stagnant_gens = 0
        else:
            self.stagnant_gens += 1

        if self.stagnant_gens >= self.patience_gens:
            print(
                f"Early stop: no objective improvement in {self.stagnant_gens} generations."
            )
            algorithm.termination.force_termination = True


def _build_policy_state(
    policy, mh_levels, baseline_kw_t, if_kw_now, queue_len, price_t
):
    eta_a = mh_levels["A"] / max(MH_CONSUMPTION_RATE_KG_PER_MIN["A"], 1e-9)
    eta_b = mh_levels["B"] / max(MH_CONSUMPTION_RATE_KG_PER_MIN["B"], 1e-9)
    eta_min = min(eta_a, eta_b)
    lookahead_ref = max(10.0, float(policy["lookahead_min"]))
    depletion_urgency = float(np.clip(1.0 - eta_min / lookahead_ref, 0.0, 1.0))

    avg_level_frac = 0.5 * (
        mh_levels["A"] / max(MH_MAX_CAPACITY_KG["A"], 1e-9)
        + mh_levels["B"] / max(MH_MAX_CAPACITY_KG["B"], 1e-9)
    )
    trigger_hit = float(
        (mh_levels["A"] <= policy["L_trigger_A"])
        or (mh_levels["B"] <= policy["L_trigger_B"])
    )
    queue_pressure = float(min(2.0, queue_len / 2.0))
    projected_no_new = baseline_kw_t + if_kw_now
    margin_kw = CONTRACT_DEMAND_KW - projected_no_new
    margin_ratio = float(np.clip(margin_kw / max(CONTRACT_DEMAND_KW, 1e-9), -1.0, 1.0))
    price_norm = float(
        np.clip(
            (price_t - np.min(TOU_PRICE_BY_HOUR)) / (np.ptp(TOU_PRICE_BY_HOUR) + 1e-9),
            0.0,
            1.0,
        )
    )
    return {
        "eta_min": eta_min,
        "depletion_urgency": depletion_urgency,
        "avg_level_frac": avg_level_frac,
        "trigger_hit": trigger_hit,
        "queue_pressure": queue_pressure,
        "margin_kw": margin_kw,
        "margin_ratio": margin_ratio,
        "price_norm": price_norm,
    }


def _start_score(policy, state):
    if SIMPLE_POLICY_MODE:
        # NEW/CHANGED: simpler and smoother score for NSGA-II search.
        score = 0.0
        score += 2.2 * state["depletion_urgency"]
        score += 1.0 * state["trigger_hit"]
        score += 0.7 * state["queue_pressure"]
        score -= policy["peak_avoid_weight"] * 2.0 * max(0.0, -state["margin_ratio"])
        score -= policy["tou_weight"] * state["price_norm"]
        return float(score)

    shortage_term = (1.0 - state["avg_level_frac"]) * policy["wait_vs_rush"]
    score = policy["start_score_bias"]
    score += policy["start_w_depletion"] * state["depletion_urgency"]
    score += policy["start_w_queue"] * state["queue_pressure"]
    score += 0.8 * state["trigger_hit"]
    score += 0.7 * shortage_term
    score -= policy["start_w_peak"] * max(0.0, -state["margin_ratio"])
    score -= policy["tou_weight"] * state["price_norm"] * (1.0 - policy["wait_vs_rush"])
    return float(score)


def _dynamic_min_start_gap(policy, state):
    peak_risk = max(0.0, -state["margin_ratio"])
    if SIMPLE_POLICY_MODE:
        return float(policy["min_start_gap_min"] + 50.0 * peak_risk)
    return float(
        policy["min_start_gap_min"] + policy["gap_peak_coeff"] * peak_risk * 60.0
    )


def _parallel_allowed(policy, state, start_score):
    return bool(
        ALLOW_PARALLEL_IF
        and state["margin_kw"] >= policy["parallel_open_margin_kw"]
        and start_score >= policy["parallel_score_bias"]
    )


def _select_if_power(
    policy, mh_levels, baseline_kw_t, if_kw_now, minute_idx, force_mode=False
):
    level_min = min(mh_levels["A"], mh_levels["B"])
    crit_level = min(policy["level_crit_A"], policy["level_crit_B"])
    top_level = min(MH_MAX_CAPACITY_KG["A"], MH_MAX_CAPACITY_KG["B"])
    mid_level = crit_level + (top_level - crit_level) * policy["level_mid_frac"]

    # urgency from M&H level
    if level_min <= crit_level:
        urgency_pref = 2
    elif level_min <= mid_level:
        urgency_pref = 1
    else:
        urgency_pref = 0

    # evaluate options with peak + TOU aware score
    best_p = 450.0
    best_score = float("inf")
    price_t = _hour_price(minute_idx)
    soft_guard = CONTRACT_DEMAND_KW - policy["demand_headroom_kw"]
    wait_weight = 1.0 - policy["wait_vs_rush"]
    peak_weight = policy["peak_avoid_weight"]
    if force_mode:
        # NEW/CHANGED: when forcing feasibility, de-emphasize TOU/peak and push power.
        wait_weight *= 0.25
        peak_weight *= 0.35
        urgency_pref = min(2, urgency_pref + int(policy.get("power_aggressiveness", 0.5) >= 0.4))

    for idx, p in enumerate(IF_POWER_OPTIONS):
        # Encourage high power when urgent, low power when not urgent.
        urgency_score = abs(idx - urgency_pref) * (
            120.0 + 220.0 * policy["wait_vs_rush"]
        )
        projected_kw = baseline_kw_t + if_kw_now + p
        soft_peak = max(0.0, projected_kw - soft_guard)
        peak_score = peak_weight * (soft_peak**2) / 50.0
        tou_score = wait_weight * policy["tou_weight"] * price_t * (p / 60.0)
        score = urgency_score + peak_score + tou_score
        if score < best_score:
            best_score = score
            best_p = p
    return best_p


def _select_if_furnace(
    policy,
    idle_furnaces,
    current_level,
    t,
    last_if_release_time,
    furnace_use_count,
    last_started_furnace=None,
):
    if len(idle_furnaces) <= 1:
        return idle_furnaces[0]

    best_f = idle_furnaces[0]
    best_score = float("inf")
    for f in idle_furnaces:
        # map furnace to associated holding-furnace risk proxy
        assoc = "A" if f == 0 else "B"
        level = current_level[assoc]
        deficit = max(0.0, MH_MIN_OPERATIONAL_LEVEL_KG[assoc] - level)
        low_risk = deficit / max(MH_MIN_OPERATIONAL_LEVEL_KG[assoc], 1e-9)

        last_release = last_if_release_time.get(f)
        cooldown_gap = 0.0 if last_release is None else max(0.0, t - last_release)
        setup_risk = 0.0 if cooldown_gap < COLD_START_GAP_THRESHOLD_MIN else 1.0

        # furnace bias: +1 prefers B (f=1), -1 prefers A (f=0)
        bias_term = -policy["furnace_bias"] if f == 1 else policy["furnace_bias"]
        count_a = furnace_use_count.get(0, 0)
        count_b = furnace_use_count.get(1, 0)
        imbalance = (count_b - count_a) if f == 1 else (count_a - count_b)
        balance_term = (
            policy["balance_weight"] * imbalance / max(1, count_a + count_b + 1)
        )

        score = (
            2.8 * low_risk
            + policy["coldstart_weight"] * setup_risk
            + 0.8 * bias_term
            + balance_term
        )
        if last_started_furnace is not None:
            # NEW/CHANGED: alternation bias -> positive prefers switching furnaces.
            alt_bias = float(policy.get("alternation_bias", 0.0))
            if f == last_started_furnace:
                score += 0.8 * alt_bias
            else:
                score -= 0.8 * alt_bias
        if score < best_score:
            best_score = score
            best_f = f
    return best_f


def _total_idle_minutes_from_intervals(intervals):
    if not intervals:
        return 0.0
    items = sorted(intervals, key=lambda w: w[0])
    merged = [[items[0][0], items[0][1]]]
    for s, e in items[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    idle = 0.0
    for i in range(1, len(merged)):
        idle += max(0.0, merged[i][0] - merged[i - 1][1])
    return idle


def quick_capacity_sanity_check():
    """Coarse feasibility check before running GA/NSGA-II."""
    daily_demand_kg = (
        MH_CONSUMPTION_RATE_KG_PER_MIN["A"] + MH_CONSUMPTION_RATE_KG_PER_MIN["B"]
    ) * SIM_DURATION_MIN
    initial_inventory_kg = MH_INITIAL_LEVEL_KG["A"] + MH_INITIAL_LEVEL_KG["B"]
    required_from_if_kg = max(0.0, daily_demand_kg - initial_inventory_kg)

    min_melt_duration = min(v["duration_min_hot"] for v in POWER_PROFILE.values())
    available_if = int(USE_FURNACE_A) + int(USE_FURNACE_B)
    parallel_if_slots = available_if if ALLOW_PARALLEL_IF else min(1, available_if)

    batches_time_limit = int(
        (SIM_DURATION_MIN / max(min_melt_duration, 1e-9)) * parallel_if_slots
    )
    supply_time_limit_kg = batches_time_limit * IF_BATCH_OUTPUT_KG

    # Receiver-side bottleneck: need 500 kg free capacity before each pour.
    total_consume_rate = (
        MH_CONSUMPTION_RATE_KG_PER_MIN["A"] + MH_CONSUMPTION_RATE_KG_PER_MIN["B"]
    )
    minutes_to_free_one_batch = IF_BATCH_OUTPUT_KG / max(total_consume_rate, 1e-9)
    batches_receive_limit = int(SIM_DURATION_MIN / max(minutes_to_free_one_batch, 1e-9))
    supply_receive_limit_kg = batches_receive_limit * IF_BATCH_OUTPUT_KG

    max_batches_current_config = int(NUM_BATCHES)
    supply_config_cap_kg = max_batches_current_config * IF_BATCH_OUTPUT_KG
    effective_max_batches = min(
        max_batches_current_config, batches_time_limit, batches_receive_limit
    )
    effective_max_supply_kg = effective_max_batches * IF_BATCH_OUTPUT_KG
    effective_total_available_kg = initial_inventory_kg + effective_max_supply_kg
    daily_shortfall_kg = max(0.0, daily_demand_kg - effective_total_available_kg)

    min_batches_required = int(np.ceil(required_from_if_kg / IF_BATCH_OUTPUT_KG))
    overtime_needed_min = max(
        0.0, min_batches_required * min_melt_duration - SIM_DURATION_MIN
    )

    report = {
        "daily_demand_kg": float(daily_demand_kg),
        "initial_inventory_kg": float(initial_inventory_kg),
        "required_from_if_kg": float(required_from_if_kg),
        "config_num_batches": int(max_batches_current_config),
        "config_supply_cap_kg": float(supply_config_cap_kg),
        "time_limit_batches": int(batches_time_limit),
        "receive_limit_batches": int(batches_receive_limit),
        "effective_max_batches": int(effective_max_batches),
        "effective_max_supply_kg": float(effective_max_supply_kg),
        "effective_total_available_kg": float(effective_total_available_kg),
        "daily_shortfall_kg": float(daily_shortfall_kg),
        "min_batches_required": int(min_batches_required),
        "min_overtime_needed_min": float(overtime_needed_min),
        "is_feasible_coarse": bool(daily_shortfall_kg <= 1e-9),
    }
    return report


def print_capacity_sanity_report(report):
    print("\n=== Capacity Sanity Check (coarse) ===")
    print("Demand/day (kg):", round(report["daily_demand_kg"], 2))
    print("Initial inventory (kg):", round(report["initial_inventory_kg"], 2))
    print("Required from IF (kg):", round(report["required_from_if_kg"], 2))
    print(
        "Config batches/cap (kg):",
        report["config_num_batches"],
        "/",
        round(report["config_supply_cap_kg"], 2),
    )
    print(
        "Batch limits [time, receive, effective]:",
        report["time_limit_batches"],
        report["receive_limit_batches"],
        report["effective_max_batches"],
    )
    print("Effective IF supply (kg):", round(report["effective_max_supply_kg"], 2))
    print(
        "Effective total available (kg):",
        round(report["effective_total_available_kg"], 2),
    )
    print("Daily shortfall (kg):", round(report["daily_shortfall_kg"], 2))
    print(
        "Min batches required / min overtime:",
        report["min_batches_required"],
        "batches /",
        round(report["min_overtime_needed_min"], 1),
        "min",
    )
    print("Coarse feasibility:", "PASS" if report["is_feasible_coarse"] else "FAIL")


def simulate_policy_day(policy_params):
    policy = _decode_policy_vector(policy_params)
    baseline_kw = _build_baseline_load_kw(SIM_DURATION_MIN)

    if_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    melt_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    reheat_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    total_plant_kw = np.zeros(SIM_DURATION_MIN, dtype=float)
    energy_cost_series = np.zeros(SIM_DURATION_MIN, dtype=float)

    mh_levels_series = {
        "A": np.zeros(SIM_DURATION_MIN, dtype=float),
        "B": np.zeros(SIM_DURATION_MIN, dtype=float),
    }
    current_level = MH_INITIAL_LEVEL_KG.copy()
    downtime_remaining = {"A": 0, "B": 0}

    if_states = {
        0: {"active": False, "status": "idle", "batch_id": None},
        1: {"active": False, "status": "idle", "batch_id": None},
    }
    available_if_furnaces = []
    if USE_FURNACE_A:
        available_if_furnaces.append(0)
    if USE_FURNACE_B:
        available_if_furnaces.append(1)

    batches = {}
    batch_id_counter = 0
    ready_to_pour_queue = []
    actual_pour_events = []

    total_if_holding_minutes = 0
    mh_empty_minutes = {"A": 0, "B": 0}
    mh_low_level_minutes = {"A": 0, "B": 0}
    mh_low_level_penalty = 0.0
    min_level_reached = {"A": current_level["A"], "B": current_level["B"]}

    overflow_kg_total = 0.0
    unpoured_batches = []
    last_if_release_time = {0: None, 1: None}
    furnace_use_count = {0: 0, 1: 0}
    if_active_intervals = []
    last_start_min = -1_000_000
    parallel_peak_minutes = 0.0
    prep_ready_time = {0: 0, 1: 0}  # NEW/CHANGED: prep readiness per IF furnace.
    prep_wait_minutes = 0.0  # NEW/CHANGED
    start_blocked_by_prep_count = 0  # NEW/CHANGED
    forced_start_count = 0  # NEW/CHANGED
    non_alternation_count = 0  # NEW/CHANGED
    alternation_count = 0  # NEW/CHANGED
    last_started_furnace = None  # NEW/CHANGED
    total_poured_kg_so_far = 0.0  # NEW/CHANGED
    daily_demand_kg = float(
        (MH_CONSUMPTION_RATE_KG_PER_MIN["A"] + MH_CONSUMPTION_RATE_KG_PER_MIN["B"])
        * SIM_DURATION_MIN
    )  # NEW/CHANGED
    initial_inventory_kg = float(MH_INITIAL_LEVEL_KG["A"] + MH_INITIAL_LEVEL_KG["B"])  # NEW/CHANGED
    required_from_if_kg = float(max(0.0, daily_demand_kg - initial_inventory_kg))  # NEW/CHANGED
    min_duration_per_batch = float(min(v["duration_min_hot"] for v in POWER_PROFILE.values()))  # NEW/CHANGED

    for t in range(SIM_DURATION_MIN):
        # 1) progress melting -> holding
        for f_idx in available_if_furnaces:
            st = if_states[f_idx]
            if st["active"] and st["status"] == "melting":
                b_id = st["batch_id"]
                if t >= batches[b_id]["melt_finish_min"]:
                    st["status"] = "holding"
                    batches[b_id]["status"] = "holding"
                    ready_to_pour_queue.append(b_id)

        # 2) pour with feasibility repair (no overflow)
        keep_pouring = True
        while keep_pouring and ready_to_pour_queue:
            keep_pouring = False
            b_id = ready_to_pour_queue[0]

            # Respect post-pour downtime for receiving in M&H.
            if downtime_remaining["A"] > 0 or downtime_remaining["B"] > 0:
                break

            available_A = MH_MAX_CAPACITY_KG["A"] - current_level["A"]
            available_B = MH_MAX_CAPACITY_KG["B"] - current_level["B"]
            total_available = available_A + available_B
            if total_available < IF_BATCH_OUTPUT_KG:
                break

            remaining = IF_BATCH_OUTPUT_KG
            poured_A = 0.0
            poured_B = 0.0
            fill_order = (
                ["A", "B"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "A" else ["B", "A"]
            )
            for f_id in fill_order:
                if remaining <= 0:
                    break
                space = MH_MAX_CAPACITY_KG[f_id] - current_level[f_id]
                put = min(remaining, space)
                if put > 0:
                    if f_id == "A":
                        poured_A += put
                    else:
                        poured_B += put
                    remaining -= put

            if remaining > 1e-9:
                overflow_kg_total += float(
                    remaining
                )  # CHANGED (Step C): accumulate real overflow before failing repair.
                raise RuntimeError(
                    "Repair failed: attempted pour with insufficient capacity."
                )

            current_level["A"] = np.clip(
                current_level["A"] + poured_A, 0.0, MH_MAX_CAPACITY_KG["A"]
            )
            current_level["B"] = np.clip(
                current_level["B"] + poured_B, 0.0, MH_MAX_CAPACITY_KG["B"]
            )

            downtime_remaining["A"] = POST_POUR_DOWNTIME_MIN
            downtime_remaining["B"] = POST_POUR_DOWNTIME_MIN

            batches[b_id]["poured_A_kg"] = poured_A
            batches[b_id]["poured_B_kg"] = poured_B
            batches[b_id]["overflow_kg"] = 0.0
            batches[b_id]["pour_min"] = t
            batches[b_id]["status"] = "poured"
            total_poured_kg_so_far += float(poured_A + poured_B)  # NEW/CHANGED

            f_idx = batches[b_id]["if_furnace"]
            if_states[f_idx] = {"active": False, "status": "idle", "batch_id": None}
            last_if_release_time[f_idx] = t
            actual_pour_events.append((t, b_id))
            ready_to_pour_queue.pop(0)
            keep_pouring = True

        # 3) holding minutes only for batches that still could not pour.
        total_if_holding_minutes += len(ready_to_pour_queue)

        # 4) policy start decision (score-based + dynamic spacing + demand-gating)
        if batch_id_counter < NUM_BATCHES:
            expected_required_remaining_kg = max(
                0.0, required_from_if_kg - total_poured_kg_so_far
            )  # NEW/CHANGED
            remaining_time_min = float(SIM_DURATION_MIN - t)  # NEW/CHANGED
            batch_needed = int(np.ceil(expected_required_remaining_kg / IF_BATCH_OUTPUT_KG))  # NEW/CHANGED
            force_buffer = min_duration_per_batch + (
                PREP_LOAD_TIME_MIN * 0.5 if ENABLE_PREP_MODEL else 0.0
            )  # NEW/CHANGED
            force_mode = bool(
                batch_needed > 0 and remaining_time_min < batch_needed * force_buffer
            )  # NEW/CHANGED

            any_if_active = any(if_states[f]["active"] for f in available_if_furnaces)
            idle = [f for f in available_if_furnaces if not if_states[f]["active"]]
            if idle:
                if_kw_now = 0.0
                for f in available_if_furnaces:
                    st = if_states[f]
                    if st["active"]:
                        if_kw_now += batches[st["batch_id"]]["power_kw"]
                state = _build_policy_state(
                    policy,
                    current_level,
                    baseline_kw[t],
                    if_kw_now,
                    len(ready_to_pour_queue),
                    _hour_price(t),
                )
                start_score = _start_score(policy, state)
                min_gap_now = _dynamic_min_start_gap(policy, state)
                gap_ok = (t - last_start_min) >= min_gap_now
                can_parallel_now = _parallel_allowed(policy, state, start_score)
                if (not any_if_active) or can_parallel_now:
                    start_allowed = (start_score >= 0.0 and gap_ok) or force_mode  # NEW/CHANGED
                    if start_allowed:
                        if force_mode and ENABLE_PREP_MODEL and ENFORCE_ALTERNATION_FLAG and len(idle) > 1 and last_started_furnace in idle:
                            alt_idle = [f for f in idle if f != last_started_furnace]
                            if alt_idle:
                                idle = alt_idle + [f for f in idle if f not in alt_idle]  # NEW/CHANGED

                        chosen_if = _select_if_furnace(
                            policy,
                            idle,
                            current_level,
                            t,
                            last_if_release_time,
                            furnace_use_count,
                            last_started_furnace=last_started_furnace,
                        )

                        blocked_by_prep = ENABLE_PREP_MODEL and t < prep_ready_time.get(chosen_if, 0)
                        if blocked_by_prep:
                            start_blocked_by_prep_count += 1  # NEW/CHANGED
                            prep_wait_minutes += float(prep_ready_time[chosen_if] - t)  # NEW/CHANGED
                        if not blocked_by_prep:
                            gap = (
                                None
                                if last_if_release_time[chosen_if] is None
                                else t - last_if_release_time[chosen_if]
                            )
                            is_cold_start = (
                                gap is None or gap >= COLD_START_GAP_THRESHOLD_MIN
                            )

                            selected_power = _select_if_power(
                                policy, current_level, baseline_kw[t], if_kw_now, t, force_mode=force_mode
                            )
                            projected_total = baseline_kw[t] + if_kw_now + selected_power
                            hard_guard = CONTRACT_DEMAND_KW - policy["demand_headroom_kw"]
                            shortage_override = (
                                force_mode
                                and state["depletion_urgency"]
                                >= policy.get("force_urgency_threshold", 0.92)
                            )
                            if projected_total <= hard_guard or shortage_override:
                                profile = _compute_batch_profile(
                                    selected_power, is_cold_start, chosen_if
                                )
                                duration = int(round(profile["duration_min"]))
                                duration = max(1, duration)
                                melt_finish = min(SIM_DURATION_MIN, t + duration)

                                b_id = batch_id_counter + 1
                                batches[b_id] = {
                                    "batch_id": b_id,
                                    "if_furnace": chosen_if,
                                    "start_min": t,
                                    "melt_finish_min": melt_finish,
                                    "pour_min": None,
                                    "power_kw": selected_power,
                                    "is_cold_start": is_cold_start,
                                    "status": "melting",
                                    "energy_kwh_profile": profile["energy_kwh"],
                                }
                                if_states[chosen_if] = {
                                    "active": True,
                                    "status": "melting",
                                    "batch_id": b_id,
                                }
                                furnace_use_count[chosen_if] += 1
                                batch_id_counter += 1
                                last_start_min = t
                                if force_mode:
                                    forced_start_count += 1  # NEW/CHANGED

                                if last_started_furnace is not None:
                                    if chosen_if == last_started_furnace:
                                        non_alternation_count += 1  # NEW/CHANGED
                                    else:
                                        alternation_count += 1  # NEW/CHANGED
                                last_started_furnace = chosen_if  # NEW/CHANGED

                                if ENABLE_PREP_MODEL and len(available_if_furnaces) > 1:
                                    other_f = 1 - chosen_if
                                    prep_ready_time[other_f] = max(
                                        prep_ready_time.get(other_f, 0),
                                        t + PREP_LOAD_TIME_MIN,
                                    )  # NEW/CHANGED

        # 5) IF load minute
        minute_if_kw = 0.0
        minute_melt_kw = 0.0
        minute_reheat_kw = 0.0
        for f_idx in available_if_furnaces:
            st = if_states[f_idx]
            if not st["active"]:
                continue
            b = batches[st["batch_id"]]
            p_kw = b["power_kw"]
            if st["status"] == "melting":
                minute_melt_kw += p_kw
                minute_if_kw += p_kw
            elif st["status"] == "holding":
                minute_reheat_kw += p_kw
                minute_if_kw += p_kw
        if_kw_series[t] = minute_if_kw
        melt_kw_series[t] = minute_melt_kw
        reheat_kw_series[t] = minute_reheat_kw

        if minute_if_kw > 0:
            if_active_intervals.append((t, t + 1))

        # 6) M&H consumption + continuity penalties
        for f_id in ["A", "B"]:
            if current_level[f_id] > MH_EMPTY_THRESHOLD_KG:
                current_level[f_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[f_id]
                current_level[f_id] = max(current_level[f_id], 0.0)

            min_level_reached[f_id] = min(min_level_reached[f_id], current_level[f_id])

            if current_level[f_id] <= MH_EMPTY_THRESHOLD_KG:
                mh_empty_minutes[f_id] += 1
            elif current_level[f_id] < MH_MIN_OPERATIONAL_LEVEL_KG[f_id]:
                mh_low_level_minutes[f_id] += 1
                deficit = MH_MIN_OPERATIONAL_LEVEL_KG[f_id] - current_level[f_id]
                deficit_ratio = deficit / max(MH_MIN_OPERATIONAL_LEVEL_KG[f_id], 1e-9)
                mh_low_level_penalty += MH_LOW_LEVEL_PENALTY_RATE * (
                    1.0 + LOW_LEVEL_NONLINEAR_FACTOR * (deficit_ratio**2)
                )

            current_level[f_id] = np.clip(
                current_level[f_id], 0.0, MH_MAX_CAPACITY_KG[f_id]
            )
            mh_levels_series[f_id][t] = current_level[f_id]

            if downtime_remaining[f_id] > 0:
                downtime_remaining[f_id] -= 1

        total_plant_kw[t] = baseline_kw[t] + minute_if_kw
        energy_cost_series[t] = (minute_if_kw / 60.0) * _hour_price(t)
        active_if_count = sum(
            1 for f in available_if_furnaces if if_states[f]["active"]
        )
        if active_if_count >= 2 and total_plant_kw[t] > (0.95 * CONTRACT_DEMAND_KW):
            parallel_peak_minutes += 1.0

    # finalize unpoured batches
    for b_id, b in batches.items():
        if b["pour_min"] is None:
            b["pour_min"] = SIM_DURATION_MIN
            b["status"] = "unpoured"
            b["poured_A_kg"] = 0.0
            b["poured_B_kg"] = 0.0
            b["overflow_kg"] = IF_BATCH_OUTPUT_KG
            overflow_kg_total += IF_BATCH_OUTPUT_KG  # CHANGED (Step C): unpoured batch contributes full overflow.
            unpoured_batches.append(b_id)

    total_melt_kwh = float(np.sum(melt_kw_series) / 60.0)
    total_reheat_kwh = float(np.sum(reheat_kw_series) / 60.0)
    total_if_kwh = float(np.sum(if_kw_series) / 60.0)
    total_energy_cost = float(np.sum(energy_cost_series))

    peak_kw = float(np.max(total_plant_kw)) if len(total_plant_kw) else 0.0
    demand_excess = max(0.0, peak_kw - CONTRACT_DEMAND_KW)
    soft_zone = max(0.0, peak_kw - CONTRACT_DEMAND_KW * DEMAND_SOFT_ZONE_RATIO)
    demand_penalty = (
        DEMAND_CHARGE_RATE_PER_KW * demand_excess
        + 0.25
        * DEMAND_CHARGE_RATE_PER_KW
        * (soft_zone**2)
        / max(CONTRACT_DEMAND_KW, 1.0)
    )

    # CHANGED (Step B): compute poured kg from actual poured mass per batch.
    total_poured_kg = float(
        sum(
            b.get("poured_A_kg", 0.0) + b.get("poured_B_kg", 0.0)
            for b in batches.values()
            if b.get("status") == "poured"
        )
    )
    # CHANGED (Step B): compute required IF contribution and resulting daily shortfall.
    shortfall_kg = float(max(0.0, required_from_if_kg - total_poured_kg))
    alternation_ratio = float(
        alternation_count / max(1, alternation_count + non_alternation_count)
    )  # NEW/CHANGED

    if actual_pour_events:
        makespan_minutes = float(max(t for t, _ in actual_pour_events))
    else:
        # CHANGED (Step A): if no pour happened, makespan is full-day, never zero.
        makespan_minutes = float(SIM_DURATION_MIN)

    schedule = []
    for b_id in sorted(batches.keys()):
        b = batches[b_id]
        schedule.append(
            {
                "batch_id": b_id,
                "if_furnace": b["if_furnace"],
                "start_min": b["start_min"],
                "melt_finish_min": b["melt_finish_min"],
                "pour_min": b["pour_min"],
                "power_kw": b["power_kw"],
                "is_cold_start": b["is_cold_start"],
                "status": b["status"],
            }
        )

    total_if_idle_minutes = _total_idle_minutes_from_intervals(if_active_intervals)

    return {
        "policy": policy,
        "schedule": schedule,
        "batch_timing": {
            b["batch_id"]: {
                "start_min": b["start_min"],
                "melt_finish_min": b["melt_finish_min"],
                "pour_min": b["pour_min"],
                "furnace": b["if_furnace"],
            }
            for b in schedule
        },
        "mh_levels": mh_levels_series,
        "baseline_kw": baseline_kw,
        "if_kw": if_kw_series,
        "total_plant_kw": total_plant_kw,
        "metrics": {
            "melt_kwh": total_melt_kwh,
            "reheat_kwh": total_reheat_kwh,
            "total_if_kwh": total_if_kwh,
            "total_energy_cost": total_energy_cost,
            "peak_kw": peak_kw,
            "demand_excess_kw": demand_excess,
            "demand_penalty": demand_penalty,
            "mh_empty_minutes": mh_empty_minutes,
            "mh_low_level_minutes": mh_low_level_minutes,
            "mh_low_level_penalty": mh_low_level_penalty,
            "overflow_kg_total": overflow_kg_total,
            "holding_minutes_total": float(total_if_holding_minutes),
            "unpoured_batches_count": len(unpoured_batches),
            "makespan_minutes": makespan_minutes,
            "total_poured_kg": total_poured_kg,  # CHANGED (Step A)
            "daily_demand_kg": daily_demand_kg,  # CHANGED (Step B)
            "required_from_if_kg": required_from_if_kg,  # CHANGED (Step B)
            "shortfall_kg": shortfall_kg,  # CHANGED (Step B)
            "prep_wait_minutes": float(prep_wait_minutes),  # NEW/CHANGED
            "non_alternation_count": float(non_alternation_count),  # NEW/CHANGED
            "alternation_ratio": alternation_ratio,  # NEW/CHANGED
            "forced_start_count": float(forced_start_count),  # NEW/CHANGED
            "start_blocked_by_prep_count": float(start_blocked_by_prep_count),  # NEW/CHANGED
            "min_level_reached": min_level_reached,
            "if_use_count_A": furnace_use_count[0],
            "if_use_count_B": furnace_use_count[1],
            "if_idle_minutes_total": total_if_idle_minutes,
            "parallel_peak_minutes": float(parallel_peak_minutes),
        },
    }


def evaluate_policy(policy_params):
    sim = simulate_policy_day(policy_params)
    m = sim["metrics"]

    comp = _compute_obj1_components(m)
    comp_for_obj = dict(comp)
    if USE_CONSTRAINT_HANDLING and OBJ1_EXCLUDE_SERVICE_PENALTIES_WHEN_CONSTRAINED:
        # CHANGED (Step D): avoid penalty pile-up in Obj1 when violations are already constrained.
        comp_for_obj["overflow_penalty"] = 0.0
        comp_for_obj["empty_penalty"] = 0.0
        comp_for_obj["low_level_min_penalty"] = 0.0
        comp_for_obj["unpoured_penalty"] = 0.0

    obj1 = _aggregate_obj1(comp_for_obj)
    obj2 = float(m["makespan_minutes"])

    raw_obj1_total = float(sum(comp.values()))
    normalized_terms = {}
    for k, v in comp.items():
        ref = max(1e-9, float(OBJ1_COMPONENT_REFS.get(k, 1.0)))
        w = float(OBJ1_COMPONENT_WEIGHTS.get(k, 1.0))
        normalized_terms[k] = w * (float(v) / ref)
    normalized_sum = float(sum(normalized_terms.values()))
    contribution_pct = {}
    if raw_obj1_total > 1e-9:
        for k, v in comp.items():
            contribution_pct[k] = 100.0 * float(v) / raw_obj1_total
    else:
        for k in comp:
            contribution_pct[k] = 0.0

    violation_g = _violation_vector(m)
    overflow_violation = float(max(0.0, violation_g[0]))
    shortfall_violation = float(max(0.0, violation_g[1]))  # CHANGED (Step B)
    empty_violation = float(max(0.0, violation_g[2]))
    low_violation = float(max(0.0, violation_g[3]))
    parallel_peak_violation = float(max(0.0, violation_g[4]))  # CHANGED (Step E)
    parallel_peak_minutes = float(max(0.0, m.get("parallel_peak_minutes", 0.0)))
    total_violation = float(
        np.sum(
            [
                overflow_violation,
                shortfall_violation,  # CHANGED (Step B)
                empty_violation,
                low_violation,
                parallel_peak_violation,
            ]
        )
    )
    obj0 = (
        OBJ0_WEIGHTS["overflow_kg"] * overflow_violation
        + OBJ0_WEIGHTS["shortfall_kg"] * shortfall_violation  # CHANGED (Step B)
        + OBJ0_WEIGHTS["empty_min"] * empty_violation
        + OBJ0_WEIGHTS["low_level_min"] * low_violation
        + OBJ0_WEIGHTS["parallel_peak_min"] * parallel_peak_violation
    )

    cost_components = {
        "obj0_violation": obj0,
        "obj1_total": obj1,
        "obj1_raw_total": raw_obj1_total,
        "obj1_mode": OBJ1_AGGREGATION_MODE,
        "obj1_normalized_sum": normalized_sum,
        "obj2_makespan": obj2,
        "comp_energy_cost": comp["energy_cost"],
        "comp_demand_penalty": comp["demand_penalty"],
        "comp_holding_penalty": comp["holding_penalty"],
        "comp_empty_penalty": comp["empty_penalty"],
        "comp_low_level_min_penalty": comp["low_level_min_penalty"],
        "comp_low_level_shape_penalty": comp["low_level_shape_penalty"],
        "comp_overflow_penalty": comp["overflow_penalty"],
        "comp_unpoured_penalty": comp["unpoured_penalty"],
        "comp_prep_wait_penalty": comp["prep_wait_penalty"],  # NEW/CHANGED
        "comp_switch_penalty": comp["switch_penalty"],  # NEW/CHANGED
        "pct_energy_cost": contribution_pct["energy_cost"],
        "pct_demand_penalty": contribution_pct["demand_penalty"],
        "pct_holding_penalty": contribution_pct["holding_penalty"],
        "pct_empty_penalty": contribution_pct["empty_penalty"],
        "pct_low_level_min_penalty": contribution_pct["low_level_min_penalty"],
        "pct_low_level_shape_penalty": contribution_pct["low_level_shape_penalty"],
        "pct_overflow_penalty": contribution_pct["overflow_penalty"],
        "pct_unpoured_penalty": contribution_pct["unpoured_penalty"],
        "violation_overflow": overflow_violation,
        "violation_shortfall_kg": shortfall_violation,  # CHANGED (Step B)
        "violation_empty_min": empty_violation,
        "violation_low_min": low_violation,
        "parallel_peak_minutes": parallel_peak_minutes,
        "violation_parallel_peak_min": float(parallel_peak_violation),
        "total_violation": total_violation,
        "total_if_kwh": m["total_if_kwh"],
        "melt_kwh": m["melt_kwh"],
        "reheat_kwh": m["reheat_kwh"],
        "total_energy_cost": m["total_energy_cost"],
        "peak_kw": m["peak_kw"],
        "demand_excess_kw": m["demand_excess_kw"],
        "demand_penalty": m["demand_penalty"],
        "mh_empty_minutes_A": m["mh_empty_minutes"]["A"],
        "mh_empty_minutes_B": m["mh_empty_minutes"]["B"],
        "mh_low_level_minutes_A": m["mh_low_level_minutes"]["A"],
        "mh_low_level_minutes_B": m["mh_low_level_minutes"]["B"],
        "mh_low_level_penalty": m["mh_low_level_penalty"],
        "overflow_kg_total": m["overflow_kg_total"],
        "holding_minutes_total": m["holding_minutes_total"],
        "if_idle_minutes_total": m["if_idle_minutes_total"],
        "if_use_count_A": m["if_use_count_A"],
        "if_use_count_B": m["if_use_count_B"],
        "unpoured_batches_count": m["unpoured_batches_count"],
        "makespan_minutes": m["makespan_minutes"],
        "total_poured_kg": m["total_poured_kg"],  # CHANGED (Step A)
        "daily_demand_kg": m["daily_demand_kg"],  # CHANGED (Step B)
        "required_from_if_kg": m["required_from_if_kg"],  # CHANGED (Step B)
        "shortfall_kg": m["shortfall_kg"],  # CHANGED (Step B)
        "prep_wait_minutes": m.get("prep_wait_minutes", 0.0),  # NEW/CHANGED
        "non_alternation_count": m.get("non_alternation_count", 0.0),  # NEW/CHANGED
        "alternation_ratio": m.get("alternation_ratio", 0.0),  # NEW/CHANGED
        "forced_start_count": m.get("forced_start_count", 0.0),  # NEW/CHANGED
        "start_blocked_by_prep_count": m.get("start_blocked_by_prep_count", 0.0),  # NEW/CHANGED
        "min_level_A": m["min_level_reached"]["A"],
        "min_level_B": m["min_level_reached"]["B"],
        "policy": sim["policy"],
        "schedule": sim["schedule"],
        "batch_timing": sim["batch_timing"],
        "mh_levels": sim["mh_levels"],
        "baseline_kw": sim["baseline_kw"],
        "if_kw": sim["if_kw"],
        "total_plant_kw": sim["total_plant_kw"],
    }
    return obj0, obj1, obj2, cost_components


class PolicyProblem(Problem):
    def __init__(self):
        if SIMPLE_POLICY_MODE:
            # NEW/CHANGED: compact 10-variable policy for smoother NSGA-II trade-offs.
            xl = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.70, 0.0],
                dtype=float,
            )
            xu = np.array(
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.98, 1.0],
                dtype=float,
            )
            n_var = 10
        else:
        # [trigA_frac, trigB_frac, lookahead_norm, critA_frac, critB_frac, mid_frac,
        #  headroom_norm, peak_weight, furnace_bias, wait_vs_rush,
        #  start_bias, w_dep, w_queue, w_peak, min_gap_norm, gap_peak_coeff,
        #  parallel_margin_norm, parallel_bias, coldstart_w, tou_w, balance_w]
            xl = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    -2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=float,
            )
            xu = np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    4.0,
                    4.0,
                    4.0,
                    1.0,
                    3.0,
                    1.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ],
                dtype=float,
            )
            n_var = 21
        # CHANGED (Step E): include shortfall + parallel-peak in constraints when enabled.
        if HARD_FORBID_OVERFLOW and USE_CONSTRAINT_HANDLING:
            n_constr = 5  # overflow, shortfall, empty, low-level, parallel-peak
        elif HARD_FORBID_OVERFLOW:
            n_constr = 1  # overflow only
        elif USE_CONSTRAINT_HANDLING:
            n_constr = 5  # overflow, shortfall, empty, low-level, parallel-peak
        else:
            n_constr = 0
        super().__init__(n_var=n_var, n_obj=3, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        details = []
        generation_cache = {}
        key_list = []
        G = []
        for x in X:
            key = _policy_cache_key(x)
            key_list.append(key)
            cached = generation_cache.get(key)
            if cached is not None:
                o0, o1, o2, d = cached
            elif DETERMINISTIC_SIMULATION:
                cached = _EVAL_CACHE.get(key)
                if cached is None:
                    o0, o1, o2, d = evaluate_policy(x)
                    _EVAL_CACHE[key] = (o0, o1, o2, d)
                else:
                    o0, o1, o2, d = cached
            else:
                # Keep caching disabled in stochastic mode to avoid bias.
                o0, o1, o2, d = evaluate_policy(x)
            generation_cache[key] = (o0, o1, o2, d)
            F.append([o0, o1, o2])
            details.append(d)
            if HARD_FORBID_OVERFLOW:
                if USE_CONSTRAINT_HANDLING:
                    # CHANGED (Step E): keep overflow hard and add shortfall/empty/low/parallel constraints.
                    G.append(
                        [
                            d.get("violation_overflow", 0.0),
                            d.get("violation_shortfall_kg", 0.0),
                            d.get("violation_empty_min", 0.0),
                            d.get("violation_low_min", 0.0),
                            d.get("violation_parallel_peak_min", 0.0),
                        ]
                    )
                else:
                    G.append([d.get("violation_overflow", 0.0)])
            elif USE_CONSTRAINT_HANDLING:
                G.append(
                    [
                        d.get("violation_overflow", 0.0),
                        d.get("violation_shortfall_kg", 0.0),  # CHANGED (Step B)
                        d.get("violation_empty_min", 0.0),
                        d.get("violation_low_min", 0.0),
                        d.get("violation_parallel_peak_min", 0.0),  # CHANGED (Step E)
                    ]
                )

        out["F"] = np.asarray(F, dtype=float)
        if HARD_FORBID_OVERFLOW or USE_CONSTRAINT_HANDLING:
            out["G"] = np.asarray(G, dtype=float)
        out["cost_details"] = details
        out["schedules"] = np.array([d["schedule"] for d in details], dtype=object)

        if DEBUG and len(F) > 0:
            unique = np.unique(np.round(out["F"], 2), axis=0).shape[0]
            f_min = np.min(out["F"], axis=0)
            f_max = np.max(out["F"], axis=0)
            avg_peak = float(np.mean([d["peak_kw"] for d in details]))
            avg_use_b = float(np.mean([d["if_use_count_B"] for d in details]))
            avg_poured = float(
                np.mean([d.get("total_poured_kg", 0.0) for d in details])
            )  # NEW/CHANGED
            avg_prep_wait = float(
                np.mean([d.get("prep_wait_minutes", 0.0) for d in details])
            )  # NEW/CHANGED
            avg_alt_ratio = float(
                np.mean([d.get("alternation_ratio", 0.0) for d in details])
            )  # NEW/CHANGED
            shortfalls = np.asarray(
                [d.get("shortfall_kg", 0.0) for d in details], dtype=float
            )  # CHANGED (Step F)
            shortfalls = np.where(np.isfinite(shortfalls), shortfalls, 0.0)  # CHANGED (Step F)
            avg_shortfall = float(np.mean(shortfalls)) if len(shortfalls) > 0 else 0.0  # CHANGED (Step F)
            min_shortfall = float(np.min(shortfalls)) if len(shortfalls) > 0 else 0.0  # CHANGED (Step F)
            max_shortfall = float(np.max(shortfalls)) if len(shortfalls) > 0 else 0.0  # CHANGED (Step F)
            shortfall_ok_pct = (
                float(np.mean(shortfalls <= MAX_SHORTFALL_KG_ALLOW) * 100.0)
                if len(shortfalls) > 0
                else 0.0
            )  # CHANGED (Step F)
            unique_policy = len(set(key_list))
            unique_x = np.unique(np.round(np.asarray(X, dtype=float), 4), axis=0).shape[
                0
            ]
            print("DEBUG unique F:", unique)
            print("DEBUG F min:", np.round(f_min, 2), "max:", np.round(f_max, 2))
            print(
                "DEBUG avg use_B / avg peak:",
                round(avg_use_b, 3),
                round(avg_peak, 3),
            )
            print(
                "DEBUG avg poured / avg shortfall:",
                round(avg_poured, 3),
                round(avg_shortfall, 3),
            )  # NEW/CHANGED
            print(
                "DEBUG avg prep_wait / avg alternation_ratio:",
                round(avg_prep_wait, 3),
                round(avg_alt_ratio, 3),
            )  # NEW/CHANGED
            print(  # CHANGED (Step F)
                "DEBUG shortfall avg/min/max:",
                round(avg_shortfall, 3),
                round(min_shortfall, 3),
                round(max_shortfall, 3),
            )
            print(  # CHANGED (Step F)
                "DEBUG shortfall<=allow (%):",
                round(shortfall_ok_pct, 2),
            )
            print(
                "DEBUG unique policy keys:", unique_policy, "unique X(4dp):", unique_x
            )


def format_policy_breakdown(cost):
    if not cost:
        return "No cost details available."
    lines = [
        "Cost Component Details:",
        f"  Obj0 Violation            : {cost.get('obj0_violation', 0.0):.2f}",
        f"  Obj1 Total                : {cost.get('obj1_total', 0.0):.2f}",
        f"  Obj1 Raw Sum              : {cost.get('obj1_raw_total', 0.0):.2f}",
        f"  Obj1 Mode                 : {cost.get('obj1_mode', 'raw')}",
        f"  Obj1 Normalized Sum       : {cost.get('obj1_normalized_sum', 0.0):.4f}",
        f"  Obj2 Makespan (min)       : {cost.get('obj2_makespan', 0.0):.2f}",
        f"  IF Total kWh              : {cost.get('total_if_kwh', 0.0):.2f}",
        f"    - Melt kWh              : {cost.get('melt_kwh', 0.0):.2f}",
        f"    - Reheat kWh            : {cost.get('reheat_kwh', 0.0):.2f}",
        f"  Energy Cost (TOU)         : {cost.get('total_energy_cost', 0.0):.2f}",
        f"  Peak kW                   : {cost.get('peak_kw', 0.0):.2f}",
        f"  Demand Excess kW          : {cost.get('demand_excess_kw', 0.0):.2f}",
        f"  Demand Penalty            : {cost.get('demand_penalty', 0.0):.2f}",
        f"  MH Empty Minutes A/B      : {cost.get('mh_empty_minutes_A', 0.0):.2f} / {cost.get('mh_empty_minutes_B', 0.0):.2f}",
        f"  MH Low-Level Minutes A/B  : {cost.get('mh_low_level_minutes_A', 0.0):.2f} / {cost.get('mh_low_level_minutes_B', 0.0):.2f}",
        f"  MH Low-Level Penalty      : {cost.get('mh_low_level_penalty', 0.0):.2f}",
        f"  Overflow kg (total)       : {cost.get('overflow_kg_total', 0.0):.2f}",
        f"  Holding Minutes (total)   : {cost.get('holding_minutes_total', 0.0):.2f}",
        f"  IF Idle Minutes (system)  : {cost.get('if_idle_minutes_total', 0.0):.2f}",
        f"  IF Use Count A/B          : {cost.get('if_use_count_A', 0)} / {cost.get('if_use_count_B', 0)}",
        f"  Unpoured Batches          : {cost.get('unpoured_batches_count', 0)}",
        f"  Makespan (min)            : {cost.get('makespan_minutes', 0.0):.2f}",
        f"  Total Poured (kg)         : {cost.get('total_poured_kg', 0.0):.2f}",  # CHANGED (Step A)
        f"  Required From IF (kg)     : {cost.get('required_from_if_kg', 0.0):.2f}",  # CHANGED (Step B)
        f"  Daily Shortfall (kg)      : {cost.get('shortfall_kg', 0.0):.2f}",  # CHANGED (Step B)
        f"  Prep Wait Minutes         : {cost.get('prep_wait_minutes', 0.0):.2f}",  # NEW/CHANGED
        f"  Non-Alternation Count     : {cost.get('non_alternation_count', 0.0):.2f}",  # NEW/CHANGED
        f"  Alternation Ratio         : {cost.get('alternation_ratio', 0.0):.3f}",  # NEW/CHANGED
        f"  Forced Start Count        : {cost.get('forced_start_count', 0.0):.2f}",  # NEW/CHANGED
        f"  Start Blocked By Prep     : {cost.get('start_blocked_by_prep_count', 0.0):.2f}",  # NEW/CHANGED
        f"  Min Level Reached A/B     : {cost.get('min_level_A', 0.0):.2f} / {cost.get('min_level_B', 0.0):.2f}",
        "  Obj1 Decomposition (raw / %raw):",
        f"    - Energy Cost           : {cost.get('comp_energy_cost', 0.0):.2f} / {cost.get('pct_energy_cost', 0.0):.2f}%",
        f"    - Demand Penalty        : {cost.get('comp_demand_penalty', 0.0):.2f} / {cost.get('pct_demand_penalty', 0.0):.2f}%",
        f"    - Holding Penalty       : {cost.get('comp_holding_penalty', 0.0):.2f} / {cost.get('pct_holding_penalty', 0.0):.2f}%",
        f"    - Empty Penalty         : {cost.get('comp_empty_penalty', 0.0):.2f} / {cost.get('pct_empty_penalty', 0.0):.2f}%",
        f"    - Low-Level Min Penalty : {cost.get('comp_low_level_min_penalty', 0.0):.2f} / {cost.get('pct_low_level_min_penalty', 0.0):.2f}%",
        f"    - Low-Level Shape       : {cost.get('comp_low_level_shape_penalty', 0.0):.2f} / {cost.get('pct_low_level_shape_penalty', 0.0):.2f}%",
        f"    - Overflow Penalty      : {cost.get('comp_overflow_penalty', 0.0):.2f} / {cost.get('pct_overflow_penalty', 0.0):.2f}%",
        f"    - Unpoured Penalty      : {cost.get('comp_unpoured_penalty', 0.0):.2f} / {cost.get('pct_unpoured_penalty', 0.0):.2f}%",
        f"    - Prep Wait Penalty     : {cost.get('comp_prep_wait_penalty', 0.0):.2f}",
        f"    - Switch Penalty        : {cost.get('comp_switch_penalty', 0.0):.2f}",
        "  Constraint Violations:",
        f"    - Overflow (kg)         : {cost.get('violation_overflow', 0.0):.2f}",
        f"    - Shortfall (kg)        : {cost.get('violation_shortfall_kg', 0.0):.2f}",  # CHANGED (Step B)
        f"    - Empty Minutes         : {cost.get('violation_empty_min', 0.0):.2f}",
        f"    - Low-Level Minutes     : {cost.get('violation_low_min', 0.0):.2f}",
        f"    - Parallel Peak Minutes : {cost.get('violation_parallel_peak_min', 0.0):.2f}",
        f"    - Total Violation       : {cost.get('total_violation', 0.0):.2f}",
    ]
    return "\n".join(lines)


class ViolationFirstProblem(Problem):
    def __init__(self):
        base = PolicyProblem()
        super().__init__(n_var=base.n_var, n_obj=2, n_constr=0, xl=base.xl, xu=base.xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        details = []
        for x in X:
            _, _, o2, d = evaluate_policy(x)
            v = d.get("total_violation", 0.0)
            F.append([v, o2])
            details.append(d)
        out["F"] = np.asarray(F, dtype=float)
        out["cost_details"] = details


def plot_policy_result(cost_details, title_prefix="Policy Result"):
    schedule = cost_details.get("schedule", [])
    mh_levels = cost_details.get("mh_levels")
    baseline_kw = cost_details.get("baseline_kw")
    if_kw = cost_details.get("if_kw")
    total_kw = cost_details.get("total_plant_kw")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 11), sharex=True)

    for item in schedule:
        b_id = item["batch_id"]
        f_idx = item["if_furnace"]
        if f_idx not in furnace_y:
            continue
        start = SHIFT_START + item["start_min"]
        melt_finish = SHIFT_START + item["melt_finish_min"]
        pour = SHIFT_START + item["pour_min"]
        melt_dur = max(0.0, melt_finish - start)
        hold_dur = max(0.0, pour - melt_finish)

        ax1.broken_barh(
            [(start, melt_dur)],
            (furnace_y[f_idx], height),
            facecolors="gray",
            edgecolor="black",
        )
        if hold_dur > 0:
            ax1.broken_barh(
                [(melt_finish, hold_dur)],
                (furnace_y[f_idx], height),
                facecolors="red",
                edgecolor="black",
            )
        ax1.text(
            start + max(1.0, melt_dur) / 2.0,
            furnace_y[f_idx] + height / 2.0,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )

    ax1.set_ylabel("IF Furnace")
    ax1.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax1.set_yticklabels(["Furnace A", "Furnace B"])
    ax1.grid(True, axis="y", alpha=0.4)
    ax1.set_title(f"{title_prefix} - IF Gantt (melt=gray, holding/reheat=red)")

    t_shifted = np.arange(SHIFT_START, SHIFT_START + SIM_DURATION_MIN)
    if mh_levels is not None:
        for mh in ["A", "B"]:
            ax2.plot(
                t_shifted,
                mh_levels[mh],
                linewidth=2,
                color=MH_FURNACE_COLORS[mh],
                label=f"M&H {mh}",
            )
            ax2.axhline(
                MH_MIN_OPERATIONAL_LEVEL_KG[mh],
                linestyle="--",
                color=MH_FURNACE_COLORS[mh],
                alpha=0.5,
            )
    ax2.set_ylabel("M&H Level (kg)")
    ax2.set_title("M&H Levels")
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc="upper right")

    if baseline_kw is not None and if_kw is not None and total_kw is not None:
        ax3.plot(
            t_shifted, baseline_kw, label="Baseline kW", color="gray", linewidth=1.5
        )
        ax3.plot(t_shifted, if_kw, label="IF kW", color="blue", linewidth=1.5)
        ax3.plot(t_shifted, total_kw, label="Total Plant kW", color="red", linewidth=2)
        ax3.axhline(
            CONTRACT_DEMAND_KW, color="black", linestyle="--", label="Contract kW"
        )
        peak_idx = int(np.argmax(total_kw))
        ax3.scatter(
            [SHIFT_START + peak_idx],
            [total_kw[peak_idx]],
            color="magenta",
            zorder=5,
            label=f"Peak {total_kw[peak_idx]:.1f} kW",
        )
    ax3.set_ylabel("kW")
    ax3.set_xlabel("Time (HH:MM)")
    ax3.set_title("Plant Load")
    ax3.grid(True, alpha=0.4)
    ax3.legend(loc="upper right")

    xticks = np.arange(SHIFT_START, SHIFT_START + SIM_DURATION_MIN + 1, 60)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels([f"{(x // 60) % 24:02d}:{x % 60:02d}" for x in xticks])

    plt.tight_layout()
    plt.show()


def _pick_display_population(result):
    if result is None:
        return None
    if result.opt is not None and len(result.opt) > 0:
        return result.opt
    if result.pop is not None and len(result.pop) > 0:
        return result.pop
    return None


def main():
    np.random.seed(42)
    sanity = quick_capacity_sanity_check()
    print_capacity_sanity_report(sanity)

    problem = PolicyProblem()
    sampling = FloatRandomSampling()
    # More exploratory variation while remaining stable for policy tuning.
    crossover = SBX(prob=0.92, eta=8)
    mutation = PolynomialMutation(prob=0.45, eta=8)

    algorithm = NSGA2(
        pop_size=80,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=PolicyDuplicateElimination(),
    )
    termination = get_termination("n_gen", 100)
    callback = StagnationEarlyStopCallback(
        patience_gens=EARLY_STOP_PATIENCE_GENS,
        delta_obj0=EARLY_STOP_DELTA_OBJ0,
        delta_obj1=EARLY_STOP_DELTA_OBJ1,
        delta_obj2=EARLY_STOP_DELTA_OBJ2,
    )

    if USE_TWO_STAGE_OPTIMIZATION:
        print("Running stage-1 (violation-first) optimization...")
        stage1_problem = ViolationFirstProblem()
        stage1_result = minimize(
            stage1_problem,
            NSGA2(
                pop_size=80,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.90, eta=10),
                mutation=PolynomialMutation(prob=0.50, eta=10),
                eliminate_duplicates=PolicyDuplicateElimination(),
            ),
            get_termination("n_gen", STAGE1_GENS),
            seed=42,
            verbose=True,
            save_history=False,
        )
        seed_X = stage1_result.pop.get("X") if stage1_result.pop is not None else None
        if seed_X is not None and len(seed_X) > 0:
            algorithm = NSGA2(
                pop_size=80,
                sampling=seed_X,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=PolicyDuplicateElimination(),
            )

    print("Running Policy NSGA-II optimization...")
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=False,
        callback=callback,
    )
    print("Optimization finished.")

    F_results = result.F
    if F_results is None and result.pop is not None:
        F_results = result.pop.get("F")

    if F_results is None or len(F_results) == 0:
        print("Warning: no objective points available for plotting Pareto front.")
    else:
        unique_pareto = np.unique(np.round(F_results, 2), axis=0).shape[0]
        print("Pareto unique points (rounded 2dp):", unique_pareto)
        print(
            "Obj0 range:",
            float(np.min(F_results[:, 0])),
            "->",
            float(np.max(F_results[:, 0])),
        )
        print(
            "Obj1 range:",
            float(np.min(F_results[:, 1])),
            "->",
            float(np.max(F_results[:, 1])),
        )
        print(
            "Obj2 range:",
            float(np.min(F_results[:, 2])),
            "->",
            float(np.max(F_results[:, 2])),
        )

        plt.figure(figsize=(10, 6))
        plt.scatter(
            F_results[:, 1],
            F_results[:, 2],
            c="red",
            s=35,
            edgecolors="k",
            label="Policy Pareto Front",
        )
        plt.xlabel("Obj1 (Cost)")
        plt.ylabel("Obj2 (Makespan min)")
        plt.title("Policy Pareto Front (3-objective; shown Obj1 vs Obj2)")
        plt.grid(True)
        plt.legend()
        plt.show()

    show_pop = _pick_display_population(result)
    if show_pop is None:
        print("No individuals to display.")
        return

    num_to_show = min(5, len(show_pop))
    print(f"\nShowing {num_to_show} solution(s):")
    for i in range(num_to_show):
        ind = show_pop[i]
        fvals = ind.F if ind.F is not None else (None, None, None)
        f0, f1, f2 = fvals
        details = ind.get("cost_details")
        print(f"\nSolution #{i+1}")
        print("  Policy X:", np.round(ind.X, 3))
        if f0 is not None:
            print(f"  Obj0 (Violation): {f0:.3f}")
        if f1 is not None:
            print(f"  Obj1: {f1:.3f}")
        if f2 is not None:
            print(f"  Obj2: {f2:.3f}")
        if details:
            print(
                "  use_B:",
                details.get("if_use_count_B", 0),
                "idle_min:",
                round(details.get("if_idle_minutes_total", 0.0), 2),
                "peak_kW:",
                round(details.get("peak_kw", 0.0), 2),
            )
            print(
                "  prep_wait_min:",
                round(details.get("prep_wait_minutes", 0.0), 2),
                "alt_ratio:",
                round(details.get("alternation_ratio", 0.0), 3),
                "non_alt:",
                round(details.get("non_alternation_count", 0.0), 2),
                "forced_start:",
                round(details.get("forced_start_count", 0.0), 2),
                "blocked_by_prep:",
                round(details.get("start_blocked_by_prep_count", 0.0), 2),
            )  # NEW/CHANGED
            print(format_policy_breakdown(details))
            plot_policy_result(details, title_prefix=f"Policy #{i+1}")


# NEW/CHANGED: ================== CONFIG SUMMARY ==================
# SIMPLE_POLICY_MODE:
#   True  -> compact 10-variable policy (recommended for smoother Pareto front)
#   False -> legacy/full policy variable set
#
# ENABLE_PREP_MODEL:
#   True  -> enforce PREP_LOAD_TIME_MIN readiness per furnace.
#            Starting furnace f prepares the other furnace during melt.
#   False -> disable prep-ready blocking dynamics.
#
# ENFORCE_ALTERNATION_FLAG:
#   False -> soft alternation via alternation_bias + switch penalty.
#   True  -> in force mode, prefer strict A/B alternation when feasible.
#
# PREP_LOAD_TIME_MIN / PREP_WAIT_COST_PER_MIN / SWITCH_PENALTY_PER_REPEAT:
#   Tune these to match plant prep behavior and alternation economics.
# ================================================================
if __name__ == "__main__":
    main()
