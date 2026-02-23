import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from pymoo.core.problem import Problem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.callback import Callback
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.ga import GA


# =============== CONFIG ==================
HOURS_A_DAY = 24 * 60
SIM_DURATION_MIN = HOURS_A_DAY
NUM_BATCHES = 5
NUM_BATCHES_TARGET_MONTHLY_AVG = 5
# Set to int to override one-day simulation batch target.
NUM_BATCHES_RUN_OVERRIDE = 12
# NEW/CHANGED: optimization mode switch ("energy" or "service").
OPT_MODE = "service"
EPS_LEVEL = 1.0
# NEW/CHANGED: service-mode scalarization weights (lexicographic-like priorities).
SERVICE_W_MISSING_BATCHES = 1e9
SERVICE_W_OVERFLOW = 1e8
SERVICE_W_ZERO_MINUTES = 1e5
SERVICE_W_EPS_DEFICIT_AREA = 1e2
SERVICE_W_LOW_LEVEL_MINUTES = 1.0
SERVICE_W_HOLDING_MINUTES = 1e4
SERVICE_W_REHEAT_KWH = 1e3
SERVICE_W_ENERGY_TIEBREAKER = 1e-6

# Energy-mode behavior tuning (price-first scheduling)
ENERGY_MODE_START_W_DEP = 0.9
ENERGY_MODE_START_W_TRIGGER = 0.3
ENERGY_MODE_START_W_QUEUE = 0.2
ENERGY_MODE_START_W_PEAK = 1.2
ENERGY_MODE_START_W_TOU = 1.0
ENERGY_MODE_START_W_INV_FLOOR = 0.5
ENERGY_MODE_INV_FLOOR_FRAC = 0.45
ENERGY_PRICE_LOOKAHEAD_MIN = 60
ENERGY_MODE_START_W_URGENCY_PER_HOUR = 0.8
ENERGY_MODE_START_W_POST_SOLAR_URGENCY = 0.6
ENERGY_MAKESPAN_COST_PER_MIN = 0.6
ENERGY_IDLE_GAP_COST_PER_MIN = 1.5
ENERGY_IDLE_GAP_SUPERLINEAR_COEFF = 0.15
ENERGY_CHEAP_MELT_CREDIT_PER_MIN = 5.0
ENERGY_MODE_MIN_JIT_SLACK_MIN = 25.0

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
MH_CONSUMPTION_RATE_KG_PER_MIN = {"A": 2.20, "B": 2.30}
MH_EMPTY_THRESHOLD_KG = 0.0
MH_MIN_OPERATIONAL_LEVEL_KG = {"A": 200.0, "B": 125.0}
MH_LOW_LEVEL_PENALTY_RATE = 200.0
LOW_LEVEL_NONLINEAR_FACTOR = 3.0

# TH tariff defaults: TOU 4.2.2 (22-33kV) + Ft (THB units, excl. VAT)
TARIFF_NAME = "TOU_22-33kV_4.2.2"
TOU_ONPEAK_BAHT_PER_KWH_BASE = 4.1839
TOU_OFFPEAK_BAHT_PER_KWH_BASE = 2.6037
FT_BAHT_PER_KWH = 0.0972
TOU_ONPEAK_BAHT_PER_KWH = TOU_ONPEAK_BAHT_PER_KWH_BASE + FT_BAHT_PER_KWH
TOU_OFFPEAK_BAHT_PER_KWH = TOU_OFFPEAK_BAHT_PER_KWH_BASE + FT_BAHT_PER_KWH
DEMAND_CHARGE_BAHT_PER_KW_MONTH = 132.93
SERVICE_FEE_BAHT_PER_MONTH = 312.24
DEMAND_INTERVAL_MIN = 15
BILLING_DAYS_PER_MONTH = 30
ASSUME_WEEKDAY = True
INCLUDE_SERVICE_FEE_IN_ENERGY_OBJECTIVE = False

# Increased to reflect realistic plant contract and allow higher baseline window.
CONTRACT_DEMAND_KW = 1600.0

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
ENFORCE_ALTERNATION_FLAG = (
    False  # If True, enforce A/B alternation in force mode when feasible.
)
SWITCH_PENALTY_PER_REPEAT = 8.0
PREP_WAIT_COST_PER_MIN = 25.0

SHIFT_START = 8 * 60
# NEW/CHANGED: solar window + offset for effective grid energy pricing.
SOLAR_START = 12 * 60  # 12:00
SOLAR_END = 13 * 60  # 13:00
SOLAR_OFFSET_KW = 500.0
# NEW/CHANGED: effective TOU discount during solar window.
SOLAR_EFFECTIVE_PRICE_FACTOR = 0.35
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
    "jit_delay_penalty": 20000.0,  # NEW/CHANGED
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
    "jit_delay_penalty": 0.25,  # NEW/CHANGED
}

# Optional constraint-handling mode to prevent "all-penalty" domination.
USE_CONSTRAINT_HANDLING = (
    True  # NEW/CHANGED: prioritize feasible/service-respecting solutions.
)
OBJ1_EXCLUDE_SERVICE_PENALTIES_WHEN_CONSTRAINED = (
    True  # CHANGED (Step D): keep Obj1 focused on cost when using constraints.
)
MAX_OVERFLOW_KG_ALLOW = 1e-6
MAX_EMPTY_MIN_ALLOW = 120.0
MAX_LOW_LEVEL_MIN_ALLOW = 240.0
HARD_FORBID_OVERFLOW = True
SIMPLE_POLICY_MODE = (
    True  # NEW/CHANGED: use reduced 10-variable policy for smoother Pareto fronts.
)
JIT_DELAY_COST_PER_MIN = 6.0  # NEW/CHANGED: mild cost for excessive JIT deferral.

# Legacy constants retained (unused in single-objective mode).
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
    "jit_slack_min": 1.0,  # NEW/CHANGED
    "start_delay_min": 5.0,  # NEW/CHANGED
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
DEBUG_DUMP_MH_TRACE = True
DEBUG_MH_TRACE_STEP_MIN = 1


def _current_num_batches():
    if NUM_BATCHES_RUN_OVERRIDE is None:
        return int(NUM_BATCHES)
    return int(NUM_BATCHES_RUN_OVERRIDE)


def _build_baseline_load_kw(duration_min=SIM_DURATION_MIN):
    # NEW/CHANGED: align baseline time-of-day with SHIFT_START (t=0 means SHIFT_START).
    t = np.arange(duration_min, dtype=float)
    tod = (SHIFT_START + t).astype(int) % 1440
    baseline = np.zeros(duration_min, dtype=float)

    # Step profile by operation windows (kW):
    # 00:00-07:00=450, 07:00-12:00=800, 12:00-13:00=600,
    # 13:00-17:00=850, 17:00-22:00=750, 22:00-24:00=450
    baseline[(tod >= 0) & (tod < 7 * 60)] = 450.0
    baseline[(tod >= 7 * 60) & (tod < 12 * 60)] = 800.0
    baseline[(tod >= 12 * 60) & (tod < 13 * 60)] = 600.0
    baseline[(tod >= 13 * 60) & (tod < 17 * 60)] = 850.0
    baseline[(tod >= 17 * 60) & (tod < 22 * 60)] = 750.0
    baseline[(tod >= 22 * 60) & (tod < 24 * 60)] = 450.0

    # Small deterministic variability (+/-2.5%) to avoid perfectly flat lines.
    noise_pct = 0.025
    if DETERMINISTIC_SIMULATION:
        rng = np.random.default_rng(42)
        noise = rng.uniform(-noise_pct, noise_pct, size=duration_min)
    else:
        noise = np.random.uniform(-noise_pct, noise_pct, size=duration_min)
    baseline *= 1.0 + noise
    return np.clip(baseline, 220.0, None)


def is_onpeak(tod_min, assume_weekday=True):
    """TOU on-peak window: 09:00-22:00 on weekdays."""
    hour = int((int(tod_min) % 1440) // 60)
    weekday_ok = bool(assume_weekday)
    return bool(weekday_ok and (9 <= hour < 22))


def get_tou_price(minute_idx, assume_weekday=ASSUME_WEEKDAY):
    """Minute-level TOU price aligned with SHIFT_START (THB/kWh)."""
    tod = int((SHIFT_START + int(minute_idx)) % 1440)
    if is_onpeak(tod, assume_weekday=assume_weekday):
        return float(TOU_ONPEAK_BAHT_PER_KWH)
    return float(TOU_OFFPEAK_BAHT_PER_KWH)


def _build_tou_price_series(duration_min=SIM_DURATION_MIN):
    # NEW/CHANGED: raw/effective TOU series aligned to SHIFT_START.
    raw = np.zeros(duration_min, dtype=float)
    effective = np.zeros(duration_min, dtype=float)
    for t in range(duration_min):
        p = float(get_tou_price(t))
        raw[t] = p
        if _is_in_solar_window(t):
            effective[t] = p * SOLAR_EFFECTIVE_PRICE_FACTOR
        else:
            effective[t] = p
    return raw, effective


def _is_in_solar_window(minute_idx):
    # NEW/CHANGED: solar window aligned to time-of-day.
    tod = int((SHIFT_START + int(minute_idx)) % 1440)
    if SOLAR_START <= SOLAR_END:
        return SOLAR_START <= tod < SOLAR_END
    return tod >= SOLAR_START or tod < SOLAR_END


def _effective_if_grid_kw(minute_idx, if_kw):
    # NEW/CHANGED: behind-the-meter solar offsets IF grid draw only.
    if_kw = float(max(0.0, if_kw))
    if _is_in_solar_window(minute_idx):
        return float(max(0.0, if_kw - SOLAR_OFFSET_KW))
    return float(if_kw)


def _effective_if_price(minute_idx, if_kw=None):
    # NEW/CHANGED: effective TOU seen by IF (solar-window discounted price).
    raw_price = float(get_tou_price(minute_idx))
    if _is_in_solar_window(minute_idx):
        return float(raw_price * SOLAR_EFFECTIVE_PRICE_FACTOR)
    return raw_price


def compute_md_15min_kw(total_plant_kw, interval=DEMAND_INTERVAL_MIN):
    """Maximum 15-min average demand (kW) from minute-resolution load."""
    x = np.asarray(total_plant_kw, dtype=float).ravel()
    if len(x) == 0:
        return 0.0
    window = max(1, int(interval))
    if len(x) < window:
        return float(np.mean(x))
    kernel = np.ones(window, dtype=float) / float(window)
    rolling = np.convolve(x, kernel, mode="valid")
    return float(np.max(rolling))


def compute_if_cost_share_day(if_kw_series, baseline_kw_series, tou_price_series):
    """Sanity check: IF cost share of total (IF + baseline) daily energy cost."""
    if_kw = np.asarray(if_kw_series, dtype=float).ravel()
    base_kw = np.asarray(baseline_kw_series, dtype=float).ravel()
    price = np.asarray(tou_price_series, dtype=float).ravel()
    n = min(len(if_kw), len(base_kw), len(price))
    if n <= 0:
        return 0.0, 0.0, 0.0

    if_cost = float(np.sum((if_kw[:n] / 60.0) * price[:n]))
    base_cost = float(np.sum((base_kw[:n] / 60.0) * price[:n]))
    denom = max(1e-9, if_cost + base_cost)
    share = float(if_cost / denom)
    return if_cost, base_cost, share


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
    # NEW/CHANGED: simple interpretable policy mode (12 vars) for smoother trade-offs.
    if SIMPLE_POLICY_MODE:
        x = np.asarray(x, dtype=float)
        lookahead_min = int(np.clip(round(np.clip(x[2], 0.0, 1.0) * 180.0), 0, 180))
        return {
            "trig_a_frac": float(np.clip(x[0], 0.0, 1.0)),
            "trig_b_frac": float(np.clip(x[1], 0.0, 1.0)),
            "L_trigger_A": float(np.clip(x[0], 0.0, 1.0) * MH_MAX_CAPACITY_KG["A"]),
            "L_trigger_B": float(np.clip(x[1], 0.0, 1.0) * MH_MAX_CAPACITY_KG["B"]),
            "lookahead_min": lookahead_min,
            "demand_headroom_kw": float(np.clip(x[3], 0.0, 1.0) * 100.0),
            "tou_weight": float(np.clip(x[4], 0.0, 2.0)),
            "peak_avoid_weight": float(np.clip(x[5], 0.0, 1.0)),
            "min_start_gap_min": float(np.clip(x[6], 0.0, 1.0) * 20.0),
            "alternation_bias": float(np.clip(x[7], -1.0, 1.0)),
            "force_urgency_threshold": float(np.clip(x[8], 0.70, 0.98)),
            "power_aggressiveness": float(np.clip(x[9], 0.0, 1.0)),
            "jit_slack_min": float(np.clip(x[10], 0.0, 1.0) * 60.0),  # NEW/CHANGED
            "start_delay_min": float(np.clip(x[11], 0.0, 1.0) * 60.0),  # NEW/CHANGED
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
            _quantize_to_step(
                p["jit_slack_min"], POLICY_KEY_STEPS["jit_slack_min"]
            ),  # NEW/CHANGED
            _quantize_to_step(
                p.get("start_delay_min", 0.0), POLICY_KEY_STEPS["start_delay_min"]
            ),  # NEW/CHANGED
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
        "jit_delay_penalty": float(
            m.get("jit_delay_minutes", 0.0) * JIT_DELAY_COST_PER_MIN
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


def compute_total_cost(m, comp):
    # Energy mode objective: focus on electricity bill components.
    service_fee_day_equiv = (
        float(m.get("service_fee_day_equiv", 0.0))
        if INCLUDE_SERVICE_FEE_IN_ENERGY_OBJECTIVE
        else 0.0
    )
    makespan_term = ENERGY_MAKESPAN_COST_PER_MIN * float(
        m.get("makespan_minutes", SIM_DURATION_MIN)
    )
    idle_min = float(m.get("if_idle_minutes_total", 0.0))
    idle_super = float(m.get("if_idle_gap_superlinear_minutes", idle_min))
    idle_gap_term = (
        ENERGY_IDLE_GAP_COST_PER_MIN * idle_min
        + ENERGY_IDLE_GAP_SUPERLINEAR_COEFF * idle_super
    )
    cheap_melt_credit = ENERGY_CHEAP_MELT_CREDIT_PER_MIN * float(
        m.get("cheap_melt_minutes", 0.0)
    )
    return float(
        m.get("total_energy_cost_day", m.get("total_energy_cost", 0.0))
        + m.get("demand_charge_day_equiv", m.get("demand_penalty", 0.0))
        + service_fee_day_equiv
        + makespan_term
        + idle_gap_term
        - cheap_melt_credit
    )


def _violation_vector(m):
    overflow_total = float(m.get("overflow_kg_total", 0.0))
    if not np.isfinite(overflow_total) or overflow_total < 0.0:
        overflow_total = 0.0
    overflow_violation = float(overflow_total - MAX_OVERFLOW_KG_ALLOW)

    poured_batches = int(m.get("poured_batches_count", 0))
    target_batches = int(m.get("target_num_batches", _current_num_batches()))
    missing_batches = float(max(0, target_batches - poured_batches))
    # Constraints are the same for both modes.
    return np.array([overflow_violation, missing_batches], dtype=float)


class PolicyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        xa = a.get("X") if hasattr(a, "get") else a.X
        xb = b.get("X") if hasattr(b, "get") else b.X
        return _policy_cache_key(xa) == _policy_cache_key(xb)


class StagnationEarlyStopCallback(Callback):
    # NEW/CHANGED: single-objective stagnation callback for GA.
    def __init__(self, patience_gens, delta_obj=0.0):
        super().__init__()
        self.patience_gens = int(patience_gens)
        self.delta_obj = float(delta_obj)
        self.best_obj = None
        self.stagnant_gens = 0

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        if F is None or len(F) == 0:
            return

        cur_obj = float(np.min(F[:, 0]))

        if self.best_obj is None:
            self.best_obj = cur_obj
            self.stagnant_gens = 0
            return

        improved = cur_obj < (self.best_obj - self.delta_obj)
        if improved:
            self.best_obj = min(self.best_obj, cur_obj)
            self.stagnant_gens = 0
        else:
            self.stagnant_gens += 1

        if self.stagnant_gens >= self.patience_gens:
            print(
                f"Early stop: no objective improvement in {self.stagnant_gens} generations."
            )
            algorithm.termination.force_termination = True


def _build_policy_state(
    policy,
    mh_levels,
    baseline_kw_t,
    if_kw_now,
    queue_len,
    effective_price_ref_t,
    remaining_batches,
    remaining_time_min,
    minute_idx,
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
            effective_price_ref_t
            / (max(TOU_ONPEAK_BAHT_PER_KWH, TOU_OFFPEAK_BAHT_PER_KWH) + 1e-9),
            0.0,
            1.0,
        )
    )
    urgency = float(max(0.0, remaining_batches) / max(1.0, remaining_time_min))
    urgency_per_hour = float(
        np.clip(
            (max(0.0, remaining_batches) * 60.0) / max(1.0, remaining_time_min),
            0.0,
            4.0,
        )
    )
    tod = int((SHIFT_START + int(minute_idx)) % 1440)
    post_solar = float(tod >= SOLAR_END)
    return {
        "eta_min": eta_min,
        "depletion_urgency": depletion_urgency,
        "avg_level_frac": avg_level_frac,
        "trigger_hit": trigger_hit,
        "queue_pressure": queue_pressure,
        "margin_kw": margin_kw,
        "margin_ratio": margin_ratio,
        "price_norm": price_norm,
        "urgency": urgency,
        "urgency_per_hour": urgency_per_hour,
        "post_solar": post_solar,
    }


def _start_score(policy, state):
    if SIMPLE_POLICY_MODE:
        # Simpler score; energy mode uses stronger TOU bias.
        score = 0.0
        if OPT_MODE == "energy":
            score += ENERGY_MODE_START_W_DEP * state["depletion_urgency"]
            score += ENERGY_MODE_START_W_TRIGGER * state["trigger_hit"]
            score += ENERGY_MODE_START_W_QUEUE * state["queue_pressure"]
            score += 3.0 * state.get("urgency", 0.0)
            score += ENERGY_MODE_START_W_URGENCY_PER_HOUR * state.get(
                "urgency_per_hour", 0.0
            )
            score += (
                ENERGY_MODE_START_W_POST_SOLAR_URGENCY
                * state.get("post_solar", 0.0)
                * state.get("urgency_per_hour", 0.0)
            )
            score += ENERGY_MODE_START_W_INV_FLOOR * max(
                0.0, ENERGY_MODE_INV_FLOOR_FRAC - state["avg_level_frac"]
            )
            score -= (
                policy["peak_avoid_weight"]
                * ENERGY_MODE_START_W_PEAK
                * max(0.0, -state["margin_ratio"])
            )
            score -= (
                policy["tou_weight"] * ENERGY_MODE_START_W_TOU * state["price_norm"]
            )
        else:
            score += 2.2 * state["depletion_urgency"]
            score += 1.0 * state["trigger_hit"]
            score += 0.7 * state["queue_pressure"]
            score -= (
                policy["peak_avoid_weight"] * 2.0 * max(0.0, -state["margin_ratio"])
            )
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
    # NEW/CHANGED: use effective solar-adjusted IF price in power selection.
    soft_guard = CONTRACT_DEMAND_KW - policy["demand_headroom_kw"]
    wait_weight = 1.0 - policy["wait_vs_rush"]
    peak_weight = policy["peak_avoid_weight"]
    if force_mode:
        # NEW/CHANGED: when forcing feasibility, de-emphasize TOU/peak and push power.
        wait_weight *= 0.25
        peak_weight *= 0.35
        urgency_pref = min(
            2, urgency_pref + int(policy.get("power_aggressiveness", 0.5) >= 0.4)
        )

    for idx, p in enumerate(IF_POWER_OPTIONS):
        # Encourage high power when urgent, low power when not urgent.
        urgency_score = abs(idx - urgency_pref) * (
            120.0 + 220.0 * policy["wait_vs_rush"]
        )
        projected_kw = baseline_kw_t + if_kw_now + p
        soft_peak = max(0.0, projected_kw - soft_guard)
        peak_score = peak_weight * (soft_peak**2) / 50.0
        eff_price_t = _effective_if_price(minute_idx, p)
        tou_score = wait_weight * policy["tou_weight"] * eff_price_t * (p / 60.0)
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


def _idle_gap_superlinear_minutes_from_intervals(intervals):
    """Penalize long idle gaps more than short gaps (thermal/cold-start proxy)."""
    if not intervals:
        return 0.0
    items = sorted(intervals, key=lambda w: w[0])
    merged = [[items[0][0], items[0][1]]]
    for s, e in items[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    penalty_minutes = 0.0
    for i in range(1, len(merged)):
        gap = max(0.0, merged[i][0] - merged[i - 1][1])
        # Superlinear weighting for longer idle stretches.
        penalty_minutes += gap * (1.0 + gap / 60.0)
    return float(penalty_minutes)


def _pour_ready_eta_min(current_level, downtime_remaining):
    # free capacity (physical space) should NOT become 0 just because of downtime
    cap_free_now = (MH_MAX_CAPACITY_KG["A"] - current_level["A"]) + (
        MH_MAX_CAPACITY_KG["B"] - current_level["B"]
    )

    need_cap = max(0.0, IF_BATCH_OUTPUT_KG - cap_free_now)

    total_cons = (
        MH_CONSUMPTION_RATE_KG_PER_MIN["A"] + MH_CONSUMPTION_RATE_KG_PER_MIN["B"]
    )
    time_to_free = need_cap / max(total_cons, 1e-9)

    # downtime just means "cannot pour until this many minutes pass"
    dt_downtime = float(max(downtime_remaining["A"], downtime_remaining["B"]))
    return float(max(time_to_free, dt_downtime))


def quick_capacity_sanity_check():
    """Coarse feasibility check for finishing NUM_BATCHES within the day."""
    target_batches = _current_num_batches()
    min_melt_duration = min(v["duration_min_hot"] for v in POWER_PROFILE.values())
    available_if = int(USE_FURNACE_A) + int(USE_FURNACE_B)
    parallel_if_slots = available_if if ALLOW_PARALLEL_IF else min(1, available_if)

    batches_time_limit = int(
        (SIM_DURATION_MIN / max(min_melt_duration, 1e-9)) * parallel_if_slots
    )
    # Receiver-side bottleneck: need 500 kg free capacity before each pour.
    total_consume_rate = (
        MH_CONSUMPTION_RATE_KG_PER_MIN["A"] + MH_CONSUMPTION_RATE_KG_PER_MIN["B"]
    )
    minutes_to_free_one_batch = IF_BATCH_OUTPUT_KG / max(total_consume_rate, 1e-9)
    batches_receive_limit = int(SIM_DURATION_MIN / max(minutes_to_free_one_batch, 1e-9))
    effective_batch_limit = min(batches_time_limit, batches_receive_limit)
    is_feasible = bool(target_batches <= effective_batch_limit)

    report = {
        "target_num_batches": int(target_batches),
        "time_limit_batches": int(batches_time_limit),
        "receive_limit_batches": int(batches_receive_limit),
        "effective_batch_limit": int(effective_batch_limit),
        "is_feasible_coarse": is_feasible,
    }
    return report


def print_capacity_sanity_report(report):
    print("\n=== Capacity Sanity Check (coarse) ===")
    print("Target NUM_BATCHES:", report["target_num_batches"])
    print(
        "Batch limits [time, receive, effective_limit]:",
        report["time_limit_batches"],
        report["receive_limit_batches"],
        report["effective_batch_limit"],
    )
    print("Coarse feasibility:", "PASS" if report["is_feasible_coarse"] else "FAIL")


def simulate_policy_day(policy_params, controller=None):
    target_num_batches = _current_num_batches()
    if isinstance(policy_params, dict):
        policy = dict(policy_params)
    elif policy_params is None:
        n = 12 if SIMPLE_POLICY_MODE else 21
        policy = _decode_policy_vector(np.zeros(n, dtype=float))
    else:
        policy = _decode_policy_vector(policy_params)
    baseline_kw = _build_baseline_load_kw(SIM_DURATION_MIN)
    tou_raw_price_series, tou_effective_price_series = _build_tou_price_series(
        SIM_DURATION_MIN
    )  # NEW/CHANGED

    if_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    melt_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    reheat_kw_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    total_plant_kw = np.zeros(SIM_DURATION_MIN, dtype=float)
    energy_cost_series = np.zeros(SIM_DURATION_MIN, dtype=float)
    reheat_energy_cost_series = np.zeros(SIM_DURATION_MIN, dtype=float)  # NEW/CHANGED
    solar_melt_minutes = 0.0  # NEW/CHANGED
    cheap_melt_minutes = 0.0
    solar_cost_saving = 0.0  # NEW/CHANGED

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
    zero_minutes_total = 0.0  # NEW/CHANGED
    eps_deficit_area = 0.0  # NEW/CHANGED
    min_level_reached = {"A": current_level["A"], "B": current_level["B"]}

    overflow_kg_total = 0.0
    unpoured_batches = []
    last_if_release_time = {0: None, 1: None}
    furnace_use_count = {0: 0, 1: 0}
    if_active_intervals = []
    last_start_min = -1_000_000
    parallel_peak_minutes = 0.0
    prep_remaining = {
        0: 0,
        1: 0,
    }  # NEW/CHANGED: prep load remaining minutes per IF furnace.
    prep_wait_minutes = 0.0  # NEW/CHANGED
    start_blocked_by_prep_count = 0  # NEW/CHANGED
    forced_start_count = 0  # NEW/CHANGED
    non_alternation_count = 0  # NEW/CHANGED
    alternation_count = 0  # NEW/CHANGED
    last_started_furnace = None  # NEW/CHANGED
    jit_delay_minutes = 0.0  # NEW/CHANGED
    delay_reason_counts = {}  # NEW/CHANGED
    min_duration_per_batch = float(
        min(v["duration_min_hot"] for v in POWER_PROFILE.values())
    )  # NEW/CHANGED

    def _advance_minute_dynamics(t):
        nonlocal mh_low_level_penalty
        nonlocal zero_minutes_total
        nonlocal eps_deficit_area
        nonlocal solar_cost_saving
        nonlocal solar_melt_minutes
        nonlocal cheap_melt_minutes
        nonlocal parallel_peak_minutes

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

        # service metrics after per-minute consumption update
        if (
            current_level["A"] <= MH_EMPTY_THRESHOLD_KG
            or current_level["B"] <= MH_EMPTY_THRESHOLD_KG
        ):
            zero_minutes_total += 1.0
        eps_deficit_area += max(0.0, EPS_LEVEL - current_level["A"]) + max(
            0.0, EPS_LEVEL - current_level["B"]
        )

        total_plant_kw[t] = baseline_kw[t] + minute_if_kw
        raw_price_t = tou_raw_price_series[t]
        eff_price_t = tou_effective_price_series[t]
        energy_cost_series[t] = (minute_if_kw / 60.0) * eff_price_t
        reheat_energy_cost_series[t] = (minute_reheat_kw / 60.0) * eff_price_t
        solar_cost_saving += (minute_if_kw / 60.0) * max(0.0, raw_price_t - eff_price_t)
        if minute_melt_kw > 1e-9 and _is_in_solar_window(t):
            solar_melt_minutes += 1.0
        if minute_melt_kw > 1e-9 and eff_price_t <= (TOU_OFFPEAK_BAHT_PER_KWH + 1e-9):
            cheap_melt_minutes += 1.0
        active_if_count = sum(
            1 for f in available_if_furnaces if if_states[f]["active"]
        )
        if active_if_count >= 2 and total_plant_kw[t] > (0.95 * CONTRACT_DEMAND_KW):
            parallel_peak_minutes += 1.0

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

        # NEW/CHANGED: reduce prep remaining when another furnace is actively melting.
        melting_furnaces = [
            f_idx
            for f_idx in available_if_furnaces
            if if_states[f_idx]["active"] and if_states[f_idx]["status"] == "melting"
        ]
        if ENABLE_PREP_MODEL:
            for f_idx in available_if_furnaces:
                if if_states[f_idx]["active"]:
                    continue
                if prep_remaining[f_idx] <= 0:
                    continue
                if any(mf != f_idx for mf in melting_furnaces):
                    prep_remaining[f_idx] = max(0, prep_remaining[f_idx] - 1)

        # 2) pour with feasibility repair (no overflow)
        keep_pouring = True
        while keep_pouring and ready_to_pour_queue:
            keep_pouring = False
            b_id = ready_to_pour_queue[0]

            # NEW/CHANGED: hard "no partial pour" -> pour only when full 500kg can be received now.
            available_A = (
                MH_MAX_CAPACITY_KG["A"] - current_level["A"]
                if downtime_remaining["A"] <= 0
                else 0.0
            )
            available_B = (
                MH_MAX_CAPACITY_KG["B"] - current_level["B"]
                if downtime_remaining["B"] <= 0
                else 0.0
            )
            total_available = available_A + available_B
            if total_available < IF_BATCH_OUTPUT_KG:
                break

            remaining = IF_BATCH_OUTPUT_KG
            fill_order = (
                ["A", "B"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "A" else ["B", "A"]
            )
            poured_A = 0.0
            poured_B = 0.0
            for f_id in fill_order:
                if remaining <= 0:
                    break
                if downtime_remaining[f_id] > 0:
                    continue
                space = MH_MAX_CAPACITY_KG[f_id] - current_level[f_id]
                put = min(remaining, space)
                if put > 0:
                    if f_id == "A":
                        poured_A += put
                    else:
                        poured_B += put
                    remaining -= put
            if remaining > 1e-9:
                # Should be unreachable due to total_available guard, keep robust.
                break

            current_level["A"] = np.clip(
                current_level["A"] + poured_A, 0.0, MH_MAX_CAPACITY_KG["A"]
            )
            current_level["B"] = np.clip(
                current_level["B"] + poured_B, 0.0, MH_MAX_CAPACITY_KG["B"]
            )

            if poured_A > 0:
                downtime_remaining["A"] = POST_POUR_DOWNTIME_MIN  # NEW/CHANGED
            if poured_B > 0:
                downtime_remaining["B"] = POST_POUR_DOWNTIME_MIN  # NEW/CHANGED

            batches[b_id]["poured_A_kg"] = poured_A
            batches[b_id]["poured_B_kg"] = poured_B
            batches[b_id]["overflow_kg"] = 0.0
            batches[b_id]["pour_min"] = t
            batches[b_id]["status"] = "poured"

            f_idx = batches[b_id]["if_furnace"]
            if_states[f_idx] = {"active": False, "status": "idle", "batch_id": None}
            if ENABLE_PREP_MODEL:
                prep_remaining[f_idx] = (
                    PREP_LOAD_TIME_MIN  # NEW/CHANGED: prep starts after pour releases furnace.
                )
            last_if_release_time[f_idx] = t
            actual_pour_events.append((t, b_id))
            ready_to_pour_queue.pop(0)
            keep_pouring = True

        # 3) holding minutes only for batches that still could not pour.
        total_if_holding_minutes += len(ready_to_pour_queue)

        # 4) policy start decision (score-based + dynamic spacing + demand-gating)
        if batch_id_counter < target_num_batches:
            remaining_time_min = float(SIM_DURATION_MIN - t)  # NEW/CHANGED
            remaining_batches = int(
                max(0, target_num_batches - batch_id_counter)
            )  # NEW/CHANGED
            force_buffer = min_duration_per_batch + (
                PREP_LOAD_TIME_MIN * 0.5 if ENABLE_PREP_MODEL else 0.0
            )  # NEW/CHANGED
            force_mode = bool(
                remaining_batches > 0
                and remaining_time_min < remaining_batches * force_buffer
            )  # NEW/CHANGED

            any_if_active = any(if_states[f]["active"] for f in available_if_furnaces)
            any_holding_active = any(
                if_states[f]["active"] and if_states[f]["status"] == "holding"
                for f in available_if_furnaces
            )  # NEW/CHANGED: hard guard - no new start while any IF is holding.
            idle_all = [f for f in available_if_furnaces if not if_states[f]["active"]]
            if idle_all:
                if_kw_now = 0.0
                for f in available_if_furnaces:
                    st = if_states[f]
                    if st["active"]:
                        if_kw_now += batches[st["batch_id"]]["power_kw"]
                t_end = min(
                    SIM_DURATION_MIN, t + max(1, int(ENERGY_PRICE_LOOKAHEAD_MIN))
                )
                effective_price_ref = float(
                    np.mean(tou_effective_price_series[t:t_end])
                )
                state = _build_policy_state(
                    policy,
                    current_level,
                    baseline_kw[t],
                    if_kw_now,
                    len(ready_to_pour_queue),
                    effective_price_ref,
                    remaining_batches,
                    remaining_time_min,
                    t,
                )
                start_score = _start_score(policy, state)
                min_gap_now = _dynamic_min_start_gap(policy, state)
                gap_ok = (t - last_start_min) >= min_gap_now
                can_parallel_now = _parallel_allowed(policy, state, start_score)
                controller_decision = {}
                if controller is not None:
                    controller_state = {
                        "t": t,
                        "policy_state": state,
                        "current_level": {
                            "A": current_level["A"],
                            "B": current_level["B"],
                        },
                        "baseline_kw": float(baseline_kw[t]),
                        "if_kw_now": float(if_kw_now),
                        "tou_effective_price": float(tou_effective_price_series[t]),
                        "remaining_batches": int(
                            max(0, target_num_batches - batch_id_counter)
                        ),
                        "ready_queue_len": int(len(ready_to_pour_queue)),
                        "idle_furnaces": list(idle_all),
                        "available_if_furnaces": list(available_if_furnaces),
                        "any_if_active": bool(any_if_active),
                        "any_holding_active": bool(any_holding_active),
                        "last_started_furnace": last_started_furnace,
                        "force_mode": bool(force_mode),
                        "gap_ok": bool(gap_ok),
                        "sim_duration_min": int(SIM_DURATION_MIN),
                        "num_batches": int(target_num_batches),
                    }
                    try:
                        controller_decision = controller(controller_state) or {}
                    except Exception as exc:
                        controller_decision = {
                            "start_allowed": False,
                            "delay_reason": f"controller_error:{type(exc).__name__}",
                        }
                    if controller_decision.get("force_mode_override") is not None:
                        force_mode = bool(
                            controller_decision.get("force_mode_override")
                        )
                if (not any_if_active) or can_parallel_now:
                    if controller is None:
                        # Keep one primary reason per minute to avoid noisy overlap.
                        block_reason = None
                        if not force_mode:
                            if t < int(policy.get("start_delay_min", 0.0)):
                                block_reason = "start_delay_block"
                            elif any_holding_active and (not ALLOW_PARALLEL_IF):
                                block_reason = "holding_guard"
                            elif not gap_ok:
                                block_reason = "gap_block"
                            elif start_score < 0.0:
                                block_reason = "tou_negative"
                        start_allowed = force_mode or (block_reason is None)
                        if block_reason is not None:
                            delay_reason_counts[block_reason] = (
                                delay_reason_counts.get(block_reason, 0) + 1
                            )
                    else:
                        start_allowed = bool(
                            controller_decision.get("start_allowed", False)
                        )
                        if (
                            any_holding_active
                            and (not ALLOW_PARALLEL_IF)
                            and (not force_mode)
                        ):
                            start_allowed = False
                            controller_decision["delay_reason"] = "holding_guard"
                    if start_allowed:
                        idle_ready = [
                            f
                            for f in idle_all
                            if (not ENABLE_PREP_MODEL) or prep_remaining[f] <= 0
                        ]  # NEW/CHANGED
                        if not idle_ready:
                            start_blocked_by_prep_count += 1  # NEW/CHANGED
                            if ENABLE_PREP_MODEL:
                                prep_wait_minutes += float(
                                    max(0, min(prep_remaining[f] for f in idle_all))
                                )  # NEW/CHANGED
                            delay_reason_counts["prep_not_ready"] = (
                                delay_reason_counts.get("prep_not_ready", 0) + 1
                            )
                            _advance_minute_dynamics(t)
                            continue

                        if (
                            force_mode
                            and ENABLE_PREP_MODEL
                            and ENFORCE_ALTERNATION_FLAG
                            and len(idle_ready) > 1
                            and last_started_furnace in idle_ready
                        ):
                            alt_idle = [
                                f for f in idle_ready if f != last_started_furnace
                            ]
                            if alt_idle:
                                idle_ready = alt_idle + [
                                    f for f in idle_ready if f not in alt_idle
                                ]  # NEW/CHANGED

                        chosen_if = None
                        if controller is not None:
                            req_if = controller_decision.get("chosen_if")
                            if req_if in idle_ready:
                                chosen_if = int(req_if)
                        if chosen_if is None:
                            chosen_if = _select_if_furnace(
                                policy,
                                idle_ready,
                                current_level,
                                t,
                                last_if_release_time,
                                furnace_use_count,
                                last_started_furnace=last_started_furnace,
                            )

                        blocked_by_prep = (
                            ENABLE_PREP_MODEL and prep_remaining.get(chosen_if, 0) > 0
                        )
                        if blocked_by_prep:
                            start_blocked_by_prep_count += 1  # NEW/CHANGED
                            prep_wait_minutes += float(
                                prep_remaining[chosen_if]
                            )  # NEW/CHANGED
                        if not blocked_by_prep:
                            gap = (
                                None
                                if last_if_release_time[chosen_if] is None
                                else t - last_if_release_time[chosen_if]
                            )
                            is_cold_start = (
                                gap is None or gap >= COLD_START_GAP_THRESHOLD_MIN
                            )

                            selected_power = None
                            if controller is not None and (
                                controller_decision.get("selected_power") is not None
                            ):
                                req_p = float(controller_decision.get("selected_power"))
                                selected_power = min(
                                    IF_POWER_OPTIONS,
                                    key=lambda v: abs(float(v) - req_p),
                                )
                            if selected_power is None:
                                selected_power = _select_if_power(
                                    policy,
                                    current_level,
                                    baseline_kw[t],
                                    if_kw_now,
                                    t,
                                    force_mode=force_mode,
                                )
                            # NEW/CHANGED: strict JIT gate from pour-ready ETA.
                            pour_ready_eta_min = _pour_ready_eta_min(
                                current_level, downtime_remaining
                            )
                            est_duration = _compute_batch_profile(
                                selected_power, is_cold_start, chosen_if
                            )["duration_min"]
                            hold_risk = max(0.0, pour_ready_eta_min - est_duration)
                            jit_slack = float(policy.get("jit_slack_min", 0.0))
                            bypass_jit_gate = bool(
                                controller_decision.get("bypass_jit_gate", False)
                            )
                            service_replenish_urgent = bool(
                                OPT_MODE == "service"
                                and (
                                    state.get("depletion_urgency", 0.0) >= 0.75
                                    or current_level["A"]
                                    <= MH_MIN_OPERATIONAL_LEVEL_KG["A"]
                                    or current_level["B"]
                                    <= MH_MIN_OPERATIONAL_LEVEL_KG["B"]
                                )
                            )
                            if (not bypass_jit_gate) and OPT_MODE == "service":
                                # NEW/CHANGED: stricter JIT in service mode to reduce holding/reheat.
                                jit_slack = min(jit_slack, 5.0)
                            elif (not bypass_jit_gate) and OPT_MODE == "energy":
                                # Energy mode: allow wider JIT slack so schedules can shift toward
                                # cheap windows without being over-blocked by minute-level JIT gate.
                                jit_slack = max(
                                    jit_slack, ENERGY_MODE_MIN_JIT_SLACK_MIN
                                )
                            if (
                                (not bypass_jit_gate)
                                and (not force_mode)
                                and (not service_replenish_urgent)
                                and hold_risk > jit_slack
                            ):
                                jit_delay_minutes += 1.0
                                delay_reason_counts["jit_gate"] = (
                                    delay_reason_counts.get("jit_gate", 0) + 1
                                )
                                _advance_minute_dynamics(t)
                                continue
                            projected_total = (
                                baseline_kw[t] + if_kw_now + selected_power
                            )
                            if projected_total > CONTRACT_DEMAND_KW + 1e-9:
                                # Hard contract-demand ceiling (no exceed).
                                delay_reason_counts["demand_guard"] = (
                                    delay_reason_counts.get("demand_guard", 0) + 1
                                )
                                _advance_minute_dynamics(t)
                                continue
                            hard_guard = (
                                CONTRACT_DEMAND_KW - policy["demand_headroom_kw"]
                            )
                            shortage_override = force_mode and state[
                                "depletion_urgency"
                            ] >= policy.get("force_urgency_threshold", 0.92)
                            service_urgency_override = bool(
                                OPT_MODE == "service"
                                and (
                                    state.get("depletion_urgency", 0.0) >= 0.75
                                    or current_level["A"]
                                    <= MH_MIN_OPERATIONAL_LEVEL_KG["A"]
                                    or current_level["B"]
                                    <= MH_MIN_OPERATIONAL_LEVEL_KG["B"]
                                )
                            )
                            bypass_demand_guard = bool(
                                controller_decision.get("bypass_demand_guard", False)
                            )
                            if (
                                bypass_demand_guard
                                or projected_total <= hard_guard
                                or shortage_override
                                or service_urgency_override
                            ):
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
                                    # NEW/CHANGED: do not set prep for other on start; prep accrues via melting minute updates.
                                    prep_remaining[other_f] = max(
                                        0, prep_remaining.get(other_f, 0)
                                    )
                            else:
                                delay_reason_counts["demand_guard"] = (
                                    delay_reason_counts.get("demand_guard", 0) + 1
                                )
                    else:
                        if controller is None:
                            reason = "policy_not_start"
                        else:
                            reason = controller_decision.get(
                                "delay_reason", "controller_not_start"
                            )
                        delay_reason_counts[reason] = (
                            delay_reason_counts.get(reason, 0) + 1
                        )
                else:
                    delay_reason_counts["active_if_no_parallel"] = (
                        delay_reason_counts.get("active_if_no_parallel", 0) + 1
                    )

        _advance_minute_dynamics(t)

    # NEW/CHANGED: ensure missing non-started batches are explicitly represented.
    # Without this, GA can "skip starting" some batches and avoid overflow accounting.
    if batch_id_counter < target_num_batches:
        for b_id in range(batch_id_counter + 1, target_num_batches + 1):
            if b_id in batches:
                continue
            batches[b_id] = {
                "batch_id": b_id,
                "if_furnace": -1,
                "start_min": SIM_DURATION_MIN,
                "melt_finish_min": SIM_DURATION_MIN,
                "pour_min": None,
                "power_kw": 0.0,
                "is_cold_start": False,
                "status": "unpoured",
                "poured_A_kg": 0.0,
                "poured_B_kg": 0.0,
                "overflow_kg": IF_BATCH_OUTPUT_KG,
            }

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
    total_energy_cost_day = float(np.sum(energy_cost_series))
    reheat_energy_cost = float(np.sum(reheat_energy_cost_series))  # NEW/CHANGED

    peak_kw = float(np.max(total_plant_kw)) if len(total_plant_kw) else 0.0
    md_15_kw = float(compute_md_15min_kw(total_plant_kw, interval=DEMAND_INTERVAL_MIN))
    baseline_md15_kw = float(
        compute_md_15min_kw(baseline_kw, interval=DEMAND_INTERVAL_MIN)
    )
    demand_penalty_kw = float(max(0.0, md_15_kw - baseline_md15_kw))
    demand_charge_month = float(demand_penalty_kw * DEMAND_CHARGE_BAHT_PER_KW_MONTH)
    demand_charge_day_equiv = float(
        demand_charge_month / max(1.0, BILLING_DAYS_PER_MONTH)
    )
    service_fee_day_equiv = float(
        SERVICE_FEE_BAHT_PER_MONTH / max(1.0, BILLING_DAYS_PER_MONTH)
    )
    demand_excess = max(0.0, md_15_kw - CONTRACT_DEMAND_KW)
    demand_penalty = demand_charge_day_equiv  # backward-compatible alias

    # CHANGED (Step B): compute poured kg from actual poured mass per batch.
    total_poured_kg = float(
        sum(
            b.get("poured_A_kg", 0.0) + b.get("poured_B_kg", 0.0)
            for b in batches.values()
            if b.get("status") == "poured"
        )
    )
    poured_batches_count = int(
        sum(1 for b in batches.values() if b.get("status") == "poured")
    )  # NEW/CHANGED
    missing_batches = int(
        max(0, target_num_batches - poured_batches_count)
    )  # NEW/CHANGED
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
    if_idle_gap_superlinear_minutes = _idle_gap_superlinear_minutes_from_intervals(
        if_active_intervals
    )
    # Sanity: IF share of daily energy cost against baseline energy cost.
    effective_if_grid_kw_series = np.array(
        [_effective_if_grid_kw(ti, if_kw_series[ti]) for ti in range(SIM_DURATION_MIN)],
        dtype=float,
    )
    energy_cost_if_day, energy_cost_baseline_day, share_if_cost = (
        compute_if_cost_share_day(
            effective_if_grid_kw_series, baseline_kw, tou_effective_price_series
        )
    )

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
        "tou_raw_price": tou_raw_price_series,  # NEW/CHANGED
        "tou_effective_price": tou_effective_price_series,  # NEW/CHANGED
        "total_plant_kw": total_plant_kw,
        "metrics": {
            "melt_kwh": total_melt_kwh,
            "reheat_kwh": total_reheat_kwh,
            "reheat_energy_cost": reheat_energy_cost,  # NEW/CHANGED
            "total_if_kwh": total_if_kwh,
            "total_energy_cost": total_energy_cost_day,  # backward-compatible alias
            "total_energy_cost_day": total_energy_cost_day,
            "peak_kw": peak_kw,
            "md_15_kw": md_15_kw,
            "baseline_md15_kw": baseline_md15_kw,
            "demand_penalty_kw": demand_penalty_kw,
            "demand_charge_month": demand_charge_month,
            "demand_charge_day_equiv": demand_charge_day_equiv,
            "service_fee_day_equiv": service_fee_day_equiv,
            "demand_excess_kw": demand_excess,
            "demand_penalty": demand_penalty,  # backward-compatible alias
            "energy_cost_if_day": energy_cost_if_day,
            "energy_cost_baseline_day": energy_cost_baseline_day,
            "share_if_cost": share_if_cost,
            "mh_empty_minutes": mh_empty_minutes,
            "mh_low_level_minutes": mh_low_level_minutes,
            "mh_low_level_penalty": mh_low_level_penalty,
            "zero_minutes_total": float(zero_minutes_total),  # NEW/CHANGED
            "eps_deficit_area": float(eps_deficit_area),  # NEW/CHANGED
            "overflow_kg_total": overflow_kg_total,
            "holding_minutes_total": float(total_if_holding_minutes),
            "unpoured_batches_count": len(unpoured_batches),
            "poured_batches_count": poured_batches_count,  # NEW/CHANGED
            "target_num_batches": int(target_num_batches),
            "missing_batches": missing_batches,  # NEW/CHANGED
            "makespan_minutes": makespan_minutes,
            "total_poured_kg": total_poured_kg,  # CHANGED (Step A)
            "prep_wait_minutes": float(prep_wait_minutes),  # NEW/CHANGED
            "non_alternation_count": float(non_alternation_count),  # NEW/CHANGED
            "alternation_ratio": alternation_ratio,  # NEW/CHANGED
            "forced_start_count": float(forced_start_count),  # NEW/CHANGED
            "start_blocked_by_prep_count": float(
                start_blocked_by_prep_count
            ),  # NEW/CHANGED
            "jit_delay_minutes": float(jit_delay_minutes),  # NEW/CHANGED
            "solar_melt_minutes": float(solar_melt_minutes),  # NEW/CHANGED
            "cheap_melt_minutes": float(cheap_melt_minutes),
            "solar_cost_saving": float(solar_cost_saving),  # NEW/CHANGED
            "start_delay_min": float(policy.get("start_delay_min", 0.0)),  # NEW/CHANGED
            "min_level_reached": min_level_reached,
            "if_use_count_A": furnace_use_count[0],
            "if_use_count_B": furnace_use_count[1],
            "if_idle_minutes_total": total_if_idle_minutes,
            "if_idle_gap_superlinear_minutes": if_idle_gap_superlinear_minutes,
            "parallel_peak_minutes": float(parallel_peak_minutes),
            "controller_name": (
                getattr(controller, "__name__", str(controller))
                if controller is not None
                else "ga_policy"
            ),
            "delay_reason_counts": dict(delay_reason_counts),
        },
    }


def evaluate_policy(policy_params):
    sim = simulate_policy_day(policy_params)
    m = sim["metrics"]

    comp = _compute_obj1_components(m)
    total_cost_energy = compute_total_cost(m, comp)
    low_level_minutes_total = float(
        m["mh_low_level_minutes"]["A"] + m["mh_low_level_minutes"]["B"]
    )
    zero_minutes_total = float(m.get("zero_minutes_total", 0.0))
    eps_deficit_area = float(m.get("eps_deficit_area", 0.0))
    holding_minutes_total = float(m.get("holding_minutes_total", 0.0))
    reheat_kwh = float(m.get("reheat_kwh", 0.0))
    if OPT_MODE == "energy":
        f_scalar = float(total_cost_energy)
    elif OPT_MODE == "service":
        # NEW/CHANGED: service as soft objective (best-effort even if strict feasibility impossible).
        tmp_g = _violation_vector(m)
        overflow_v = float(max(0.0, tmp_g[0]))
        missing_v = float(max(0.0, tmp_g[1]))
        f_scalar = float(
            SERVICE_W_MISSING_BATCHES * missing_v
            + SERVICE_W_OVERFLOW * overflow_v
            + SERVICE_W_ZERO_MINUTES * zero_minutes_total
            + SERVICE_W_EPS_DEFICIT_AREA * eps_deficit_area
            + SERVICE_W_LOW_LEVEL_MINUTES * low_level_minutes_total
            + SERVICE_W_HOLDING_MINUTES * holding_minutes_total
            + SERVICE_W_REHEAT_KWH * reheat_kwh
            + SERVICE_W_ENERGY_TIEBREAKER * m["total_energy_cost"]
        )
    else:
        raise ValueError(f"Invalid OPT_MODE: {OPT_MODE}")

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

    g_constraints = _violation_vector(m)
    overflow_violation = float(max(0.0, g_constraints[0]))
    missing_batches_violation = float(max(0.0, g_constraints[1]))
    min_level_a = float(m["min_level_reached"]["A"])
    min_level_b = float(m["min_level_reached"]["B"])
    zero_level_violation = float(max(0.0, EPS_LEVEL - min(min_level_a, min_level_b)))
    total_violation = float(np.sum(np.maximum(0.0, g_constraints)))

    cost_components = {
        "opt_mode": OPT_MODE,  # NEW/CHANGED
        "objective_total_cost": f_scalar,
        "energy_mode_total_cost": total_cost_energy,  # NEW/CHANGED
        "obj1_total": f_scalar,
        "obj1_raw_total": raw_obj1_total,
        "obj1_mode": OBJ1_AGGREGATION_MODE,
        "obj1_normalized_sum": normalized_sum,
        "obj2_makespan": float(m["makespan_minutes"]),
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
        "comp_jit_delay_penalty": comp["jit_delay_penalty"],  # NEW/CHANGED
        "pct_energy_cost": contribution_pct["energy_cost"],
        "pct_demand_penalty": contribution_pct["demand_penalty"],
        "pct_holding_penalty": contribution_pct["holding_penalty"],
        "pct_empty_penalty": contribution_pct["empty_penalty"],
        "pct_low_level_min_penalty": contribution_pct["low_level_min_penalty"],
        "pct_low_level_shape_penalty": contribution_pct["low_level_shape_penalty"],
        "pct_overflow_penalty": contribution_pct["overflow_penalty"],
        "pct_unpoured_penalty": contribution_pct["unpoured_penalty"],
        "violation_overflow": overflow_violation,
        "violation_missing_batches": missing_batches_violation,  # NEW/CHANGED
        "violation_zero_level": zero_level_violation,  # NEW/CHANGED
        "total_violation": total_violation,
        "zero_minutes_total": zero_minutes_total,  # NEW/CHANGED
        "eps_deficit_area": eps_deficit_area,  # NEW/CHANGED
        "low_level_minutes_total": low_level_minutes_total,  # NEW/CHANGED
        "service_obj_holding_term": SERVICE_W_HOLDING_MINUTES
        * holding_minutes_total,  # NEW/CHANGED
        "service_obj_reheat_term": SERVICE_W_REHEAT_KWH * reheat_kwh,  # NEW/CHANGED
        "total_if_kwh": m["total_if_kwh"],
        "melt_kwh": m["melt_kwh"],
        "reheat_kwh": m["reheat_kwh"],
        "reheat_energy_cost": m.get("reheat_energy_cost", 0.0),  # NEW/CHANGED
        "total_energy_cost": m["total_energy_cost"],  # backward-compatible alias
        "total_energy_cost_day": m.get("total_energy_cost_day", m["total_energy_cost"]),
        "peak_kw": m["peak_kw"],
        "md_15_kw": m.get("md_15_kw", 0.0),
        "baseline_md15_kw": m.get("baseline_md15_kw", 0.0),
        "demand_penalty_kw": m.get("demand_penalty_kw", 0.0),
        "demand_charge_month": m.get("demand_charge_month", 0.0),
        "demand_charge_day_equiv": m.get(
            "demand_charge_day_equiv", m.get("demand_penalty", 0.0)
        ),
        "service_fee_day_equiv": m.get("service_fee_day_equiv", 0.0),
        "energy_cost_if_day": m.get("energy_cost_if_day", 0.0),
        "energy_cost_baseline_day": m.get("energy_cost_baseline_day", 0.0),
        "share_if_cost": m.get("share_if_cost", 0.0),
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
        "if_idle_gap_superlinear_minutes": m.get(
            "if_idle_gap_superlinear_minutes", 0.0
        ),
        "if_use_count_A": m["if_use_count_A"],
        "if_use_count_B": m["if_use_count_B"],
        "poured_batches_count": m.get("poured_batches_count", 0),  # NEW/CHANGED
        "target_num_batches": m.get("target_num_batches", _current_num_batches()),
        "missing_batches": m.get("missing_batches", 0),  # NEW/CHANGED
        "unpoured_batches_count": m["unpoured_batches_count"],
        "makespan_minutes": m["makespan_minutes"],
        "total_poured_kg": m["total_poured_kg"],  # CHANGED (Step A)
        "prep_wait_minutes": m.get("prep_wait_minutes", 0.0),  # NEW/CHANGED
        "non_alternation_count": m.get("non_alternation_count", 0.0),  # NEW/CHANGED
        "alternation_ratio": m.get("alternation_ratio", 0.0),  # NEW/CHANGED
        "forced_start_count": m.get("forced_start_count", 0.0),  # NEW/CHANGED
        "start_blocked_by_prep_count": m.get(
            "start_blocked_by_prep_count", 0.0
        ),  # NEW/CHANGED
        "jit_delay_minutes": m.get("jit_delay_minutes", 0.0),  # NEW/CHANGED
        "solar_melt_minutes": m.get("solar_melt_minutes", 0.0),  # NEW/CHANGED
        "cheap_melt_minutes": m.get("cheap_melt_minutes", 0.0),
        "solar_cost_saving": m.get("solar_cost_saving", 0.0),  # NEW/CHANGED
        "start_delay_min": m.get("start_delay_min", 0.0),  # NEW/CHANGED
        "min_level_A": min_level_a,
        "min_level_B": min_level_b,
        "policy": sim["policy"],
        "schedule": sim["schedule"],
        "batch_timing": sim["batch_timing"],
        "mh_levels": sim["mh_levels"],
        "baseline_kw": sim["baseline_kw"],
        "if_kw": sim["if_kw"],
        "tou_raw_price": sim.get("tou_raw_price"),
        "tou_effective_price": sim.get("tou_effective_price"),
        "total_plant_kw": sim["total_plant_kw"],
        "delay_reason_counts": m.get("delay_reason_counts", {}),
        "obj_term_makespan": ENERGY_MAKESPAN_COST_PER_MIN
        * float(m.get("makespan_minutes", SIM_DURATION_MIN)),
        "obj_term_idle_gap": (
            ENERGY_IDLE_GAP_COST_PER_MIN * float(m.get("if_idle_minutes_total", 0.0))
            + ENERGY_IDLE_GAP_SUPERLINEAR_COEFF
            * float(
                m.get(
                    "if_idle_gap_superlinear_minutes",
                    m.get("if_idle_minutes_total", 0.0),
                )
            )
        ),
        "obj_term_cheap_melt_credit": ENERGY_CHEAP_MELT_CREDIT_PER_MIN
        * float(m.get("cheap_melt_minutes", 0.0)),
    }
    return f_scalar, np.asarray(g_constraints, dtype=float), cost_components


class PolicyProblem(Problem):
    def __init__(self):
        if SIMPLE_POLICY_MODE:
            # NEW/CHANGED: compact 12-variable policy (added start_delay_min).
            xl = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.70, 0.0, 0.0, 0.0],
                dtype=float,
            )
            xu = np.array(
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.98, 1.0, 1.0, 1.0],
                dtype=float,
            )
            n_var = 12
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
        n_constr = 2  # [overflow, missing_batches] for both modes
        super().__init__(n_var=n_var, n_obj=1, n_constr=n_constr, xl=xl, xu=xu)

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
                f_scalar, g_vals, d = cached
            elif DETERMINISTIC_SIMULATION:
                cached = _EVAL_CACHE.get(key)
                if cached is None:
                    f_scalar, g_vals, d = evaluate_policy(x)
                    _EVAL_CACHE[key] = (f_scalar, g_vals, d)
                else:
                    f_scalar, g_vals, d = cached
            else:
                # Keep caching disabled in stochastic mode to avoid bias.
                f_scalar, g_vals, d = evaluate_policy(x)
            generation_cache[key] = (f_scalar, g_vals, d)
            F.append([f_scalar])
            details.append(d)
            G.append(list(np.asarray(g_vals, dtype=float).ravel()))

        out["F"] = np.asarray(F, dtype=float)
        out["G"] = np.asarray(G, dtype=float)
        out["cost_details"] = details
        out["schedules"] = np.array([d["schedule"] for d in details], dtype=object)

        if DEBUG and len(F) > 0:
            unique = np.unique(np.round(out["F"], 2), axis=0).shape[0]
            f_min = float(np.min(out["F"][:, 0]))
            f_max = float(np.max(out["F"][:, 0]))
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
            avg_unpoured = float(
                np.mean([d.get("unpoured_batches_count", 0.0) for d in details])
            )
            avg_overflow = float(
                np.mean([d.get("overflow_kg_total", 0.0) for d in details])
            )
            unique_policy = len(set(key_list))
            unique_x = np.unique(np.round(np.asarray(X, dtype=float), 4), axis=0).shape[
                0
            ]
            print("DEBUG unique F:", unique)
            print("DEBUG F min/max:", round(f_min, 2), round(f_max, 2))
            print(
                "DEBUG avg use_B / avg peak:",
                round(avg_use_b, 3),
                round(avg_peak, 3),
            )
            print(
                "DEBUG avg poured / avg unpoured:",
                round(avg_poured, 3),
                round(avg_unpoured, 3),
            )  # NEW/CHANGED
            print(
                "DEBUG avg prep_wait / avg alternation_ratio:",
                round(avg_prep_wait, 3),
                round(avg_alt_ratio, 3),
            )  # NEW/CHANGED
            print("DEBUG avg overflow kg:", round(avg_overflow, 3))
            print(
                "DEBUG unique policy keys:", unique_policy, "unique X(4dp):", unique_x
            )


def format_policy_breakdown(cost):
    if not cost:
        return "No cost details available."
    lines = [
        "Cost Component Details:",
        f"  OPT_MODE                  : {cost.get('opt_mode', 'energy')}",
        f"  Objective TotalCost       : {cost.get('objective_total_cost', cost.get('obj1_total', 0.0)):.2f}",
        f"  Energy-Mode Cost Proxy    : {cost.get('energy_mode_total_cost', 0.0):.2f}",
        f"  Obj1 Raw Sum              : {cost.get('obj1_raw_total', 0.0):.2f}",
        f"  Obj1 Mode                 : {cost.get('obj1_mode', 'raw')}",
        f"  Obj1 Normalized Sum       : {cost.get('obj1_normalized_sum', 0.0):.4f}",
        f"  Obj2 Makespan (min)       : {cost.get('obj2_makespan', 0.0):.2f}",
        f"  IF Total kWh              : {cost.get('total_if_kwh', 0.0):.2f}",
        f"    - Melt kWh              : {cost.get('melt_kwh', 0.0):.2f}",
        f"    - Reheat kWh            : {cost.get('reheat_kwh', 0.0):.2f}",
        f"  Energy Cost/Day (TOU)     : {cost.get('total_energy_cost_day', cost.get('total_energy_cost', 0.0)):.2f}",
        f"  Reheat Energy Cost        : {cost.get('reheat_energy_cost', 0.0):.2f}",  # NEW/CHANGED
        f"  Peak kW (1-min instant)   : {cost.get('peak_kw', 0.0):.2f}",
        f"  MD 15-min kW (billing)    : {cost.get('md_15_kw', 0.0):.2f}",
        f"  Baseline MD 15-min kW     : {cost.get('baseline_md15_kw', 0.0):.2f}",
        f"  Incremental Demand kW     : {cost.get('demand_penalty_kw', 0.0):.2f}",
        f"  Demand Charge/Month       : {cost.get('demand_charge_month', 0.0):.2f}",
        f"  Demand Charge/Day equiv   : {cost.get('demand_charge_day_equiv', cost.get('demand_penalty', 0.0)):.2f}",
        f"  Service Fee/Day equiv     : {cost.get('service_fee_day_equiv', 0.0):.2f}",
        f"  IF Energy Cost/Day        : {cost.get('energy_cost_if_day', 0.0):.2f}",
        f"  Baseline Energy Cost/Day  : {cost.get('energy_cost_baseline_day', 0.0):.2f}",
        f"  IF Cost Share (target 0.20): {cost.get('share_if_cost', 0.0):.3f}",
        f"  Demand Excess kW          : {cost.get('demand_excess_kw', 0.0):.2f}",
        f"  Demand Cost Alias         : {cost.get('demand_penalty', 0.0):.2f}",
        f"  MH Empty Minutes A/B      : {cost.get('mh_empty_minutes_A', 0.0):.2f} / {cost.get('mh_empty_minutes_B', 0.0):.2f}",
        f"  MH Low-Level Minutes A/B  : {cost.get('mh_low_level_minutes_A', 0.0):.2f} / {cost.get('mh_low_level_minutes_B', 0.0):.2f}",
        f"  MH Low-Level Penalty      : {cost.get('mh_low_level_penalty', 0.0):.2f}",
        f"  Overflow kg (total)       : {cost.get('overflow_kg_total', 0.0):.2f}",
        f"  Holding Minutes (total)   : {cost.get('holding_minutes_total', 0.0):.2f}",
        f"  IF Idle Minutes (system)  : {cost.get('if_idle_minutes_total', 0.0):.2f}",
        f"  IF Idle Gap Superlinear   : {cost.get('if_idle_gap_superlinear_minutes', 0.0):.2f}",
        f"  IF Use Count A/B          : {cost.get('if_use_count_A', 0)} / {cost.get('if_use_count_B', 0)}",
        f"  Poured Batches            : {cost.get('poured_batches_count', 0)} / {cost.get('target_num_batches', _current_num_batches())}",
        f"  Missing Batches           : {cost.get('missing_batches', 0)}",
        f"  Unpoured Batches          : {cost.get('unpoured_batches_count', 0)}",
        f"  Makespan (min)            : {cost.get('makespan_minutes', 0.0):.2f}",
        f"  Total Poured (kg)         : {cost.get('total_poured_kg', 0.0):.2f}",  # CHANGED (Step A)
        f"  Prep Wait Minutes         : {cost.get('prep_wait_minutes', 0.0):.2f}",  # NEW/CHANGED
        f"  Non-Alternation Count     : {cost.get('non_alternation_count', 0.0):.2f}",  # NEW/CHANGED
        f"  Alternation Ratio         : {cost.get('alternation_ratio', 0.0):.3f}",  # NEW/CHANGED
        f"  Forced Start Count        : {cost.get('forced_start_count', 0.0):.2f}",  # NEW/CHANGED
        f"  Start Blocked By Prep     : {cost.get('start_blocked_by_prep_count', 0.0):.2f}",  # NEW/CHANGED
        f"  JIT Delay Minutes         : {cost.get('jit_delay_minutes', 0.0):.2f}",  # NEW/CHANGED
        f"  Solar Melt Minutes        : {cost.get('solar_melt_minutes', 0.0):.2f}",  # NEW/CHANGED
        f"  Cheap Melt Minutes        : {cost.get('cheap_melt_minutes', 0.0):.2f}",
        f"  Solar Cost Saving         : {cost.get('solar_cost_saving', 0.0):.2f}",  # NEW/CHANGED
        f"  Obj Term Makespan         : {cost.get('obj_term_makespan', 0.0):.2f}",
        f"  Obj Term Idle Gap         : {cost.get('obj_term_idle_gap', 0.0):.2f}",
        f"  Obj Term Cheap Credit     : -{cost.get('obj_term_cheap_melt_credit', 0.0):.2f}",
        f"  Start Delay (min)         : {cost.get('start_delay_min', 0.0):.2f}",  # NEW/CHANGED
        f"  Zero Minutes Total        : {cost.get('zero_minutes_total', 0.0):.2f}",  # NEW/CHANGED
        f"  EPS Deficit Area          : {cost.get('eps_deficit_area', 0.0):.2f}",  # NEW/CHANGED
        f"  Low-Level Minutes Total   : {cost.get('low_level_minutes_total', 0.0):.2f}",  # NEW/CHANGED
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
        f"    - JIT Delay Penalty     : {cost.get('comp_jit_delay_penalty', 0.0):.2f}",
        "  Constraint Violations:",
        f"    - Overflow (kg)         : {cost.get('violation_overflow', 0.0):.2f}",
        f"    - Missing Batches       : {cost.get('violation_missing_batches', 0.0):.2f}",
        f"    - Zero-Level Violation  : {cost.get('violation_zero_level', 0.0):.2f}",
        f"    - Total Violation       : {cost.get('total_violation', 0.0):.2f}",
    ]
    return "\n".join(lines)


def print_delay_reason_summary(cost_details, top_k=5):
    counts = dict(cost_details.get("delay_reason_counts", {}) or {})
    if not counts:
        print("Delay reasons (top): none")
        return

    tracked = [
        "active_if_no_parallel",
        "gap_block",
        "tou_negative",
        "holding_guard",
        "start_delay_block",
        "demand_guard",
        "jit_gate",
        "prep_not_ready",
        "policy_not_start",
        "controller_not_start",
    ]
    for key in tracked:
        counts.setdefault(key, 0)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    txt = ", ".join([f"{k}={int(v)}" for k, v in ranked])
    print(f"Delay reasons top {top_k}: {txt}")


class ViolationFirstProblem(Problem):
    def __init__(self):
        base = PolicyProblem()
        super().__init__(n_var=base.n_var, n_obj=2, n_constr=0, xl=base.xl, xu=base.xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        details = []
        for x in X:
            f_scalar, g_vals, d = evaluate_policy(x)
            v = float(np.sum(np.maximum(0.0, np.asarray(g_vals, dtype=float))))
            F.append([v, f_scalar])
            details.append(d)
        out["F"] = np.asarray(F, dtype=float)
        out["cost_details"] = details


def plot_policy_result(cost_details, title_prefix="Policy Result"):
    schedule = cost_details.get("schedule", [])
    mh_levels = cost_details.get("mh_levels")
    baseline_kw = cost_details.get("baseline_kw")
    if_kw = cost_details.get("if_kw")
    total_kw = cost_details.get("total_plant_kw")
    tou_raw = cost_details.get("tou_raw_price")
    tou_effective = cost_details.get("tou_effective_price")

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 14), sharex=True)

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
            # Align discrete minute series with an explicit initial point at SHIFT_START.
            t_shifted_mh = np.arange(SHIFT_START, SHIFT_START + SIM_DURATION_MIN + 1)
            mh_series = np.asarray(mh_levels[mh], dtype=float).ravel()
            mh_plot = np.concatenate(
                ([float(MH_INITIAL_LEVEL_KG[mh])], mh_series[:SIM_DURATION_MIN])
            )
            ax2.step(
                t_shifted_mh,
                mh_plot,
                where="post",
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
        md_15_line = float(cost_details.get("md_15_kw", 0.0))
        ax3.axhline(
            md_15_line,
            color="purple",
            linestyle=":",
            linewidth=1.6,
            label=f"MD15 {md_15_line:.1f} kW",
        )
        md_15 = float(cost_details.get("md_15_kw", 0.0))
        dc_month = float(cost_details.get("demand_charge_month", 0.0))
        dc_day = float(cost_details.get("demand_charge_day_equiv", 0.0))
        ax3.text(
            0.01,
            0.98,
            f"MD15={md_15:.1f} kW | DC month={dc_month:.1f} THB | DC day~{dc_day:.1f} THB",
            transform=ax3.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )
    ax3.set_ylabel("kW")
    ax3.set_xlabel("Time (HH:MM)")
    ax3.set_title("Plant Load")
    ax3.grid(True, alpha=0.4)
    ax3.legend(loc="upper right")

    if tou_raw is not None:
        ax4.plot(
            t_shifted,
            tou_raw,
            label="TOU Raw Price",
            color="black",
            linewidth=1.8,
        )
    if tou_effective is not None:
        ax4.plot(
            t_shifted,
            tou_effective,
            label="TOU Effective Price (after solar)",
            color="purple",
            linewidth=2.0,
        )
    if if_kw is not None and tou_effective is not None:
        y0 = float(np.min(tou_effective))
        y1 = y0 + max(0.05, 0.08 * float(np.ptp(tou_effective) + 1e-9))
        active_mask = np.asarray(if_kw) > 1e-9
        ax4.fill_between(
            t_shifted,
            y0,
            y1,
            where=active_mask,
            color="steelblue",
            alpha=0.22,
            step="pre",
            label="IF active window",
        )
    ax4.set_ylabel("Price")
    ax4.set_title("TOU Raw vs Effective")
    ax4.grid(True, alpha=0.4)
    ax4.legend(loc="upper right")

    # NEW/CHANGED: highlight solar window on every subplot.
    plot_start = SHIFT_START
    plot_end = SHIFT_START + SIM_DURATION_MIN
    for day in range(-1, 3):
        s = day * 1440 + SOLAR_START
        e = day * 1440 + SOLAR_END
        if e <= plot_start or s >= plot_end:
            continue
        ss = max(s, plot_start)
        ee = min(e, plot_end)
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvspan(ss, ee, color="gold", alpha=0.18, linewidth=0)

    xticks = np.arange(SHIFT_START, SHIFT_START + SIM_DURATION_MIN + 1, 60)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels([f"{(x // 60) % 24:02d}:{x % 60:02d}" for x in xticks])

    plt.tight_layout()
    plt.show()


def dump_mh_trace(cost_details, out_csv_path, step_min=1):
    """Dump minute-level MH values used for plotting to CSV."""
    mh_levels = cost_details.get("mh_levels")
    if mh_levels is None:
        print("MH trace dump skipped: no mh_levels found.")
        return

    mh_a = np.asarray(mh_levels.get("A", []), dtype=float).ravel()
    mh_b = np.asarray(mh_levels.get("B", []), dtype=float).ravel()
    if len(mh_a) == 0 or len(mh_b) == 0:
        print("MH trace dump skipped: empty MH series.")
        return

    baseline_kw = np.asarray(
        cost_details.get("baseline_kw", np.zeros_like(mh_a)), dtype=float
    )
    if_kw = np.asarray(cost_details.get("if_kw", np.zeros_like(mh_a)), dtype=float)
    total_kw = np.asarray(
        cost_details.get("total_plant_kw", baseline_kw + if_kw), dtype=float
    )

    n = int(min(len(mh_a), len(mh_b), len(baseline_kw), len(if_kw), len(total_kw)))
    step = max(1, int(step_min))

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "minute",
                "time_of_day",
                "mh_A_kg",
                "mh_B_kg",
                "baseline_kw",
                "if_kw",
                "total_plant_kw",
            ]
        )
        for t in range(0, n, step):
            tod = int((SHIFT_START + t) % 1440)
            writer.writerow(
                [
                    t,
                    f"{tod // 60:02d}:{tod % 60:02d}",
                    float(mh_a[t]),
                    float(mh_b[t]),
                    float(baseline_kw[t]),
                    float(if_kw[t]),
                    float(total_kw[t]),
                ]
            )

    print(f"Saved MH trace CSV: {out_csv_path}")

    # Extra debug window around pour events to inspect sudden drops/jumps.
    schedule = cost_details.get("schedule", [])
    pour_events = sorted(
        [int(x.get("pour_min", -1)) for x in schedule if x.get("status") == "poured"]
    )
    if len(pour_events) > 0:
        print("MH around pour events (minute, A, B):")
        for i, pm in enumerate(pour_events, start=1):
            left = max(0, pm - 3)
            right = min(n - 1, pm + 3)
            window = ", ".join(
                [f"{tt}:{mh_a[tt]:.1f}/{mh_b[tt]:.1f}" for tt in range(left, right + 1)]
            )
            print(f"  pour#{i} at t={pm}: {window}")


def main():
    np.random.seed(42)
    sanity = quick_capacity_sanity_check()
    print_capacity_sanity_report(sanity)
    if not sanity["is_feasible_coarse"]:
        raise ValueError(
            f"Target batches ({_current_num_batches()}) exceed coarse throughput limits. "
            "Try reducing target batches (NUM_BATCHES/NUM_BATCHES_RUN_OVERRIDE), "
            "enabling ALLOW_PARALLEL_IF, or increasing receive/melt capacity."
        )

    problem = PolicyProblem()
    sampling = FloatRandomSampling()
    # More exploratory variation while remaining stable for policy tuning.
    crossover = SBX(prob=0.92, eta=8)
    mutation = PolynomialMutation(prob=0.45, eta=8)

    algorithm = GA(
        pop_size=80,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=PolicyDuplicateElimination(),
    )
    termination = get_termination("n_gen", 100)
    callback = StagnationEarlyStopCallback(
        patience_gens=EARLY_STOP_PATIENCE_GENS,
        delta_obj=EARLY_STOP_DELTA_OBJ1,
    )

    print("Running Policy GA optimization...")
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

    best_x = None
    if result is not None and result.X is not None:
        best_x = np.asarray(result.X, dtype=float).ravel()
    else:
        # NEW/CHANGED: fallback to least-infeasible individual from final population.
        pop = None
        if result is not None and getattr(result, "pop", None) is not None:
            pop = result.pop
        elif getattr(algorithm, "pop", None) is not None:
            pop = algorithm.pop

        if pop is None:
            print("No valid best solution returned by GA and no population available.")
            return

        pop_X = pop.get("X")
        pop_F = pop.get("F")
        pop_G = pop.get("G")
        if pop_X is None or len(pop_X) == 0:
            print("No valid best solution returned by GA and empty population.")
            return

        if pop_G is None:
            cv = np.zeros(len(pop_X), dtype=float)
        else:
            g = np.asarray(pop_G, dtype=float)
            if g.ndim == 1:
                g = g.reshape(-1, 1)
            cv = np.sum(np.maximum(0.0, g), axis=1)

        fvals = (
            np.asarray(pop_F, dtype=float).reshape(-1)
            if pop_F is not None
            else np.full(len(pop_X), np.inf, dtype=float)
        )
        # Argmin by CV first, then objective tie-break.
        candidate_idxs = np.where(cv == np.min(cv))[0]
        if len(candidate_idxs) > 1:
            idx = int(candidate_idxs[np.argmin(fvals[candidate_idxs])])
        else:
            idx = int(candidate_idxs[0])
        best_x = np.asarray(pop_X[idx], dtype=float).ravel()
        print(
            f"Fallback selection: picked least-infeasible individual idx={idx}, CV={cv[idx]:.6f}"
        )

    best_f, best_g, best_details = evaluate_policy(best_x)
    print("\nBest GA solution:")
    print("  Policy X:", np.round(best_x, 4))
    print(f"  Objective TotalCost: {best_f:.3f}")
    print(f"  Constraints: {np.round(np.asarray(best_g, dtype=float), 6)}")
    print(
        "  objective_total_cost:",
        round(best_details.get("objective_total_cost", best_f), 2),
        "total_energy_cost_day:",
        round(
            best_details.get(
                "total_energy_cost_day", best_details.get("total_energy_cost", 0.0)
            ),
            2,
        ),
        "md_15_kw:",
        round(best_details.get("md_15_kw", 0.0), 2),
        "demand_charge_month:",
        round(best_details.get("demand_charge_month", 0.0), 2),
        "demand_charge_day_equiv:",
        round(
            best_details.get(
                "demand_charge_day_equiv", best_details.get("demand_penalty", 0.0)
            ),
            2,
        ),
        "solar_cost_saving:",
        round(best_details.get("solar_cost_saving", 0.0), 2),
        "reheat_energy_cost:",
        round(best_details.get("reheat_energy_cost", 0.0), 2),
        "holding_minutes_total:",
        round(best_details.get("holding_minutes_total", 0.0), 2),
        "if_share_cost:",
        round(best_details.get("share_if_cost", 0.0), 4),
    )
    print(
        "  IF share sanity:",
        round(best_details.get("share_if_cost", 0.0), 4),
        "(target ~0.20)",
        "if_cost_day:",
        round(best_details.get("energy_cost_if_day", 0.0), 2),
        "baseline_cost_day:",
        round(best_details.get("energy_cost_baseline_day", 0.0), 2),
    )
    print(
        "  total_poured_kg:",
        round(best_details.get("total_poured_kg", 0.0), 2),
        "unpoured_batches_count:",
        int(best_details.get("unpoured_batches_count", 0)),
        "overflow_kg_total:",
        round(best_details.get("overflow_kg_total", 0.0), 2),
    )
    print(
        "  peak_kw:",
        round(best_details.get("peak_kw", 0.0), 2),
        "md_15_kw:",
        round(best_details.get("md_15_kw", 0.0), 2),
        "baseline_md15_kw:",
        round(best_details.get("baseline_md15_kw", 0.0), 2),
        "demand_penalty_kw:",
        round(best_details.get("demand_penalty_kw", 0.0), 2),
        "demand_charge_day_equiv:",
        round(
            best_details.get(
                "demand_charge_day_equiv", best_details.get("demand_penalty", 0.0)
            ),
            2,
        ),
    )
    print_delay_reason_summary(best_details, top_k=5)
    print(
        "  KPI empty_min(A/B):",
        round(best_details.get("mh_empty_minutes_A", 0.0), 2),
        "/",
        round(best_details.get("mh_empty_minutes_B", 0.0), 2),
        "low_min(A/B):",
        round(best_details.get("mh_low_level_minutes_A", 0.0), 2),
        "/",
        round(best_details.get("mh_low_level_minutes_B", 0.0), 2),
        "min_level(A/B):",
        round(best_details.get("min_level_A", 0.0), 2),
        "/",
        round(best_details.get("min_level_B", 0.0), 2),
    )
    print(
        "  KPI zero_minutes_total:",
        round(best_details.get("zero_minutes_total", 0.0), 2),
        "eps_deficit_area:",
        round(best_details.get("eps_deficit_area", 0.0), 2),
    )
    print(
        f"OPT_MODE={OPT_MODE} | poured_batches={int(best_details.get('poured_batches_count', 0))}/{int(best_details.get('target_num_batches', _current_num_batches()))} | "
        f"missing={int(best_details.get('missing_batches', 0))} | overflow={round(best_details.get('overflow_kg_total', 0.0), 3)} | "
        f"min_level(A,B)={round(best_details.get('min_level_A', 0.0), 3)},{round(best_details.get('min_level_B', 0.0), 3)} | "
        f"zero_minutes={round(best_details.get('zero_minutes_total', 0.0), 3)} | "
        f"eps_deficit_area={round(best_details.get('eps_deficit_area', 0.0), 3)} | "
        f"energy_day={round(best_details.get('total_energy_cost_day', best_details.get('total_energy_cost', 0.0)), 3)} | "
        f"md15={round(best_details.get('md_15_kw', 0.0), 3)} | "
        f"demand_day={round(best_details.get('demand_charge_day_equiv', best_details.get('demand_penalty', 0.0)), 3)} | "
        f"if_share={round(best_details.get('share_if_cost', 0.0), 3)} | "
        f"solar_saving={round(best_details.get('solar_cost_saving', 0.0), 3)} | "
        f"peak={round(best_details.get('peak_kw', 0.0), 3)}"
    )
    print(format_policy_breakdown(best_details))
    if DEBUG_DUMP_MH_TRACE:
        trace_path = os.path.join(os.path.dirname(__file__), "mh_trace_debug.csv")
        dump_mh_trace(
            best_details, out_csv_path=trace_path, step_min=DEBUG_MH_TRACE_STEP_MIN
        )
    plot_policy_result(best_details, title_prefix="Best GA Policy")


# NEW/CHANGED: ================== CONFIG SUMMARY ==================
# SIMPLE_POLICY_MODE:
#   True  -> compact 12-variable policy (includes start_delay_min)
#   False -> legacy/full policy variable set
#
# ENABLE_PREP_MODEL:
#   True  -> prep_remaining model is active:
#            - when furnace pour completes: prep_remaining[f] = PREP_LOAD_TIME_MIN
#            - while the other furnace is melting: prep_remaining on idle furnace counts down
#            - a furnace can start melting only when prep_remaining[f] == 0
#   False -> disable prep-ready blocking dynamics.
#
# ENFORCE_ALTERNATION_FLAG:
#   False -> soft alternation via alternation_bias + switch penalty.
#   True  -> in force mode, prefer strict A/B alternation when feasible.
#
# PREP_LOAD_TIME_MIN / PREP_WAIT_COST_PER_MIN / SWITCH_PENALTY_PER_REPEAT / JIT_DELAY_COST_PER_MIN:
#   Tune prep-delay, repetition cost, and JIT-delay economics for your plant.
# ================================================================
if __name__ == "__main__":
    main()
