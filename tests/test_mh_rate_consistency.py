"""
MH consumption rate consistency tests.

Field bug: changing the M&H consumption rate on a plan made the Metal Level
chart show abnormal values (negative levels, big jumps, slope not matching the
new rate). These tests pin down the simulator side of that contract so the
chart's source data (`mh_levels` series) is provably consistent with the rate
that was applied.

Each test snapshots/restores `app_v9` module globals — same pattern as
`tests/test_jit_holding_scenarios.py` and `tests/test_single_furnace_mode.py`.

Properties verified per scenario:
  - Per-minute slope of mh_*_levels matches the configured rate while the
    furnace is consuming (not paused at empty, not just topped up by a pour).
  - All sample points stay within [0, MAX_CAPACITY_KG].
  - No NaN / inf values.
  - No "jump" exceeding what one pour event can legitimately add
    (capped by the IF batch output sent to that furnace).
  - Boundary: zero rate keeps the level monotonically non-decreasing.

We run the same simulator multiple times with different rates and assert that
the *first* downward segment's slope tracks the configured rate. That's the
clean window before any pour event muddies the trace.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.app_v9 as app_v9  # noqa: E402


# ---------------------------------------------------------------------------
# Globals snapshot/restore — mirrors ga_service._apply_settings_overrides
# ---------------------------------------------------------------------------

_PLANT_GLOBALS = (
    "MH_MAX_CAPACITY_KG",
    "MH_INITIAL_LEVEL_KG",
    "MH_MIN_OPERATIONAL_LEVEL_KG",
    "MH_CONSUMPTION_RATE_KG_PER_MIN",
    "IF_BATCH_OUTPUT_KG",
    "OPT_MODE",
    "USE_FURNACE_A",
    "USE_FURNACE_B",
    "ENERGY_MODE_MIN_JIT_SLACK_MIN",
    "ENERGY_MODE_HIGH_TOU_GAP_WEIGHT",
    "TOU_ONPEAK_BAHT_PER_KWH",
    "TOU_OFFPEAK_BAHT_PER_KWH",
    "TOU_ONPEAK_BAHT_PER_KWH_BASE",
    "TOU_OFFPEAK_BAHT_PER_KWH_BASE",
    "DEMAND_CHARGE_BAHT_PER_KW_MONTH",
    "IF_HOLDING_PENALTY_PER_MIN",
    "LOW_LEVEL_NONLINEAR_FACTOR",
    "NUM_BATCHES_RUN_OVERRIDE",
)


def _snapshot_globals():
    snap = {name: getattr(app_v9, name, None) for name in _PLANT_GLOBALS}
    for k, v in list(snap.items()):
        if isinstance(v, dict):
            snap[k] = dict(v)
    snap["_obj1_weights"] = dict(getattr(app_v9, "OBJ1_COMPONENT_WEIGHTS", {}))
    snap["_cache"] = dict(getattr(app_v9, "_EVAL_CACHE", {}))
    return snap


def _restore_globals(snap):
    for name in _PLANT_GLOBALS:
        v = snap.get(name)
        if v is None:
            continue
        if isinstance(v, dict):
            getattr(app_v9, name).clear()
            getattr(app_v9, name).update(v)
        else:
            setattr(app_v9, name, v)
    app_v9.OBJ1_COMPONENT_WEIGHTS.clear()
    app_v9.OBJ1_COMPONENT_WEIGHTS.update(snap["_obj1_weights"])
    app_v9._EVAL_CACHE.clear()
    app_v9._EVAL_CACHE.update(snap["_cache"])


def _patch_plant_config(
    *,
    cons_a: float,
    cons_b: float,
    init_a: float = 800.0,
    init_b: float = 1100.0,
    max_a: float = 800.0,
    max_b: float = 1100.0,
    min_a: float = 400.0,
    min_b: float = 550.0,
    target_batches: int = 9,
    if_batch_output: float = 600.0,
):
    """Patch globals the same way ga_service does, with flags-off so the
    simulator produces deterministic continuous-melt traces."""
    app_v9.MH_MAX_CAPACITY_KG.clear()
    app_v9.MH_MAX_CAPACITY_KG.update({"A": max_a, "B": max_b})
    app_v9.MH_INITIAL_LEVEL_KG.clear()
    app_v9.MH_INITIAL_LEVEL_KG.update({"A": init_a, "B": init_b})
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.clear()
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.update({"A": min_a, "B": min_b})
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.clear()
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.update({"A": cons_a, "B": cons_b})

    app_v9.IF_BATCH_OUTPUT_KG = float(if_batch_output)
    app_v9.OPT_MODE = "energy"
    app_v9.USE_FURNACE_A = True
    app_v9.USE_FURNACE_B = True
    app_v9.NUM_BATCHES_RUN_OVERRIDE = int(target_batches)

    # Flags-off mirror — eliminates JIT-slack-floor noise so the level trace
    # is just (initial − rate * minutes) until a pour lands.
    app_v9.TOU_ONPEAK_BAHT_PER_KWH_BASE = app_v9.TOU_OFFPEAK_BAHT_PER_KWH_BASE
    app_v9.TOU_ONPEAK_BAHT_PER_KWH = app_v9.TOU_OFFPEAK_BAHT_PER_KWH
    app_v9.ENERGY_MODE_HIGH_TOU_GAP_WEIGHT = 0.0
    app_v9.DEMAND_CHARGE_BAHT_PER_KW_MONTH = 0.0
    app_v9.IF_HOLDING_PENALTY_PER_MIN = 0.0
    if "holding_penalty" in app_v9.OBJ1_COMPONENT_WEIGHTS:
        app_v9.OBJ1_COMPONENT_WEIGHTS["holding_penalty"] = 0.0
    app_v9.LOW_LEVEL_NONLINEAR_FACTOR = 0.0
    app_v9.ENERGY_MODE_MIN_JIT_SLACK_MIN = 0.0

    app_v9._EVAL_CACHE.clear()


@pytest.fixture(autouse=True)
def _isolate_globals():
    snap = _snapshot_globals()
    try:
        yield
    finally:
        _restore_globals(snap)


# ---------------------------------------------------------------------------
# Helpers — analyse the mh_levels series the GA service hands to charts
# ---------------------------------------------------------------------------

def _run_default():
    return app_v9.simulate_policy_day(None)


def _series(result, furnace: str):
    """Return the mh_levels series for one furnace as a list of floats."""
    levels = result.get("mh_levels", {}).get(furnace)
    if levels is None:
        raise AssertionError(f"mh_levels[{furnace!r}] missing from simulator output")
    return [float(x) for x in levels]


def _first_downward_slope(series: list[float], window: int = 15) -> float:
    """Average kg/min of the first sustained decline. Picks the longest
    monotonically non-increasing run starting from t=0 and returns its
    per-minute slope. `window` caps how many minutes we look at — keeps us
    away from the first pour event in case the simulator is busy at minute 0."""
    if len(series) < 2:
        raise AssertionError("series too short to measure slope")
    end = 1
    while end < min(len(series), window + 1) and series[end] <= series[end - 1]:
        end += 1
    drop = series[0] - series[end - 1]
    minutes = end - 1
    assert minutes >= 1, "no downward minute found in initial window"
    return drop / minutes


def _largest_jump(series: list[float]) -> float:
    """Max absolute one-step change. Positive jumps come from pours; should
    never exceed IF_BATCH_OUTPUT_KG. Negative single-step drops should never
    exceed the per-minute rate (allow small float slack)."""
    return max(abs(b - a) for a, b in zip(series, series[1:]))


# ---------------------------------------------------------------------------
# T1 — Baseline slope matches the configured rate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cons_a,cons_b", [
    (2.5, 2.6),   # default
    (3.5, 3.0),
    (5.0, 4.5),
    (2.0, 5.5),   # asymmetric — A slow, B fast
])
def test_initial_slope_tracks_rate(cons_a, cons_b):
    _patch_plant_config(cons_a=cons_a, cons_b=cons_b)
    result = _run_default()

    slope_a = _first_downward_slope(_series(result, "A"))
    slope_b = _first_downward_slope(_series(result, "B"))

    assert math.isclose(slope_a, cons_a, abs_tol=0.05), (
        f"MH-A slope {slope_a:.3f} kg/min does not match configured rate {cons_a}"
    )
    assert math.isclose(slope_b, cons_b, abs_tol=0.05), (
        f"MH-B slope {slope_b:.3f} kg/min does not match configured rate {cons_b}"
    )


# ---------------------------------------------------------------------------
# T2 — Doubling the rate doubles the slope (sanity / proportionality)
# ---------------------------------------------------------------------------
def test_doubling_rate_doubles_slope():
    _patch_plant_config(cons_a=2.5, cons_b=2.6)
    base_a = _first_downward_slope(_series(_run_default(), "A"))

    _patch_plant_config(cons_a=5.0, cons_b=2.6)
    doubled_a = _first_downward_slope(_series(_run_default(), "A"))

    ratio = doubled_a / base_a if base_a else 0
    assert 1.85 <= ratio <= 2.15, (
        f"Doubled rate produced slope ratio {ratio:.3f}; expected ~2.0"
    )


# ---------------------------------------------------------------------------
# T3 — Levels never go negative or above capacity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cons_a,cons_b", [
    (2.5, 2.6),
    (5.0, 5.0),
    (0.1, 0.1),
])
def test_levels_stay_within_bounds(cons_a, cons_b):
    _patch_plant_config(cons_a=cons_a, cons_b=cons_b)
    result = _run_default()

    for furnace, max_kg in (("A", app_v9.MH_MAX_CAPACITY_KG["A"]),
                            ("B", app_v9.MH_MAX_CAPACITY_KG["B"])):
        series = _series(result, furnace)
        lo, hi = min(series), max(series)
        assert lo >= -1e-6, f"MH-{furnace} level dipped to {lo:.3f} kg (negative)"
        assert hi <= max_kg + 1e-6, (
            f"MH-{furnace} level reached {hi:.3f} kg, exceeds max {max_kg}"
        )


# ---------------------------------------------------------------------------
# T4 — No NaN / inf anywhere in the series
# ---------------------------------------------------------------------------
def test_no_nan_or_inf_in_series():
    _patch_plant_config(cons_a=2.5, cons_b=2.6)
    result = _run_default()
    for furnace in ("A", "B"):
        series = _series(result, furnace)
        bad = [(i, x) for i, x in enumerate(series) if not math.isfinite(x)]
        assert not bad, f"MH-{furnace} has non-finite values at {bad[:5]}"


# ---------------------------------------------------------------------------
# T5 — No single-minute jump larger than one pour event
# ---------------------------------------------------------------------------
def test_jumps_bounded_by_pour_size():
    _patch_plant_config(cons_a=2.5, cons_b=2.6, if_batch_output=600.0)
    result = _run_default()
    # A pour can be split A/B but neither side gets more than IF_BATCH_OUTPUT_KG
    # in a single minute. Add a small slack for float rounding.
    cap = app_v9.IF_BATCH_OUTPUT_KG + 1.0
    for furnace in ("A", "B"):
        series = _series(result, furnace)
        jump = _largest_jump(series)
        assert jump <= cap, (
            f"MH-{furnace} had a {jump:.1f} kg single-step change; "
            f"larger than one full IF batch ({cap:.1f})"
        )


# ---------------------------------------------------------------------------
# T6 — Zero rate: level should be monotonically non-decreasing (only pours add)
# ---------------------------------------------------------------------------
def test_zero_rate_no_consumption():
    _patch_plant_config(cons_a=0.0, cons_b=0.0, init_a=400.0, init_b=550.0)
    result = _run_default()
    for furnace in ("A", "B"):
        series = _series(result, furnace)
        deltas = [b - a for a, b in zip(series, series[1:])]
        worst_drop = min(deltas) if deltas else 0.0
        assert worst_drop >= -1e-6, (
            f"MH-{furnace} dropped by {worst_drop:.4f} kg with rate=0; "
            "no consumption should happen"
        )


# ---------------------------------------------------------------------------
# T7 — Higher rate ⇒ strictly lower minimum level (when no pour intervenes)
# ---------------------------------------------------------------------------
def test_higher_rate_drains_more():
    _patch_plant_config(cons_a=2.0, cons_b=2.6)
    low_rate_min = min(_series(_run_default(), "A"))

    _patch_plant_config(cons_a=4.0, cons_b=2.6)
    high_rate_min = min(_series(_run_default(), "A"))

    assert high_rate_min < low_rate_min - 1.0, (
        f"Higher rate (4.0) reached min {high_rate_min:.1f}, "
        f"not lower than rate=2.0 min {low_rate_min:.1f}"
    )


# ---------------------------------------------------------------------------
# T8 — Series length matches simulator duration
# ---------------------------------------------------------------------------
def test_series_length_matches_duration():
    _patch_plant_config(cons_a=2.5, cons_b=2.6)
    result = _run_default()
    for furnace in ("A", "B"):
        series = _series(result, furnace)
        assert len(series) == app_v9.SIM_DURATION_MIN, (
            f"MH-{furnace} series length {len(series)} != SIM_DURATION_MIN "
            f"{app_v9.SIM_DURATION_MIN}"
        )
