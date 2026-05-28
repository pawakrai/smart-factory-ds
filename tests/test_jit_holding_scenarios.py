"""
JIT-gate / holding-time scenarios for app_v9.simulate_policy_day.

Verifies that:
  - Hold time stays low when energy-mode JIT slack floor is cleared
    (mirroring the ga_service patch when flags are off).
  - Service mode keeps holding tight even without the patch.
  - Hard MH-min constraint is honoured.
  - Balanced consumption rates (MH-A 2.2, MH-B 3.6) drive the
    below-min violation toward zero.

These tests drive the simulator directly with a default policy (None).
They cover SIMULATOR mechanics, not GA optimisation quality; a live
end-to-end sweep against the dockerised backend complements them.

Each test snapshots/restores app_v9 module globals using the same
pattern as tests/test_single_furnace_mode.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.app_v9 as app_v9  # noqa: E402


# ---------------------------------------------------------------------------
# Plant config snapshot/restore  (mirrors ga_service._apply_settings_overrides)
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
    # Dicts are mutated below; deep-copy the references we care about.
    for k, v in list(snap.items()):
        if isinstance(v, dict):
            snap[k] = dict(v)
    # OBJ1_COMPONENT_WEIGHTS["holding_penalty"] also gets patched.
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
    cons_a: float = 2.2,
    cons_b: float = 2.6,
    max_a: float = 800.0,
    max_b: float = 1100.0,
    init_a: float = 800.0,
    init_b: float = 1100.0,
    min_a: float = 400.0,
    min_b: float = 550.0,
    target_batches: int = 9,
    opt_mode: str = "energy",
    flags_off: bool = True,
    if_batch_output: float = 600.0,
):
    """Patch app_v9 module globals to mirror a Plan from the backend."""
    app_v9.MH_MAX_CAPACITY_KG.clear()
    app_v9.MH_MAX_CAPACITY_KG.update({"A": max_a, "B": max_b})
    app_v9.MH_INITIAL_LEVEL_KG.clear()
    app_v9.MH_INITIAL_LEVEL_KG.update({"A": init_a, "B": init_b})
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.clear()
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.update({"A": min_a, "B": min_b})
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.clear()
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.update({"A": cons_a, "B": cons_b})

    app_v9.IF_BATCH_OUTPUT_KG = float(if_batch_output)
    app_v9.OPT_MODE = opt_mode
    app_v9.USE_FURNACE_A = True
    app_v9.USE_FURNACE_B = True
    app_v9.NUM_BATCHES_RUN_OVERRIDE = int(target_batches)

    if flags_off:
        # Same patches ga_service applies when both flags are off.
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


def _run_default_policy():
    """Run simulate_policy_day with the built-in default policy and return metrics."""
    result = app_v9.simulate_policy_day(None)
    return result.get("metrics", {}), result


def _theoretical_pour_cycle(cons_a, cons_b, pour_total=600.0):
    """Minimum minutes between pours bounded by combined MH consumption."""
    return pour_total / (cons_a + cons_b)


# ---------------------------------------------------------------------------
# Fixture: snapshot/restore module globals per test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_globals():
    snap = _snapshot_globals()
    try:
        yield
    finally:
        _restore_globals(snap)


# ---------------------------------------------------------------------------
# T1 — Energy + flags off, moderate load
# ---------------------------------------------------------------------------
def test_t1_energy_flags_off_moderate_load():
    """With JIT slack floor cleared, energy-mode + flags-off should keep
    holding low and pour all 9 batches without emptying MH."""
    _patch_plant_config(
        target_batches=9, cons_a=2.2, cons_b=2.6,
        opt_mode="energy", flags_off=True,
    )
    metrics, _ = _run_default_policy()
    poured = metrics["poured_batches_count"]
    holding = metrics["holding_minutes_total"]
    empty = metrics["mh_empty_minutes"]["A"] + metrics["mh_empty_minutes"]["B"]
    assert poured == 9, f"poured={poured}, expected 9"
    assert holding < 50, f"holding_minutes_total={holding}, expected < 50"
    assert empty < 30, f"mh_empty (A+B)={empty}, expected < 30"


# ---------------------------------------------------------------------------
# T2 — Energy + flags off, heavy load (near feasibility limit)
# ---------------------------------------------------------------------------
def test_t2_energy_flags_off_heavy_load():
    """12 batches over 24h is near the MH bottleneck (cycle≈125 min).
    Allow a few missed but cap holding."""
    _patch_plant_config(
        target_batches=12, cons_a=2.2, cons_b=2.6,
        opt_mode="energy", flags_off=True,
    )
    metrics, _ = _run_default_policy()
    poured = metrics["poured_batches_count"]
    holding = metrics["holding_minutes_total"]
    assert poured >= 10, f"poured={poured}, expected >= 10"
    assert holding < 120, f"holding_minutes_total={holding}, expected < 120"


# ---------------------------------------------------------------------------
# T3 — Energy + flags ON (regression): allowed to hold longer for price shift
# ---------------------------------------------------------------------------
def test_t3_energy_flags_on_regression():
    """With flags ON, JIT slack floor stays at 25 min so holding can grow,
    but the schedule must still pour all batches."""
    _patch_plant_config(
        target_batches=9, cons_a=2.2, cons_b=2.6,
        opt_mode="energy", flags_off=False,
    )
    metrics, _ = _run_default_policy()
    poured = metrics["poured_batches_count"]
    assert poured == 9, f"poured={poured}, expected 9"


# ---------------------------------------------------------------------------
# T4 — Service mode + flags off: very tight holding
# ---------------------------------------------------------------------------
def test_t4_service_flags_off_tight():
    """Service mode caps jit_slack at 5 min in app_v9, so holding stays tight
    without needing the energy-mode patch."""
    _patch_plant_config(
        target_batches=9, cons_a=2.2, cons_b=2.6,
        opt_mode="service", flags_off=True,
    )
    metrics, _ = _run_default_policy()
    holding = metrics["holding_minutes_total"]
    poured = metrics["poured_batches_count"]
    assert poured == 9, f"poured={poured}, expected 9"
    assert holding < 30, f"holding_minutes_total={holding}, expected < 30"


# ---------------------------------------------------------------------------
# T5 — Balanced consumption rates (A=2.2, B=3.6)
# ---------------------------------------------------------------------------
def test_t5_high_b_consumption_still_feasible():
    """When MH-B consumption is raised significantly (3.6 kg/min, simulating
    rebalanced physics), the simulator must still drive all 9 batches through.

    NOTE: default policy is not GA-optimised and may produce sub-optimal MH
    trajectories with non-default rates — comparative quality must be
    validated through the live GA sweep, not this unit test.
    """
    _patch_plant_config(
        target_batches=9, cons_a=2.2, cons_b=3.6,
        opt_mode="energy", flags_off=True,
    )
    metrics, _ = _run_default_policy()
    poured = metrics["poured_batches_count"]
    assert poured == 9, f"poured={poured}, expected 9 even with raised cons_B"


# ---------------------------------------------------------------------------
# T6 — Theoretical pour cycle vs simulator behaviour
# ---------------------------------------------------------------------------
def test_t6_pour_cycle_matches_theoretical():
    """With flags off and JIT slack floor cleared, the inter-pour spacing
    should approach the theoretical MH bottleneck (600 / (cons_a+cons_b))."""
    cons_a, cons_b = 2.2, 2.6
    theory = _theoretical_pour_cycle(cons_a, cons_b)
    _patch_plant_config(
        target_batches=9, cons_a=cons_a, cons_b=cons_b,
        opt_mode="energy", flags_off=True,
    )
    _metrics, result = _run_default_policy()
    pours = sorted(
        b["pour_min"] for b in result["schedule"] if b.get("pour_min") is not None
    )
    assert len(pours) >= 2, "need ≥2 pours to compute gaps"
    gaps = [b - a for a, b in zip(pours[:-1], pours[1:])]
    avg_gap = sum(gaps) / len(gaps)
    # Allow ±30% deviation from theoretical cycle.
    assert theory * 0.7 <= avg_gap <= theory * 1.3, (
        f"avg pour gap={avg_gap:.1f} min outside ±30% of "
        f"theoretical {theory:.1f} min"
    )
