"""
Verify ga_service._apply_settings_overrides behaves correctly for the
both-flags-off path, split between energy mode and service mode.

Bug context: a plan with `opt_mode="service"`, `consider_tou_price=False`,
`consider_plant_load=False` was producing schedules where MH-A drained well
below `MH_MIN_OPERATIONAL_LEVEL_KG` and the GA happily inserted long idle
gaps between charges — the opposite of what the operator wants in
"continuous-melt" mode.

The override block now distinguishes service vs energy:

* service + flags off:
    - LOW_LEVEL_NONLINEAR_FACTOR stays at default (quadratic depth penalty)
    - OBJ1 safety weights boosted to ≥ 3.0 / 2.5
    - MAX_LOW_LEVEL_MIN_ALLOW tightened to 60 min
    - POLICY_OVERRIDE_MIN_START_GAP_MAX forced to 0 (back-to-back charges)

* energy + flags off (regression):
    - LOW_LEVEL_NONLINEAR_FACTOR = 0 (linearized — original tuning)
    - POLICY_OVERRIDE_MIN_START_GAP_MAX = 5 (the old cap)
    - holding_penalty = 0

These tests invoke `_apply_settings_overrides` directly with a fake plan
object so we can pin the side effects on `app_v9` globals without running
the GA. The snapshot fixture restores everything between tests.
"""
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.app_v9 as app_v9  # noqa: E402
from backend.services.ga_service import _apply_settings_overrides  # noqa: E402


# ---------------------------------------------------------------------------
# Snapshot/restore — same pattern as test_jit_holding_scenarios
# ---------------------------------------------------------------------------

_GLOBALS = (
    "MH_MAX_CAPACITY_KG", "MH_INITIAL_LEVEL_KG", "MH_MIN_OPERATIONAL_LEVEL_KG",
    "MH_CONSUMPTION_RATE_KG_PER_MIN", "IF_BATCH_OUTPUT_KG", "OPT_MODE",
    "USE_FURNACE_A", "USE_FURNACE_B", "ENERGY_MODE_MIN_JIT_SLACK_MIN",
    "ENERGY_MODE_HIGH_TOU_GAP_WEIGHT", "TOU_ONPEAK_BAHT_PER_KWH",
    "TOU_OFFPEAK_BAHT_PER_KWH", "TOU_ONPEAK_BAHT_PER_KWH_BASE",
    "TOU_OFFPEAK_BAHT_PER_KWH_BASE", "DEMAND_CHARGE_BAHT_PER_KW_MONTH",
    "IF_HOLDING_PENALTY_PER_MIN", "LOW_LEVEL_NONLINEAR_FACTOR",
    "NUM_BATCHES_RUN_OVERRIDE", "PREFERRED_START_FURNACE",
    "MAX_LOW_LEVEL_MIN_ALLOW", "POLICY_OVERRIDE_MIN_START_GAP_MAX",
    "ENERGY_IDLE_GAP_COST_PER_MIN", "SOLAR_EFFECTIVE_PRICE_FACTOR",
    "SERVICE_W_MAKESPAN",
    "SERVICE_W_HOLDING_MINUTES", "CONTRACT_DEMAND_KW",
)


def _snap():
    out = {n: getattr(app_v9, n, None) for n in _GLOBALS}
    for k, v in list(out.items()):
        if isinstance(v, dict):
            out[k] = dict(v)
    out["_obj1"] = dict(getattr(app_v9, "OBJ1_COMPONENT_WEIGHTS", {}))
    out["_cache"] = dict(getattr(app_v9, "_EVAL_CACHE", {}))
    return out


def _restore(s):
    for n in _GLOBALS:
        v = s.get(n)
        if isinstance(v, dict):
            getattr(app_v9, n).clear()
            getattr(app_v9, n).update(v)
        else:
            setattr(app_v9, n, v)
    app_v9.OBJ1_COMPONENT_WEIGHTS.clear()
    app_v9.OBJ1_COMPONENT_WEIGHTS.update(s["_obj1"])
    app_v9._EVAL_CACHE.clear()
    app_v9._EVAL_CACHE.update(s["_cache"])


@pytest.fixture(autouse=True)
def _iso():
    s = _snap()
    try:
        yield
    finally:
        _restore(s)


# ---------------------------------------------------------------------------
# Fake plan + canonical settings dict (mirrors the production seed)
# ---------------------------------------------------------------------------

@dataclass
class FakePlan:
    target_batches: int = 9
    shift_start: datetime = datetime(2026, 6, 4, 8, 0, 0)
    opt_mode: str = "service"
    if_a_enabled: bool = True
    if_b_enabled: bool = True
    mh_a_consumption_rate: float | None = 2.5
    mh_b_consumption_rate: float | None = 2.6
    mh_a_initial_level_kg: float | None = 800.0
    mh_b_initial_level_kg: float | None = 1100.0
    consider_tou_price: bool = True
    consider_plant_load: bool = True
    preferred_start_furnace: str = "A"


_SETTINGS = {
    "mh_a_consumption_rate_kg_per_min": "2.5",
    "mh_b_consumption_rate_kg_per_min": "2.6",
    "mh_a_capacity_kg": "800",
    "mh_b_capacity_kg": "1100",
    "mh_a_initial_level_kg": "800",
    "mh_b_initial_level_kg": "1100",
    "mh_a_min_operational_level_kg": "400",
    "mh_b_min_operational_level_kg": "550",
    "mh_low_level_nonlinear_factor": "3.0",
    "mh_max_low_level_min_allow": "240",
    "ga_obj_weight_empty_penalty": "1.10",
    "ga_obj_weight_low_level_min": "0.80",
    "ga_obj_weight_low_level_shape": "0.90",
    # rest of the keys fall through to in-code defaults via `f(key, default)`.
}


# ---------------------------------------------------------------------------
# T1 — Service mode + flags off: MH protection STRONG, gaps forbidden
# ---------------------------------------------------------------------------
def test_service_flags_off_keeps_mh_protection_strong():
    plan = FakePlan(
        opt_mode="service",
        consider_tou_price=False,
        consider_plant_load=False,
    )
    _apply_settings_overrides(app_v9, _SETTINGS, plan)

    # MH protection NOT linearized — quadratic depth penalty still applies
    assert app_v9.LOW_LEVEL_NONLINEAR_FACTOR > 0, (
        "service+flags off must keep LOW_LEVEL_NONLINEAR_FACTOR > 0; "
        "linearizing lets GA happily drain MH below Min."
    )
    # OBJ1 safety weights boosted
    w = app_v9.OBJ1_COMPONENT_WEIGHTS
    # Fix 12: boosted ×3 from previous values so GA never tolerates dipping near Min.
    assert w["empty_penalty"] >= 30.0, f"empty_penalty={w['empty_penalty']}"
    assert w["low_level_min_penalty"] >= 30.0, f"low_level_min={w['low_level_min_penalty']}"
    assert w["low_level_shape_penalty"] >= 15.0, f"low_level_shape={w['low_level_shape_penalty']}"
    # Hard MH-min budget tightened to ≤ 5 min (Fix 12 — near zero tolerance)
    assert app_v9.MAX_LOW_LEVEL_MIN_ALLOW <= 5.0, (
        f"MAX_LOW_LEVEL_MIN_ALLOW={app_v9.MAX_LOW_LEVEL_MIN_ALLOW}; "
        f"service+flags off should tighten this from default 240."
    )
    # Empty budget also tightened
    assert app_v9.MAX_EMPTY_MIN_ALLOW <= 15.0, (
        f"MAX_EMPTY_MIN_ALLOW={app_v9.MAX_EMPTY_MIN_ALLOW}; "
        f"empty MH is unsafe — must be ≤ 15 min in service+flags off."
    )
    # Continuous melt: zero inter-start gap
    assert app_v9.POLICY_OVERRIDE_MIN_START_GAP_MAX == 0.0, (
        f"POLICY_OVERRIDE_MIN_START_GAP_MAX={app_v9.POLICY_OVERRIDE_MIN_START_GAP_MAX}"
    )
    # Cost terms still zeroed
    assert app_v9.IF_HOLDING_PENALTY_PER_MIN == 0.0
    assert app_v9.ENERGY_IDLE_GAP_COST_PER_MIN == 0.0
    assert app_v9.SOLAR_EFFECTIVE_PRICE_FACTOR == 1.0
    # Pour distribution is now proportional to per-furnace free space at the
    # simulator level (see _distribute_pour_proportional). The old static
    # PREFERRED_MH_FURNACE_TO_FILL_FIRST flag is no longer read.
    # Makespan reward — service objective normally lacks a makespan term.
    assert app_v9.SERVICE_W_MAKESPAN >= 1000.0, (
        f"expected SERVICE_W_MAKESPAN ≥ 1000 to dominate hold + reheat costs, "
        f"got {app_v9.SERVICE_W_MAKESPAN}"
    )
    # Holding-minutes weight RAISED in Fix 12.1 — combined with the cap on
    # start_delay_min the GA picks JIT dispatch (no hold) rather than "start
    # immediately and hold".
    assert app_v9.SERVICE_W_HOLDING_MINUTES >= 20.0, (
        f"SERVICE_W_HOLDING_MINUTES={app_v9.SERVICE_W_HOLDING_MINUTES}; "
        f"need ≥ 20 so GA prefers JIT dispatch over start-at-t=0-then-hold."
    )
    # CONTRACT_DEMAND_KW effectively infinite → margin_ratio stays positive →
    # peak_risk = 0 → _dynamic_min_start_gap returns 0 (no auto-gap from
    # baseline + IF projection).
    assert app_v9.CONTRACT_DEMAND_KW >= 1e6, (
        f"CONTRACT_DEMAND_KW={app_v9.CONTRACT_DEMAND_KW}; "
        f"must be ≫ baseline+IF kW to defeat _dynamic_min_start_gap."
    )


# ---------------------------------------------------------------------------
# T2 — Energy mode + flags off: same safety + continuous-melt as service mode,
#      but the makespan/hold REWARD is wired through the energy-mode cost
#      terms instead of the service-mode weights.
# ---------------------------------------------------------------------------
def test_energy_flags_off_drives_continuous_melt_via_energy_terms():
    plan = FakePlan(
        opt_mode="energy",
        consider_tou_price=False,
        consider_plant_load=False,
    )
    _apply_settings_overrides(app_v9, _SETTINGS, plan)

    # Safety overrides identical to service+flags off
    assert app_v9.LOW_LEVEL_NONLINEAR_FACTOR == 3.0, (
        "energy+flags off must keep quadratic low-level penalty; the old "
        "linearization was the proximate cause of MH-A draining unchecked."
    )
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] >= 30.0
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["low_level_min_penalty"] >= 30.0
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["low_level_shape_penalty"] >= 15.0
    assert app_v9.MAX_LOW_LEVEL_MIN_ALLOW <= 5.0
    assert app_v9.MAX_EMPTY_MIN_ALLOW <= 15.0

    # Continuous-melt geometry identical
    assert app_v9.POLICY_OVERRIDE_MIN_START_GAP_MAX == 0.0
    assert app_v9.CONTRACT_DEMAND_KW >= 1e6
    # Pour distribution is proportional (helper-level), no per-plan override.

    # ── Mode-specific: energy mode drives makespan via ENERGY_MAKESPAN_COST_PER_MIN ──
    # Fix 12: boosted to 100 so makespan dominates; IF hold dropped to 1 so
    # GA can hold briefly for early-start without lockout.
    assert app_v9.ENERGY_MAKESPAN_COST_PER_MIN >= 100.0, (
        f"energy+flags off must keep makespan cost ≥ 100 to push GA toward "
        f"tight schedules; got {app_v9.ENERGY_MAKESPAN_COST_PER_MIN}."
    )
    assert app_v9.ENERGY_IDLE_GAP_COST_PER_MIN >= 100.0
    assert app_v9.IF_HOLDING_PENALTY_PER_MIN >= 5.0, (
        f"IF_HOLDING_PENALTY_PER_MIN={app_v9.IF_HOLDING_PENALTY_PER_MIN}; "
        f"Fix 12.1 raises this so GA prefers JIT dispatch over start+hold."
    )
    # Service weights NOT activated in energy mode
    assert app_v9.SERVICE_W_MAKESPAN == 0.0
    assert app_v9.SERVICE_W_HOLDING_MINUTES == 1e1


# ---------------------------------------------------------------------------
# T3 — Flags ON (regression): no overrides kick in at all
# ---------------------------------------------------------------------------
def test_flags_on_leaves_safety_at_defaults():
    plan = FakePlan(
        opt_mode="service",
        consider_tou_price=True,
        consider_plant_load=True,
    )
    _apply_settings_overrides(app_v9, _SETTINGS, plan)

    # Quadratic penalty at settings default 3.0
    assert app_v9.LOW_LEVEL_NONLINEAR_FACTOR == 3.0
    # OBJ1 weights at settings defaults
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] == 1.10
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["low_level_min_penalty"] == 0.80
    # No POLICY_OVERRIDE: stays at the in-code default (None)
    # (we don't pin a value here because the in-code default is None and
    # the override block doesn't touch it when both flags are ON)


# ---------------------------------------------------------------------------
# T4 — Only ONE flag off: cost overrides for that flag, MH untouched
# ---------------------------------------------------------------------------
def test_single_flag_off_does_not_touch_mh_protection():
    plan = FakePlan(
        opt_mode="service",
        consider_tou_price=False,
        consider_plant_load=True,  # only TOU off
    )
    _apply_settings_overrides(app_v9, _SETTINGS, plan)

    # Quadratic penalty unchanged
    assert app_v9.LOW_LEVEL_NONLINEAR_FACTOR == 3.0
    # OBJ1 weights at settings defaults
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] == 1.10
    # MH-min budget at default
    assert app_v9.MAX_LOW_LEVEL_MIN_ALLOW == 240.0
    # But onpeak collapsed to offpeak
    assert app_v9.TOU_ONPEAK_BAHT_PER_KWH == app_v9.TOU_OFFPEAK_BAHT_PER_KWH


# ---------------------------------------------------------------------------
# T5 — Idempotency: two plans back-to-back don't leak service-mode overrides
#       into the next energy-mode plan
# ---------------------------------------------------------------------------
def test_overrides_do_not_leak_across_plans():
    # First: service + flags off, weights boosted to 3.0+
    p1 = FakePlan(
        opt_mode="service",
        consider_tou_price=False,
        consider_plant_load=False,
    )
    _apply_settings_overrides(app_v9, _SETTINGS, p1)
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] >= 30.0

    # Then: energy + flags on, weights MUST go back to settings defaults
    p2 = FakePlan(
        opt_mode="energy",
        consider_tou_price=True,
        consider_plant_load=True,
    )
    _apply_settings_overrides(app_v9, _SETTINGS, p2)
    assert app_v9.OBJ1_COMPONENT_WEIGHTS["empty_penalty"] == 1.10, (
        "service+flags off boosted weights leaked into a fresh energy+flags-on plan"
    )
    assert app_v9.MAX_LOW_LEVEL_MIN_ALLOW == 240.0, (
        "MAX_LOW_LEVEL_MIN_ALLOW leaked between plans"
    )
    assert app_v9.LOW_LEVEL_NONLINEAR_FACTOR == 3.0
    # Service-mode-specific overrides must NOT leak — CONTRACT_DEMAND_KW
    # comes from settings (1600 here), and SERVICE_W_MAKESPAN goes back
    # to module-default 0 because ga_service doesn't touch it outside the
    # service+flags-off branch.
    # Note: SERVICE_W_MAKESPAN doesn't get explicitly reset; if it leaked,
    # the snapshot fixture would clean it between tests but the next plan
    # in the same process could see it. We document this contract:
    assert app_v9.CONTRACT_DEMAND_KW == 1600.0, (
        f"CONTRACT_DEMAND_KW leaked: expected 1600 (from settings), got "
        f"{app_v9.CONTRACT_DEMAND_KW}"
    )
    assert app_v9.SERVICE_W_MAKESPAN == 0.0, (
        f"SERVICE_W_MAKESPAN leaked across plans: got {app_v9.SERVICE_W_MAKESPAN}"
    )
    assert app_v9.SERVICE_W_HOLDING_MINUTES == 1e1, (
        f"SERVICE_W_HOLDING_MINUTES leaked: got {app_v9.SERVICE_W_HOLDING_MINUTES}"
    )
    # PREFERRED_MH_FURNACE_TO_FILL_FIRST is no longer touched by ga_service
    # (pour distribution is proportional at the simulator level).
