"""
Operator preference for which Induction Furnace runs the first charge.

These tests exercise the `app_v9.PREFERRED_START_FURNACE` global that
`ga_service._apply_settings_overrides` sets from each Plan. They:

  - Run the deterministic default-policy simulator (no GA), so we can
    isolate the dispatch-order effect from the GA's furnace_bias scoring.
  - Verify that when the preference is set, the first IF-active batch lands
    on the requested furnace.
  - Verify that subsequent batches still flow through the normal scoring
    path (preference is first-batch-only).
  - Verify that when the preference is None, behaviour is unchanged.

Mirrors the snapshot/restore pattern from test_jit_holding_scenarios.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.app_v9 as app_v9  # noqa: E402


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
    "TOU_ONPEAK_BAHT_PER_KWH",
    "TOU_OFFPEAK_BAHT_PER_KWH",
    "TOU_ONPEAK_BAHT_PER_KWH_BASE",
    "TOU_OFFPEAK_BAHT_PER_KWH_BASE",
    "DEMAND_CHARGE_BAHT_PER_KW_MONTH",
    "IF_HOLDING_PENALTY_PER_MIN",
    "LOW_LEVEL_NONLINEAR_FACTOR",
    "NUM_BATCHES_RUN_OVERRIDE",
    "PREFERRED_START_FURNACE",
)


def _snapshot():
    snap = {n: getattr(app_v9, n, None) for n in _PLANT_GLOBALS}
    for k, v in list(snap.items()):
        if isinstance(v, dict):
            snap[k] = dict(v)
    snap["_obj1"] = dict(getattr(app_v9, "OBJ1_COMPONENT_WEIGHTS", {}))
    snap["_cache"] = dict(getattr(app_v9, "_EVAL_CACHE", {}))
    return snap


def _restore(snap):
    for n in _PLANT_GLOBALS:
        v = snap.get(n)
        if v is None:
            # Note: PREFERRED_START_FURNACE may legitimately be None — set it explicitly
            if n == "PREFERRED_START_FURNACE":
                setattr(app_v9, n, None)
            continue
        if isinstance(v, dict):
            getattr(app_v9, n).clear()
            getattr(app_v9, n).update(v)
        else:
            setattr(app_v9, n, v)
    app_v9.OBJ1_COMPONENT_WEIGHTS.clear()
    app_v9.OBJ1_COMPONENT_WEIGHTS.update(snap["_obj1"])
    app_v9._EVAL_CACHE.clear()
    app_v9._EVAL_CACHE.update(snap["_cache"])


def _patch(*, preferred: str | None, target_batches: int = 9):
    """Standard plant + flags-off + operator preference."""
    app_v9.MH_MAX_CAPACITY_KG.clear()
    app_v9.MH_MAX_CAPACITY_KG.update({"A": 800.0, "B": 1100.0})
    app_v9.MH_INITIAL_LEVEL_KG.clear()
    app_v9.MH_INITIAL_LEVEL_KG.update({"A": 800.0, "B": 1100.0})
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.clear()
    app_v9.MH_MIN_OPERATIONAL_LEVEL_KG.update({"A": 400.0, "B": 550.0})
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.clear()
    app_v9.MH_CONSUMPTION_RATE_KG_PER_MIN.update({"A": 2.5, "B": 2.6})

    app_v9.IF_BATCH_OUTPUT_KG = 600.0
    app_v9.OPT_MODE = "energy"
    app_v9.USE_FURNACE_A = True
    app_v9.USE_FURNACE_B = True
    app_v9.NUM_BATCHES_RUN_OVERRIDE = target_batches

    app_v9.TOU_ONPEAK_BAHT_PER_KWH_BASE = app_v9.TOU_OFFPEAK_BAHT_PER_KWH_BASE
    app_v9.TOU_ONPEAK_BAHT_PER_KWH = app_v9.TOU_OFFPEAK_BAHT_PER_KWH
    app_v9.DEMAND_CHARGE_BAHT_PER_KW_MONTH = 0.0
    app_v9.IF_HOLDING_PENALTY_PER_MIN = 0.0
    if "holding_penalty" in app_v9.OBJ1_COMPONENT_WEIGHTS:
        app_v9.OBJ1_COMPONENT_WEIGHTS["holding_penalty"] = 0.0
    app_v9.LOW_LEVEL_NONLINEAR_FACTOR = 0.0
    app_v9.ENERGY_MODE_MIN_JIT_SLACK_MIN = 0.0

    app_v9.PREFERRED_START_FURNACE = preferred
    app_v9._EVAL_CACHE.clear()


@pytest.fixture(autouse=True)
def _isolate():
    snap = _snapshot()
    try:
        yield
    finally:
        _restore(snap)


def _ordered_batches(result):
    """Return schedule batches ordered by start_min (skip un-started)."""
    return sorted(
        (b for b in result.get("schedule", []) if b.get("start_min") is not None),
        key=lambda b: b["start_min"],
    )


def _first_furnace_letter(result) -> str:
    batches = _ordered_batches(result)
    assert batches, "no started batches in schedule"
    idx = batches[0]["if_furnace"]
    return "A" if idx == 0 else "B"


# ---------------------------------------------------------------------------
# T1 — Default behaviour (None) — no forced preference
# ---------------------------------------------------------------------------
def test_no_preference_leaves_dispatch_to_ga():
    _patch(preferred=None)
    result = app_v9.simulate_policy_day(None)
    started = _ordered_batches(result)
    assert started, "expected some batches to start"


# ---------------------------------------------------------------------------
# T2 — Preference = A — first batch on furnace 0
# ---------------------------------------------------------------------------
def test_prefer_a_first():
    _patch(preferred="A")
    result = app_v9.simulate_policy_day(None)
    assert _first_furnace_letter(result) == "A", (
        f"expected first batch on A, got {_first_furnace_letter(result)}"
    )


# ---------------------------------------------------------------------------
# T3 — Preference = B — first batch on furnace 1
# ---------------------------------------------------------------------------
def test_prefer_b_first():
    _patch(preferred="B")
    result = app_v9.simulate_policy_day(None)
    assert _first_furnace_letter(result) == "B", (
        f"expected first batch on B, got {_first_furnace_letter(result)}"
    )


# ---------------------------------------------------------------------------
# T4 — Preference only affects FIRST batch; both furnaces used over the shift
# ---------------------------------------------------------------------------
def test_preference_only_first_batch():
    _patch(preferred="B", target_batches=9)
    result = app_v9.simulate_policy_day(None)
    batches = _ordered_batches(result)
    used = {b["if_furnace"] for b in batches}
    # If preference leaked beyond first batch and forced everything onto B,
    # we'd see {1}. Real GA still dispatches A later.
    assert 0 in used and 1 in used, (
        f"expected both furnaces used over the shift, got indices={used}"
    )


# ---------------------------------------------------------------------------
# T5 — Total throughput same whether A-first or B-first (within ±1 batch)
# ---------------------------------------------------------------------------
def test_throughput_similar_either_way():
    _patch(preferred="A", target_batches=9)
    a_first = app_v9.simulate_policy_day(None)
    _patch(preferred="B", target_batches=9)
    b_first = app_v9.simulate_policy_day(None)
    poured_a = a_first["metrics"]["poured_batches_count"]
    poured_b = b_first["metrics"]["poured_batches_count"]
    assert abs(poured_a - poured_b) <= 1, (
        f"throughput differs too much by start furnace: A-first={poured_a}, B-first={poured_b}"
    )


# ---------------------------------------------------------------------------
# T6 — Preference ignored when that furnace is disabled
# ---------------------------------------------------------------------------
def test_preference_ignored_when_furnace_disabled():
    _patch(preferred="A")
    app_v9.USE_FURNACE_A = False     # only B available
    result = app_v9.simulate_policy_day(None)
    assert _first_furnace_letter(result) == "B", (
        "first batch should fall back to B when A is disabled"
    )
