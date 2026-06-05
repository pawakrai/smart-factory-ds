"""
Unit tests for the free-space proportional pour distribution.

Background: the original simulator filled MH-A to capacity first, then sent
whatever remained to MH-B (or vice versa depending on a global flag). When
the two furnaces had unequal consumption rates the second-priority furnace
ran dry — operators reported MH-B reaching zero with A=2.2 / B=2.95 kg/min.

The replacement function `_distribute_pour_proportional` allocates the 600 kg
batch in proportion to each furnace's current free space, so whichever MH
has drained more receives more of the pour.

These tests pin the algorithm's invariants without involving the GA or the
minute-by-minute simulator — fast, deterministic, and catch any regression
in the math layer immediately.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.app_v9 import _distribute_pour_proportional  # noqa: E402


# Standard IF batch — matches IF_BATCH_OUTPUT_KG default in the simulator.
TOTAL = 600.0


def _check_invariants(a_free, b_free, total, poured_a, poured_b):
    """Properties the helper must satisfy for any non-degenerate input."""
    # 1. Non-negative
    assert poured_a >= -1e-9, f"poured_A={poured_a} is negative"
    assert poured_b >= -1e-9, f"poured_B={poured_b} is negative"
    # 2. Never exceeds the furnace's free space
    assert poured_a <= a_free + 1e-9, f"poured_A={poured_a} > free_A={a_free}"
    assert poured_b <= b_free + 1e-9, f"poured_B={poured_b} > free_B={b_free}"
    # 3. Sum equals the pour total (no mass loss/creation)
    assert math.isclose(poured_a + poured_b, total, abs_tol=1e-6), (
        f"sum {poured_a + poured_b} != total {total}"
    )
    # 4. Finite
    assert math.isfinite(poured_a) and math.isfinite(poured_b)


# ---------------------------------------------------------------------------
# T1 — Balanced free space → equal split
# ---------------------------------------------------------------------------
def test_balanced_free_space_splits_evenly():
    poured_a, poured_b = _distribute_pour_proportional(600.0, 600.0, TOTAL)
    assert math.isclose(poured_a, 300.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 300.0, abs_tol=1e-6)
    _check_invariants(600.0, 600.0, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T2 — A has 2× the free space of B → A gets 2× the pour
# ---------------------------------------------------------------------------
def test_a_double_free_space_gets_double_pour():
    poured_a, poured_b = _distribute_pour_proportional(400.0, 200.0, TOTAL)
    # share_A = 600 * 400 / 600 = 400, share_B = 600 * 200 / 600 = 200
    assert math.isclose(poured_a, 400.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 200.0, abs_tol=1e-6)
    _check_invariants(400.0, 200.0, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T3 — B is full → A absorbs the whole pour
# ---------------------------------------------------------------------------
def test_b_full_routes_all_to_a():
    poured_a, poured_b = _distribute_pour_proportional(600.0, 0.0, TOTAL)
    assert math.isclose(poured_a, 600.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 0.0, abs_tol=1e-6)
    _check_invariants(600.0, 0.0, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T4 — A is full → B absorbs the whole pour
# ---------------------------------------------------------------------------
def test_a_full_routes_all_to_b():
    poured_a, poured_b = _distribute_pour_proportional(0.0, 600.0, TOTAL)
    assert math.isclose(poured_a, 0.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 600.0, abs_tol=1e-6)
    _check_invariants(0.0, 600.0, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T5 — Rate-imbalance scenario (operator's actual case)
#       free_A=300 (A drains slowly), free_B=500 (B drains fast) → B gets more
# ---------------------------------------------------------------------------
def test_rate_imbalance_distributes_correctly():
    poured_a, poured_b = _distribute_pour_proportional(300.0, 500.0, TOTAL)
    # share_A = 600 * 300 / 800 = 225, share_B = 600 * 500 / 800 = 375
    assert math.isclose(poured_a, 225.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 375.0, abs_tol=1e-6)
    _check_invariants(300.0, 500.0, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T6 — Float-precision edge: B has microscopic free space — no negative pours
# ---------------------------------------------------------------------------
def test_float_precision_no_negative_pour():
    poured_a, poured_b = _distribute_pour_proportional(600.0, 1e-7, TOTAL)
    _check_invariants(600.0, 1e-7, TOTAL, poured_a, poured_b)
    # Nearly all goes to A, B gets at most its 1e-7 free space
    assert poured_b <= 1e-7 + 1e-9
    assert poured_a >= TOTAL - 1e-6


# ---------------------------------------------------------------------------
# T7 — Sum-invariance across many random splits (parametric)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("free_a,free_b", [
    (350.0, 450.0),
    (700.0, 100.0),
    (100.0, 700.0),
    (250.0, 350.0),  # if_pour_amount_mh_a/b_kg defaults from settings
    (799.5, 0.5),
    (1.0, 599.0),
    (123.4, 567.8),
])
def test_sum_invariance(free_a, free_b):
    """For any feasible (free_a + free_b >= total) input, sum == total."""
    poured_a, poured_b = _distribute_pour_proportional(free_a, free_b, TOTAL)
    _check_invariants(free_a, free_b, TOTAL, poured_a, poured_b)


# ---------------------------------------------------------------------------
# T8 — Downtime on A → caller passes available_A = 0 → all goes to B
# ---------------------------------------------------------------------------
def test_downtime_on_a_routes_to_b():
    """The simulator's `available_A` is set to 0.0 when A is in post-pour
    downtime — same input shape as A being full."""
    poured_a, poured_b = _distribute_pour_proportional(0.0, 700.0, TOTAL)
    assert math.isclose(poured_a, 0.0, abs_tol=1e-6)
    assert math.isclose(poured_b, 600.0, abs_tol=1e-6)
    _check_invariants(0.0, 700.0, TOTAL, poured_a, poured_b)
