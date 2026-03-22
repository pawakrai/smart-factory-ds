"""
RL Service — wraps trained DQN models to generate a power profile guideline.
Returns a list of {time_min, power_kw} dicts.
"""

from __future__ import annotations
from typing import Any


def get_power_profile(batch_id: str, duration_min: int = 120) -> list[dict[str, Any]]:
    """
    Returns a recommended power profile for a batch.
    Attempts to use the trained DQN model; falls back to a mock profile.
    """
    try:
        return _run_rl_model(duration_min)
    except Exception:
        return _mock_profile(duration_min)


def _run_rl_model(duration_min: int) -> list[dict[str, Any]]:
    """Placeholder — wire to src/agents/ + models/*.pth in Phase 4."""
    raise NotImplementedError("RL model integration pending Phase 4")


def _mock_profile(duration_min: int) -> list[dict[str, Any]]:
    """
    Induction furnace power profile:
      0–5 min  : 50 kW  (initial heat-up)
      5–10 min : 150 kW
      10–15 min: 250 kW
      15–35 min: 350 kW
      35 min   : 0 kW   (Si+Fe addition pause)
      36–40 min: 400 kW (recovery ramp)
      40–end   : 450 kW (max power hold)
      end      : 0 kW   (batch complete)
    """
    def _kw_at(t: int) -> float:
        if t < 5:            return 50.0
        if t < 10:           return 150.0
        if t < 15:           return 250.0
        if t < 35:           return 350.0
        if t < 36:           return 0.0    # Si+Fe addition
        if t < 40:           return 400.0
        if t < duration_min: return 450.0
        return 0.0  # sharp drop at batch end

    # Emit all key transition points + every 5 min after t=40
    key_points: set[int] = {0, 5, 10, 15, 35, 36, 40, duration_min}
    for t in range(45, duration_min, 5):
        key_points.add(t)

    return [
        {"time_min": t, "power_kw": _kw_at(t)}
        for t in sorted(key_points)
        if t <= duration_min
    ]
