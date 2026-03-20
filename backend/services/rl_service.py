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
    """Ramp-up → hold → ramp-down power profile."""
    profile = []
    for t in range(0, duration_min + 1, 5):
        if t < 20:
            kw = 100 + t * 15
        elif t < 90:
            kw = 400
        else:
            kw = max(100, 400 - (t - 90) * 10)
        profile.append({"time_min": t, "power_kw": kw})
    return profile
