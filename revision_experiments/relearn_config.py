"""
relearn_config.py — Shared Configuration for RL Reward-Retuning Study
======================================================================
Provides constants, tuning grid, PatchedEnvV2 subclass, and parity gate
for the controlled RL relearning study (journal revision).

All training uses start_mode="hot" only, preserving the original training
distribution from train_with_env_11.py.  Generalization to cold-start and
mixed conditions is assessed exclusively via the evaluation protocol.

No files under src/ are modified.
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "revision_relearn"

# ── Training config ──────────────────────────────────────────────────────────
TRAIN_EPISODES = 1500
SEEDS = [42, 2024, 7]

# ── Evaluation config ────────────────────────────────────────────────────────
EVAL_SEED = 2024
EVAL_EPISODES = 100
HOT_IDLE_RANGE = (0.0, 30.0)
HOT_FRACTION = 0.80

# ── Environment config ───────────────────────────────────────────────────────
TARGET_TEMP_C = 950.0
STATE_DIM = 8
ACTION_DIM = 10
MAX_POWER_KW = 450.0

ENV_KWARGS = dict(
    target_temp_c=TARGET_TEMP_C,
    start_mode="hot",
    initial_weight_kg=350,
    max_time_min=120,
)

# ── Agent config (identical to original train_with_env_11.py) ────────────────
AGENT_KWARGS = dict(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    learning_rate=0.0005,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    memory_size=10000,
    batch_size=64,
    target_update_freq=100,
    hidden_size=512,
)

# ── Baseline reward coefficients (must match env_11 lines 488-520 exactly) ───
BASE_COEFFS = {
    "tracking_scale":        0.5,
    "safety_scale":          5.0,
    "progress_scale":        2.0,
    "energy_scale":          0.02,
    "time_scale":            0.001,
    "terminal_success_base": 10.0,
    "terminal_success_eff":  5.0,
    "terminal_fail":         5.0,
    "optimal_efficiency":    1.8,
    "overshoot_scale":       0.0,   # new: no overshoot penalty at baseline
}

# ── Tuning grid (pre-registered, 6 configurations) ──────────────────────────
TUNING_GRID = {
    "baseline_reward": {
        **BASE_COEFFS,
    },
    "energy_x1.25": {
        **BASE_COEFFS,
        "energy_scale": 0.025,
    },
    "energy_x1.50": {
        **BASE_COEFFS,
        "energy_scale": 0.030,
    },
    "energy_x2.00": {
        **BASE_COEFFS,
        "energy_scale": 0.040,
    },
    "energy_x1.50_plus_overshoot": {
        **BASE_COEFFS,
        "energy_scale": 0.030,
        "overshoot_scale": 0.5,
    },
    "energy_x1.25_plus_overshoot": {
        **BASE_COEFFS,
        "energy_scale": 0.025,
        "overshoot_scale": 0.5,
    },
}

# ── Model path for original DQN final ───────────────────────────────────────
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "models" / "dqn_final_model_env11.pth"

# ── Duration guard for model selection ───────────────────────────────────────
DURATION_GUARD_MINUTES = 2.0  # max allowed duration increase vs original DQN


# ─────────────────────────────────────────────────────────────────────────────
# PatchedEnvV2 — reward-only override, physics unchanged
# ─────────────────────────────────────────────────────────────────────────────

class PatchedEnvV2(AluminumMeltingEnvironment):
    """
    Subclass of AluminumMeltingEnvironment that replaces the reward function
    with a parameterized version while keeping all physics 100% unchanged.

    Physics update: delegated entirely to super().step().
    Reward computation: replicated from env_11 lines 488-520 with substituted
    coefficients plus an optional overshoot penalty term.

    The dprog term requires the pre-step progress value, which must be captured
    BEFORE calling super().step() (parent updates self._prev_progress internally).
    """

    def __init__(self, *args, reward_coeffs: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._coeffs = BASE_COEFFS.copy()
        if reward_coeffs:
            self._coeffs.update(reward_coeffs)

    def step(self, action, power_profile_kw=None):
        # ── Capture pre-step values before parent modifies state ─────────
        t_min_before = float(self.state["time"]) / 60.0
        if power_profile_kw is None:
            target_power = self.action_space.get(int(action), 0.0)
        else:
            target_power = float(power_profile_kw(t_min_before))
        prev_progress = self._progress()

        # ── Delegate physics to parent ───────────────────────────────────
        state_arr, _orig_reward, done = super().step(action, power_profile_kw=power_profile_kw)

        # ── Recompute reward with patched coefficients ───────────────────
        t_min = t_min_before  # alloying check uses pre-step time
        is_alloying_time = 30.0 <= t_min < 31.0

        # 5.1 Expert tracking penalty
        expert_pwr = self._expert_power_profile(t_min)
        power_error = abs(target_power - expert_pwr)
        tracking_penalty = -(power_error / self.max_power_kw) * self._coeffs["tracking_scale"]

        # 5.2 Safety penalty
        safety_penalty = -self._coeffs["safety_scale"] if (is_alloying_time and target_power > 0.0) else 0.0

        # 5.3 Progress reward
        prog = self._progress()
        dprog = prog - prev_progress
        progress_reward = dprog * self._coeffs["progress_scale"]

        # 5.4 Energy efficiency penalty
        applied_power_kw = float(self.state["power"]) + self.auxiliary_power_kw
        energy_used_kwh = float((applied_power_kw * self.dt) / 3600.0) * self.energy_consumption_scale
        energy_penalty = -energy_used_kwh * self._coeffs["energy_scale"]

        # 5.5 Time penalty
        time_penalty = -self._coeffs["time_scale"]

        # 5.6 Terminal reward
        terminal = 0.0
        success = False
        if done:
            final_temp = float(self.state["temperature"])
            success = final_temp >= self.target_temp
            if success:
                weight = float(self.state["weight"])
                total_energy = float(self.state["energy_consumption"])
                efficiency = weight / max(1e-6, total_energy)
                norm_eff = min(efficiency / self._coeffs["optimal_efficiency"], 1.0)
                terminal = self._coeffs["terminal_success_base"] + norm_eff * self._coeffs["terminal_success_eff"]

                # Overshoot penalty (new additive term)
                overshoot_scale = self._coeffs.get("overshoot_scale", 0.0)
                if overshoot_scale > 0.0:
                    overshoot_c = max(0.0, final_temp - self.target_temp)
                    terminal -= overshoot_c * overshoot_scale
            else:
                terminal = -self._coeffs["terminal_fail"]

        reward = tracking_penalty + safety_penalty + progress_reward + energy_penalty + time_penalty + terminal
        return state_arr, float(reward), done


# ─────────────────────────────────────────────────────────────────────────────
# Parity Gate
# ─────────────────────────────────────────────────────────────────────────────

def run_parity_gate(env_kwargs: dict | None = None, verbose: bool = True) -> dict:
    """
    Verify that PatchedEnvV2 at baseline coefficients (overshoot_scale=0)
    produces identical state transitions and rewards to the original
    AluminumMeltingEnvironment on a fixed action sequence.

    Returns a dict with 'passed' (bool) and 'details'.
    """
    if verbose:
        print("\n--- Mandatory Parity Gate ---")

    ek = env_kwargs or ENV_KWARGS
    TEST_SEED = 0
    TEST_ACTIONS = [9, 5, 9, 0, 7, 9]

    env_orig = AluminumMeltingEnvironment(seed=TEST_SEED, **ek)
    env_patch = PatchedEnvV2(seed=TEST_SEED, reward_coeffs=BASE_COEFFS, **ek)

    state_o = env_orig.reset()
    state_p = env_patch.reset()

    details = []
    passed = True

    if not np.allclose(state_o, state_p, atol=1e-5):
        passed = False
        details.append(f"FAIL reset: max_diff={np.abs(np.asarray(state_o) - np.asarray(state_p)).max():.8f}")
    else:
        details.append("PASS reset state match")

    for i, action in enumerate(TEST_ACTIONS):
        so, ro, do = env_orig.step(action)
        sp, rp, dp = env_patch.step(action)

        state_ok = np.allclose(so, sp, atol=1e-4)
        reward_ok = abs(ro - rp) < 1e-3
        done_ok = (do == dp)

        if not (state_ok and reward_ok and done_ok):
            passed = False
            details.append(
                f"FAIL step {i+1} action={action}: "
                f"state_diff={np.abs(np.asarray(so) - np.asarray(sp)).max():.6f} "
                f"reward_orig={ro:.6f} reward_patch={rp:.6f} "
                f"done_orig={do} done_patch={dp}"
            )
        else:
            details.append(
                f"PASS step {i+1} action={action}: "
                f"reward={ro:.6f} (patch={rp:.6f})"
            )

        if do or dp:
            break

    result = {
        "passed": passed,
        "test_seed": TEST_SEED,
        "test_actions": TEST_ACTIONS,
        "details": details,
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        for line in details:
            print(f"  {line}")
        if passed:
            print("  ✓ Parity PASSED — PatchedEnvV2 matches original env")
        else:
            print("  ✗ Parity FAILED — DO NOT proceed with training")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (reusable by eval script)
# ─────────────────────────────────────────────────────────────────────────────

ENV_EVAL_BASE_KWARGS = dict(
    target_temp_c=TARGET_TEMP_C,
    dt_sec=60,
)


def build_episode_configs(rng, scenario: str, n: int) -> list:
    """
    Build per-episode start configs for a given scenario.
    Identical logic to eval_rl_cold_hot_mixed.py lines 66-89.
    """
    configs = []
    if scenario == "cold_start":
        for _ in range(n):
            configs.append({"start_mode": "cold", "idle_time_min": None, "sub_condition": "cold"})

    elif scenario == "hot_start":
        for _ in range(n):
            idle = float(rng.uniform(*HOT_IDLE_RANGE))
            configs.append({"start_mode": "hot", "idle_time_min": idle, "sub_condition": "hot"})

    elif scenario == "mixed_80_20":
        n_hot = int(round(n * HOT_FRACTION))
        n_cold = n - n_hot
        labels = ["hot"] * n_hot + ["cold"] * n_cold
        rng.shuffle(labels)
        for lbl in labels:
            if lbl == "hot":
                idle = float(rng.uniform(*HOT_IDLE_RANGE))
                configs.append({"start_mode": "hot", "idle_time_min": idle, "sub_condition": "hot"})
            else:
                configs.append({"start_mode": "cold", "idle_time_min": None, "sub_condition": "cold"})
    return configs


def make_eval_env(cfg: dict) -> AluminumMeltingEnvironment:
    """Create evaluation environment from episode config (original env, not patched)."""
    kwargs = dict(ENV_EVAL_BASE_KWARGS)
    kwargs["start_mode"] = cfg["start_mode"]
    if cfg["idle_time_min"] is not None:
        kwargs["idle_time_min"] = cfg["idle_time_min"]
    return AluminumMeltingEnvironment(**kwargs)


def seed_all(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit_short() -> str:
    """Get short git commit hash, or 'unknown' if unavailable."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"
