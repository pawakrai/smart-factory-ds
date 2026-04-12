"""
Script D-tier1 — Post-hoc Reward Weight Sensitivity Probe (Reviewer Item D)
=============================================================================
Records full (state, action, next_state) trajectories from the final DQN model
and analytically recomputes cumulative reward under OFAT (one-factor-at-a-time)
perturbations of the six reward component coefficients.

IMPORTANT FRAMING: This is a post-hoc sensitivity probe of the EVALUATION SCORE,
not a study of policy robustness under retraining. It answers: "If we had used
different reward weights, how would this policy's cumulative return change?"
This must be clearly stated in the paper.

The six reward components in env_11 step() lines 488-518:
  1. tracking_penalty  = -(|P_agent - P_expert| / max_power) * tracking_scale (0.5)
  2. safety_penalty    = -safety_scale (5.0) if is_alloying and action > 0
  3. progress_reward   = dprog * progress_scale (2.0)
  4. energy_penalty    = -energy_delta_kwh * energy_scale (0.02)
  5. time_penalty      = -time_scale (0.001) per step
  6. terminal_bonus    = +success_base (10.0) + norm_eff * success_eff (5.0) on success
                         OR -terminal_fail (5.0) on timeout

Perturbation: vary one coefficient at a time, multipliers [0.5, 0.75, 1.0, 1.25, 1.5].

Outputs (outputs/revision_phase1/rl_reward_sensitivity/):
  trajectories.json               — raw per-step trajectory data (100 episodes)
  relabeling_table.csv            — mean cumulative reward per (coefficient, multiplier)
  sensitivity_heatmap.png         — visual summary
  rl_reward_sensitivity_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/eval_rl_reward_sensitivity.py
"""

import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "rl_reward_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_final_model_env11.pth"
SEED = 2024
NUM_EPISODES = 100
IDLE_TIME_MIN_RANGE = (0, 30)
TARGET_TEMP_C = 950.0
AMBIENT_TEMP_C = 25.0
MAX_POWER_KW = 450.0
STATE_DIM = 8
ACTION_DIM = 10

ENV_KWARGS = dict(
    target_temp_c=TARGET_TEMP_C,
    start_mode="hot",
    initial_weight_kg=350,
    max_time_min=120,
)

# Baseline reward coefficients (matching env_11 lines 488-518)
BASE_COEFFICIENTS = {
    "tracking_scale":   0.5,    # line 488
    "safety_scale":     5.0,    # line 492
    "progress_scale":   2.0,    # line 499
    "energy_scale":     0.02,   # line 502
    "time_scale":       0.001,  # line 505
    "terminal_success_base": 10.0,   # line 518
    "terminal_success_eff":  5.0,    # line 515
    "terminal_fail":    5.0,    # line 520
}

# Perturb these four (most paper-relevant) one at a time
PERTURB_TARGETS = ["tracking_scale", "safety_scale", "progress_scale", "energy_scale"]
PERTURBATION_FACTORS = [0.50, 0.75, 1.00, 1.25, 1.50]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_agent() -> DQNAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    agent.model.load_state_dict(state_dict)
    agent.model.to(device)
    agent.model.eval()
    agent.epsilon = 0.0
    return agent


def _recompute_reward(
    state: np.ndarray,
    action: int,
    next_state: np.ndarray,
    done: bool,
    coeffs: dict,
    env_ref: AluminumMeltingEnvironment,
) -> float:
    """
    Analytically recompute the reward for a single transition using given coefficients.
    All inputs are derived from stored trajectory state arrays — no env calls.

    State layout (from env_11):
      [0] temperature_c
      [1] weight_kg
      [2] time_sec
      [3] power_kw (post-override, actually applied)
      [4] status
      [5] energy_kwh
      [6] scrap_added_kg
      [7] furnace_wall_temp_c
    """
    t_min = float(state[2]) / 60.0

    # The action that was chosen; target_power is what agent requested (= action's kW level)
    # Because env already has action_space, use it for mapping
    target_power = env_ref.action_space.get(action, 0.0)

    # 1. Tracking penalty
    expert_pwr = env_ref._expert_power_profile(t_min)
    power_error = abs(target_power - expert_pwr)
    tracking_penalty = -(power_error / MAX_POWER_KW) * coeffs["tracking_scale"]

    # 2. Safety penalty
    is_alloying = (30.0 <= t_min < 31.0)
    safety_penalty = -coeffs["safety_scale"] if (is_alloying and target_power > 0.0) else 0.0

    # 3. Progress reward — dprog uses next_state[0] vs state[0]
    denom = max(1e-9, float(TARGET_TEMP_C - AMBIENT_TEMP_C))
    prog_before = (float(state[0]) - AMBIENT_TEMP_C) / denom
    prog_after  = (float(next_state[0]) - AMBIENT_TEMP_C) / denom
    dprog = float(np.clip(prog_after, 0, 1)) - float(np.clip(prog_before, 0, 1))
    progress_reward = dprog * coeffs["progress_scale"]

    # 4. Energy penalty
    energy_delta = float(next_state[5]) - float(state[5])
    energy_penalty = -max(0.0, energy_delta) * coeffs["energy_scale"]

    # 5. Time penalty (always -1 per step)
    time_penalty = -coeffs["time_scale"]

    # 6. Terminal reward
    terminal = 0.0
    if done:
        final_temp = float(next_state[0])
        success = final_temp >= TARGET_TEMP_C
        if success:
            weight = float(next_state[1])
            total_energy = float(next_state[5])
            efficiency = weight / max(1e-6, total_energy)
            optimal_efficiency = 1.8
            norm_eff = min(efficiency / optimal_efficiency, 1.0)
            terminal = coeffs["terminal_success_base"] + norm_eff * coeffs["terminal_success_eff"]
        else:
            terminal = -coeffs["terminal_fail"]

    return (tracking_penalty + safety_penalty + progress_reward
            + energy_penalty + time_penalty + terminal)


# ── Phase 1: Record trajectories ──────────────────────────────────────────────

def record_trajectories(agent: DQNAgent, idle_times: list[float]) -> list[list[dict]]:
    """
    Run NUM_EPISODES episodes; for each step store (state, action, next_state, done).
    Returns a list of episodes, each a list of step dicts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_episodes = []

    for ep_idx, idle_time in enumerate(idle_times):
        env = AluminumMeltingEnvironment(idle_time_min=idle_time, **ENV_KWARGS)
        state = env.reset()
        done = False
        episode_steps = []

        while not done:
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = int(torch.argmax(agent.model(s_t)).item())
            next_state, _orig_reward, done = env.step(action)
            episode_steps.append({
                "state": state.tolist(),
                "action": action,
                "next_state": next_state.tolist(),
                "done": done,
            })
            state = next_state

        all_episodes.append(episode_steps)
        if (ep_idx + 1) % 20 == 0:
            print(f"  Recorded {ep_idx + 1}/{NUM_EPISODES} episodes "
                  f"({sum(len(e) for e in all_episodes)} steps total)")

    return all_episodes


# ── Phase 2: Relabel trajectories ─────────────────────────────────────────────

def relabel_episodes(
    episodes: list[list[dict]],
    coeffs: dict,
    env_ref: AluminumMeltingEnvironment,
) -> list[float]:
    """Recompute cumulative reward per episode under given coefficients."""
    cumulative_rewards = []
    for ep_steps in episodes:
        ep_reward = sum(
            _recompute_reward(
                np.array(step["state"], dtype=np.float32),
                step["action"],
                np.array(step["next_state"], dtype=np.float32),
                step["done"],
                coeffs,
                env_ref,
            )
            for step in ep_steps
        )
        cumulative_rewards.append(ep_reward)
    return cumulative_rewards


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("eval_rl_reward_sensitivity.py — Reviewer Item D (Tier 1)")
    print("=" * 65)
    print("FRAMING: Post-hoc sensitivity probe of evaluation score.")
    print("Does NOT reflect policy behavior under alternative training.\n")

    rng = np.random.default_rng(SEED)
    idle_times = [float(rng.uniform(*IDLE_TIME_MIN_RANGE)) for _ in range(NUM_EPISODES)]

    agent = _load_agent()
    # Reference env for action_space and _expert_power_profile (no physics needed)
    env_ref = AluminumMeltingEnvironment(**ENV_KWARGS)

    # ── Phase 1: Record trajectories ─────────────────────────────────────────
    print("Phase 1: Recording trajectories...")
    episodes = record_trajectories(agent, idle_times)
    total_steps = sum(len(e) for e in episodes)
    print(f"  {NUM_EPISODES} episodes, {total_steps} steps total")

    traj_path = OUTPUT_DIR / "trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(episodes, f)
    print(f"  Saved: {traj_path.relative_to(PROJECT_ROOT)}")

    # ── Phase 2: OFAT relabeling ──────────────────────────────────────────────
    print("\nPhase 2: OFAT coefficient relabeling...")
    rows = []

    for coeff_name in PERTURB_TARGETS:
        baseline_val = BASE_COEFFICIENTS[coeff_name]
        for factor in PERTURBATION_FACTORS:
            coeffs = BASE_COEFFICIENTS.copy()
            coeffs[coeff_name] = baseline_val * factor
            rewards = relabel_episodes(episodes, coeffs, env_ref)
            rewards_arr = np.array(rewards)
            row = {
                "perturbed_coefficient": coeff_name,
                "baseline_value": baseline_val,
                "factor": factor,
                "actual_value": coeffs[coeff_name],
                "mean_cumulative_reward": float(rewards_arr.mean()),
                "sd_cumulative_reward": float(rewards_arr.std()),
                "median_cumulative_reward": float(np.median(rewards_arr)),
                "min_cumulative_reward": float(rewards_arr.min()),
                "max_cumulative_reward": float(rewards_arr.max()),
            }
            rows.append(row)
            marker = " ← baseline" if factor == 1.0 else ""
            print(f"  {coeff_name} × {factor:.2f}: "
                  f"mean={row['mean_cumulative_reward']:+.3f} ± {row['sd_cumulative_reward']:.3f}{marker}")

    # ── Save relabeling table ─────────────────────────────────────────────────
    table_path = OUTPUT_DIR / "relabeling_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {table_path.relative_to(PROJECT_ROOT)}")

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Build matrix: rows = coefficients, cols = factors
        matrix = np.zeros((len(PERTURB_TARGETS), len(PERTURBATION_FACTORS)))
        for i, name in enumerate(PERTURB_TARGETS):
            for j, factor in enumerate(PERTURBATION_FACTORS):
                matched = [r for r in rows
                           if r["perturbed_coefficient"] == name
                           and abs(r["factor"] - factor) < 1e-6]
                if matched:
                    matrix[i, j] = matched[0]["mean_cumulative_reward"]

        # Normalize relative to baseline (factor=1.0)
        baseline_col = PERTURBATION_FACTORS.index(1.0)
        rel_matrix = (matrix - matrix[:, baseline_col:baseline_col+1]) / (
            np.abs(matrix[:, baseline_col:baseline_col+1]) + 1e-6
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

        im1 = ax1.imshow(matrix, aspect="auto", cmap="RdYlGn")
        ax1.set_xticks(range(len(PERTURBATION_FACTORS)))
        ax1.set_xticklabels([f"×{f}" for f in PERTURBATION_FACTORS])
        ax1.set_yticks(range(len(PERTURB_TARGETS)))
        ax1.set_yticklabels(PERTURB_TARGETS)
        ax1.set_title("Absolute Mean Cumulative Reward")
        plt.colorbar(im1, ax=ax1)
        for i in range(len(PERTURB_TARGETS)):
            for j in range(len(PERTURBATION_FACTORS)):
                ax1.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=8)

        im2 = ax2.imshow(rel_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        ax2.set_xticks(range(len(PERTURBATION_FACTORS)))
        ax2.set_xticklabels([f"×{f}" for f in PERTURBATION_FACTORS])
        ax2.set_yticks(range(len(PERTURB_TARGETS)))
        ax2.set_yticklabels(PERTURB_TARGETS)
        ax2.set_title("Relative Change from Baseline (×1.0)")
        plt.colorbar(im2, ax=ax2)
        for i in range(len(PERTURB_TARGETS)):
            for j in range(len(PERTURBATION_FACTORS)):
                ax2.text(j, i, f"{rel_matrix[i,j]:+.2f}", ha="center", va="center", fontsize=8)

        plt.suptitle(
            "Post-hoc Reward Sensitivity Probe\n"
            "(Fixed policy; reward score recomputed under alternative weights)",
            fontsize=10,
        )
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "sensitivity_heatmap.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {fig_path.relative_to(PROJECT_ROOT)}")
    except ImportError:
        print("  [SKIP] matplotlib not available.")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "eval_rl_reward_sensitivity.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "idle_time_min_range": list(IDLE_TIME_MIN_RANGE),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "env_kwargs": ENV_KWARGS,
        "episode_conditions": [
            {"episode": i + 1, "idle_time_min": t}
            for i, t in enumerate(idle_times)
        ],
        "base_coefficients": BASE_COEFFICIENTS,
        "perturb_targets": PERTURB_TARGETS,
        "perturbation_factors": PERTURBATION_FACTORS,
        "framing_note": (
            "Post-hoc sensitivity: fixed trained policy; reward score recomputed under "
            "alternative weights. Does NOT reflect behavior if policy were retrained. "
            "See train_rl_reward_variants.py for empirical policy sensitivity."
        ),
        "filters_applied": [],
    }
    with open(OUTPUT_DIR / "rl_reward_sensitivity_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/rl_reward_sensitivity/")
    print("=" * 65)


if __name__ == "__main__":
    main()
