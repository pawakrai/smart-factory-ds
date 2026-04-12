"""
Script B — Extended RL Evaluation (Reviewer Item B)
=====================================================
Replaces the original 50-episode, no-seed evaluation in
src/training/evaluate_model_env11.py with a larger, seeded, and more
comprehensive evaluation that supports reviewer claims:

  B1. Multi-episode robustness summary with fixed seed (100 episodes)
  B2. Additional metrics: success rate, overshoot, final temperature
  B3. Comparison against the expert heuristic profile (simulation baseline)

NOTE: This script is NOT the source for reviewer item A (historical baseline).
      Historical baseline statistics come only from real plant data —
      see revision_experiments/extract_historical_baseline.py.

Outputs (outputs/revision_phase1/rl_eval/):
  rl_extended_per_episode.json   — per-episode raw data for both policies
  rl_extended_summary.csv        — mean/SD/median/min/max/success_rate
  rl_eval_manifest.json          — reproducibility sidecar

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/eval_rl_extended.py
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

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "rl_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_final_model_env11.pth"
SEED = 2024
NUM_EPISODES = 100
IDLE_TIME_MIN_RANGE = (0, 30)
TARGET_TEMP_C = 950.0
STATE_DIM = 8
ACTION_DIM = 10

ENV_KWARGS = dict(
    target_temp_c=TARGET_TEMP_C,
    start_mode="hot",
    initial_weight_kg=350,
    max_time_min=120,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_dqn_agent(model_path: Path) -> DQNAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    state_dict = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(state_dict)
    agent.model.to(device)
    agent.model.eval()
    agent.epsilon = 0.0
    return agent


def _run_episodes(
    policy_name: str,
    agent_or_profile,
    idle_times: list[float],
    rng: np.random.Generator,
) -> list[dict]:
    """
    Run NUM_EPISODES episodes.

    agent_or_profile: either a DQNAgent (uses argmax Q-values) or
                      a callable f(t_min) -> power_kw (uses env.step power_profile_kw).
    idle_times: pre-sampled idle_time_min values (one per episode).
    """
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_agent = isinstance(agent_or_profile, DQNAgent)

    for ep_idx, idle_time in enumerate(idle_times):
        env = AluminumMeltingEnvironment(
            idle_time_min=idle_time,
            **ENV_KWARGS,
        )
        state = env.reset()
        done = False
        steps = 0

        while not done:
            if is_agent:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = int(torch.argmax(agent_or_profile.model(s_t)).item())
                state, _reward, done = env.step(action)
            else:
                # Fixed power-profile: pass the callable directly to env.step
                state, _reward, done = env.step(action=0, power_profile_kw=agent_or_profile)
            steps += 1

        final_temp = float(env.state["temperature"])
        success = bool(final_temp >= TARGET_TEMP_C)
        energy = float(env.state["energy_consumption"])
        duration_min = steps * env.dt / 60.0
        overshoot_c = max(0.0, final_temp - TARGET_TEMP_C)

        ep_result = {
            "policy": policy_name,
            "episode": ep_idx + 1,
            "idle_time_min": float(idle_time),
            "duration_min": float(duration_min),
            "total_energy_kwh": float(energy),
            "final_temp_c": float(final_temp),
            "success": success,
            "overshoot_c": float(overshoot_c),
        }
        results.append(ep_result)

        if (ep_idx + 1) % 10 == 0:
            print(
                f"  [{policy_name}] Episode {ep_idx + 1:>3d}/{NUM_EPISODES} | "
                f"Duration: {duration_min:6.1f} min | "
                f"Energy: {energy:7.2f} kWh | "
                f"Success: {success} | "
                f"Temp: {final_temp:.1f}°C"
            )

    return results


def _compute_summary(results: list[dict], policy_name: str) -> dict:
    durations = np.array([r["duration_min"] for r in results])
    energies = np.array([r["total_energy_kwh"] for r in results])
    successes = np.array([r["success"] for r in results], dtype=float)
    overshoots = np.array([r["overshoot_c"] for r in results])
    final_temps = np.array([r["final_temp_c"] for r in results])

    return {
        "policy": policy_name,
        "n_episodes": len(results),
        # Duration
        "duration_mean": float(np.mean(durations)),
        "duration_sd": float(np.std(durations)),
        "duration_median": float(np.median(durations)),
        "duration_min": float(np.min(durations)),
        "duration_max": float(np.max(durations)),
        # Energy
        "energy_mean_kwh": float(np.mean(energies)),
        "energy_sd_kwh": float(np.std(energies)),
        "energy_median_kwh": float(np.median(energies)),
        "energy_min_kwh": float(np.min(energies)),
        "energy_max_kwh": float(np.max(energies)),
        # Success / temperature
        "success_rate": float(np.mean(successes)),
        "success_count": int(np.sum(successes)),
        "mean_final_temp_c": float(np.mean(final_temps)),
        "sd_final_temp_c": float(np.std(final_temps)),
        "mean_overshoot_c": float(np.mean(overshoots[overshoots > 0])) if any(overshoots > 0) else 0.0,
        "overshoot_rate": float(np.mean(overshoots > 0)),
    }


def _print_summary(s: dict) -> None:
    print(f"\n  Policy: {s['policy']}")
    print(f"    N episodes     : {s['n_episodes']}")
    print(f"    Success rate   : {s['success_rate']:.1%}  ({s['success_count']} / {s['n_episodes']})")
    print(f"    Duration (min) : {s['duration_mean']:.1f} ± {s['duration_sd']:.1f}  "
          f"[{s['duration_min']:.0f}, {s['duration_max']:.0f}]")
    print(f"    Energy (kWh)   : {s['energy_mean_kwh']:.2f} ± {s['energy_sd_kwh']:.2f}  "
          f"[{s['energy_min_kwh']:.2f}, {s['energy_max_kwh']:.2f}]")
    print(f"    Final temp (°C): {s['mean_final_temp_c']:.1f} ± {s['sd_final_temp_c']:.1f}")
    print(f"    Overshoot rate : {s['overshoot_rate']:.1%}  (mean overshoot: {s['mean_overshoot_c']:.2f}°C)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("eval_rl_extended.py — Reviewer Item B")
    print("=" * 65)
    print(f"Seed        : {SEED}")
    print(f"Episodes    : {NUM_EPISODES}")
    print(f"Model       : {MODEL_PATH.relative_to(PROJECT_ROOT)}")

    # Reproducible idle_time samples — shared across both policies for fair comparison
    rng = np.random.default_rng(SEED)
    idle_times = [float(rng.uniform(*IDLE_TIME_MIN_RANGE)) for _ in range(NUM_EPISODES)]

    # ── Policy 1: DQN agent ──────────────────────────────────────────────────
    print(f"\n--- Policy: DQN (final model) ---")
    agent = _load_dqn_agent(MODEL_PATH)
    dqn_results = _run_episodes("dqn_final", agent, idle_times, rng)
    dqn_summary = _compute_summary(dqn_results, "dqn_final")
    _print_summary(dqn_summary)

    # ── Policy 2: Expert profile (simulation-level heuristic baseline) ───────
    # NOTE: this is a simulation ablation baseline, NOT the plant historical baseline.
    # Use a fresh env just to access the method reference.
    _ref_env = AluminumMeltingEnvironment(**ENV_KWARGS)
    expert_profile = _ref_env._expert_power_profile  # safe: no side effects

    print(f"\n--- Policy: Expert profile (simulation heuristic) ---")
    expert_results = _run_episodes("expert_profile", expert_profile, idle_times, rng)
    expert_summary = _compute_summary(expert_results, "expert_profile")
    _print_summary(expert_summary)

    # ── Save per-episode records ──────────────────────────────────────────────
    all_results = dqn_results + expert_results
    per_ep_path = OUTPUT_DIR / "rl_extended_per_episode.json"
    with open(per_ep_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summaries = [dqn_summary, expert_summary]
    summary_path = OUTPUT_DIR / "rl_extended_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\n  Saved per-episode : {per_ep_path.relative_to(PROJECT_ROOT)}")
    print(f"  Saved summary     : {summary_path.relative_to(PROJECT_ROOT)}")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "eval_rl_extended.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "idle_time_min_range": list(IDLE_TIME_MIN_RANGE),
        "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "env_kwargs": ENV_KWARGS,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "target_temp_c": TARGET_TEMP_C,
        "episode_conditions": [
            {"episode": i + 1, "idle_time_min": t}
            for i, t in enumerate(idle_times)
        ],
        "filters_applied": [],
        "config_values": {
            "max_power_kw": 450.0,
            "dt_sec": 60,
            "ambient_temp_c": 25.0,
        },
        "policies_evaluated": ["dqn_final", "expert_profile"],
    }
    with open(OUTPUT_DIR / "rl_eval_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/rl_eval/")
    print("=" * 65)


if __name__ == "__main__":
    main()
