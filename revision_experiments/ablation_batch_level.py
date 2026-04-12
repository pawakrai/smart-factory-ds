"""
Script F — Two-Level Ablation Study, Batch Level (Reviewer Item F)
===================================================================
Compares RL power control against fixed power-profile baselines in the
thermal simulation environment (env_11) to isolate the contribution of the
RL batch controller.

FRAMING (must appear in paper):
  This is a TWO-LEVEL ablation:
  (1) Batch level [this script]: RL DQN vs. fixed power profiles in env_11.
      Isolates the contribution of RL to within-batch thermal control.
  (2) Day level [existing src/experiment_compare_results.csv]: GA scheduler
      vs. greedy and rule-based scheduling policies in app_v9.
      Isolates the contribution of GA to across-day scheduling.

  No single integrated system couples both levels; each contribution is
  evaluated independently. Do NOT fabricate an end-to-end RL-only system.

Policies evaluated (batch level):
  1. DQN (final model)     — argmax Q-value control, epsilon=0
  2. Always-max power       — constant 450 kW throughout; no RL ("no RL" ablation)
  3. Expert profile         — env._expert_power_profile(t_min); human heuristic

Note: This script reuses the same idle_time samples as eval_rl_extended.py
(SEED=2024) so results are directly comparable. Where eval_rl_extended.py
outputs exist, this script can import them (see --reuse flag below).

Outputs (outputs/revision_phase1/ablation/):
  batch_level_ablation.csv
  batch_level_barplot.png
  day_level_from_existing.csv     — copy of src/experiment_compare_results.csv
  ablation_batch_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/ablation_batch_level.py
"""

import os
import sys
import json
import csv
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "dqn_final_model_env11.pth"
EXISTING_EVAL_PATH = (
    PROJECT_ROOT / "outputs" / "revision_phase1" / "rl_eval" / "rl_extended_per_episode.json"
)
EXISTING_DAY_CSV = PROJECT_ROOT / "src" / "experiment_compare_results.csv"

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

ALWAYS_MAX_KW = 450.0   # highest discrete action level in env_11


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


def _run_policy(
    policy_name: str,
    idle_times: list[float],
    agent: DQNAgent | None = None,
    power_profile_fn=None,
) -> list[dict]:
    """Run NUM_EPISODES for the given policy. Exactly one of agent/power_profile_fn must be set."""
    assert (agent is None) != (power_profile_fn is None), \
        "Provide exactly one of agent or power_profile_fn"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for ep_idx, idle_time in enumerate(idle_times):
        env = AluminumMeltingEnvironment(idle_time_min=idle_time, **ENV_KWARGS)
        state = env.reset()
        done = False
        steps = 0

        while not done:
            if agent is not None:
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = int(torch.argmax(agent.model(s_t)).item())
                state, _r, done = env.step(action)
            else:
                state, _r, done = env.step(action=0, power_profile_kw=power_profile_fn)
            steps += 1

        final_temp = float(env.state["temperature"])
        success = final_temp >= TARGET_TEMP_C
        energy = float(env.state["energy_consumption"])
        duration_min = steps * env.dt / 60.0
        overshoot_c = max(0.0, final_temp - TARGET_TEMP_C)

        results.append({
            "policy": policy_name,
            "episode": ep_idx + 1,
            "idle_time_min": float(idle_time),
            "duration_min": float(duration_min),
            "total_energy_kwh": float(energy),
            "final_temp_c": float(final_temp),
            "success": success,
            "overshoot_c": float(overshoot_c),
        })

    return results


def _load_existing_eval_results(policy_name: str) -> list[dict] | None:
    """Try to load results from a previously saved eval_rl_extended.py output."""
    if not EXISTING_EVAL_PATH.exists():
        return None
    with open(EXISTING_EVAL_PATH) as f:
        all_results = json.load(f)
    matched = [r for r in all_results if r.get("policy") == policy_name]
    return matched if matched else None


def _summarize(results: list[dict]) -> dict:
    policy = results[0]["policy"]
    durations  = np.array([r["duration_min"]      for r in results])
    energies   = np.array([r["total_energy_kwh"]  for r in results])
    successes  = np.array([r["success"]            for r in results], dtype=float)
    overshoots = np.array([r["overshoot_c"]        for r in results])
    return {
        "policy":              policy,
        "n_episodes":          len(results),
        "duration_mean_min":   float(durations.mean()),
        "duration_sd_min":     float(durations.std()),
        "energy_mean_kwh":     float(energies.mean()),
        "energy_sd_kwh":       float(energies.std()),
        "success_rate":        float(successes.mean()),
        "mean_overshoot_c":    float(overshoots[overshoots > 0].mean()) if any(overshoots > 0) else 0.0,
        "overshoot_rate":      float((overshoots > 0).mean()),
    }


def _print_summary(s: dict) -> None:
    print(f"\n  Policy: {s['policy']}")
    print(f"    Success rate   : {s['success_rate']:.1%}")
    print(f"    Duration (min) : {s['duration_mean_min']:.1f} ± {s['duration_sd_min']:.1f}")
    print(f"    Energy  (kWh)  : {s['energy_mean_kwh']:.2f} ± {s['energy_sd_kwh']:.2f}")
    print(f"    Overshoot rate : {s['overshoot_rate']:.1%}  (mean: {s['mean_overshoot_c']:.2f}°C)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ablation_batch_level.py — Reviewer Item F (Batch Level)")
    print("=" * 65)
    print("FRAMING: Two-level ablation. This script covers batch level only.")
    print("Day-level comparison is in src/experiment_compare_results.csv.\n")

    rng = np.random.default_rng(SEED)
    idle_times = [float(rng.uniform(*IDLE_TIME_MIN_RANGE)) for _ in range(NUM_EPISODES)]

    all_summaries = []
    all_per_ep = []

    # ── Policy 1: DQN — try to reuse eval_rl_extended.py results ─────────────
    print("--- Policy 1: DQN (final model) ---")
    dqn_results = _load_existing_eval_results("dqn_final")
    if dqn_results and len(dqn_results) == NUM_EPISODES:
        print(f"  Reusing results from {EXISTING_EVAL_PATH.relative_to(PROJECT_ROOT)}")
    else:
        print("  eval_rl_extended.py results not found or incomplete — running fresh evaluation")
        agent = _load_agent()
        dqn_results = _run_policy("dqn_final", idle_times, agent=agent)
    all_per_ep.extend(dqn_results)
    dqn_summary = _summarize(dqn_results)
    _print_summary(dqn_summary)
    all_summaries.append(dqn_summary)

    # ── Policy 2: Always-max power (no RL) ───────────────────────────────────
    print(f"\n--- Policy 2: Always-max ({ALWAYS_MAX_KW} kW constant) ---")
    always_max_fn = lambda t_min: ALWAYS_MAX_KW
    always_max_results = _run_policy("always_max_450kw", idle_times, power_profile_fn=always_max_fn)
    all_per_ep.extend(always_max_results)
    always_max_summary = _summarize(always_max_results)
    _print_summary(always_max_summary)
    all_summaries.append(always_max_summary)

    # ── Policy 3: Expert profile — try to reuse ───────────────────────────────
    print("\n--- Policy 3: Expert profile (heuristic baseline) ---")
    expert_results = _load_existing_eval_results("expert_profile")
    if expert_results and len(expert_results) == NUM_EPISODES:
        print(f"  Reusing results from {EXISTING_EVAL_PATH.relative_to(PROJECT_ROOT)}")
    else:
        print("  Running fresh evaluation")
        _ref_env = AluminumMeltingEnvironment(**ENV_KWARGS)
        expert_fn = _ref_env._expert_power_profile
        expert_results = _run_policy("expert_profile", idle_times, power_profile_fn=expert_fn)
    all_per_ep.extend(expert_results)
    expert_summary = _summarize(expert_results)
    _print_summary(expert_summary)
    all_summaries.append(expert_summary)

    # ── Save batch-level ablation CSV ─────────────────────────────────────────
    ablation_path = OUTPUT_DIR / "batch_level_ablation.csv"
    with open(ablation_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n  Saved: {ablation_path.relative_to(PROJECT_ROOT)}")

    # ── Copy day-level results for completeness ───────────────────────────────
    day_level_dest = OUTPUT_DIR / "day_level_from_existing.csv"
    if EXISTING_DAY_CSV.exists():
        shutil.copy2(EXISTING_DAY_CSV, day_level_dest)
        print(f"  Copied day-level results: {day_level_dest.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [WARN] Day-level CSV not found at {EXISTING_DAY_CSV}")

    # ── Print consolidated ablation table ─────────────────────────────────────
    print("\n  === Batch-Level Ablation Summary ===")
    print(f"  {'Policy':<30} {'Success':>8} {'Energy (kWh)':>18} {'Duration (min)':>16}")
    print(f"  {'-'*74}")
    for s in all_summaries:
        print(f"  {s['policy']:<30} {s['success_rate']:>7.1%}  "
              f"{s['energy_mean_kwh']:>7.2f} ± {s['energy_sd_kwh']:<8.2f} "
              f"{s['duration_mean_min']:>6.1f} ± {s['duration_sd_min']:.1f}")

    print("\n  Day-level comparison: see src/experiment_compare_results.csv")
    print("  (GA vs. continuous_baseline vs. rule_based — single run, seed=42)")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [s["policy"] for s in all_summaries]
        energies_mean = [s["energy_mean_kwh"] for s in all_summaries]
        energies_sd = [s["energy_sd_kwh"] for s in all_summaries]
        success_rates = [s["success_rate"] * 100 for s in all_summaries]
        durations_mean = [s["duration_mean_min"] for s in all_summaries]
        durations_sd = [s["duration_sd_min"] for s in all_summaries]

        colors = ["#E3000F", "#3F3F46", "#3B82F6"]
        x = np.arange(len(labels))
        width = 0.6

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        axes[0].bar(x, energies_mean, width, yerr=energies_sd, capsize=5,
                    color=colors, alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=15, ha="right")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].set_title("Energy per Batch")
        axes[0].grid(axis="y", alpha=0.3)

        axes[1].bar(x, success_rates, width, color=colors, alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15, ha="right")
        axes[1].set_ylabel("Success Rate (%)")
        axes[1].set_ylim(0, 105)
        axes[1].set_title("Target Attainment Rate")
        axes[1].grid(axis="y", alpha=0.3)

        axes[2].bar(x, durations_mean, width, yerr=durations_sd, capsize=5,
                    color=colors, alpha=0.85)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=15, ha="right")
        axes[2].set_ylabel("Duration (min)")
        axes[2].set_title("Batch Duration")
        axes[2].grid(axis="y", alpha=0.3)

        plt.suptitle("Batch-Level Ablation: RL vs Fixed Power Profiles\n"
                     f"(100 episodes, seed={SEED})", fontsize=11)
        plt.tight_layout()
        fig_path = OUTPUT_DIR / "batch_level_barplot.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {fig_path.relative_to(PROJECT_ROOT)}")
    except ImportError:
        print("  [SKIP] matplotlib not available.")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "ablation_batch_level.py",
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
        "policies_evaluated": [
            "dqn_final (final model, epsilon=0)",
            f"always_max_450kw (constant {ALWAYS_MAX_KW} kW)",
            "expert_profile (env._expert_power_profile)",
        ],
        "day_level_source": str(EXISTING_DAY_CSV.relative_to(PROJECT_ROOT)),
        "framing_note": (
            "Two-level ablation. Batch level (this script): RL vs fixed profiles. "
            "Day level (experiment_compare_results.csv): GA vs baselines. "
            "No integrated system; each level evaluated independently."
        ),
        "filters_applied": [],
    }
    with open(OUTPUT_DIR / "ablation_batch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/ablation/")
    print("=" * 65)


if __name__ == "__main__":
    main()
