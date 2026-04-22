"""
train_rl_relearn_grid.py — RL Reward-Retuning Training Grid
=============================================================
Trains 6 reward configurations × 3 seeds = 18 DQN models for the
controlled RL relearning study (journal revision).

All training uses start_mode="hot" only, preserving the original
training distribution from train_with_env_11.py.  The relearning study
keeps the training distribution fixed so that any energy differences are
attributable solely to the reward signal.

MANDATORY PARITY GATE runs first.  If it fails, training is aborted.

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/train_rl_relearn_grid.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from revision_experiments.relearn_config import (
    OUTPUT_ROOT, TRAIN_EPISODES, SEEDS, BASE_COEFFS, TUNING_GRID,
    ENV_KWARGS, AGENT_KWARGS, PatchedEnvV2, run_parity_gate,
    seed_all, git_commit_short,
)
from src.agents.agent2 import DQNAgent


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_single(config_name: str, coeffs: dict, seed: int) -> tuple:
    """
    Train a DQN agent on PatchedEnvV2 for TRAIN_EPISODES episodes.
    Returns (agent, episode_rewards, episode_energies, wall_time_sec).
    """
    seed_all(seed)

    env = PatchedEnvV2(seed=seed, reward_coeffs=coeffs, **ENV_KWARGS)
    agent = DQNAgent(**AGENT_KWARGS)

    episode_rewards = []
    episode_energies = []
    t0 = time.time()

    for ep in range(TRAIN_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.end_episode()
        episode_rewards.append(float(total_reward))
        episode_energies.append(float(env.state["energy_consumption"]))

        if (ep + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            print(f"    [{config_name}/seed={seed}] ep {ep+1}/{TRAIN_EPISODES} | "
                  f"avg_reward(100)={np.mean(recent):.2f} | "
                  f"avg_energy(100)={np.mean(episode_energies[-100:]):.1f} kWh | "
                  f"ε={agent.epsilon:.3f}")

    wall_time = time.time() - t0
    return agent, episode_rewards, episode_energies, wall_time


# ─────────────────────────────────────────────────────────────────────────────
# Audit report
# ─────────────────────────────────────────────────────────────────────────────

def write_audit(parity_result: dict, training_summary: list, wall_total: float):
    """Write relearn_audit.md documenting the reward structure and study setup."""
    audit_path = OUTPUT_ROOT / "relearn_audit.md"
    lines = [
        "# RL Reward-Retuning Audit",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        f"**Git commit**: {git_commit_short()}",
        "",
        "## 1. Current Reward Structure (env_11, lines 488-520)",
        "",
        "| Component | Coefficient | env_11 Line | Tuned? |",
        "|---|---|---|---|",
        "| Expert tracking penalty | 0.5 | 488 | No |",
        "| Safety penalty | 5.0 | 492-493 | No |",
        "| Progress reward | 2.0 | 499 | No |",
        "| **Energy penalty** | **0.02** | **502** | **Yes** |",
        "| Time penalty | 0.001 | 505 | No |",
        "| Terminal success base | 10.0 | 518 | No |",
        "| Terminal success efficiency | 5.0 | 518 | No |",
        "| Terminal failure | 5.0 | 520 | No |",
        "| **Overshoot penalty (new)** | **0.0** | **N/A** | **Yes** |",
        "",
        "## 2. Tuning Rationale",
        "",
        "- `energy_scale` was identified as the most sensitive coefficient by prior OFAT analysis",
        "  (range 11.82 vs ≤2.45 for all others).",
        "- `overshoot_scale` is a new additive term in the terminal reward, penalizing temperature",
        "  overshoot beyond the 950°C target.  Current DQN mean overshoot = 5.58°C (SD 2.69).",
        "- All other coefficients are held fixed to isolate the effect of energy/overshoot tuning.",
        "",
        "## 3. Training Distribution",
        "",
        "All training uses `start_mode='hot'` only, preserving the original training distribution",
        "from `train_with_env_11.py`.  Generalization to cold-start and mixed conditions is",
        "assessed exclusively via the evaluation protocol (eval_rl_relearn_grid.py).",
        "",
        "## 4. Parity Gate",
        "",
        f"- Result: **{'PASSED' if parity_result['passed'] else 'FAILED'}**",
        f"- Timestamp: {parity_result['timestamp']}",
        "",
        "Details:",
    ]
    for d in parity_result["details"]:
        lines.append(f"  - {d}")

    lines.extend([
        "",
        "## 5. Training Summary",
        "",
        "| Config | Seed | Episodes | Wall Time (s) | Final Avg Reward (100) | Final Avg Energy (100) |",
        "|---|---|---|---|---|---|",
    ])
    for s in training_summary:
        lines.append(
            f"| {s['config']} | {s['seed']} | {s['episodes']} | {s['wall_sec']:.0f} | "
            f"{s['final_avg_reward']:.2f} | {s['final_avg_energy']:.1f} kWh |"
        )
    lines.extend([
        "",
        f"**Total wall time**: {wall_total:.0f} s ({wall_total/3600:.1f} h)",
    ])

    audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved audit: {audit_path.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("train_rl_relearn_grid.py — RL Reward-Retuning Study")
    print("=" * 70)
    print(f"Configs       : {list(TUNING_GRID.keys())}")
    print(f"Seeds         : {SEEDS}")
    print(f"Episodes/run  : {TRAIN_EPISODES}")
    print(f"Total runs    : {len(TUNING_GRID) * len(SEEDS)}")
    print(f"Total episodes: {len(TUNING_GRID) * len(SEEDS) * TRAIN_EPISODES}")
    print()

    # ── Step 1: Parity gate ──────────────────────────────────────────────────
    parity_result = run_parity_gate()
    parity_path = OUTPUT_ROOT / "parity_test_result.json"
    with open(parity_path, "w") as f:
        json.dump(parity_result, f, indent=2)

    if not parity_result["passed"]:
        print("\n[ABORT] Parity check failed. Training aborted.")
        manifest = {
            "script": "train_rl_relearn_grid.py",
            "run_timestamp": datetime.now().isoformat(),
            "git_commit": git_commit_short(),
            "parity_test": {"result": "FAILED"},
            "training_status": "ABORTED",
        }
        with open(OUTPUT_ROOT / "relearn_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # ── Step 2: Write tuning plan ────────────────────────────────────────────
    tuning_plan = {
        "description": "Pre-registered tuning grid for RL reward-retuning study",
        "timestamp": datetime.now().isoformat(),
        "base_coefficients": BASE_COEFFS,
        "grid": {
            name: {k: v for k, v in coeffs.items() if k in ("energy_scale", "overshoot_scale")}
            for name, coeffs in TUNING_GRID.items()
        },
        "full_grid": TUNING_GRID,
        "seeds": SEEDS,
        "train_episodes": TRAIN_EPISODES,
        "env_kwargs": ENV_KWARGS,
        "agent_kwargs": AGENT_KWARGS,
    }
    plan_path = OUTPUT_ROOT / "tuning_plan.json"
    with open(plan_path, "w") as f:
        json.dump(tuning_plan, f, indent=2, default=str)
    print(f"Saved tuning plan: {plan_path.relative_to(PROJECT_ROOT)}")

    # ── Step 3: Train all configs × seeds ────────────────────────────────────
    training_summary = []
    wall_total_start = time.time()

    for config_name, coeffs in TUNING_GRID.items():
        for seed in SEEDS:
            print(f"\n{'='*70}")
            print(f"Training: {config_name} | seed={seed}")
            energy_label = f"energy_scale={coeffs['energy_scale']}"
            overshoot_label = f"overshoot_scale={coeffs.get('overshoot_scale', 0.0)}"
            print(f"  {energy_label}, {overshoot_label}")

            agent, ep_rewards, ep_energies, wall_sec = train_single(config_name, coeffs, seed)

            # Save model
            model_dir = OUTPUT_ROOT / "models" / config_name / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pth"
            torch.save(agent.model.state_dict(), model_path)
            print(f"  Saved model: {model_path.relative_to(PROJECT_ROOT)}")

            # Save training log
            log_dir = OUTPUT_ROOT / "logs" / config_name / f"seed_{seed}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_data = {
                "config_name": config_name,
                "seed": seed,
                "coefficients": coeffs,
                "train_episodes": TRAIN_EPISODES,
                "wall_time_sec": wall_sec,
                "episode_rewards": ep_rewards,
                "episode_energies": ep_energies,
                "final_epsilon": float(agent.epsilon),
                "timestamp": datetime.now().isoformat(),
            }
            log_path = log_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=2)

            final_100_reward = float(np.mean(ep_rewards[-100:])) if len(ep_rewards) >= 100 else float(np.mean(ep_rewards))
            final_100_energy = float(np.mean(ep_energies[-100:])) if len(ep_energies) >= 100 else float(np.mean(ep_energies))

            training_summary.append({
                "config": config_name,
                "seed": seed,
                "episodes": TRAIN_EPISODES,
                "wall_sec": wall_sec,
                "final_avg_reward": final_100_reward,
                "final_avg_energy": final_100_energy,
                "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            })

            print(f"  Wall time: {wall_sec:.0f}s | Final avg energy (100): {final_100_energy:.1f} kWh")

    wall_total = time.time() - wall_total_start

    # ── Step 4: Write audit ──────────────────────────────────────────────────
    write_audit(parity_result, training_summary, wall_total)

    # ── Step 5: Write manifest ───────────────────────────────────────────────
    manifest = {
        "script": "train_rl_relearn_grid.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": git_commit_short(),
        "parity_test": {"result": "PASSED"},
        "training_status": "COMPLETED",
        "seeds": SEEDS,
        "train_episodes": TRAIN_EPISODES,
        "env_kwargs": ENV_KWARGS,
        "agent_kwargs": AGENT_KWARGS,
        "tuning_grid": {
            name: {k: v for k, v in coeffs.items() if k in ("energy_scale", "overshoot_scale")}
            for name, coeffs in TUNING_GRID.items()
        },
        "training_summary": training_summary,
        "total_wall_time_sec": wall_total,
    }
    manifest_path = OUTPUT_ROOT / "relearn_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Training complete. Total wall time: {wall_total:.0f}s ({wall_total/3600:.1f}h)")
    print(f"Models:   {OUTPUT_ROOT / 'models'}")
    print(f"Logs:     {OUTPUT_ROOT / 'logs'}")
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
