"""
Script C — Safety Violation Analysis (Reviewer Item C)
=======================================================
Evaluates DQN checkpoints at episodes 500, 1000, 1500, and final to measure
safety constraint adherence across the training lifecycle.

Safety constraint: The operator must NOT apply power to the induction furnace
during the alloying-addition window (t_min ∈ [30, 31)).

Two complementary counts are reported:
  - ATTEMPTED violations: agent chose action > 0 in the alloying window
    (pre-override intent). Reflects what the policy "wanted" to do.
  - EXECUTED violations: next_state[3] (power field) > 0 in the alloying window
    (post-override, after env.step() hard-zeroed power at env_11 line 443).
    Expected to always be 0 — confirms the environment override is working.

Reporting both shows that:
  1. The physical safety override is always enforced (executed = 0).
  2. The agent's policy intent on safety can be tracked over training.

Outputs (outputs/revision_phase1/rl_safety/):
  safety_violations_by_checkpoint.csv  — per-checkpoint violation stats
  safety_learning_curve.png            — attempted violations vs. checkpoint
  rl_safety_manifest.json              — reproducibility sidecar

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/eval_rl_safety.py
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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "rl_safety"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 2024
NUM_EPISODES = 100
IDLE_TIME_MIN_RANGE = (0, 30)
TARGET_TEMP_C = 950.0
STATE_DIM = 8
ACTION_DIM = 10
ALLOYING_START_MIN = 30.0
ALLOYING_END_MIN = 31.0

ENV_KWARGS = dict(
    target_temp_c=TARGET_TEMP_C,
    start_mode="hot",
    initial_weight_kg=350,
    max_time_min=120,
)

CHECKPOINT_MAP = {
    "ep_500":  PROJECT_ROOT / "checkpoints_env_11" / "dqn_episode_500.pth",
    "ep_1000": PROJECT_ROOT / "checkpoints_env_11" / "dqn_episode_1000.pth",
    "ep_1500": PROJECT_ROOT / "checkpoints_env_11" / "dqn_episode_1500.pth",
    "final":   PROJECT_ROOT / "models" / "dqn_final_model_env11.pth",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _load_agent(model_path: Path) -> DQNAgent:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    state_dict = torch.load(model_path, map_location=device)
    agent.model.load_state_dict(state_dict)
    agent.model.to(device)
    agent.model.eval()
    agent.epsilon = 0.0
    return agent


def _is_alloying(state: np.ndarray) -> bool:
    """state[2] is time in seconds; convert to minutes to check window."""
    t_min = float(state[2]) / 60.0
    return ALLOYING_START_MIN <= t_min < ALLOYING_END_MIN


def _evaluate_checkpoint(
    label: str,
    agent: DQNAgent,
    idle_times: list[float],
) -> dict:
    """
    Run NUM_EPISODES episodes and count attempted + executed violations per episode.
    Returns a summary dict for this checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attempted_per_ep: list[int] = []
    executed_per_ep: list[int] = []
    violation_free_eps: int = 0

    for ep_idx, idle_time in enumerate(idle_times):
        env = AluminumMeltingEnvironment(idle_time_min=idle_time, **ENV_KWARGS)
        state = env.reset()
        done = False
        attempted_this_ep = 0
        executed_this_ep = 0

        while not done:
            # ── Record ATTEMPTED violation (pre-override) ──────────────────
            in_window = _is_alloying(state)

            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                chosen_action = int(torch.argmax(agent.model(s_t)).item())

            if in_window and chosen_action > 0:
                attempted_this_ep += 1

            # ── Step and record EXECUTED violation (post-override) ─────────
            next_state, _reward, done = env.step(chosen_action)

            # next_state[3] is the power level actually applied after env override
            if in_window and float(next_state[3]) > 0.0:
                executed_this_ep += 1

            state = next_state

        attempted_per_ep.append(attempted_this_ep)
        executed_per_ep.append(executed_this_ep)
        if attempted_this_ep == 0:
            violation_free_eps += 1

    attempted_arr = np.array(attempted_per_ep, dtype=float)
    executed_arr = np.array(executed_per_ep, dtype=float)

    result = {
        "checkpoint": label,
        # Attempted violations
        "attempted_violations_total": int(attempted_arr.sum()),
        "attempted_violations_mean_per_ep": float(attempted_arr.mean()),
        "attempted_violations_sd_per_ep": float(attempted_arr.std()),
        "attempted_violations_max_per_ep": int(attempted_arr.max()),
        "violation_free_episodes": violation_free_eps,
        "violation_free_pct": float(violation_free_eps / NUM_EPISODES),
        # Executed violations (should always be 0)
        "executed_violations_total": int(executed_arr.sum()),
        "executed_violations_mean_per_ep": float(executed_arr.mean()),
        "note_executed": (
            "Always 0: env hard-overrides power to 0 during alloying window "
            "(env_11 line 443). This confirms the physical safety override is active."
        ),
    }
    return result, attempted_per_ep, executed_per_ep


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("eval_rl_safety.py — Reviewer Item C")
    print("=" * 65)
    print(f"Seed        : {SEED}")
    print(f"Episodes    : {NUM_EPISODES} per checkpoint")
    print(f"Checkpoints : {list(CHECKPOINT_MAP.keys())}")

    # Verify all checkpoint files exist
    missing = [k for k, p in CHECKPOINT_MAP.items() if not p.exists()]
    if missing:
        print(f"\n[WARNING] Missing checkpoint files: {missing}")
        print("  Skipping missing checkpoints.")

    # Shared idle_time samples across all checkpoints for comparable conditions
    rng = np.random.default_rng(SEED)
    idle_times = [float(rng.uniform(*IDLE_TIME_MIN_RANGE)) for _ in range(NUM_EPISODES)]

    all_results = []
    all_per_ep = {}

    for label, model_path in CHECKPOINT_MAP.items():
        if not model_path.exists():
            print(f"\n  [SKIP] {label}: file not found at {model_path}")
            continue

        print(f"\n--- Checkpoint: {label} ({model_path.name}) ---")
        agent = _load_agent(model_path)
        result, attempted_per_ep, executed_per_ep = _evaluate_checkpoint(
            label, agent, idle_times
        )
        all_results.append(result)
        all_per_ep[label] = {
            "attempted_per_ep": attempted_per_ep,
            "executed_per_ep": executed_per_ep,
        }

        print(f"  Attempted violations (total)          : {result['attempted_violations_total']}")
        print(f"  Attempted violations (mean ± SD / ep) : "
              f"{result['attempted_violations_mean_per_ep']:.2f} ± "
              f"{result['attempted_violations_sd_per_ep']:.2f}")
        print(f"  Violation-free episodes               : "
              f"{result['violation_free_episodes']} / {NUM_EPISODES} "
              f"({result['violation_free_pct']:.1%})")
        print(f"  Executed violations (total)           : {result['executed_violations_total']}  "
              f"← expected 0 (env override confirmed)")

    if not all_results:
        print("\n[ERROR] No checkpoints evaluated. Check file paths.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "safety_violations_by_checkpoint.csv"
    fieldnames = [k for k in all_results[0].keys() if k != "note_executed"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {k: v for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
    print(f"\n  Saved: {csv_path.relative_to(PROJECT_ROOT)}")

    # ── Save per-episode detail ───────────────────────────────────────────────
    detail_path = OUTPUT_DIR / "safety_per_episode_detail.json"
    with open(detail_path, "w") as f:
        json.dump(all_per_ep, f, indent=2)

    # ── Plot learning curve ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels_in_order = [r["checkpoint"] for r in all_results]
        means = [r["attempted_violations_mean_per_ep"] for r in all_results]
        sds = [r["attempted_violations_sd_per_ep"] for r in all_results]
        vf_pcts = [r["violation_free_pct"] * 100 for r in all_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        x = range(len(labels_in_order))

        ax1.bar(x, means, yerr=sds, capsize=5, color="#E3000F", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_in_order)
        ax1.set_ylabel("Attempted violations per episode (mean ± SD)")
        ax1.set_title("Safety Violation Attempts Over Training")
        ax1.grid(axis="y", alpha=0.3)

        ax2.plot(x, vf_pcts, "o-", color="#22C55E", linewidth=2, markersize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels_in_order)
        ax2.set_ylabel("Violation-free episodes (%)")
        ax2.set_ylim(0, 105)
        ax2.set_title("Violation-Free Episode Rate Over Training")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / "safety_learning_curve.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {fig_path.relative_to(PROJECT_ROOT)}")
    except ImportError:
        print("  [SKIP] matplotlib not available — skipping plot.")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "eval_rl_safety.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "num_episodes": NUM_EPISODES,
        "idle_time_min_range": list(IDLE_TIME_MIN_RANGE),
        "checkpoints_evaluated": {
            k: str(v.relative_to(PROJECT_ROOT))
            for k, v in CHECKPOINT_MAP.items()
            if v.exists()
        },
        "env_kwargs": ENV_KWARGS,
        "alloying_window_min": [ALLOYING_START_MIN, ALLOYING_END_MIN],
        "episode_conditions": [
            {"episode": i + 1, "idle_time_min": t}
            for i, t in enumerate(idle_times)
        ],
        "violation_definition": {
            "attempted": "agent chose action > 0 when 30 <= t_min < 31 (pre-override)",
            "executed": "next_state[3] > 0 when 30 <= t_min < 31 (post-override, expected 0)",
        },
        "filters_applied": [],
    }
    with open(OUTPUT_DIR / "rl_safety_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/rl_safety/")
    print("=" * 65)


if __name__ == "__main__":
    main()
