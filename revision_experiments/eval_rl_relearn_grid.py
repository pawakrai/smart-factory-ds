"""
eval_rl_relearn_grid.py — Evaluate All Relearn Candidates Under Fixed Protocol
================================================================================
Evaluates every trained model from the reward-retuning grid plus comparison
baselines (original DQN final, expert profile, always-max) under the same
3-scenario evaluation protocol used in eval_rl_cold_hot_mixed.py.

Scenarios: cold_start | hot_start | mixed_80_20  (100 episodes each)
Seed: 2024 (identical to Phase 1/2 evaluations)

All outputs → outputs/revision_relearn/eval/
No src/ modifications.

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/eval_rl_relearn_grid.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from revision_experiments.relearn_config import (
    OUTPUT_ROOT, EVAL_SEED, EVAL_EPISODES, ORIGINAL_MODEL_PATH,
    TARGET_TEMP_C, STATE_DIM, ACTION_DIM,
    build_episode_configs, make_eval_env, git_commit_short,
)
from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

EVAL_DIR = OUTPUT_ROOT / "eval"
SCENARIOS = ["cold_start", "hot_start", "mixed_80_20"]


# ─────────────────────────────────────────────────────────────────────────────
# Candidate discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_candidates() -> list[dict]:
    """
    Scan outputs/revision_relearn/models/ for trained models.
    Returns list of dicts: {name, config, seed, model_path}.
    """
    models_dir = OUTPUT_ROOT / "models"
    candidates = []
    if not models_dir.exists():
        return candidates

    for config_dir in sorted(models_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        for seed_dir in sorted(config_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            model_path = seed_dir / "model.pth"
            if model_path.exists():
                seed_str = seed_dir.name.replace("seed_", "")
                candidates.append({
                    "name": f"{config_name}_seed{seed_str}",
                    "config": config_name,
                    "seed": int(seed_str),
                    "model_path": str(model_path),
                })
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Policy runners
# ─────────────────────────────────────────────────────────────────────────────

def run_dqn_episode(agent: DQNAgent, cfg: dict) -> dict:
    """Run one episode with a DQN agent, tracking violations."""
    env = make_eval_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp = float(state[7])
    done = False
    attempted_violations = 0
    executed_violations = 0

    while not done:
        t_min = float(state[2]) / 60.0
        in_alloying = 30.0 <= t_min < 31.0

        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0)
            action = int(torch.argmax(agent.model(s_t)).item())

        if in_alloying and action > 0:
            attempted_violations += 1

        next_state, _, done = env.step(action)

        # Check executed violation (post-step power during alloying)
        if in_alloying and float(next_state[3]) > 0:
            executed_violations += 1

        state = next_state

    return _build_result(env, cfg, initial_metal_temp, initial_wall_temp,
                         attempted_violations, executed_violations)


def run_expert_episode(cfg: dict) -> dict:
    """Run one episode with expert profile."""
    env = make_eval_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp = float(state[7])
    done = False

    while not done:
        state, _, done = env.step(None, power_profile_kw=env._expert_power_profile)

    return _build_result(env, cfg, initial_metal_temp, initial_wall_temp, 0, 0)


def run_always_max_episode(cfg: dict) -> dict:
    """Run one episode with always-max 450 kW."""
    env = make_eval_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp = float(state[7])
    done = False
    attempted_violations = 0

    while not done:
        t_min = float(state[2]) / 60.0
        in_alloying = 30.0 <= t_min < 31.0
        if in_alloying:
            attempted_violations += 1  # always-max always requests 450 kW

        state, _, done = env.step(9)  # action 9 = 450 kW

    return _build_result(env, cfg, initial_metal_temp, initial_wall_temp,
                         attempted_violations, 0)


def _build_result(env, cfg, initial_metal_temp, initial_wall_temp,
                  attempted_violations, executed_violations) -> dict:
    final_temp = float(env.state["temperature"])
    return {
        "idle_time_min": cfg["idle_time_min"],
        "sub_condition": cfg["sub_condition"],
        "initial_metal_temp_c": initial_metal_temp,
        "initial_wall_temp_c": initial_wall_temp,
        "duration_min": float(env.state["time"]) / 60.0,
        "total_energy_kwh": float(env.state["energy_consumption"]),
        "success": final_temp >= env.target_temp,
        "final_temp_c": final_temp,
        "overshoot_c": max(0.0, final_temp - env.target_temp),
        "attempted_violations": attempted_violations,
        "executed_violations": executed_violations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(EVAL_SEED)

    print("=" * 70)
    print("eval_rl_relearn_grid.py — Fixed-Protocol Evaluation")
    print("=" * 70)
    print(f"Seed       : {EVAL_SEED}")
    print(f"Episodes   : {EVAL_EPISODES} per scenario × per policy")
    print(f"Scenarios  : {SCENARIOS}")

    # ── Discover candidates ──────────────────────────────────────────────────
    candidates = discover_candidates()
    print(f"\nDiscovered {len(candidates)} trained candidates:")
    for c in candidates:
        print(f"  - {c['name']}")

    # ── Load original DQN final ──────────────────────────────────────────────
    print(f"\nLoading original DQN final: {ORIGINAL_MODEL_PATH}")
    orig_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    orig_agent.model.load_state_dict(torch.load(str(ORIGINAL_MODEL_PATH), map_location="cpu"))
    orig_agent.model.eval()
    orig_agent.epsilon = 0.0
    orig_md5 = hashlib.md5(open(str(ORIGINAL_MODEL_PATH), "rb").read()).hexdigest()

    # ── Build episode configs (shared across all policies) ───────────────────
    all_configs = {}
    for sc in SCENARIOS:
        all_configs[sc] = build_episode_configs(rng, sc, EVAL_EPISODES)

    # ── Evaluate all policies ────────────────────────────────────────────────
    all_records = []

    # Helper to evaluate a DQN agent across all scenarios
    def eval_dqn_all_scenarios(policy_name: str, agent: DQNAgent):
        agent.model.eval()
        agent.epsilon = 0.0
        for sc in SCENARIOS:
            configs = all_configs[sc]
            print(f"  --- {sc} | {policy_name} ---")
            for ep_i, cfg in enumerate(configs):
                res = run_dqn_episode(agent, cfg)
                res.update({"scenario": sc, "policy": policy_name, "episode": ep_i + 1})
                all_records.append(res)
            _print_scenario_summary(policy_name, sc, all_records)

    # 1. Original DQN final
    print(f"\n{'='*70}")
    print("Evaluating: dqn_final (original)")
    eval_dqn_all_scenarios("dqn_final", orig_agent)

    # 2. Expert profile
    print(f"\n{'='*70}")
    print("Evaluating: expert_profile")
    for sc in SCENARIOS:
        configs = all_configs[sc]
        print(f"  --- {sc} | expert_profile ---")
        for ep_i, cfg in enumerate(configs):
            res = run_expert_episode(cfg)
            res.update({"scenario": sc, "policy": "expert_profile", "episode": ep_i + 1})
            all_records.append(res)
        _print_scenario_summary("expert_profile", sc, all_records)

    # 3. Always-max
    print(f"\n{'='*70}")
    print("Evaluating: always_max_450kw")
    for sc in SCENARIOS:
        configs = all_configs[sc]
        print(f"  --- {sc} | always_max_450kw ---")
        for ep_i, cfg in enumerate(configs):
            res = run_always_max_episode(cfg)
            res.update({"scenario": sc, "policy": "always_max_450kw", "episode": ep_i + 1})
            all_records.append(res)
        _print_scenario_summary("always_max_450kw", sc, all_records)

    # 4. All relearn candidates
    for cand in candidates:
        print(f"\n{'='*70}")
        print(f"Evaluating: {cand['name']}")
        agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        agent.model.load_state_dict(torch.load(cand["model_path"], map_location="cpu"))
        eval_dqn_all_scenarios(cand["name"], agent)

    # ── Save per-episode CSV ─────────────────────────────────────────────────
    col_order = [
        "scenario", "policy", "episode", "sub_condition", "idle_time_min",
        "initial_metal_temp_c", "initial_wall_temp_c",
        "duration_min", "total_energy_kwh", "success", "final_temp_c", "overshoot_c",
        "attempted_violations", "executed_violations",
    ]
    df_ep = pd.DataFrame(all_records)[col_order]
    ep_path = EVAL_DIR / "all_candidates_per_episode.csv"
    df_ep.to_csv(ep_path, index=False)
    print(f"\nSaved per-episode CSV ({len(df_ep)} rows): {ep_path.relative_to(PROJECT_ROOT)}")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_rows = []
    for sc in SCENARIOS:
        policies = df_ep[df_ep["scenario"] == sc]["policy"].unique()
        for pol in sorted(policies):
            sub = df_ep[(df_ep["scenario"] == sc) & (df_ep["policy"] == pol)]
            summary_rows.append({
                "scenario": sc,
                "policy": pol,
                "n_episodes": len(sub),
                "energy_mean": sub["total_energy_kwh"].mean(),
                "energy_sd": sub["total_energy_kwh"].std(ddof=1),
                "energy_min": sub["total_energy_kwh"].min(),
                "energy_max": sub["total_energy_kwh"].max(),
                "duration_mean": sub["duration_min"].mean(),
                "duration_sd": sub["duration_min"].std(ddof=1),
                "success_rate": sub["success"].mean(),
                "overshoot_mean": sub["overshoot_c"].mean(),
                "overshoot_sd": sub["overshoot_c"].std(ddof=1),
                "attempted_violations_mean": sub["attempted_violations"].mean(),
                "executed_violations_mean": sub["executed_violations"].mean(),
                "violation_free_pct": (sub["attempted_violations"] == 0).mean(),
                "n_hot": (sub["sub_condition"] == "hot").sum(),
                "n_cold": (sub["sub_condition"] == "cold").sum(),
            })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = EVAL_DIR / "all_candidates_summary.csv"
    df_sum.to_csv(sum_path, index=False)
    print(f"Saved summary CSV ({len(df_sum)} rows): {sum_path.relative_to(PROJECT_ROOT)}")

    # ── Manifest ─────────────────────────────────────────────────────────────
    manifest = {
        "script": "eval_rl_relearn_grid.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": git_commit_short(),
        "eval_seed": EVAL_SEED,
        "n_episodes_per_scenario_per_policy": EVAL_EPISODES,
        "scenarios": SCENARIOS,
        "original_model_path": str(ORIGINAL_MODEL_PATH),
        "original_model_md5": orig_md5,
        "candidates_evaluated": [c["name"] for c in candidates],
        "baseline_policies": ["dqn_final", "expert_profile", "always_max_450kw"],
        "episode_configs": {
            sc: [
                {"sub_condition": c["sub_condition"], "idle_time_min": c["idle_time_min"]}
                for c in all_configs[sc]
            ]
            for sc in SCENARIOS
        },
    }
    mfst_path = EVAL_DIR / "eval_manifest.json"
    with open(mfst_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest: {mfst_path.relative_to(PROJECT_ROOT)}")
    print(f"\n{'='*70}")
    print("Evaluation complete.")
    print(f"{'='*70}")


def _print_scenario_summary(policy: str, scenario: str, records: list):
    """Print quick summary for a (policy, scenario) block."""
    sub = [r for r in records if r["policy"] == policy and r["scenario"] == scenario]
    if not sub:
        return
    energies = [r["total_energy_kwh"] for r in sub]
    durations = [r["duration_min"] for r in sub]
    successes = [r["success"] for r in sub]
    print(f"    Energy  : {np.mean(energies):.2f} ± {np.std(energies, ddof=1):.2f} kWh")
    print(f"    Duration: {np.mean(durations):.1f} ± {np.std(durations, ddof=1):.1f} min")
    print(f"    Success : {np.mean(successes)*100:.1f}%")


if __name__ == "__main__":
    main()
