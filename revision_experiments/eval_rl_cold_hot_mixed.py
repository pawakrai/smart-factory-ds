"""
eval_rl_cold_hot_mixed.py — Phase 2 Revised Start-State Evaluation
====================================================================
Evaluates DQN, expert profile, and always-max policies under three
start-state scenarios to better reflect real plant operating conditions.

Scenarios
---------
  cold_start   : start_mode="cold"  → metal=wall=25°C (ambient)
  hot_start    : start_mode="hot",  idle_time ~ Uniform(0,30) min
                 (identical to Phase 1 eval_rl_extended.py)
  mixed_80_20  : 80% hot-start + 20% cold-start  ← user-specified ratio
                 Documented assumption: plant mostly runs intra-shift
                 (hot) batches with ~20% cold-start events (first batch
                 of shift, post-maintenance).

Plant historical baseline (primary, N=103):
  energy = 587.31 ± 34.85 kWh
  duration = 101.93 ± 7.14 min

No src/ modifications.  All outputs → outputs/revision_phase1/rl_eval/
"""

import os
import sys
import json
import time
import datetime
import hashlib
import numpy as np
import pandas as pd

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_ROOT)

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent
import torch

# ─── Config ────────────────────────────────────────────────────────────────
SEED = 2024
N_EPISODES = 100          # per scenario × per policy
MODEL_PATH = os.path.join(PROJ_ROOT, "models", "dqn_final_model_env11.pth")
OUTPUT_DIR = os.path.join(PROJ_ROOT, "outputs", "revision_phase1", "rl_eval")

# Plant baseline (N=103)
PLANT_ENERGY_MEAN = 587.31
PLANT_ENERGY_SD   = 34.85
PLANT_DURATION_MEAN = 101.93
PLANT_DURATION_SD   = 7.14
PLANT_N = 103

HOT_IDLE_RANGE = (0.0, 30.0)   # minutes
HOT_FRACTION   = 0.80           # for mixed_80_20
COLD_FRACTION  = 0.20

ENV_BASE_KWARGS = dict(
    target_temp_c=950.0,
    dt_sec=60,
)

# ─── Scenario definitions ────────────────────────────────────────────────────
# Each scenario defines per-episode start configs for N_EPISODES episodes.
# Returns list of dicts: {start_mode, idle_time_min (or None), sub_condition}

def build_episode_configs(rng, scenario: str, n: int) -> list:
    configs = []
    if scenario == "cold_start":
        for _ in range(n):
            configs.append({"start_mode": "cold", "idle_time_min": None, "sub_condition": "cold"})

    elif scenario == "hot_start":
        for _ in range(n):
            idle = float(rng.uniform(*HOT_IDLE_RANGE))
            configs.append({"start_mode": "hot", "idle_time_min": idle, "sub_condition": "hot"})

    elif scenario == "mixed_80_20":
        n_hot  = int(round(n * HOT_FRACTION))
        n_cold = n - n_hot
        # interleave: hot every 5, cold every 5th slot → shuffle with fixed rng
        labels = ["hot"] * n_hot + ["cold"] * n_cold
        rng.shuffle(labels)
        for lbl in labels:
            if lbl == "hot":
                idle = float(rng.uniform(*HOT_IDLE_RANGE))
                configs.append({"start_mode": "hot", "idle_time_min": idle, "sub_condition": "hot"})
            else:
                configs.append({"start_mode": "cold", "idle_time_min": None, "sub_condition": "cold"})
    return configs


# ─── Policy runners ──────────────────────────────────────────────────────────

def make_env(cfg: dict) -> AluminumMeltingEnvironment:
    kwargs = dict(ENV_BASE_KWARGS)
    kwargs["start_mode"] = cfg["start_mode"]
    if cfg["idle_time_min"] is not None:
        kwargs["idle_time_min"] = cfg["idle_time_min"]
    return AluminumMeltingEnvironment(**kwargs)


def run_dqn(agent: DQNAgent, cfg: dict) -> dict:
    env = make_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp  = float(state[7])
    done = False
    while not done:
        action = agent.select_action(state, explore=False)
        state, _, done = env.step(action)
    return _episode_result(env, cfg, initial_metal_temp, initial_wall_temp)


def run_expert(cfg: dict) -> dict:
    env = make_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp  = float(state[7])
    done = False
    while not done:
        t_min = float(state[2]) / 60.0
        state, _, done = env.step(None, power_profile_kw=env._expert_power_profile)
    return _episode_result(env, cfg, initial_metal_temp, initial_wall_temp)


def run_always_max(cfg: dict, max_action: int = 9) -> dict:
    """Always select action 9 = 450 kW."""
    env = make_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp  = float(state[7])
    done = False
    while not done:
        state, _, done = env.step(max_action)
    return _episode_result(env, cfg, initial_metal_temp, initial_wall_temp)


def _episode_result(env, cfg, initial_metal_temp, initial_wall_temp) -> dict:
    final_temp = float(env.state["temperature"])
    energy     = float(env.state["energy_consumption"])
    duration   = float(env.state["time"]) / 60.0
    success    = final_temp >= env.target_temp
    overshoot  = max(0.0, final_temp - env.target_temp)
    return {
        "idle_time_min":        cfg["idle_time_min"],
        "sub_condition":        cfg["sub_condition"],
        "initial_metal_temp_c": initial_metal_temp,
        "initial_wall_temp_c":  initial_wall_temp,
        "duration_min":         duration,
        "total_energy_kwh":     energy,
        "success":              success,
        "final_temp_c":         final_temp,
        "overshoot_c":          overshoot,
    }


# ─── Expert step wrapper ─────────────────────────────────────────────────────
# env.step() accepts power_profile_kw callable — confirm by checking the signature
# If not accepted, fall back to computing action index from power level.

def _check_expert_step_support() -> bool:
    """Check whether env.step() accepts power_profile_kw kwarg."""
    import inspect
    sig = inspect.signature(AluminumMeltingEnvironment.step)
    return "power_profile_kw" in sig.parameters


EXPERT_USES_KW_ARG = _check_expert_step_support()


def run_expert_v2(cfg: dict) -> dict:
    """Expert profile run — handles both step() signatures."""
    env = make_env(cfg)
    state = env.reset()
    initial_metal_temp = float(state[0])
    initial_wall_temp  = float(state[7])
    done = False
    while not done:
        if EXPERT_USES_KW_ARG:
            state, _, done = env.step(None, power_profile_kw=env._expert_power_profile)
        else:
            # Map expert kW → closest action index (0-9, 50 kW increments)
            t_min = float(state[2]) / 60.0
            expert_kw = env._expert_power_profile(t_min)
            action = int(round(expert_kw / 50.0))
            action = max(0, min(9, action))
            state, _, done = env.step(action)
    return _episode_result(env, cfg, initial_metal_temp, initial_wall_temp)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("eval_rl_cold_hot_mixed.py — Phase 2 Start-State Evaluation")
    print("=" * 70)
    print(f"Seed         : {SEED}")
    print(f"Episodes     : {N_EPISODES} per scenario × per policy")
    print(f"Scenarios    : cold_start | hot_start | mixed_80_20")
    print(f"Policies     : dqn_final | expert_profile | always_max_450kw")
    print(f"Mixed ratio  : {int(HOT_FRACTION*100)}% hot / {int(COLD_FRACTION*100)}% cold")
    print(f"Plant N=103  : energy={PLANT_ENERGY_MEAN} ± {PLANT_ENERGY_SD} kWh, "
          f"duration={PLANT_DURATION_MEAN} ± {PLANT_DURATION_SD} min")
    print()

    # Load DQN model
    agent = DQNAgent(state_dim=8, action_dim=10)
    agent.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    agent.model.eval()
    model_md5 = hashlib.md5(open(MODEL_PATH, "rb").read()).hexdigest()
    print(f"Model loaded : {MODEL_PATH}")
    print(f"Model MD5    : {model_md5}\n")

    scenarios = ["cold_start", "hot_start", "mixed_80_20"]
    policies  = ["dqn_final", "expert_profile", "always_max_450kw"]

    # Pre-build episode configs (same for all policies within a scenario, for fair comparison)
    all_configs = {}
    for sc in scenarios:
        all_configs[sc] = build_episode_configs(rng, sc, N_EPISODES)

    records = []

    for sc in scenarios:
        configs = all_configs[sc]
        for pol in policies:
            print(f"--- {sc} | {pol} ---")
            ep_results = []
            for ep_i, cfg in enumerate(configs):
                if pol == "dqn_final":
                    res = run_dqn(agent, cfg)
                elif pol == "expert_profile":
                    res = run_expert_v2(cfg)
                else:
                    res = run_always_max(cfg)

                res.update({"scenario": sc, "policy": pol, "episode": ep_i + 1})
                ep_results.append(res)
                records.append(res)

            energies  = [r["total_energy_kwh"] for r in ep_results]
            durations = [r["duration_min"] for r in ep_results]
            successes = [r["success"] for r in ep_results]
            print(f"  Energy  : {np.mean(energies):.2f} ± {np.std(energies, ddof=1):.2f} kWh"
                  f"  [{np.min(energies):.0f}, {np.max(energies):.0f}]")
            print(f"  Duration: {np.mean(durations):.1f} ± {np.std(durations, ddof=1):.1f} min")
            print(f"  Success : {np.mean(successes)*100:.1f}%\n")

    # ── Save per-episode CSV ────────────────────────────────────────────────
    col_order = [
        "scenario", "policy", "episode", "sub_condition", "idle_time_min",
        "initial_metal_temp_c", "initial_wall_temp_c",
        "duration_min", "total_energy_kwh", "success", "final_temp_c", "overshoot_c",
    ]
    df_ep = pd.DataFrame(records)[col_order]
    ep_path = os.path.join(OUTPUT_DIR, "cold_hot_mixed_per_episode.csv")
    df_ep.to_csv(ep_path, index=False)
    print(f"Saved per-episode CSV ({len(df_ep)} rows): {ep_path}")

    # ── Summary CSV ────────────────────────────────────────────────────────
    summary_rows = []
    for sc in scenarios:
        for pol in policies:
            sub = df_ep[(df_ep["scenario"] == sc) & (df_ep["policy"] == pol)]
            summary_rows.append({
                "scenario":        sc,
                "policy":          pol,
                "n_episodes":      len(sub),
                "energy_mean":     sub["total_energy_kwh"].mean(),
                "energy_sd":       sub["total_energy_kwh"].std(ddof=1),
                "energy_min":      sub["total_energy_kwh"].min(),
                "energy_max":      sub["total_energy_kwh"].max(),
                "duration_mean":   sub["duration_min"].mean(),
                "duration_sd":     sub["duration_min"].std(ddof=1),
                "success_rate":    sub["success"].mean() * 100,
                "overshoot_mean":  sub["overshoot_c"].mean(),
                "n_hot":           (sub["sub_condition"] == "hot").sum(),
                "n_cold":          (sub["sub_condition"] == "cold").sum(),
            })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(OUTPUT_DIR, "cold_hot_mixed_summary.csv")
    df_sum.to_csv(sum_path, index=False)
    print(f"Saved summary CSV ({len(df_sum)} rows): {sum_path}")

    # ── Plant comparison (DQN mixed_80_20) ────────────────────────────────
    print("\n" + "=" * 70)
    print("PLANT BASELINE COMPARISON (DQN mixed_80_20 vs N=103 plant)")
    print("=" * 70)
    dqn_mixed = df_ep[(df_ep["scenario"] == "mixed_80_20") & (df_ep["policy"] == "dqn_final")]
    m_e  = dqn_mixed["total_energy_kwh"].mean()
    m_d  = dqn_mixed["duration_min"].mean()
    diff_e = m_e - PLANT_ENERGY_MEAN
    diff_d = m_d - PLANT_DURATION_MEAN
    print(f"DQN mixed_80_20   : energy={m_e:.2f} kWh, duration={m_d:.1f} min")
    print(f"Plant N=103       : energy={PLANT_ENERGY_MEAN:.2f} kWh, duration={PLANT_DURATION_MEAN:.1f} min")
    print(f"Energy difference : {diff_e:+.2f} kWh ({diff_e/PLANT_ENERGY_MEAN*100:+.2f}%)")
    print(f"Duration difference: {diff_d:+.1f} min ({diff_d/PLANT_DURATION_MEAN*100:+.2f}%)")
    within_1sd_e = abs(diff_e) <= PLANT_ENERGY_SD
    within_1sd_d = abs(diff_d) <= PLANT_DURATION_SD
    print(f"Within 1 SD energy: {'YES' if within_1sd_e else 'NO'}  (plant SD={PLANT_ENERGY_SD})")
    print(f"Within 1 SD duration: {'YES' if within_1sd_d else 'NO'}  (plant SD={PLANT_DURATION_SD})")

    print("\nAll scenarios × DQN energy comparison:")
    for sc in scenarios:
        sub = df_ep[(df_ep["scenario"] == sc) & (df_ep["policy"] == "dqn_final")]
        e = sub["total_energy_kwh"].mean()
        d = sub["duration_min"].mean()
        print(f"  {sc:20s}: energy={e:.2f} kWh ({e-PLANT_ENERGY_MEAN:+.2f}), "
              f"duration={d:.1f} min ({d-PLANT_DURATION_MEAN:+.1f})")

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest = {
        "script": "eval_rl_cold_hot_mixed.py",
        "run_timestamp": datetime.datetime.now().isoformat(),
        "seed": SEED,
        "model_path": MODEL_PATH,
        "model_md5": model_md5,
        "num_episodes_per_scenario_per_policy": N_EPISODES,
        "scenarios": scenarios,
        "policies": policies,
        "mixed_ratio_assumption": {
            "hot_fraction": HOT_FRACTION,
            "cold_fraction": COLD_FRACTION,
            "rationale": (
                "80% hot / 20% cold: plant mostly runs intra-shift batches (hot), "
                "with ~20% cold-start events (first batch of shift, post-maintenance). "
                "Exact proportions not available from plant records — documented assumption."
            ),
        },
        "hot_idle_time_range_min": HOT_IDLE_RANGE,
        "cold_start_definition": "start_mode='cold': metal_temp=wall_temp=25°C (ambient)",
        "hot_start_definition": "start_mode='hot': wall=25+200*exp(-idle/60)°C, metal=25+0.7*(wall-25)°C",
        "env_base_kwargs": ENV_BASE_KWARGS,
        "plant_baseline_n103": {
            "energy_mean_kwh": PLANT_ENERGY_MEAN,
            "energy_sd_kwh": PLANT_ENERGY_SD,
            "duration_mean_min": PLANT_DURATION_MEAN,
            "duration_sd_min": PLANT_DURATION_SD,
            "n": PLANT_N,
            "source": "data/raw/MDB6 (INDUCTION)_20241028_111546_missing_data.xlsx",
        },
        "episode_configs": {
            sc: [
                {"sub_condition": c["sub_condition"],
                 "idle_time_min": c["idle_time_min"]}
                for c in all_configs[sc]
            ]
            for sc in scenarios
        },
    }
    mfst_path = os.path.join(OUTPUT_DIR, "cold_hot_mixed_manifest.json")
    with open(mfst_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {mfst_path}")
    print("\n" + "=" * 70)
    print("Done. Check outputs/revision_phase1/rl_eval/")
    print("=" * 70)


if __name__ == "__main__":
    main()
