"""
Script D-tier2 — Short RL Retrains Under Alternative Reward Weights (Reviewer Item D)
=======================================================================================
Demonstrates that the choice of reward coefficients has an empirically measurable
effect on trained policy behavior (not just evaluation scoring).

METHOD:
  - Subclass AluminumMeltingEnvironment in this file (no src/ modification)
  - Override step() to recompute reward with patched coefficients
  - Train 3 variants × 500 episodes with seed=42
  - Evaluate each for 50 episodes; compare energy/duration/success/safety

MANDATORY PARITY GATE (runs first):
  Before any training, the PatchedEnv at baseline coefficients must produce
  identical states and rewards to the original env on a fixed action sequence.
  If this test fails, training is aborted and the failure is recorded.
  *** Do NOT proceed with retraining if parity check fails. ***

VARIANTS:
  - baseline_coeffs : original coefficients (500 ep retrain for fair comparison)
  - safety_low      : safety_coeff = 1.0  (×0.2 — less safety emphasis)
  - safety_high     : safety_coeff = 25.0 (×5.0 — more safety emphasis)

Outputs (outputs/revision_phase1/rl_reward_variants/):
  parity_test_result.json
  variant_{name}_model.pth
  variant_{name}_eval.json
  comparison_table.csv
  rl_reward_variants_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/train_rl_reward_variants.py
"""

import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "rl_reward_variants"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
TRAIN_EPISODES = 500
EVAL_EPISODES = 50
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

# Baseline coefficients (must match env_11 lines 488-520 exactly)
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
}

VARIANTS = {
    "baseline_coeffs": BASE_COEFFS.copy(),
    "safety_low":      {**BASE_COEFFS, "safety_scale": 1.0},    # ×0.2
    "safety_high":     {**BASE_COEFFS, "safety_scale": 25.0},   # ×5.0
}


# ─────────────────────────────────────────────────────────────────────────────
# PatchedEnv — subclass that overrides only the reward computation
# ─────────────────────────────────────────────────────────────────────────────

class PatchedEnv(AluminumMeltingEnvironment):
    """
    Subclass of AluminumMeltingEnvironment that replaces the reward function with
    a parameterized version while keeping all physics 100% unchanged.

    Physics update: delegated entirely to super().step().
    Reward computation: replicated from env_11 lines 488-520 with substituted coefficients.

    The dprog term requires the pre-step progress value, which must be captured
    BEFORE calling super().step() (parent updates self._prev_progress internally).
    """

    def __init__(self, *args, reward_coeffs: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._coeffs = BASE_COEFFS.copy()
        if reward_coeffs:
            self._coeffs.update(reward_coeffs)

    def step(self, action, power_profile_kw=None):
        # ── Capture pre-step values before parent modifies state ───────────
        t_min_before = float(self.state["time"]) / 60.0
        target_power = self.action_space.get(int(action), 0.0) if power_profile_kw is None \
                       else float(power_profile_kw(t_min_before))
        prev_progress = self._progress()   # ← must capture before super() updates _prev_progress

        # ── Delegate physics to parent (also computes & returns original reward) ─
        state_arr, _orig_reward, done = super().step(action, power_profile_kw=power_profile_kw)

        # ── Recompute reward with patched coefficients ──────────────────────
        # Re-read state after parent step
        Tm_after = float(self.state["temperature"])
        t_min = t_min_before   # alloying check uses the time BEFORE the step tick
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

        # 5.4 Energy efficiency penalty (energy used this step)
        energy_used_kwh = float(state_arr[5]) - float(prev_progress)  # will be corrected below
        # Correctly derive energy delta from state array indices
        # state_arr[5] = cumulative energy AFTER step; parent added energy_used_kwh
        # We need energy_used_kwh = metered_power × dt / 3600 × scale (same as parent)
        # Simplest: energy_used_kwh = state_arr[5] - (previous cumulative energy)
        # But we don't have pre-step energy easily from here... use parent's computation result:
        # parent step already updated self.state["energy_consumption"], so diff from prev is:
        energy_used_kwh = float(self.state["energy_consumption"]) - (
            float(state_arr[5]) - (float(self.state["energy_consumption"]) - float(self.state["energy_consumption"]))
        )
        # Simpler derivation: energy added this step = power × dt / 3600 × scale
        # parent stored applied power in self.state["power"] after the step
        applied_power_kw = float(self.state["power"]) + self.auxiliary_power_kw
        energy_used_kwh = float((applied_power_kw * self.dt) / 3600.0) * self.energy_consumption_scale
        energy_penalty = -energy_used_kwh * self._coeffs["energy_scale"]

        # 5.5 Time penalty
        time_penalty = -self._coeffs["time_scale"]

        # 5.6 Terminal reward
        terminal = 0.0
        if done:
            final_temp = float(self.state["temperature"])
            success = final_temp >= self.target_temp
            if success:
                weight = float(self.state["weight"])
                total_energy = float(self.state["energy_consumption"])
                efficiency = weight / max(1e-6, total_energy)
                norm_eff = min(efficiency / self._coeffs["optimal_efficiency"], 1.0)
                terminal = self._coeffs["terminal_success_base"] + norm_eff * self._coeffs["terminal_success_eff"]
            else:
                terminal = -self._coeffs["terminal_fail"]

        reward = tracking_penalty + safety_penalty + progress_reward + energy_penalty + time_penalty + terminal
        return state_arr, float(reward), done


# ─────────────────────────────────────────────────────────────────────────────
# Parity Gate
# ─────────────────────────────────────────────────────────────────────────────

def run_parity_test() -> dict:
    """
    Verify that PatchedEnv at baseline coefficients produces identical
    state transitions and rewards to the original AluminumMeltingEnvironment.

    Returns a dict with 'passed' (bool) and 'details'.
    """
    print("\n--- Mandatory Parity Gate ---")
    TEST_SEED = 0
    TEST_ACTIONS = [9, 5, 9, 0, 7, 9]   # representative action sequence

    env_orig = AluminumMeltingEnvironment(seed=TEST_SEED, **ENV_KWARGS)
    env_patch = PatchedEnv(seed=TEST_SEED, reward_coeffs=BASE_COEFFS, **ENV_KWARGS)

    state_o = env_orig.reset()
    state_p = env_patch.reset()

    details = []
    passed = True

    # Check reset parity
    if not np.allclose(state_o, state_p, atol=1e-5):
        passed = False
        details.append(f"FAIL reset: orig={state_o} patch={state_p}")
    else:
        details.append("PASS reset state match")

    # Check step parity
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
                f"state_diff={np.abs(so-sp).max():.6f} "
                f"reward_orig={ro:.6f} reward_patch={rp:.6f} "
                f"done_orig={do} done_patch={dp}"
            )
        else:
            details.append(
                f"PASS step {i+1} action={action}: "
                f"reward={ro:.6f} (patch={rp:.6f})"
            )

        if do or dp:
            break  # episode ended

    result = {
        "passed": passed,
        "test_seed": TEST_SEED,
        "test_actions": TEST_ACTIONS,
        "details": details,
        "timestamp": datetime.now().isoformat(),
    }

    for line in details:
        print(f"  {line}")
    if passed:
        print("  ✓ Parity PASSED — proceeding with training")
    else:
        print("  ✗ Parity FAILED — aborting training (see parity_test_result.json)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(name: str, coeffs: dict, seed: int) -> tuple[DQNAgent, list[float]]:
    """Train a DQN agent on PatchedEnv for TRAIN_EPISODES episodes."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = PatchedEnv(seed=seed, reward_coeffs=coeffs, **ENV_KWARGS)
    agent = DQNAgent(**AGENT_KWARGS)
    episode_rewards = []

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
        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            recent = episode_rewards[-100:]
            print(f"  [{name}] Episode {ep+1}/{TRAIN_EPISODES} | "
                  f"avg reward (last 100): {np.mean(recent):.2f} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name: str, agent: DQNAgent) -> dict:
    """Evaluate a trained agent for EVAL_EPISODES episodes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED + 9999)  # different from training seed
    agent.model.eval()
    agent.epsilon = 0.0

    durations, energies, successes, safety_violations = [], [], [], []

    for ep in range(EVAL_EPISODES):
        idle_time = float(rng.uniform(0, 30))
        env = AluminumMeltingEnvironment(idle_time_min=idle_time, **ENV_KWARGS)
        state = env.reset()
        done = False
        ep_violations = 0

        while not done:
            t_min = float(state[2]) / 60.0
            in_alloying = (30.0 <= t_min < 31.0)

            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = int(torch.argmax(agent.model(s_t)).item())

            if in_alloying and action > 0:
                ep_violations += 1

            state, _r, done = env.step(action)

        final_temp = float(env.state["temperature"])
        durations.append(float(env.state["time"]) / 60.0)
        energies.append(float(env.state["energy_consumption"]))
        successes.append(float(final_temp >= TARGET_TEMP_C))
        safety_violations.append(ep_violations)

    return {
        "variant": name,
        "n_eval_episodes": EVAL_EPISODES,
        "duration_mean": float(np.mean(durations)),
        "duration_sd": float(np.std(durations)),
        "energy_mean_kwh": float(np.mean(energies)),
        "energy_sd_kwh": float(np.std(energies)),
        "success_rate": float(np.mean(successes)),
        "safety_violations_mean_per_ep": float(np.mean(safety_violations)),
        "safety_violations_sd_per_ep": float(np.std(safety_violations)),
        "safety_violation_free_pct": float(np.mean(np.array(safety_violations) == 0)),
        "coefficients": coeffs if False else None,  # placeholder; set below
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    print("=" * 65)
    print("train_rl_reward_variants.py — Reviewer Item D (Tier 2)")
    print("=" * 65)
    print(f"Seed          : {SEED}")
    print(f"Train episodes: {TRAIN_EPISODES} per variant")
    print(f"Eval episodes : {EVAL_EPISODES} per variant")
    print(f"Variants      : {list(VARIANTS.keys())}")

    # ── Parity gate ──────────────────────────────────────────────────────────
    parity_result = run_parity_test()
    parity_path = OUTPUT_DIR / "parity_test_result.json"
    with open(parity_path, "w") as f:
        json.dump(parity_result, f, indent=2)

    if not parity_result["passed"]:
        print("\n[ABORT] Parity check failed. Inspect parity_test_result.json.")
        print("        Training aborted per project safety policy.")
        # Write manifest even on abort to record the failure
        manifest = {
            "script": "train_rl_reward_variants.py",
            "run_timestamp": datetime.now().isoformat(),
            "git_commit": _git_commit(),
            "parity_test": {"result": "FAILED", "path": str(parity_path.relative_to(PROJECT_ROOT))},
            "training_status": "ABORTED",
            "reason": "Mandatory parity gate failed. PatchedEnv reward does not match original.",
        }
        with open(OUTPUT_DIR / "rl_reward_variants_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # ── Train all variants ────────────────────────────────────────────────────
    all_eval_results = []
    training_histories = {}

    for variant_name, coeffs in VARIANTS.items():
        print(f"\n{'='*65}")
        print(f"Training variant: {variant_name}")
        safety_label = "baseline" if coeffs["safety_scale"] == BASE_COEFFS["safety_scale"] \
                       else f"x{coeffs['safety_scale'] / BASE_COEFFS['safety_scale']:.1f}"
        print(f"  safety_scale = {coeffs['safety_scale']} ({safety_label})")

        agent, ep_rewards = train_variant(variant_name, coeffs, SEED)

        # Save model
        model_path = OUTPUT_DIR / f"variant_{variant_name}_model.pth"
        torch.save(agent.model.state_dict(), model_path)
        print(f"  Saved model: {model_path.relative_to(PROJECT_ROOT)}")
        training_histories[variant_name] = ep_rewards

        # Evaluate
        print(f"  Evaluating {variant_name}...")
        eval_result = evaluate_variant(variant_name, agent)
        eval_result["coefficients"] = coeffs
        all_eval_results.append(eval_result)

        # Save per-variant eval
        eval_path = OUTPUT_DIR / f"variant_{variant_name}_eval.json"
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2)

        print(f"  Energy : {eval_result['energy_mean_kwh']:.2f} ± {eval_result['energy_sd_kwh']:.2f} kWh")
        print(f"  Success: {eval_result['success_rate']:.1%}")
        print(f"  Safety violations/ep: {eval_result['safety_violations_mean_per_ep']:.2f} ± "
              f"{eval_result['safety_violations_sd_per_ep']:.2f}")

    # ── Comparison table ──────────────────────────────────────────────────────
    table_rows = []
    for r in all_eval_results:
        table_rows.append({
            "variant": r["variant"],
            "safety_coeff": r["coefficients"]["safety_scale"],
            "n_eval": r["n_eval_episodes"],
            "duration_mean_min": r["duration_mean"],
            "duration_sd_min": r["duration_sd"],
            "energy_mean_kwh": r["energy_mean_kwh"],
            "energy_sd_kwh": r["energy_sd_kwh"],
            "success_rate": r["success_rate"],
            "safety_violations_mean_per_ep": r["safety_violations_mean_per_ep"],
            "violation_free_pct": r["safety_violation_free_pct"],
        })

    table_path = OUTPUT_DIR / "comparison_table.csv"
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"\n  Comparison table: {table_path.relative_to(PROJECT_ROOT)}")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "train_rl_reward_variants.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "parity_test": {"result": "PASSED"},
        "training_status": "COMPLETED",
        "env_kwargs": ENV_KWARGS,
        "agent_kwargs": AGENT_KWARGS,
        "variants": {
            name: {"coefficients": coeffs}
            for name, coeffs in VARIANTS.items()
        },
        "filters_applied": [],
        "config_values": {
            "target_temp_c": TARGET_TEMP_C,
            "max_power_kw": MAX_POWER_KW,
        },
    }
    with open(OUTPUT_DIR / "rl_reward_variants_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/rl_reward_variants/")
    print("=" * 65)


if __name__ == "__main__":
    main()
