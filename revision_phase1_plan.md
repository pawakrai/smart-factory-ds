# Revision Phase 1 Plan — Aluminum Melting RL/GA Paper

**Status:** In progress  
**Last updated:** 2026-04-12  
**Purpose:** Augment codebase with reviewer-required analyses. All new work is in
`revision_experiments/` and `outputs/revision_phase1/`. No `src/` files are modified.

---

## Codebase Map

| Component | File | Notes |
|-----------|------|-------|
| RL environment | `src/environment/aluminum_melting_env_11.py` | Reward weights are hardcoded literals (lines 488–518), not constructor params |
| DQN agent | `src/agents/agent2.py` | `DQNAgent(state_dim=8, action_dim=10)` for all .pth files |
| RL training | `src/training/train_with_env_11.py` | 5000 eps, no explicit seed |
| RL evaluation | `src/training/evaluate_model_env11.py` | 50 eps, no seed, only duration+energy |
| Final model | `models/dqn_final_model_env11.pth` | Production model |
| Checkpoints | `checkpoints_env_11/dqn_episode_{500,1000,1500}.pth` | Same architecture |
| GA scheduler | `src/app_v9.py` | `DETERMINISTIC_SIMULATION=True` |
| GA harness | `src/experiment_compare.py` | `_run_ga_mode()` is reusable |
| Baselines | `src/policies_baseline.py` | `continuous_melting_controller`, `make_rule_based_controller` |
| Plant data | `data/raw/*.xlsx`, `data/สรุปการหลอมทุก Batch new.xlsx` | Sole valid source for historical baseline |
| Existing RL sensitivity | `results/sensitivity_analysis_rl/` | **⚠ Was run on env_8 (7-dim, 5-action), NOT env_11 — not representative of final model** |
| Existing GA sensitivity | `results/sensitivity_analysis_ga/` | Usable as-is |
| Baseline comparison | `src/experiment_compare_results.csv` | One run only (seed=42) |

---

## Safety Constraints

- **Do NOT modify any file in `src/`**
- **Do NOT overwrite `models/dqn_final_model_env11.pth`**
- All outputs → `outputs/revision_phase1/`
- All new scripts → `revision_experiments/`

---

## Key Implementation Facts

- Safety alloying window: `30.0 <= state[2]/60.0 < 31.0`
  - *Attempted violation*: agent chose `action > 0` in this window (pre-override)
  - *Executed violation*: `next_state[3] > 0` in this window (always 0 due to env hard-override at line 443)
- Reward weights are **hardcoded literals** in env_11 `step()` → must subclass to vary
- `power_profile_kw` callable parameter exists in `env.step()` (lines 433–435)
- Historical baseline must come from real plant Excel data only — no simulated proxies

---

## Scripts and Status

| Script | Item | Status | Output |
|--------|------|--------|--------|
| `extract_historical_baseline.py` | A | ⬜ pending | `outputs/revision_phase1/historical_baseline/` |
| `eval_rl_extended.py` | B | ⬜ pending | `outputs/revision_phase1/rl_eval/` |
| `eval_rl_safety.py` | C | ⬜ pending | `outputs/revision_phase1/rl_safety/` |
| `eval_rl_reward_sensitivity.py` | D tier 1 | ⬜ pending | `outputs/revision_phase1/rl_reward_sensitivity/` |
| `train_rl_reward_variants.py` | D tier 2 | ⬜ pending (gated on parity) | `outputs/revision_phase1/rl_reward_variants/` |
| `ga_repeated_runs.py` | E | ⬜ pending | `outputs/revision_phase1/ga_repeated/` |
| `ablation_batch_level.py` | F | ⬜ pending | `outputs/revision_phase1/ablation/` |
| `ga_runtime.py` | H | ⬜ pending | `outputs/revision_phase1/ga_runtime/` |
| `thermal_model_validation.py` | I | ⬜ pending | `outputs/revision_phase1/thermal_validation/` |

**Item G** (rule-based baseline documentation): no new script needed.
`src/policies_baseline.py:19-95` already clearly implements two controllers:
1. `continuous_melting_controller` — greedy, always melts if idle furnace and batches remain
2. `make_rule_based_controller(price_threshold=3.2, peak_headroom_kw=120, fixed_power=475, urgency_start_threshold=0.60)` — refuses if TOU > threshold AND depletion urgency < 0.60, or if projected peak exceeds contract − 120 kW

---

## Execution Order

```
Phase 0 (now):        this file + directory structure
Phase 1 (parallel):   A0, B, C, D1, H, I
Phase 2 (sequential): D2 (gated on parity), E (long-running)
Phase 3 (final):      F (aggregation), results_summary.md
```

---

## Reproducibility Requirements

Every script writes a `*_manifest.json` sidecar containing:
- `script`, `run_timestamp`, `seed`, `num_episodes`
- `episode_conditions`: per-episode sampled parameters (e.g., idle_time_min)
- `filters_applied`, `config_values`, optional `git_commit`

---

## Risks

| Risk | Mitigation |
|------|-----------|
| Excel columns in Thai / missing | Inspect mode always runs first; output gap report if unusable |
| Reward subclass diverges | Mandatory parity gate before any Tier 2 training |
| GA repeated runs > 2 hrs | n_gen=50/pop_size=50 for repeatability; bridge at n=100/80 |
| Existing RL sensitivity not for env_11 | Flagged above; must note in paper |
