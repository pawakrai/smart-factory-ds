# RL Reward-Retuning Audit

**Generated**: 2026-04-16T12:02:14.331862
**Git commit**: 1e63d91

## 1. Current Reward Structure (env_11, lines 488-520)

| Component | Coefficient | env_11 Line | Tuned? |
|---|---|---|---|
| Expert tracking penalty | 0.5 | 488 | No |
| Safety penalty | 5.0 | 492-493 | No |
| Progress reward | 2.0 | 499 | No |
| **Energy penalty** | **0.02** | **502** | **Yes** |
| Time penalty | 0.001 | 505 | No |
| Terminal success base | 10.0 | 518 | No |
| Terminal success efficiency | 5.0 | 518 | No |
| Terminal failure | 5.0 | 520 | No |
| **Overshoot penalty (new)** | **0.0** | **N/A** | **Yes** |

## 2. Tuning Rationale

- `energy_scale` was identified as the most sensitive coefficient by prior OFAT analysis
  (range 11.82 vs ≤2.45 for all others).
- `overshoot_scale` is a new additive term in the terminal reward, penalizing temperature
  overshoot beyond the 950°C target.  Current DQN mean overshoot = 5.58°C (SD 2.69).
- All other coefficients are held fixed to isolate the effect of energy/overshoot tuning.

## 3. Training Distribution

All training uses `start_mode='hot'` only, preserving the original training distribution
from `train_with_env_11.py`.  Generalization to cold-start and mixed conditions is
assessed exclusively via the evaluation protocol (eval_rl_relearn_grid.py).

## 4. Parity Gate

- Result: **PASSED**
- Timestamp: 2026-04-16T11:12:15.722210

Details:
  - PASS reset state match
  - PASS step 1 action=9: reward=-0.452599 (patch=-0.452599)
  - PASS step 2 action=5: reward=-0.243559 (patch=-0.243559)
  - PASS step 3 action=9: reward=-0.489649 (patch=-0.489649)
  - PASS step 4 action=0: reward=-0.090929 (patch=-0.090929)
  - PASS step 5 action=7: reward=-0.384600 (patch=-0.384600)
  - PASS step 6 action=9: reward=-0.402172 (patch=-0.402172)

## 5. Training Summary

| Config | Seed | Episodes | Wall Time (s) | Final Avg Reward (100) | Final Avg Energy (100) |
|---|---|---|---|---|---|
| baseline_reward | 42 | 1500 | 185 | -0.58 | 572.7 kWh |
| baseline_reward | 2024 | 1500 | 189 | -2.13 | 568.0 kWh |
| baseline_reward | 7 | 1500 | 191 | 0.68 | 573.2 kWh |
| energy_x1.25 | 42 | 1500 | 132 | -13.09 | 550.0 kWh |
| energy_x1.25 | 2024 | 1500 | 178 | -3.13 | 573.9 kWh |
| energy_x1.25 | 7 | 1500 | 177 | -10.45 | 567.7 kWh |
| energy_x1.50 | 42 | 1500 | 182 | -7.63 | 570.2 kWh |
| energy_x1.50 | 2024 | 1500 | 180 | -8.54 | 576.3 kWh |
| energy_x1.50 | 7 | 1500 | 162 | -6.86 | 575.5 kWh |
| energy_x2.00 | 42 | 1500 | 157 | -16.71 | 573.2 kWh |
| energy_x2.00 | 2024 | 1500 | 150 | -19.20 | 580.4 kWh |
| energy_x2.00 | 7 | 1500 | 156 | -13.52 | 573.5 kWh |
| energy_x1.50_plus_overshoot | 42 | 1500 | 144 | -8.37 | 568.5 kWh |
| energy_x1.50_plus_overshoot | 2024 | 1500 | 152 | -7.81 | 574.0 kWh |
| energy_x1.50_plus_overshoot | 7 | 1500 | 162 | -13.21 | 589.4 kWh |
| energy_x1.25_plus_overshoot | 42 | 1500 | 165 | -11.15 | 586.5 kWh |
| energy_x1.25_plus_overshoot | 2024 | 1500 | 174 | -6.92 | 559.7 kWh |
| energy_x1.25_plus_overshoot | 7 | 1500 | 163 | -9.02 | 562.4 kWh |

**Total wall time**: 2999 s (0.8 h)