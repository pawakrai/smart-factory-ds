# Best Model Selection Report

**Generated**: 2026-04-16T12:48:33.805599
**Git commit**: 1e63d91

## Selection Rule (Predefined)

1. Discard `success_rate < 100%` under mixed_80_20
2. Discard `executed_violations > 0` under any scenario
3. Discard `duration_mean > 94.1 min` (original DQN + 2.0 min guard)
4. Rank remaining by lowest `energy_mean` under mixed_80_20
5. Tiebreak: lower overshoot_mean → shorter duration_mean

## Reference: Original DQN Final (mixed_80_20)

- Energy: 600.79 ± 19.96 kWh
- Duration: 92.1 ± 2.9 min
- Overshoot: 4.33 °C

## Gate Results

| Candidate | Energy (kWh) | Duration (min) | Success | Violations | Gate1 | Gate2 | Gate3 | Pass |
|---|---|---|---|---|---|---|---|---|
| baseline_reward_seed2024 | 603.92 ± 24.72 | 97.5 ± 3.0 | 100% | 0 | PASS | PASS | FAIL | FAIL |
| baseline_reward_seed42 | 601.03 ± 19.95 | 92.2 ± 2.9 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| baseline_reward_seed7 | 504.79 ± 183.14 | 95.9 ± 12.3 | 80% | 0 | FAIL | PASS | FAIL | FAIL |
| energy_x1.25_plus_overshoot_seed2024 | 599.93 ± 17.79 | 90.0 ± 1.7 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.25_plus_overshoot_seed42 | 601.93 ± 14.51 | 88.4 ± 1.5 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.25_plus_overshoot_seed7 | 604.04 ± 15.00 | 88.1 ± 1.9 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.25_seed2024 | 621.61 ± 11.44 | 87.2 ± 2.8 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.25_seed42 | 636.96 ± 39.08 | 80.5 ± 4.9 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.25_seed7 | 699.57 ± 4.12 | 120.0 ± 0.0 | 0% | 0 | FAIL | PASS | FAIL | FAIL |
| energy_x1.50_plus_overshoot_seed2024 | 597.96 ± 18.23 | 90.8 ± 2.6 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.50_plus_overshoot_seed42 | 601.46 ± 16.53 | 90.5 ± 2.4 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.50_plus_overshoot_seed7 | 613.36 ± 11.98 | 85.8 ± 1.5 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.50_seed2024 | 613.16 ± 28.00 | 97.2 ± 4.8 | 100% | 0 | PASS | PASS | FAIL | FAIL |
| energy_x1.50_seed42 | 606.07 ± 25.08 | 87.8 ± 3.1 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x1.50_seed7 | 598.48 ± 17.75 | 89.5 ± 1.4 | 100% | 0 | PASS | PASS | PASS | **PASS** |
| energy_x2.00_seed2024 | 622.98 ± 24.94 | 96.7 ± 3.1 | 100% | 0 | PASS | PASS | FAIL | FAIL |
| energy_x2.00_seed42 | 629.31 ± 30.61 | 105.8 ± 6.9 | 100% | 0 | PASS | PASS | FAIL | FAIL |
| energy_x2.00_seed7 | 610.01 ± 20.61 | 106.7 ± 6.7 | 100% | 0 | PASS | PASS | FAIL | FAIL |

## Ranked Candidates (Passing All Gates)

| Rank | Candidate | Energy (kWh) | Overshoot (°C) | Duration (min) | Δ Energy |
|---|---|---|---|---|---|
| 1 | energy_x1.50_plus_overshoot_seed2024 | 597.96 ± 18.23 | 3.41 | 90.8 | -2.83 kWh |
| 2 | energy_x1.50_seed7 | 598.48 ± 17.75 | 3.38 | 89.5 | -2.31 kWh |
| 3 | energy_x1.25_plus_overshoot_seed2024 | 599.93 ± 17.79 | 4.21 | 90.0 | -0.85 kWh |
| 4 | baseline_reward_seed42 | 601.03 ± 19.95 | 4.51 | 92.2 | +0.24 kWh |
| 5 | energy_x1.50_plus_overshoot_seed42 | 601.46 ± 16.53 | 5.12 | 90.5 | +0.68 kWh |
| 6 | energy_x1.25_plus_overshoot_seed42 | 601.93 ± 14.51 | 4.15 | 88.4 | +1.15 kWh |
| 7 | energy_x1.25_plus_overshoot_seed7 | 604.04 ± 15.00 | 3.76 | 88.1 | +3.26 kWh |
| 8 | energy_x1.50_seed42 | 606.07 ± 25.08 | 4.83 | 87.8 | +5.29 kWh |
| 9 | energy_x1.50_plus_overshoot_seed7 | 613.36 ± 11.98 | 4.96 | 85.8 | +12.58 kWh |
| 10 | energy_x1.25_seed2024 | 621.61 ± 11.44 | 5.18 | 87.2 | +20.83 kWh |
| 11 | energy_x1.25_seed42 | 636.96 ± 39.08 | 4.38 | 80.5 | +36.17 kWh |

## Winner: `energy_x1.50_plus_overshoot_seed2024`

Selected per predefined rule. No manual override applied.