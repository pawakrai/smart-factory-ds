# RL Model Validation Report
Generated on: 2026-01-28 17:29:48
Model: models/dqn_final_model_11.pth

## Performance Metrics Summary
- Average Temperature Accuracy: 98.68% ± 0.85%
- Average Energy Consumption: 549.17 ± 9.35 kWh
- Average Melting Time: 90.33 ± 1.25 minutes
- Number of Scenarios Tested: 3

## Individual Scenario Results
| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |
|----------|------------------|-------------------|------------|--------------|--------------|------------|
| 500kg_900C | 900 | 917.7 | 17.7 | 98.0 | 561.7 | 92.0 |
| 400kg_900C | 900 | 917.0 | 17.0 | 98.1 | 546.7 | 90.0 |
| 500kg_850C | 850 | 851.0 | 1.0 | 99.9 | 539.2 | 89.0 |

## Baseline Comparison
| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |
|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|
| 500kg_900C | 561.7 | 578.4 | 2.9 | 92.0 | 110.0 | 16.4 |
| 400kg_900C | 546.7 | 560.0 | 2.4 | 90.0 | 90.0 | 0.0 |
| 500kg_850C | 539.2 | 570.0 | 5.4 | 89.0 | 92.0 | 3.3 |
