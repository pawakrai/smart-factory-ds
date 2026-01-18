# RL Model Validation Report
Generated on: 2026-01-13 17:27:51
Model: models/dqn_final_model_11.pth

## Performance Metrics Summary
- Average Temperature Accuracy: 98.81% ± 0.78%
- Average Energy Consumption: 546.67 ± 10.61 kWh
- Average Melting Time: 90.00 ± 1.41 minutes
- Number of Scenarios Tested: 3

## Individual Scenario Results
| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |
|----------|------------------|-------------------|------------|--------------|--------------|------------|
| 500kg_900C | 900 | 917.7 | 17.7 | 98.0 | 561.7 | 92.0 |
| 400kg_900C | 900 | 913.4 | 13.4 | 98.5 | 539.2 | 89.0 |
| 500kg_850C | 850 | 851.0 | 1.0 | 99.9 | 539.2 | 89.0 |

## Baseline Comparison
| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |
|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|
| 500kg_900C | 561.7 | 578.4 | 2.9 | 92.0 | 110.0 | 16.4 |
| 400kg_900C | 539.2 | 560.0 | 3.7 | 89.0 | 90.0 | 1.1 |
| 500kg_850C | 539.2 | 570.0 | 5.4 | 89.0 | 92.0 | 3.3 |
