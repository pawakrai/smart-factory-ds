# RL Model Validation Report
Generated on: 2025-06-22 14:16:10
Model: models/dqn_final_model_11.pth

## Performance Metrics Summary
- Average Temperature Accuracy: 98.61% ± 0.91%
- Average Energy Consumption: 551.39 ± 9.06 kWh
- Average Melting Time: 90.67 ± 1.25 minutes
- Number of Scenarios Tested: 3

## Individual Scenario Results
| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |
|----------|------------------|-------------------|------------|--------------|--------------|------------|
| 500kg_900C | 900 | 916.7 | 16.7 | 98.1 | 560.8 | 92.0 |
| 400kg_900C | 900 | 919.7 | 19.7 | 97.8 | 554.2 | 91.0 |
| 500kg_850C | 850 | 851.0 | 1.0 | 99.9 | 539.2 | 89.0 |

## Baseline Comparison
| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |
|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|
| 500kg_900C | 560.8 | 578.4 | 3.0 | 92.0 | 110.0 | 16.4 |
| 400kg_900C | 554.2 | 560.0 | 1.0 | 91.0 | 90.0 | -1.1 |
| 500kg_850C | 539.2 | 570.0 | 5.4 | 89.0 | 92.0 | 3.3 |
