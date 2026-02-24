# RL Model Validation Report
Generated on: 2026-02-24 14:09:00
Model: models/dqn_final_model_11.pth

## Performance Metrics Summary
- Average Temperature Accuracy: 99.08% ± 0.77%
- Average Energy Consumption: 546.39 ± 10.21 kWh
- Average Melting Time: 90.00 ± 1.41 minutes
- Number of Scenarios Tested: 3

## Individual Scenario Results
| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |
|----------|------------------|-------------------|------------|--------------|--------------|------------|
| 500kg_900C | 900 | 917.6 | 17.6 | 98.0 | 560.8 | 92.0 |
| 400kg_900C | 900 | 906.3 | 6.3 | 99.3 | 539.2 | 89.0 |
| 500kg_850C | 850 | 851.0 | 1.0 | 99.9 | 539.2 | 89.0 |

## Baseline Comparison
| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |
|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|
| 500kg_900C | 560.8 | 578.4 | 3.0 | 92.0 | 110.0 | 16.4 |
| 400kg_900C | 539.2 | 560.0 | 3.7 | 89.0 | 90.0 | 1.1 |
| 500kg_850C | 539.2 | 570.0 | 5.4 | 89.0 | 92.0 | 3.3 |
