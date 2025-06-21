# RL Model Validation Report
Generated on: 2025-06-21 19:10:50
Model: models/dqn_final_model_10.pth

## Performance Metrics Summary
- Average Temperature Accuracy: 99.97% ± 0.04%
- Average Energy Consumption: 433.33 ± 85.81 kWh
- Average Melting Time: 75.33 ± 11.44 minutes
- Number of Scenarios Tested: 3

## Individual Scenario Results
| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |
|----------|------------------|-------------------|------------|--------------|--------------|------------|
| 500kg_900C | 900 | 900.0 | 0.0 | 100.0 | 535.8 | 89.0 |
| 400kg_900C | 900 | 900.0 | 0.0 | 100.0 | 438.3 | 76.0 |
| 500kg_850C | 850 | 850.7 | 0.7 | 99.9 | 325.8 | 61.0 |

## Baseline Comparison
| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |
|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|
| 500kg_900C | 535.8 | 565.3 | 5.2 | 89.0 | 90.0 | 1.1 |
| 400kg_900C | 438.3 | 450.0 | 2.6 | 76.0 | 75.0 | -1.3 |
| 500kg_850C | 325.8 | 420.0 | 22.4 | 61.0 | 70.0 | 12.9 |
