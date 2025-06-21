# RL Parameter Sensitivity Analysis Report
Generated on: 2025-06-21 13:55:02

## Executive Summary
This report presents the results of parameter sensitivity analysis for the DQN model.
The analysis evaluates how changes in hyperparameters affect training performance.

## Parameter Ranges Tested
- **learning_rate**: [0.0001, 0.0005, 0.001, 0.005]
- **gamma**: [0.9, 0.95, 0.99, 0.995]
- **epsilon_decay**: [0.995, 0.999, 0.9995, 0.9999]
- **batch_size**: [32, 64, 128, 256]
- **hidden_size**: [64, 128, 256, 512]
- **target_update_freq**: [100, 500, 1000, 2000]

## learning_rate Analysis
### Results Summary
- **Best Value**: 0.001 (Reward: 103.26)
- **Worst Value**: 0.005 (Reward: 90.28)
- **Performance Range**: 12.98

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 0.0001 | 96.11 | 605 | 0.082 | 0.799 | 70.3 |
| 0.0005 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 0.001 | 103.26 | 451 | 0.133 | 0.875 | 55.0 |
| 0.005 | 90.28 | 394 | 0.182 | 0.769 | 49.6 |

## gamma Analysis
### Results Summary
- **Best Value**: 0.995 (Reward: 102.00)
- **Worst Value**: 0.9 (Reward: 84.52)
- **Performance Range**: 17.48

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 0.9 | 84.52 | 560 | 0.109 | 0.771 | 64.7 |
| 0.95 | 95.63 | 522 | 0.101 | 0.827 | 62.0 |
| 0.99 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 0.995 | 102.00 | 537 | 0.090 | 0.853 | 62.2 |

## epsilon_decay Analysis
### Results Summary
- **Best Value**: 0.9995 (Reward: 100.86)
- **Worst Value**: 0.999 (Reward: 99.37)
- **Performance Range**: 1.49

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 0.995 | 99.70 | 501 | 0.102 | 0.845 | 59.4 |
| 0.999 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 0.9995 | 100.86 | 488 | 0.101 | 0.856 | 59.6 |
| 0.9999 | 100.76 | 505 | 0.102 | 0.849 | 59.5 |

## batch_size Analysis
### Results Summary
- **Best Value**: 256 (Reward: 100.04)
- **Worst Value**: 128 (Reward: 98.93)
- **Performance Range**: 1.11

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 32 | 99.92 | 500 | 0.103 | 0.852 | 60.1 |
| 64 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 128 | 98.93 | 506 | 0.098 | 0.850 | 60.5 |
| 256 | 100.04 | 499 | 0.100 | 0.847 | 59.7 |

## hidden_size Analysis
### Results Summary
- **Best Value**: 512 (Reward: 101.02)
- **Worst Value**: 128 (Reward: 99.37)
- **Performance Range**: 1.66

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 64 | 99.57 | 494 | 0.097 | 0.856 | 60.3 |
| 128 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 256 | 99.58 | 511 | 0.098 | 0.850 | 59.7 |
| 512 | 101.02 | 503 | 0.101 | 0.858 | 60.7 |

## target_update_freq Analysis
### Results Summary
- **Best Value**: 2000 (Reward: 100.25)
- **Worst Value**: 100 (Reward: 99.06)
- **Performance Range**: 1.18

### Detailed Results
| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |
|-------|--------------|---------------------|-------------------|----------------|---------------|
| 100 | 99.06 | 493 | 0.098 | 0.850 | 59.8 |
| 500 | 99.96 | 497 | 0.099 | 0.845 | 60.0 |
| 1000 | 99.37 | 499 | 0.100 | 0.848 | 59.8 |
| 2000 | 100.25 | 504 | 0.108 | 0.858 | 60.1 |

## Recommendations
Based on the sensitivity analysis, the following parameter values are recommended:

- **learning_rate**: 0.001
- **gamma**: 0.995
- **epsilon_decay**: 0.9995
- **batch_size**: 256
- **hidden_size**: 512
- **target_update_freq**: 2000

These recommendations are based on maximizing final reward performance.
Consider other factors such as training time and stability when making final decisions.