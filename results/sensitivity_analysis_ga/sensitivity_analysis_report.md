# GA Sensitivity Analysis Report

Generated on: 2025-06-23 12:09:35

## Analysis Configuration

- Number of batches: 8
- Default parameters: {'crossover_rate': 0.8, 'mutation_rate': 0.1, 'population_size': 150, 'num_generations': 100, 'seed': 42}

## Parameter Ranges Analyzed

- **crossover_rate**: [0.6, 0.7, 0.8, 0.9]
- **mutation_rate**: [0.01, 0.05, 0.1, 0.2]
- **population_size**: [50, 100, 150, 200]
- **num_generations**: [50, 100, 150, 200]

## Results Summary

| Parameter | Best Value | Best Fitness | Convergence Gen | Sensitivity |
|-----------|------------|--------------|-----------------|-------------|
| crossover_rate | 0.7 | 886336.00 | 37 | 0.001 |
| mutation_rate | 0.05 | 886336.00 | 40 | 0.001 |
| population_size | 100 | 888331.00 | 18 | 0.001 |
| num_generations | 50 | 889330.75 | 22 | 0.000 |

## Detailed Analysis

### crossover_rate

| Value | Best Fitness | Convergence Gen | Exec Time (s) | Diversity |
|-------|--------------|-----------------|---------------|----------|
| 0.6 | 888331.50 | 24 | 40.30 | 0.000 |
| 0.7 | 886336.00 | 37 | 39.06 | 0.075 |
| 0.8 | 889330.75 | 22 | 44.37 | 0.000 |
| 0.9 | 886336.00 | 30 | 36.48 | 0.075 |

### mutation_rate

| Value | Best Fitness | Convergence Gen | Exec Time (s) | Diversity |
|-------|--------------|-----------------|---------------|----------|
| 0.01 | 887333.75 | 35 | 37.64 | 0.000 |
| 0.05 | 886336.00 | 40 | 38.18 | 0.000 |
| 0.1 | 889330.75 | 22 | 43.74 | 0.000 |
| 0.2 | 887333.50 | 36 | 38.20 | 0.000 |

### population_size

| Value | Best Fitness | Convergence Gen | Exec Time (s) | Diversity |
|-------|--------------|-----------------|---------------|----------|
| 50 | 889328.50 | 34 | 14.58 | 0.000 |
| 100 | 888331.00 | 18 | 26.26 | 0.000 |
| 150 | 889330.75 | 22 | 43.63 | 0.000 |
| 200 | 888331.00 | 30 | 50.92 | 0.000 |

### num_generations

| Value | Best Fitness | Convergence Gen | Exec Time (s) | Diversity |
|-------|--------------|-----------------|---------------|----------|
| 50 | 889330.75 | 22 | 19.67 | 0.000 |
| 100 | 889330.75 | 22 | 43.86 | 0.000 |
| 150 | 889330.75 | 22 | 68.55 | 0.000 |
| 200 | 889330.75 | 22 | 92.14 | 0.000 |

## Recommendations

Based on the sensitivity analysis:

- **crossover_rate**: Optimal value is 0.7
- **mutation_rate**: Optimal value is 0.05
- **population_size**: Optimal value is 100
- **num_generations**: Optimal value is 50
