"""
Parameter Sensitivity Analysis for GA (Genetic Algorithm) Model
This script analyzes the sensitivity of GA hyperparameters on optimization performance
Integrates with normal_ga.py for aluminum melting scheduling optimization
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from datetime import datetime
import itertools
import random
from copy import deepcopy
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import GA components from normal_ga.py
sys.path.append(os.path.join(project_root, "src"))

from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.ga import GA

# Import from normal_ga.py
from normal_ga import (
    RealisticDirectSchedulingProblem,
    BalancedDirectSchedulingSampling,
    ScheduleRepair,
    NUM_BATCHES,
    scheduling_cost,
    decode_schedule,
    calculate_realistic_penalties,
)


class GASensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for GA hyperparameters
    """

    def __init__(self, output_dir="results/sensitivity_analysis_ga"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define parameter ranges for sensitivity analysis
        self.param_ranges = {
            "crossover_rate": [0.6, 0.7, 0.8, 0.9],
            "mutation_rate": [0.01, 0.05, 0.1, 0.2],
            "population_size": [50, 100, 150, 200],
            "num_generations": [50, 100, 150, 200],
        }

        # Default parameters (baseline)
        self.default_params = {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "population_size": 150,
            "num_generations": 100,
            "seed": 42,
        }

        self.results = {}

    def run_single_parameter_analysis(self, param_name):
        """
        Analyze sensitivity for a single parameter

        Args:
            param_name: Name of parameter to analyze
        """
        print(f"\n=== Analyzing {param_name} ===")

        param_values = self.param_ranges[param_name]
        results = []

        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")

            # Create parameter configuration
            config = self.default_params.copy()
            config[param_name] = param_value

            # Run GA optimization with this configuration
            metrics = self._run_ga_with_config(config)

            result = {
                "parameter": param_name,
                "value": param_value,
                "best_fitness": metrics["best_fitness"],
                "final_fitness": metrics["final_fitness"],
                "convergence_generation": metrics["convergence_generation"],
                "diversity_maintenance": metrics["diversity_maintenance"],
                "execution_time": metrics["execution_time"],
                "fitness_improvement": metrics["fitness_improvement"],
                "convergence_rate": metrics["convergence_rate"],
            }

            results.append(result)
            print(f"  Best fitness: {metrics['best_fitness']:.2f}")
            print(f"  Convergence generation: {metrics['convergence_generation']}")
            print(f"  Execution time: {metrics['execution_time']:.2f}s")
            print(f"  Diversity maintenance: {metrics['diversity_maintenance']:.3f}")

        # Store results
        self.results[param_name] = results

        # Save detailed results immediately
        self._save_parameter_results(param_name, results)

        return results

    def run_complete_sensitivity_analysis(self):
        """
        Run sensitivity analysis for all parameters
        """
        print("Starting Complete Sensitivity Analysis for GA Parameters...")
        start_time = time.time()

        # Analyze each parameter
        for param_name in self.param_ranges.keys():
            self.run_single_parameter_analysis(param_name)

        # Generate summary analysis
        self._generate_sensitivity_summary()

        # Save all results
        self._save_results()

        total_time = time.time() - start_time
        print(f"\nComplete sensitivity analysis finished in {total_time:.2f} seconds")
        print(f"Results saved to {self.output_dir}")

    def _run_ga_with_config(self, config):
        """
        Run GA optimization with specific configuration and return metrics

        Args:
            config: Dictionary with GA parameters

        Returns:
            Dictionary with GA performance metrics
        """
        start_time = time.time()

        # Create problem
        problem = RealisticDirectSchedulingProblem(NUM_BATCHES)

        # Setup operators
        sampling = BalancedDirectSchedulingSampling()
        crossover = SBX(prob=config["crossover_rate"])
        mutation = PolynomialMutation(prob=config["mutation_rate"])
        repair = ScheduleRepair()

        # Create algorithm
        algorithm = GA(
            pop_size=config["population_size"],
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            eliminate_duplicates=True,
        )

        # Setup termination
        termination = get_termination("n_gen", config["num_generations"])

        # Run optimization
        try:
            result = minimize(
                problem,
                algorithm,
                termination,
                seed=config["seed"],
                verbose=False,  # Disable verbose output for cleaner analysis
                save_history=True,
            )

            execution_time = time.time() - start_time

            # Extract metrics
            metrics = self._extract_metrics_from_result(result)
            metrics["execution_time"] = execution_time

        except Exception as e:
            print(f"Error during GA optimization: {e}")
            # Return default/failure metrics
            metrics = {
                "best_fitness": float("inf"),
                "final_fitness": float("inf"),
                "convergence_generation": config["num_generations"],
                "diversity_maintenance": 0.0,
                "execution_time": time.time() - start_time,
                "fitness_improvement": 0.0,
                "convergence_rate": 0.0,
            }

        return metrics

    def _extract_metrics_from_result(self, result):
        """
        Extract performance metrics from GA optimization result
        """
        metrics = {}

        # Best fitness found
        if result.F is not None:
            metrics["best_fitness"] = float(result.F[0])
        else:
            metrics["best_fitness"] = float("inf")

        # Extract convergence information from history
        if result.history and len(result.history) > 0:
            # Get fitness values over generations
            generation_best_fitness = []
            for gen in result.history:
                if gen.opt is not None and gen.opt.get("F") is not None:
                    gen_fitness = np.min(gen.opt.get("F"))
                    generation_best_fitness.append(gen_fitness)
                else:
                    # If no fitness available, use a large value
                    generation_best_fitness.append(float("inf"))

            metrics["final_fitness"] = (
                generation_best_fitness[-1] if generation_best_fitness else float("inf")
            )

            # Calculate convergence generation (when fitness improvement becomes minimal)
            convergence_generation = self._find_convergence_generation(
                generation_best_fitness
            )
            metrics["convergence_generation"] = convergence_generation

            # Calculate diversity maintenance (approximated by fitness variance in later generations)
            if len(generation_best_fitness) >= 10:
                late_gen_fitness = generation_best_fitness[-10:]
                diversity = (
                    np.std(late_gen_fitness) if len(late_gen_fitness) > 1 else 0.0
                )
                metrics["diversity_maintenance"] = diversity
            else:
                metrics["diversity_maintenance"] = 0.0

            # Calculate fitness improvement rate
            if len(generation_best_fitness) > 1:
                initial_fitness = generation_best_fitness[0]
                final_fitness = generation_best_fitness[-1]
                if initial_fitness != 0 and not np.isinf(initial_fitness):
                    fitness_improvement = (
                        initial_fitness - final_fitness
                    ) / initial_fitness
                else:
                    fitness_improvement = 0.0
                metrics["fitness_improvement"] = fitness_improvement
            else:
                metrics["fitness_improvement"] = 0.0

            # Calculate convergence rate (improvement per generation in early stages)
            if len(generation_best_fitness) >= 20:
                early_improvement = (
                    generation_best_fitness[0] - generation_best_fitness[19]
                )
                convergence_rate = (
                    early_improvement / 20 if generation_best_fitness[0] != 0 else 0.0
                )
                metrics["convergence_rate"] = max(0.0, convergence_rate)
            else:
                metrics["convergence_rate"] = 0.0

        else:
            # No history available
            metrics["final_fitness"] = metrics["best_fitness"]
            metrics["convergence_generation"] = 0
            metrics["diversity_maintenance"] = 0.0
            metrics["fitness_improvement"] = 0.0
            metrics["convergence_rate"] = 0.0

        return metrics

    def _find_convergence_generation(
        self, fitness_history, improvement_threshold=0.001
    ):
        """
        Find the generation where convergence occurs (minimal improvement)
        """
        if len(fitness_history) < 10:
            return len(fitness_history)

        # Look for a window where improvement is minimal
        window_size = 10
        for i in range(window_size, len(fitness_history)):
            window_start_fitness = fitness_history[i - window_size]
            current_fitness = fitness_history[i]

            if window_start_fitness != 0 and not np.isinf(window_start_fitness):
                improvement_rate = (
                    window_start_fitness - current_fitness
                ) / window_start_fitness
                if improvement_rate < improvement_threshold:
                    return i - window_size + 1

        return len(fitness_history)

    def _save_parameter_results(self, param_name, results):
        """
        Save results for a specific parameter
        """
        # Save as CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, f"detailed_{param_name}_results.csv")
        df.to_csv(csv_path, index=False)

        # Save as JSON for more detailed data
        json_path = os.path.join(self.output_dir, f"detailed_{param_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  Results saved: {csv_path}")

    def _generate_sensitivity_summary(self):
        """
        Generate summary analysis across all parameters
        """
        print("\n=== Sensitivity Analysis Summary ===")

        summary_data = []

        for param_name, results in self.results.items():
            # Calculate sensitivity metrics
            best_fitness_values = [r["best_fitness"] for r in results]
            convergence_generations = [r["convergence_generation"] for r in results]
            execution_times = [r["execution_time"] for r in results]
            diversity_values = [r["diversity_maintenance"] for r in results]

            # Calculate sensitivity (coefficient of variation)
            fitness_sensitivity = (
                np.std(best_fitness_values) / np.mean(best_fitness_values)
                if np.mean(best_fitness_values) != 0
                else 0
            )
            convergence_sensitivity = (
                np.std(convergence_generations) / np.mean(convergence_generations)
                if np.mean(convergence_generations) != 0
                else 0
            )
            time_sensitivity = (
                np.std(execution_times) / np.mean(execution_times)
                if np.mean(execution_times) != 0
                else 0
            )

            # Find best parameter value
            best_idx = np.argmin(best_fitness_values)
            best_value = results[best_idx]["value"]

            summary_data.append(
                {
                    "parameter": param_name,
                    "fitness_sensitivity": fitness_sensitivity,
                    "convergence_sensitivity": convergence_sensitivity,
                    "time_sensitivity": time_sensitivity,
                    "best_value": best_value,
                    "best_fitness": best_fitness_values[best_idx],
                    "best_convergence": convergence_generations[best_idx],
                    "avg_diversity": np.mean(diversity_values),
                }
            )

            print(f"{param_name}:")
            print(f"  Fitness Sensitivity: {fitness_sensitivity:.3f}")
            print(f"  Convergence Sensitivity: {convergence_sensitivity:.3f}")
            print(f"  Time Sensitivity: {time_sensitivity:.3f}")
            print(f"  Best Value: {best_value}")
            print(f"  Best Fitness: {best_fitness_values[best_idx]:.2f}")
            print(
                f"  Best Convergence: {convergence_generations[best_idx]} generations"
            )
            print()

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, "sensitivity_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"Summary saved to: {summary_path}")
        return summary_data

    def _save_results(self):
        """
        Save all results to files
        """
        # Save configuration
        config_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "parameter_ranges": self.param_ranges,
            "default_params": self.default_params,
            "num_batches": NUM_BATCHES,
        }

        config_path = os.path.join(self.output_dir, "analysis_config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Save detailed results
        detailed_path = os.path.join(self.output_dir, "detailed_results.json")
        with open(detailed_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"Configuration saved to: {config_path}")
        print(f"Detailed results saved to: {detailed_path}")

    def generate_report(self):
        """
        Generate a comprehensive markdown report
        """
        report_path = os.path.join(self.output_dir, "sensitivity_analysis_report.md")

        with open(report_path, "w") as f:
            f.write("# GA Sensitivity Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Analysis Configuration\n\n")
            f.write(f"- Number of batches: {NUM_BATCHES}\n")
            f.write(f"- Default parameters: {self.default_params}\n\n")

            f.write("## Parameter Ranges Analyzed\n\n")
            for param, values in self.param_ranges.items():
                f.write(f"- **{param}**: {values}\n")
            f.write("\n")

            if self.results:
                f.write("## Results Summary\n\n")
                f.write(
                    "| Parameter | Best Value | Best Fitness | Convergence Gen | Sensitivity |\n"
                )
                f.write(
                    "|-----------|------------|--------------|-----------------|-------------|\n"
                )

                for param_name, results in self.results.items():
                    best_fitness_values = [r["best_fitness"] for r in results]
                    convergence_generations = [
                        r["convergence_generation"] for r in results
                    ]

                    best_idx = np.argmin(best_fitness_values)
                    best_value = results[best_idx]["value"]
                    best_fitness = best_fitness_values[best_idx]
                    best_convergence = convergence_generations[best_idx]

                    # Calculate sensitivity
                    fitness_sensitivity = (
                        np.std(best_fitness_values) / np.mean(best_fitness_values)
                        if np.mean(best_fitness_values) != 0
                        else 0
                    )

                    f.write(
                        f"| {param_name} | {best_value} | {best_fitness:.2f} | {best_convergence} | {fitness_sensitivity:.3f} |\n"
                    )

                f.write("\n## Detailed Analysis\n\n")
                for param_name, results in self.results.items():
                    f.write(f"### {param_name}\n\n")
                    f.write(
                        "| Value | Best Fitness | Convergence Gen | Exec Time (s) | Diversity |\n"
                    )
                    f.write(
                        "|-------|--------------|-----------------|---------------|----------|\n"
                    )

                    for result in results:
                        f.write(
                            f"| {result['value']} | {result['best_fitness']:.2f} | {result['convergence_generation']} | {result['execution_time']:.2f} | {result['diversity_maintenance']:.3f} |\n"
                        )
                    f.write("\n")

            f.write("## Recommendations\n\n")
            if self.results:
                f.write("Based on the sensitivity analysis:\n\n")
                for param_name, results in self.results.items():
                    best_fitness_values = [r["best_fitness"] for r in results]
                    best_idx = np.argmin(best_fitness_values)
                    best_value = results[best_idx]["value"]
                    f.write(f"- **{param_name}**: Optimal value is {best_value}\n")

        print(f"Comprehensive report saved to: {report_path}")
