#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for Genetic Algorithm (GA)
This script analyzes the sensitivity of GA hyperparameters on optimization performance
Focus: Crossover Rate and Mutation Rate effects on convergence speed and solution quality
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import json
from datetime import datetime
import itertools
from multiprocessing import Pool
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class GASensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for GA hyperparameters
    Focus on Crossover Rate and Mutation Rate
    """

    def __init__(self, output_dir="results/sensitivity_analysis_ga"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define parameter ranges for sensitivity analysis
        self.param_ranges = {
            "crossover_rate": [0.6, 0.7, 0.8, 0.9],  # อัตราการผสมข้าม
            "mutation_rate": [0.01, 0.05, 0.1, 0.2],  # อัตราการกลายพันธุ์
            "population_size": [50, 100, 200],  # ขนาดประชากร
            "num_generations": [100, 200, 300],  # จำนวนรุ่น
        }

        # Default parameters (baseline)
        self.default_params = {
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "population_size": 100,
            "num_generations": 200,
            "elite_size": 20,
            "tournament_size": 5,
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
            metrics = self._run_ga_optimization(config)

            result = {
                "parameter": param_name,
                "value": param_value,
                "best_fitness": metrics["best_fitness"],
                "convergence_generation": metrics["convergence_generation"],
                "optimization_stability": metrics["optimization_stability"],
                "final_solution_quality": metrics["final_solution_quality"],
                "execution_time": metrics["execution_time"],
                "diversity_maintenance": metrics["diversity_maintenance"],
            }

            results.append(result)
            print(f"  Best fitness: {metrics['best_fitness']:.4f}")
            print(f"  Convergence generation: {metrics['convergence_generation']}")
            print(f"  Solution quality: {metrics['final_solution_quality']:.4f}")

        # Store results
        self.results[param_name] = results

        # Create visualizations
        self._plot_parameter_sensitivity(param_name, results)

        return results

    def run_complete_sensitivity_analysis(self):
        """
        Run sensitivity analysis for all parameters
        """
        print("Starting Complete Sensitivity Analysis for GA Parameters...")

        # Analyze each parameter
        for param_name in self.param_ranges.keys():
            self.run_single_parameter_analysis(param_name)

        # Generate summary analysis
        self._generate_sensitivity_summary()

        # Save results
        self._save_results()

        print(f"\nSensitivity analysis complete! Results saved to {self.output_dir}")

    def _run_ga_optimization(self, config):
        """
        Run GA optimization with specific configuration and return metrics

        Args:
            config: Dictionary with GA parameters

        Returns:
            Dictionary with optimization metrics
        """
        print(f"Starting GA optimization with config: {config}")
        start_time = datetime.now()

        # Import GA adapter for pymoo
        from ga_sensitivity_adapter_pymoo import GASensitivityAdapterPymoo

        # Create GA instance with config parameters
        ga = GASensitivityAdapterPymoo(
            population_size=config["population_size"],
            num_generations=config["num_generations"],
            crossover_rate=config["crossover_rate"],
            mutation_rate=config["mutation_rate"],
            elite_size=config.get("elite_size", 20),
            tournament_size=config.get("tournament_size", 5),
        )

        # Run optimization
        print(f"Running GA for {config['num_generations']} generations...")

        try:
            # Run complete optimization
            results = ga.run_complete_optimization()

            fitness_history = results["fitness_history"]
            diversity_history = results["diversity_history"]

            # Check for convergence
            convergence_generation = config[
                "num_generations"
            ]  # Default to max if no convergence
            convergence_threshold = 1000.0  # Cost improvement threshold (adjusted for scheduling cost scale)
            convergence_window = 20  # Generations to check for convergence

            if len(fitness_history) >= convergence_window:
                for generation in range(convergence_window, len(fitness_history)):
                    recent_fitness = fitness_history[
                        generation - convergence_window : generation
                    ]
                    fitness_improvement = max(recent_fitness) - min(recent_fitness)
                    if fitness_improvement < convergence_threshold:
                        convergence_generation = generation + 1
                        print(f"Converged at generation {convergence_generation}")
                        break

            # Progress update
            for i in range(0, len(fitness_history), max(1, len(fitness_history) // 5)):
                print(f"Generation {i+1}: Best Fitness = {fitness_history[i]:.4f}")

        except Exception as e:
            print(f"Error during GA optimization: {e}")
            # Return default/failed metrics
            fitness_history = [float("inf")]
            diversity_history = [0.0]
            convergence_generation = config["num_generations"]

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Calculate final metrics
        if fitness_history:
            final_best = ga.get_best_individual()
            best_fitness = final_best["fitness"]
        else:
            best_fitness = float("inf")

        # Calculate optimization stability (coefficient of variation of last 20% of generations)
        stable_period = max(10, len(fitness_history) // 5)
        if len(fitness_history) >= stable_period:
            stable_fitness = fitness_history[-stable_period:]
        else:
            stable_fitness = fitness_history

        optimization_stability = np.std(stable_fitness) / (
            np.mean(stable_fitness) + 1e-8
        )

        # Final solution quality (same as best fitness for GA)
        final_solution_quality = best_fitness

        # Diversity maintenance (average diversity in last 20% of generations)
        if len(diversity_history) >= stable_period:
            diversity_maintenance = np.mean(diversity_history[-stable_period:])
        else:
            diversity_maintenance = (
                np.mean(diversity_history) if diversity_history else 0.0
            )

        metrics = {
            "best_fitness": float(best_fitness),
            "convergence_generation": int(convergence_generation),
            "optimization_stability": float(optimization_stability),
            "final_solution_quality": float(final_solution_quality),
            "execution_time": float(execution_time),
            "diversity_maintenance": float(diversity_maintenance),
            "fitness_history": fitness_history,
            "diversity_history": diversity_history,
        }

        print(f"GA optimization completed in {execution_time:.1f} seconds")
        print(
            f"Final metrics: Fitness={best_fitness:.4f}, Convergence={convergence_generation}, Quality={final_solution_quality:.4f}"
        )

        return metrics

    def _plot_parameter_sensitivity(self, param_name, results):
        """
        Create sensitivity plots for a specific parameter
        """
        values = [r["value"] for r in results]
        best_fitness = [r["best_fitness"] for r in results]
        convergence_generation = [r["convergence_generation"] for r in results]
        optimization_stability = [r["optimization_stability"] for r in results]
        final_solution_quality = [r["final_solution_quality"] for r in results]
        execution_time = [r["execution_time"] for r in results]
        diversity_maintenance = [r["diversity_maintenance"] for r in results]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"GA Parameter Sensitivity Analysis: {param_name}", fontsize=16)

        # Best Fitness
        axes[0, 0].plot(
            values, best_fitness, "o-", linewidth=2, markersize=8, color="blue"
        )
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel("Best Fitness")
        axes[0, 0].set_title("Best Fitness vs Parameter Value")
        axes[0, 0].grid(True, alpha=0.3)

        # Convergence Generation
        axes[0, 1].plot(
            values,
            convergence_generation,
            "s-",
            linewidth=2,
            markersize=8,
            color="orange",
        )
        axes[0, 1].set_xlabel(param_name)
        axes[0, 1].set_ylabel("Convergence Generation")
        axes[0, 1].set_title("Convergence Speed vs Parameter Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Optimization Stability
        axes[0, 2].plot(
            values,
            optimization_stability,
            "^-",
            linewidth=2,
            markersize=8,
            color="green",
        )
        axes[0, 2].set_xlabel(param_name)
        axes[0, 2].set_ylabel("Optimization Stability")
        axes[0, 2].set_title("Stability vs Parameter Value")
        axes[0, 2].grid(True, alpha=0.3)

        # Final Solution Quality
        axes[1, 0].plot(
            values, final_solution_quality, "d-", linewidth=2, markersize=8, color="red"
        )
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel("Solution Quality")
        axes[1, 0].set_title("Solution Quality vs Parameter Value")
        axes[1, 0].grid(True, alpha=0.3)

        # Execution Time
        axes[1, 1].plot(
            values, execution_time, "v-", linewidth=2, markersize=8, color="purple"
        )
        axes[1, 1].set_xlabel(param_name)
        axes[1, 1].set_ylabel("Execution Time (seconds)")
        axes[1, 1].set_title("Execution Time vs Parameter Value")
        axes[1, 1].grid(True, alpha=0.3)

        # Diversity Maintenance
        axes[1, 2].plot(
            values,
            diversity_maintenance,
            "h-",
            linewidth=2,
            markersize=8,
            color="brown",
        )
        axes[1, 2].set_xlabel(param_name)
        axes[1, 2].set_ylabel("Diversity Maintenance")
        axes[1, 2].set_title("Diversity vs Parameter Value")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/sensitivity_{param_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _generate_sensitivity_summary(self):
        """
        Generate summary analysis across all parameters
        """
        print("\n=== GA Sensitivity Analysis Summary ===")

        summary_data = []

        for param_name, results in self.results.items():
            # Calculate sensitivity metrics
            best_fitness = [r["best_fitness"] for r in results]
            convergence_generation = [r["convergence_generation"] for r in results]
            optimization_stability = [r["optimization_stability"] for r in results]

            fitness_sensitivity = (max(best_fitness) - min(best_fitness)) / np.mean(
                best_fitness
            )
            convergence_sensitivity = (
                max(convergence_generation) - min(convergence_generation)
            ) / np.mean(convergence_generation)
            stability_sensitivity = (
                max(optimization_stability) - min(optimization_stability)
            ) / np.mean(optimization_stability)

            # Find best parameter value
            best_idx = np.argmax(best_fitness)
            best_value = results[best_idx]["value"]

            summary_data.append(
                {
                    "parameter": param_name,
                    "fitness_sensitivity": fitness_sensitivity,
                    "convergence_sensitivity": convergence_sensitivity,
                    "stability_sensitivity": stability_sensitivity,
                    "best_value": best_value,
                    "best_fitness": best_fitness[best_idx],
                }
            )

            print(f"{param_name}:")
            print(f"  Fitness Sensitivity: {fitness_sensitivity:.3f}")
            print(f"  Convergence Sensitivity: {convergence_sensitivity:.3f}")
            print(f"  Stability Sensitivity: {stability_sensitivity:.3f}")
            print(f"  Best Value: {best_value}")
            print(f"  Best Fitness: {best_fitness[best_idx]:.4f}")
            print()

        # Create summary visualization
        self._plot_sensitivity_summary(summary_data)

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_dir}/sensitivity_summary.csv", index=False)

        return summary_data

    def _plot_sensitivity_summary(self, summary_data):
        """
        Create summary visualization of parameter sensitivities
        """
        parameters = [s["parameter"] for s in summary_data]
        fitness_sensitivities = [s["fitness_sensitivity"] for s in summary_data]
        convergence_sensitivities = [s["convergence_sensitivity"] for s in summary_data]
        stability_sensitivities = [s["stability_sensitivity"] for s in summary_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sensitivity comparison
        x = np.arange(len(parameters))
        width = 0.25

        ax1.bar(
            x - width,
            fitness_sensitivities,
            width,
            label="Fitness Sensitivity",
            alpha=0.8,
        )
        ax1.bar(
            x,
            convergence_sensitivities,
            width,
            label="Convergence Sensitivity",
            alpha=0.8,
        )
        ax1.bar(
            x + width,
            stability_sensitivities,
            width,
            label="Stability Sensitivity",
            alpha=0.8,
        )

        ax1.set_xlabel("Parameters")
        ax1.set_ylabel("Sensitivity Score")
        ax1.set_title("GA Parameter Sensitivity Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(parameters, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Best values
        best_values = [s["best_value"] for s in summary_data]
        best_fitness = [s["best_fitness"] for s in summary_data]

        ax2.bar(parameters, best_fitness, alpha=0.7, color="skyblue")
        ax2.set_xlabel("Parameters")
        ax2.set_ylabel("Best Fitness")
        ax2.set_title("Best Fitness by Optimal Parameter Values")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (param, value, fitness) in enumerate(
            zip(parameters, best_values, best_fitness)
        ):
            ax2.text(i, fitness + 0.01, f"{value}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/sensitivity_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def _save_results(self):
        """
        Save detailed results to JSON files
        """
        # Save detailed results
        detailed_results = {}
        for param_name, results in self.results.items():
            detailed_results[param_name] = results

        with open(f"{self.output_dir}/detailed_results.json", "w") as f:
            json.dump(detailed_results, f, indent=2, default=str)

        # Save analysis configuration
        config = {
            "param_ranges": self.param_ranges,
            "default_params": self.default_params,
            "analysis_date": datetime.now().isoformat(),
            "output_directory": self.output_dir,
        }

        with open(f"{self.output_dir}/analysis_config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Results saved to {self.output_dir}/")

    def generate_report(self):
        """
        Generate comprehensive analysis report
        """
        report_path = f"{self.output_dir}/sensitivity_analysis_report.md"

        with open(report_path, "w") as f:
            f.write("# GA Parameter Sensitivity Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Analysis Overview\n")
            f.write(
                "This report analyzes the sensitivity of Genetic Algorithm hyperparameters "
            )
            f.write("on optimization performance, focusing on:\n")
            f.write(
                "- **Crossover Rate**: Effect on solution exploration and exploitation\n"
            )
            f.write(
                "- **Mutation Rate**: Effect on diversity maintenance and convergence\n"
            )
            f.write("- **Population Size**: Effect on search space coverage\n")
            f.write("- **Number of Generations**: Effect on convergence time\n\n")

            f.write("## Key Findings\n")
            for param_name, results in self.results.items():
                best_fitness = [r["best_fitness"] for r in results]
                best_idx = np.argmax(best_fitness)
                best_result = results[best_idx]

                f.write(f"### {param_name}\n")
                f.write(f"- **Optimal Value**: {best_result['value']}\n")
                f.write(f"- **Best Fitness**: {best_result['best_fitness']:.4f}\n")
                f.write(
                    f"- **Convergence Generation**: {best_result['convergence_generation']}\n"
                )
                f.write(
                    f"- **Solution Quality**: {best_result['final_solution_quality']:.4f}\n\n"
                )

            f.write("## Recommendations\n")
            f.write("Based on the sensitivity analysis results:\n")
            f.write(
                "1. Focus tuning efforts on parameters with highest sensitivity scores\n"
            )
            f.write("2. Use optimal values identified for each parameter\n")
            f.write("3. Consider parameter interactions for fine-tuning\n")
            f.write("4. Monitor convergence behavior during optimization\n\n")

        print(f"Report generated: {report_path}")


def main():
    """
    Main function to run GA sensitivity analysis
    """
    print("=== GA Parameter Sensitivity Analysis ===")
    print(f"Started at: {datetime.now()}")

    # Create analyzer
    analyzer = GASensitivityAnalyzer()

    # Choose analysis mode
    print("\nChoose analysis mode:")
    print("1. Crossover Rate analysis")
    print("2. Mutation Rate analysis")
    print("3. Complete analysis (all parameters)")
    print("4. Quick test (crossover_rate only)")

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        print("\nAnalyzing Crossover Rate sensitivity...")
        analyzer.run_single_parameter_analysis("crossover_rate")
    elif choice == "2":
        print("\nAnalyzing Mutation Rate sensitivity...")
        analyzer.run_single_parameter_analysis("mutation_rate")
    elif choice == "3":
        print("\nRunning complete sensitivity analysis...")
        print("This will take some time!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == "y":
            analyzer.run_complete_sensitivity_analysis()
        else:
            print("Cancelled.")
            return
    elif choice == "4":
        print("\nRunning quick test with crossover_rate...")
        analyzer.run_single_parameter_analysis("crossover_rate")
    else:
        print("Invalid choice!")
        return

    # Generate report
    analyzer.generate_report()

    print(f"\nCompleted at: {datetime.now()}")
    print(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
