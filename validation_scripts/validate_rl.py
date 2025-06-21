# This file will be used for RL model validation.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

# Adjust path to import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.environment.aluminum_melting_env_7 import AluminumMeltingEnvironment
    from src.agents.agent2 import DQNAgent
    from src.training.run_trained_agent import run_episode_and_plot

    print("Successfully imported RL components")
except ImportError as e:
    print(f"Error importing RL components: {e}")
    sys.exit(1)


class RLModelValidator:
    """
    Comprehensive validation framework for RL models
    """

    def __init__(self, model_path, state_dim=6, action_dim=5):
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent = None
        self.validation_results = {}

    def load_model(self):
        """Load trained RL model"""
        try:
            self.agent = DQNAgent.load_checkpoint(
                self.model_path, state_dim=self.state_dim, action_dim=self.action_dim
            )
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def validate_temperature_accuracy(self, scenarios):
        """
        Validate temperature accuracy against baseline/real data

        Args:
            scenarios: List of (weight_kg, target_temp_c, real_data) tuples
        """
        print("\n=== Temperature Accuracy Validation ===")

        accuracy_results = []

        for scenario_idx, (weight_kg, target_temp_c, real_data) in enumerate(scenarios):
            print(f"\nScenario {scenario_idx + 1}: {weight_kg}kg, {target_temp_c}°C")

            # Create environment
            env = AluminumMeltingEnvironment(
                initial_weight_kg=weight_kg, target_temp_c=target_temp_c
            )

            # Run episode
            state = env.reset()
            done = False

            # Data collection
            sim_times = []
            sim_temps = []
            sim_powers = []
            sim_energies = []

            # Set agent to evaluation mode
            original_epsilon = self.agent.epsilon
            self.agent.epsilon = 0.0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = env.step(action)

                # Collect data
                sim_times.append(env.state["time"] / 60.0)  # Convert to minutes
                sim_temps.append(env.state["temperature"])
                sim_powers.append(env.state["power"])
                sim_energies.append(env.state["energy_consumption"])

                state = next_state

            # Restore epsilon
            self.agent.epsilon = original_epsilon

            # Calculate metrics
            final_temp = sim_temps[-1]
            final_energy = sim_energies[-1]
            final_time = sim_times[-1]

            # Temperature accuracy
            temp_error = abs(final_temp - target_temp_c)
            temp_accuracy = max(0, 100 - (temp_error / target_temp_c) * 100)

            # Store results
            result = {
                "scenario": f"{weight_kg}kg_{target_temp_c}C",
                "target_temp": target_temp_c,
                "achieved_temp": final_temp,
                "temp_error": temp_error,
                "temp_accuracy": temp_accuracy,
                "energy_consumption": final_energy,
                "melting_time": final_time,
                "power_profile": sim_powers,
                "temp_profile": sim_temps,
                "time_profile": sim_times,
            }

            accuracy_results.append(result)

            print(f"  Target Temperature: {target_temp_c}°C")
            print(f"  Achieved Temperature: {final_temp:.1f}°C")
            print(f"  Temperature Error: {temp_error:.1f}°C")
            print(f"  Temperature Accuracy: {temp_accuracy:.1f}%")
            print(f"  Energy Consumption: {final_energy:.1f} kWh")
            print(f"  Melting Time: {final_time:.1f} minutes")

            # Plot individual scenario
            self._plot_scenario_results(result, scenario_idx + 1)

        self.validation_results["temperature_accuracy"] = accuracy_results
        return accuracy_results

    def compare_with_baseline(self, baseline_data):
        """
        Compare RL performance with baseline (real operation data)

        Args:
            baseline_data: Dictionary with baseline performance data
        """
        print("\n=== Baseline Comparison ===")

        if "temperature_accuracy" not in self.validation_results:
            print("Please run temperature accuracy validation first")
            return None

        comparison_results = []

        for rl_result in self.validation_results["temperature_accuracy"]:
            scenario = rl_result["scenario"]

            # Find matching baseline data
            baseline_match = None
            for baseline in baseline_data:
                if baseline["scenario"] == scenario:
                    baseline_match = baseline
                    break

            if baseline_match:
                energy_improvement = (
                    (
                        baseline_match["energy_consumption"]
                        - rl_result["energy_consumption"]
                    )
                    / baseline_match["energy_consumption"]
                    * 100
                )

                time_improvement = (
                    (baseline_match["melting_time"] - rl_result["melting_time"])
                    / baseline_match["melting_time"]
                    * 100
                )

                comparison = {
                    "scenario": scenario,
                    "rl_energy": rl_result["energy_consumption"],
                    "baseline_energy": baseline_match["energy_consumption"],
                    "energy_improvement": energy_improvement,
                    "rl_time": rl_result["melting_time"],
                    "baseline_time": baseline_match["melting_time"],
                    "time_improvement": time_improvement,
                    "rl_temp_accuracy": rl_result["temp_accuracy"],
                    "baseline_temp_accuracy": baseline_match.get(
                        "temp_accuracy", "N/A"
                    ),
                }

                comparison_results.append(comparison)

                print(f"\nScenario: {scenario}")
                print(
                    f"  Energy: RL={rl_result['energy_consumption']:.1f} kWh, "
                    f"Baseline={baseline_match['energy_consumption']:.1f} kWh "
                    f"(Improvement: {energy_improvement:.1f}%)"
                )
                print(
                    f"  Time: RL={rl_result['melting_time']:.1f} min, "
                    f"Baseline={baseline_match['melting_time']:.1f} min "
                    f"(Improvement: {time_improvement:.1f}%)"
                )
            else:
                print(f"No baseline data found for scenario: {scenario}")

        self.validation_results["baseline_comparison"] = comparison_results
        self._plot_baseline_comparison(comparison_results)

        return comparison_results

    def performance_metrics_analysis(self):
        """Calculate comprehensive performance metrics"""
        print("\n=== Performance Metrics Analysis ===")

        if "temperature_accuracy" not in self.validation_results:
            print("Please run temperature accuracy validation first")
            return None

        results = self.validation_results["temperature_accuracy"]

        # Calculate aggregate metrics
        metrics = {
            "avg_temp_accuracy": np.mean([r["temp_accuracy"] for r in results]),
            "avg_energy_consumption": np.mean(
                [r["energy_consumption"] for r in results]
            ),
            "avg_melting_time": np.mean([r["melting_time"] for r in results]),
            "temp_accuracy_std": np.std([r["temp_accuracy"] for r in results]),
            "energy_consumption_std": np.std(
                [r["energy_consumption"] for r in results]
            ),
            "melting_time_std": np.std([r["melting_time"] for r in results]),
            "scenarios_count": len(results),
        }

        print(
            f"Average Temperature Accuracy: {metrics['avg_temp_accuracy']:.2f}% ± {metrics['temp_accuracy_std']:.2f}%"
        )
        print(
            f"Average Energy Consumption: {metrics['avg_energy_consumption']:.2f} ± {metrics['energy_consumption_std']:.2f} kWh"
        )
        print(
            f"Average Melting Time: {metrics['avg_melting_time']:.2f} ± {metrics['melting_time_std']:.2f} minutes"
        )

        self.validation_results["performance_metrics"] = metrics
        return metrics

    def _plot_scenario_results(self, result, scenario_num):
        """Plot detailed results for a single scenario"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        times = result["time_profile"]
        temps = result["temp_profile"]
        powers = result["power_profile"]

        # Temperature plot
        axes[0].plot(times, temps, "b-", linewidth=2, label="Achieved Temperature")
        axes[0].axhline(
            y=result["target_temp"],
            color="r",
            linestyle="--",
            linewidth=2,
            label="Target Temperature",
        )
        axes[0].set_ylabel("Temperature (°C)")
        axes[0].set_title(
            f'Scenario {scenario_num}: {result["scenario"]} - Temperature Profile'
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Power plot
        axes[1].plot(times, powers, "g-", linewidth=2, label="Power Consumption")
        axes[1].set_ylabel("Power (kW)")
        axes[1].set_title("Power Profile")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Energy consumption (cumulative)
        energy_cumulative = np.cumsum(
            [p / 60 for p in powers]
        )  # Approximate cumulative energy
        axes[2].plot(
            times, energy_cumulative, "m-", linewidth=2, label="Cumulative Energy"
        )
        axes[2].set_ylabel("Energy (kWh)")
        axes[2].set_xlabel("Time (minutes)")
        axes[2].set_title("Cumulative Energy Consumption")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"validation_results_scenario_{scenario_num}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_baseline_comparison(self, comparison_results):
        """Plot comparison with baseline results"""
        if not comparison_results:
            return

        scenarios = [r["scenario"] for r in comparison_results]
        energy_improvements = [r["energy_improvement"] for r in comparison_results]
        time_improvements = [r["time_improvement"] for r in comparison_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Energy improvement
        colors = ["green" if x > 0 else "red" for x in energy_improvements]
        bars1 = ax1.bar(scenarios, energy_improvements, color=colors, alpha=0.7)
        ax1.set_ylabel("Energy Improvement (%)")
        ax1.set_title("Energy Efficiency Improvement vs Baseline")
        ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, energy_improvements):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (1 if height > 0 else -3),
                f"{value:.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

        # Time improvement
        colors = ["green" if x > 0 else "red" for x in time_improvements]
        bars2 = ax2.bar(scenarios, time_improvements, color=colors, alpha=0.7)
        ax2.set_ylabel("Time Improvement (%)")
        ax2.set_title("Time Efficiency Improvement vs Baseline")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars2, time_improvements):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (1 if height > 0 else -3),
                f"{value:.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

        plt.tight_layout()
        plt.savefig("baseline_comparison.png", dpi=300, bbox_inches="tight")
        plt.show()

    def generate_validation_report(self, output_file="rl_validation_report.md"):
        """Generate comprehensive validation report"""
        report = []
        report.append("# RL Model Validation Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_path}")
        report.append("")

        # Performance metrics
        if "performance_metrics" in self.validation_results:
            metrics = self.validation_results["performance_metrics"]
            report.append("## Performance Metrics Summary")
            report.append(
                f"- Average Temperature Accuracy: {metrics['avg_temp_accuracy']:.2f}% ± {metrics['temp_accuracy_std']:.2f}%"
            )
            report.append(
                f"- Average Energy Consumption: {metrics['avg_energy_consumption']:.2f} ± {metrics['energy_consumption_std']:.2f} kWh"
            )
            report.append(
                f"- Average Melting Time: {metrics['avg_melting_time']:.2f} ± {metrics['melting_time_std']:.2f} minutes"
            )
            report.append(f"- Number of Scenarios Tested: {metrics['scenarios_count']}")
            report.append("")

        # Individual scenario results
        if "temperature_accuracy" in self.validation_results:
            report.append("## Individual Scenario Results")
            report.append(
                "| Scenario | Target Temp (°C) | Achieved Temp (°C) | Error (°C) | Accuracy (%) | Energy (kWh) | Time (min) |"
            )
            report.append(
                "|----------|------------------|-------------------|------------|--------------|--------------|------------|"
            )

            for result in self.validation_results["temperature_accuracy"]:
                report.append(
                    f"| {result['scenario']} | {result['target_temp']} | {result['achieved_temp']:.1f} | {result['temp_error']:.1f} | {result['temp_accuracy']:.1f} | {result['energy_consumption']:.1f} | {result['melting_time']:.1f} |"
                )
            report.append("")

        # Baseline comparison
        if "baseline_comparison" in self.validation_results:
            report.append("## Baseline Comparison")
            report.append(
                "| Scenario | RL Energy (kWh) | Baseline Energy (kWh) | Energy Improvement (%) | RL Time (min) | Baseline Time (min) | Time Improvement (%) |"
            )
            report.append(
                "|----------|-----------------|----------------------|------------------------|---------------|---------------------|----------------------|"
            )

            for comp in self.validation_results["baseline_comparison"]:
                report.append(
                    f"| {comp['scenario']} | {comp['rl_energy']:.1f} | {comp['baseline_energy']:.1f} | {comp['energy_improvement']:.1f} | {comp['rl_time']:.1f} | {comp['baseline_time']:.1f} | {comp['time_improvement']:.1f} |"
                )
            report.append("")

        # Write report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"Validation report saved to: {output_file}")


def main():
    """Main validation execution"""
    print("Starting RL Model Validation...")

    # Configuration
    MODEL_PATH = "models/dqn_final_model_10.pth"

    # Initialize validator
    validator = RLModelValidator(MODEL_PATH)

    if not validator.load_model():
        print("Failed to load model. Exiting.")
        return

    # Define test scenarios
    test_scenarios = [
        (500, 900, None),  # 500kg, 900°C
        (400, 900, None),  # 400kg, 900°C
        (500, 850, None),  # 500kg, 850°C
    ]

    # Run temperature accuracy validation
    validator.validate_temperature_accuracy(test_scenarios)

    # Define baseline data for comparison (example data - replace with real data)
    baseline_data = [
        {
            "scenario": "500kg_900C",
            "energy_consumption": 565.3,
            "melting_time": 90,
            "temp_accuracy": 95.0,
        },
        {
            "scenario": "400kg_900C",
            "energy_consumption": 450.0,  # Estimated
            "melting_time": 75,  # Estimated
            "temp_accuracy": 95.0,
        },
        {
            "scenario": "500kg_850C",
            "energy_consumption": 420.0,  # Estimated
            "melting_time": 70,  # Estimated
            "temp_accuracy": 95.0,
        },
    ]

    # Run baseline comparison
    validator.compare_with_baseline(baseline_data)

    # Calculate performance metrics
    validator.performance_metrics_analysis()

    # Generate report
    validator.generate_validation_report()

    print("\nRL Model Validation Complete!")
    print("Check generated plots and validation report for detailed results.")


if __name__ == "__main__":
    main()
