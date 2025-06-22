"""
Parameter Sensitivity Analysis for RL (DQN) Model
This script analyzes the sensitivity of DQN hyperparameters on training performance
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
import subprocess

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class RLSensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for RL hyperparameters
    """

    def __init__(self, output_dir="results/sensitivity_analysis_rl"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Define parameter ranges for sensitivity analysis
        self.param_ranges = {
            "learning_rate": [0.0001, 0.0005, 0.001],
            "gamma": [0.95, 0.99, 0.995],
            "epsilon_decay": [0.995, 0.999, 0.9995],
            "batch_size": [32, 64, 128],
            "hidden_size": [128, 256, 512],
            "target_update_freq": [100, 500, 1000],
        }

        # Default parameters (baseline)
        self.default_params = {
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.999,
            "batch_size": 64,
            "hidden_size": 256,
            "target_update_freq": 500,
            "memory_size": 10000,
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

            # Run training with this configuration
            metrics = self._run_training_with_config(config)

            result = {
                "parameter": param_name,
                "value": param_value,
                "final_reward": metrics["final_reward"],
                "convergence_episodes": metrics["convergence_episodes"],
                "training_stability": metrics["training_stability"],
                "final_accuracy": metrics["final_accuracy"],
                "training_time": metrics["training_time"],
            }

            results.append(result)
            print(f"  Final reward: {metrics['final_reward']:.2f}")
            print(f"  Convergence episodes: {metrics['convergence_episodes']}")
            print(f"  Training stability: {metrics['training_stability']:.3f}")

        # Store results
        self.results[param_name] = results

        # Create visualizations
        self._plot_parameter_sensitivity(param_name, results)

        return results

    def run_complete_sensitivity_analysis(self):
        """
        Run sensitivity analysis for all parameters
        """
        print("Starting Complete Sensitivity Analysis for RL Parameters...")

        # Analyze each parameter
        for param_name in self.param_ranges.keys():
            self.run_single_parameter_analysis(param_name)

        # Generate summary analysis
        self._generate_sensitivity_summary()

        # Save results
        self._save_results()

        print(f"\nSensitivity analysis complete! Results saved to {self.output_dir}")

    def _run_training_with_config(self, config):
        """
        Run training with specific configuration and return metrics

        Args:
            config: Dictionary with training parameters

        Returns:
            Dictionary with training metrics
        """
        # This is a placeholder for actual training implementation
        # In practice, you would:
        # 1. Create environment with config
        # 2. Initialize agent with config
        # 3. Run training
        # 4. Extract metrics

        # For demonstration, we'll use synthetic data based on realistic parameter effects
        metrics = self._simulate_training_metrics(config)

        return metrics

    def _simulate_training_metrics(self, config):
        """
        Run actual training with given configuration and return real metrics
        """
        import sys
        import os
        import torch
        import numpy as np
        from datetime import datetime

        # Add project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.environment.aluminum_melting_env_8 import AluminumMeltingEnvironment
        from src.agents.agent2 import DQNAgent

        print(f"Starting actual training with config: {config}")
        start_time = datetime.now()

        # Create environment
        env = AluminumMeltingEnvironment(initial_weight_kg=350, target_temp_c=900)

        # Create agent with config parameters
        agent = DQNAgent(
            state_dim=7,
            action_dim=5,
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            epsilon=config.get("epsilon", 1.0),
            epsilon_min=config.get("epsilon_min", 0.01),
            epsilon_decay=config.get("epsilon_decay", 0.995),
            memory_size=config.get("memory_size", 10000),
            batch_size=config.get("batch_size", 32),
            target_update_freq=config.get("target_update_freq", 100),
            hidden_size=config.get("hidden_size", 512),
        )

        # Training parameters
        max_episodes = 300  # Reduced for sensitivity analysis (faster)
        episode_rewards = []
        episode_lengths = []
        convergence_threshold = 0.6  # Reward threshold for convergence
        convergence_window = 50  # Episodes to check for convergence

        print(f"Training for max {max_episodes} episodes...")

        for episode in range(max_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False

            while not done and steps < 120:  # Max 120 minutes
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)

                # Store experience
                agent.store_experience(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                state = next_state
                total_reward = reward if done else 0  # Only count final reward
                steps += 1

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            # Update target network
            if episode % agent.target_update_freq == 0:
                agent.update_target_network()

            # Check convergence
            if episode >= convergence_window:
                recent_rewards = episode_rewards[-convergence_window:]
                avg_reward = np.mean(recent_rewards)
                if avg_reward >= convergence_threshold:
                    print(
                        f"Converged at episode {episode + 1} with avg reward {avg_reward:.3f}"
                    )
                    break

            # Progress update
            if (episode + 1) % 50 == 0:
                avg_reward = (
                    np.mean(episode_rewards[-50:])
                    if len(episode_rewards) >= 50
                    else np.mean(episode_rewards)
                )
                print(
                    f"Episode {episode + 1}: Avg Reward = {avg_reward:.3f}, Epsilon = {agent.epsilon:.3f}"
                )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds() / 60.0  # minutes

        # Calculate metrics
        final_reward = (
            np.mean(episode_rewards[-20:])
            if len(episode_rewards) >= 20
            else np.mean(episode_rewards)
        )
        convergence_episodes = len(episode_rewards)

        # Calculate training stability (coefficient of variation of last 50 episodes)
        if len(episode_rewards) >= 50:
            last_rewards = episode_rewards[-50:]
        else:
            last_rewards = episode_rewards

        training_stability = np.std(last_rewards) / (np.mean(last_rewards) + 1e-8)

        # Test final accuracy (run 10 test episodes)
        print("Testing final model...")
        test_rewards = []
        agent.epsilon = 0.0  # No exploration for testing

        for _ in range(10):
            state = env.reset()
            done = False
            steps = 0

            while not done and steps < 120:
                action = agent.select_action(state)
                state, reward, done = env.step(action)
                steps += 1

            test_rewards.append(reward if done else 0)

        final_accuracy = np.mean(test_rewards)

        metrics = {
            "final_reward": float(final_reward),
            "convergence_episodes": int(convergence_episodes),
            "training_stability": float(training_stability),
            "final_accuracy": float(final_accuracy),
            "training_time": float(training_time),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "test_rewards": test_rewards,
        }

        print(f"Training completed in {training_time:.1f} minutes")
        print(
            f"Final metrics: Reward={final_reward:.3f}, Accuracy={final_accuracy:.3f}, Stability={training_stability:.3f}"
        )

        return metrics

    def _plot_parameter_sensitivity(self, param_name, results):
        """
        Create sensitivity plots for a specific parameter
        """
        values = [r["value"] for r in results]
        final_rewards = [r["final_reward"] for r in results]
        convergence_episodes = [r["convergence_episodes"] for r in results]
        training_stability = [r["training_stability"] for r in results]
        final_accuracy = [r["final_accuracy"] for r in results]
        training_time = [r["training_time"] for r in results]

        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Parameter Sensitivity Analysis: {param_name}", fontsize=16)

        # Final Reward
        axes[0, 0].plot(values, final_rewards, "o-", linewidth=2, markersize=8)
        axes[0, 0].set_xlabel(param_name)
        axes[0, 0].set_ylabel("Final Reward")
        axes[0, 0].set_title("Final Reward vs Parameter Value")
        axes[0, 0].grid(True, alpha=0.3)

        # Convergence Episodes
        axes[0, 1].plot(
            values,
            convergence_episodes,
            "s-",
            linewidth=2,
            markersize=8,
            color="orange",
        )
        axes[0, 1].set_xlabel(param_name)
        axes[0, 1].set_ylabel("Episodes to Convergence")
        axes[0, 1].set_title("Convergence Speed vs Parameter Value")
        axes[0, 1].grid(True, alpha=0.3)

        # Training Stability
        axes[0, 2].plot(
            values, training_stability, "^-", linewidth=2, markersize=8, color="green"
        )
        axes[0, 2].set_xlabel(param_name)
        axes[0, 2].set_ylabel("Training Stability (Reward Variance)")
        axes[0, 2].set_title("Training Stability vs Parameter Value")
        axes[0, 2].grid(True, alpha=0.3)

        # Final Accuracy
        axes[1, 0].plot(
            values, final_accuracy, "d-", linewidth=2, markersize=8, color="red"
        )
        axes[1, 0].set_xlabel(param_name)
        axes[1, 0].set_ylabel("Final Accuracy")
        axes[1, 0].set_title("Final Accuracy vs Parameter Value")
        axes[1, 0].grid(True, alpha=0.3)

        # Training Time
        axes[1, 1].plot(
            values, training_time, "v-", linewidth=2, markersize=8, color="purple"
        )
        axes[1, 1].set_xlabel(param_name)
        axes[1, 1].set_ylabel("Training Time (minutes)")
        axes[1, 1].set_title("Training Time vs Parameter Value")
        axes[1, 1].grid(True, alpha=0.3)

        # Summary radar chart
        self._create_radar_chart(axes[1, 2], param_name, results)

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/sensitivity_{param_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _create_radar_chart(self, ax, param_name, results):
        """
        Create radar chart for parameter comparison
        """
        # Normalize metrics for radar chart
        metrics_names = [
            "Final Reward",
            "Convergence Speed",
            "Stability",
            "Accuracy",
            "Training Time",
        ]

        # Extract and normalize values
        values = [r["value"] for r in results]
        final_rewards = [r["final_reward"] for r in results]
        convergence_episodes = [
            1000 / r["convergence_episodes"] * 100 for r in results
        ]  # Invert (faster is better)
        training_stability = [
            (1 - r["training_stability"]) * 100 for r in results
        ]  # Invert (lower variance is better)
        final_accuracy = [r["final_accuracy"] * 100 for r in results]
        training_time = [
            100 / r["training_time"] * 60 for r in results
        ]  # Invert (faster is better)

        # Normalize to 0-100 scale
        def normalize(data):
            min_val, max_val = min(data), max(data)
            if max_val == min_val:
                return [50] * len(data)
            return [(x - min_val) / (max_val - min_val) * 100 for x in data]

        final_rewards_norm = normalize(final_rewards)
        convergence_episodes_norm = normalize(convergence_episodes)
        training_stability_norm = normalize(training_stability)
        final_accuracy_norm = normalize(final_accuracy)
        training_time_norm = normalize(training_time)

        # Create radar chart for best parameter value
        best_idx = np.argmax(final_rewards)
        best_values = [
            final_rewards_norm[best_idx],
            convergence_episodes_norm[best_idx],
            training_stability_norm[best_idx],
            final_accuracy_norm[best_idx],
            training_time_norm[best_idx],
        ]

        # Radar chart setup
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        best_values += best_values[:1]  # Complete the circle
        angles += angles[:1]

        ax.plot(angles, best_values, "o-", linewidth=2, color="blue", alpha=0.7)
        ax.fill(angles, best_values, alpha=0.25, color="blue")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(
            f"Best {param_name} Performance Profile\n(Value: {values[best_idx]})"
        )
        ax.grid(True)

    def _generate_sensitivity_summary(self):
        """
        Generate summary analysis across all parameters
        """
        print("\n=== Sensitivity Analysis Summary ===")

        summary_data = []

        for param_name, results in self.results.items():
            # Calculate sensitivity metrics
            final_rewards = [r["final_reward"] for r in results]
            convergence_episodes = [r["convergence_episodes"] for r in results]
            training_stability = [r["training_stability"] for r in results]

            reward_sensitivity = (max(final_rewards) - min(final_rewards)) / np.mean(
                final_rewards
            )
            convergence_sensitivity = (
                max(convergence_episodes) - min(convergence_episodes)
            ) / np.mean(convergence_episodes)
            stability_sensitivity = (
                max(training_stability) - min(training_stability)
            ) / np.mean(training_stability)

            # Find best parameter value
            best_idx = np.argmax(final_rewards)
            best_value = results[best_idx]["value"]

            summary_data.append(
                {
                    "parameter": param_name,
                    "reward_sensitivity": reward_sensitivity,
                    "convergence_sensitivity": convergence_sensitivity,
                    "stability_sensitivity": stability_sensitivity,
                    "best_value": best_value,
                    "best_reward": final_rewards[best_idx],
                }
            )

            print(f"{param_name}:")
            print(f"  Reward Sensitivity: {reward_sensitivity:.3f}")
            print(f"  Convergence Sensitivity: {convergence_sensitivity:.3f}")
            print(f"  Stability Sensitivity: {stability_sensitivity:.3f}")
            print(f"  Best Value: {best_value}")
            print(f"  Best Reward: {final_rewards[best_idx]:.2f}")
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
        reward_sensitivities = [s["reward_sensitivity"] for s in summary_data]
        convergence_sensitivities = [s["convergence_sensitivity"] for s in summary_data]
        stability_sensitivities = [s["stability_sensitivity"] for s in summary_data]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Sensitivity comparison
        x = np.arange(len(parameters))
        width = 0.25

        ax1.bar(
            x - width,
            reward_sensitivities,
            width,
            label="Reward Sensitivity",
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
        ax1.set_title("Parameter Sensitivity Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(parameters, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Best values
        best_rewards = [s["best_reward"] for s in summary_data]
        colors = plt.cm.viridis(np.linspace(0, 1, len(parameters)))

        bars = ax2.bar(parameters, best_rewards, color=colors, alpha=0.8)
        ax2.set_xlabel("Parameters")
        ax2.set_ylabel("Best Reward Achieved")
        ax2.set_title("Best Performance by Parameter")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, best_rewards):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/sensitivity_summary.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def _save_results(self):
        """
        Save all results to files
        """
        # Save detailed results
        with open(f"{self.output_dir}/detailed_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save configuration
        config_info = {
            "param_ranges": self.param_ranges,
            "default_params": self.default_params,
            "analysis_date": datetime.now().isoformat(),
        }

        with open(f"{self.output_dir}/analysis_config.json", "w") as f:
            json.dump(config_info, f, indent=2)

        print(f"Results saved to {self.output_dir}")

    def generate_report(self):
        """
        Generate comprehensive sensitivity analysis report
        """
        report = []
        report.append("# RL Parameter Sensitivity Analysis Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("## Executive Summary")
        report.append(
            "This report presents the results of parameter sensitivity analysis for the DQN model."
        )
        report.append(
            "The analysis evaluates how changes in hyperparameters affect training performance."
        )
        report.append("")

        # Parameter Ranges
        report.append("## Parameter Ranges Tested")
        for param, values in self.param_ranges.items():
            report.append(f"- **{param}**: {values}")
        report.append("")

        # Detailed Results
        for param_name, results in self.results.items():
            report.append(f"## {param_name} Analysis")
            report.append(f"### Results Summary")

            # Find best and worst configurations
            final_rewards = [r["final_reward"] for r in results]
            best_idx = np.argmax(final_rewards)
            worst_idx = np.argmin(final_rewards)

            report.append(
                f"- **Best Value**: {results[best_idx]['value']} (Reward: {results[best_idx]['final_reward']:.2f})"
            )
            report.append(
                f"- **Worst Value**: {results[worst_idx]['value']} (Reward: {results[worst_idx]['final_reward']:.2f})"
            )
            report.append(
                f"- **Performance Range**: {max(final_rewards) - min(final_rewards):.2f}"
            )
            report.append("")

            # Detailed table
            report.append("### Detailed Results")
            report.append(
                "| Value | Final Reward | Convergence Episodes | Training Stability | Final Accuracy | Training Time |"
            )
            report.append(
                "|-------|--------------|---------------------|-------------------|----------------|---------------|"
            )

            for r in results:
                report.append(
                    f"| {r['value']} | {r['final_reward']:.2f} | {r['convergence_episodes']:.0f} | {r['training_stability']:.3f} | {r['final_accuracy']:.3f} | {r['training_time']:.1f} |"
                )

            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append(
            "Based on the sensitivity analysis, the following parameter values are recommended:"
        )
        report.append("")

        for param_name, results in self.results.items():
            final_rewards = [r["final_reward"] for r in results]
            best_idx = np.argmax(final_rewards)
            report.append(f"- **{param_name}**: {results[best_idx]['value']}")

        report.append("")
        report.append(
            "These recommendations are based on maximizing final reward performance."
        )
        report.append(
            "Consider other factors such as training time and stability when making final decisions."
        )

        # Save report
        with open(
            f"{self.output_dir}/sensitivity_analysis_report.md", "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(report))

        print(f"Report saved to {self.output_dir}/sensitivity_analysis_report.md")


def main():
    """
    Main execution function
    """
    print("Starting RL Parameter Sensitivity Analysis...")

    # Create analyzer
    analyzer = RLSensitivityAnalyzer()

    # Run complete analysis
    analyzer.run_complete_sensitivity_analysis()

    # Generate report
    analyzer.generate_report()

    print("\nSensitivity Analysis Complete!")
    print("Check the results directory for detailed analysis and visualizations.")


if __name__ == "__main__":
    main()
