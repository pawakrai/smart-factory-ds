#!/usr/bin/env python3
"""
Visualize GA Sensitivity Analysis Results
Creates graphs similar to the sensitivity_summary.png example
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json


def load_ga_results():
    """
    Load GA sensitivity analysis results
    """
    results_dir = "results/sensitivity_analysis_ga"

    # Load summary data
    summary_df = pd.read_csv(os.path.join(results_dir, "sensitivity_summary.csv"))

    # Load detailed results
    with open(os.path.join(results_dir, "detailed_results.json"), "r") as f:
        detailed_results = json.load(f)

    return summary_df, detailed_results


def create_sensitivity_comparison_graph(summary_df):
    """
    Create Parameter Sensitivity Comparison graph (left side)
    """
    # Map parameter names to more readable names
    param_mapping = {
        "crossover_rate": "crossover_rate",
        "mutation_rate": "mutation_rate",
        "population_size": "population_size",
        "num_generations": "num_generations",
    }

    # Prepare data
    parameters = [param_mapping.get(p, p) for p in summary_df["parameter"]]

    # Use sensitivity metrics from the data
    fitness_sensitivity = (
        summary_df["fitness_sensitivity"] * 1000
    )  # Scale up for visibility
    convergence_sensitivity = (
        summary_df["convergence_sensitivity"] * 100
    )  # Scale to percentage
    time_sensitivity = summary_df["time_sensitivity"] * 100  # Scale to percentage

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar positions
    x = np.arange(len(parameters))
    width = 0.25

    # Create bars
    bars1 = ax1.bar(
        x - width,
        fitness_sensitivity,
        width,
        label="Fitness Sensitivity (×1000)",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        x,
        convergence_sensitivity,
        width,
        label="Convergence Sensitivity (%)",
        color="orange",
        alpha=0.8,
    )
    bars3 = ax1.bar(
        x + width,
        time_sensitivity,
        width,
        label="Time Sensitivity (%)",
        color="green",
        alpha=0.8,
    )

    # Customize first plot
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Sensitivity Score")
    ax1.set_title("Parameter Sensitivity Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(parameters, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    return fig, ax1, ax2


def create_best_performance_graph(summary_df, detailed_results, ax2):
    """
    Create Best Performance by Parameter graph (right side)
    """
    # Map parameter names and get best values
    param_mapping = {
        "crossover_rate": "crossover_rate",
        "mutation_rate": "mutation_rate",
        "population_size": "population_size",
        "num_generations": "num_generations",
    }

    parameters = [param_mapping.get(p, p) for p in summary_df["parameter"]]
    best_values = summary_df["best_value"].values

    # Create color map
    colors = ["#8B4A9C", "#6B5B95", "#4A90A4", "#50C7A4", "#A8D08D", "#FFD93D"]

    # Create bars
    bars = ax2.bar(parameters, best_values, color=colors[: len(parameters)], alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, best_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Customize second plot
    ax2.set_xlabel("Parameters")
    ax2.set_ylabel("Best Reward Achieved")
    ax2.set_title("Best Performance by Parameter")
    ax2.set_xticklabels(parameters, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(best_values) * 1.1)


def create_detailed_analysis_graphs(detailed_results):
    """
    Create additional detailed analysis graphs
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Crossover Rate Analysis
    crossover_data = detailed_results["crossover_rate"]
    values = [d["value"] for d in crossover_data]
    fitness = [d["best_fitness"] for d in crossover_data]

    ax1.plot(values, fitness, "o-", color="blue", linewidth=2, markersize=8)
    ax1.set_xlabel("Crossover Rate")
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Crossover Rate vs Performance")
    ax1.grid(True, alpha=0.3)

    # 2. Mutation Rate Analysis
    mutation_data = detailed_results["mutation_rate"]
    values = [d["value"] for d in mutation_data]
    fitness = [d["best_fitness"] for d in mutation_data]

    ax2.plot(values, fitness, "o-", color="orange", linewidth=2, markersize=8)
    ax2.set_xlabel("Mutation Rate")
    ax2.set_ylabel("Best Fitness")
    ax2.set_title("Mutation Rate vs Performance")
    ax2.grid(True, alpha=0.3)

    # 3. Population Size Analysis
    pop_data = detailed_results["population_size"]
    values = [d["value"] for d in pop_data]
    fitness = [d["best_fitness"] for d in pop_data]
    time = [d["execution_time"] for d in pop_data]

    ax3_twin = ax3.twinx()
    line1 = ax3.plot(
        values,
        fitness,
        "o-",
        color="green",
        linewidth=2,
        markersize=8,
        label="Best Fitness",
    )
    line2 = ax3_twin.plot(
        values,
        time,
        "s-",
        color="red",
        linewidth=2,
        markersize=6,
        label="Execution Time",
    )

    ax3.set_xlabel("Population Size")
    ax3.set_ylabel("Best Fitness", color="green")
    ax3_twin.set_ylabel("Execution Time (s)", color="red")
    ax3.set_title("Population Size vs Performance & Time")
    ax3.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # 4. Generations Analysis
    gen_data = detailed_results["num_generations"]
    values = [d["value"] for d in gen_data]
    fitness = [d["best_fitness"] for d in gen_data]
    time = [d["execution_time"] for d in gen_data]
    convergence = [d["convergence_generation"] for d in gen_data]

    ax4_twin = ax4.twinx()
    line1 = ax4.plot(
        values,
        time,
        "o-",
        color="purple",
        linewidth=2,
        markersize=8,
        label="Execution Time",
    )
    line2 = ax4_twin.plot(
        values,
        convergence,
        "s-",
        color="brown",
        linewidth=2,
        markersize=6,
        label="Convergence Gen",
    )

    ax4.set_xlabel("Number of Generations")
    ax4.set_ylabel("Execution Time (s)", color="purple")
    ax4_twin.set_ylabel("Convergence Generation", color="brown")
    ax4.set_title("Generations vs Time & Convergence")
    ax4.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    return fig


def create_summary_table(summary_df, detailed_results):
    """
    Create a summary table with recommendations
    """
    print("\n" + "=" * 80)
    print("🧬 GA SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\n📊 PARAMETER SENSITIVITY RANKING (Higher = More Sensitive):")
    # Sort by fitness sensitivity
    sorted_df = summary_df.sort_values("fitness_sensitivity", ascending=False)

    for idx, row in sorted_df.iterrows():
        sensitivity_level = (
            "HIGH"
            if row["fitness_sensitivity"] > 0.001
            else "MEDIUM" if row["fitness_sensitivity"] > 0.0005 else "LOW"
        )
        print(
            f"  {idx+1}. {row['parameter']:15} | Sensitivity: {row['fitness_sensitivity']:.4f} ({sensitivity_level})"
        )

    print(f"\n🎯 OPTIMAL PARAMETER CONFIGURATION:")
    for idx, row in summary_df.iterrows():
        improvement = calculate_improvement(row["parameter"], detailed_results)
        print(
            f"  • {row['parameter']:15}: {row['best_value']} (improves performance by {improvement:.1f}%)"
        )

    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    time_sorted = summary_df.sort_values("time_sensitivity", ascending=False)
    for idx, row in time_sorted.iterrows():
        if row["time_sensitivity"] > 0.3:
            print(
                f"  • {row['parameter']:15}: HIGH time impact ({row['time_sensitivity']:.1%})"
            )


def calculate_improvement(param_name, detailed_results):
    """
    Calculate performance improvement of best vs worst value
    """
    param_data = detailed_results[param_name]
    fitness_values = [d["best_fitness"] for d in param_data]

    best_fitness = min(fitness_values)
    worst_fitness = max(fitness_values)

    if worst_fitness > 0:
        improvement = ((worst_fitness - best_fitness) / worst_fitness) * 100
        return max(0, improvement)
    return 0


def main():
    """
    Main function to create all visualizations
    """
    print("🎨 Creating GA Sensitivity Analysis Visualizations...")

    # Load data
    summary_df, detailed_results = load_ga_results()

    # Create main comparison graph (like the example)
    fig, ax1, ax2 = create_sensitivity_comparison_graph(summary_df)
    create_best_performance_graph(summary_df, detailed_results, ax2)

    plt.tight_layout()
    plt.savefig(
        "results/sensitivity_analysis_ga/ga_sensitivity_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Create detailed analysis graphs
    fig2 = create_detailed_analysis_graphs(detailed_results)
    plt.savefig(
        "results/sensitivity_analysis_ga/ga_detailed_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Print summary table
    create_summary_table(summary_df, detailed_results)

    print(f"\n📁 Graphs saved to:")
    print(f"  • results/sensitivity_analysis_ga/ga_sensitivity_summary.png")
    print(f"  • results/sensitivity_analysis_ga/ga_detailed_analysis.png")


if __name__ == "__main__":
    main()
