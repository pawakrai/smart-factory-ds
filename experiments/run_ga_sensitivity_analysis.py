#!/usr/bin/env python3
"""
Run GA Sensitivity Analysis
This script performs sensitivity analysis for Genetic Algorithm hyperparameters
Focus: Crossover Rate and Mutation Rate effects
"""

import sys
import os
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sensitivity_analysis_ga import GASensitivityAnalyzer


def main():
    """
    Run GA sensitivity analysis
    """
    print("=== GA Sensitivity Analysis ===")
    print(f"Started at: {datetime.now()}")
    print("\nFocus: Crossover Rate and Mutation Rate Analysis")
    print(
        "Goal: Understand parameter effects on convergence speed and solution quality"
    )

    # Create analyzer
    analyzer = GASensitivityAnalyzer(output_dir="results/sensitivity_analysis_ga")

    # Show parameter ranges
    print(f"\nParameter Ranges:")
    for param, values in analyzer.param_ranges.items():
        print(f"  {param}: {values}")

    print("\nChoose analysis mode:")
    print("1. Crossover Rate analysis (อัตราการผสมข้าม)")
    print("2. Mutation Rate analysis (อัตราการกลายพันธุ์)")
    print("3. Population Size analysis")
    print("4. Complete analysis (all parameters)")
    print("5. Quick test (crossover_rate only)")

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        print("\n🧬 Analyzing Crossover Rate sensitivity...")
        print("This will test how crossover rate affects:")
        print("- Solution quality (best fitness)")
        print("- Convergence speed")
        print("- Population diversity")
        analyzer.run_single_parameter_analysis("crossover_rate")

    elif choice == "2":
        print("\n🔀 Analyzing Mutation Rate sensitivity...")
        print("This will test how mutation rate affects:")
        print("- Exploration vs exploitation balance")
        print("- Diversity maintenance")
        print("- Convergence behavior")
        analyzer.run_single_parameter_analysis("mutation_rate")

    elif choice == "3":
        print("\n👥 Analyzing Population Size sensitivity...")
        analyzer.run_single_parameter_analysis("population_size")

    elif choice == "4":
        print("\n📊 Running complete sensitivity analysis...")
        print("This will analyze all parameters:")
        print("- Crossover Rate (0.6, 0.7, 0.8, 0.9)")
        print("- Mutation Rate (0.01, 0.05, 0.1, 0.2)")
        print("- Population Size (50, 100, 200)")
        print("- Number of Generations (100, 200, 300)")
        print("\nThis will take significant time!")

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == "y":
            analyzer.run_complete_sensitivity_analysis()
        else:
            print("Cancelled.")
            return

    elif choice == "5":
        print("\n⚡ Running quick test with crossover_rate...")
        # Reduce parameter range for quick test
        analyzer.param_ranges["crossover_rate"] = [0.7, 0.8, 0.9]
        analyzer.run_single_parameter_analysis("crossover_rate")

    else:
        print("Invalid choice!")
        return

    # Generate comprehensive report
    analyzer.generate_report()

    print(f"\n✅ Analysis completed at: {datetime.now()}")
    print(f"📁 Results saved to: {analyzer.output_dir}")

    # Show key insights
    print(f"\n🎯 Key Insights:")
    if analyzer.results:
        for param_name, results in analyzer.results.items():
            best_fitness = [r["best_fitness"] for r in results]
            best_idx = np.argmin(best_fitness)  # Lower fitness is better
            best_result = results[best_idx]

            print(f"\n{param_name}:")
            print(f"  🏆 Optimal Value: {best_result['value']}")
            print(f"  🎯 Best Fitness: {best_result['best_fitness']:.4f}")
            print(
                f"  ⚡ Convergence: Generation {best_result['convergence_generation']}"
            )
            print(f"  🔄 Diversity: {best_result['diversity_maintenance']:.4f}")


if __name__ == "__main__":
    import numpy as np

    main()
