#!/usr/bin/env python3
"""
Run Real Sensitivity Analysis for DQN Hyperparameters
This script performs actual training with different parameter configurations
"""

import sys
import os
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sensitivity_analysis_rl import RLSensitivityAnalyzer


def main():
    """
    Run real sensitivity analysis with actual training
    """
    print("=== Real DQN Sensitivity Analysis ===")
    print(f"Started at: {datetime.now()}")

    # Create analyzer
    analyzer = RLSensitivityAnalyzer(output_dir="results/sensitivity_analysis_rl")

    # Option 1: Run analysis for a single parameter (faster)
    print("\nChoose analysis mode:")
    print("1. Single parameter analysis (faster)")
    print("2. Complete analysis (all parameters)")
    print("3. Quick test (learning_rate only)")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        print("\nAvailable parameters:")
        for i, param in enumerate(analyzer.param_ranges.keys(), 1):
            print(f"{i}. {param}")

        param_choice = input("Enter parameter number: ").strip()
        try:
            param_idx = int(param_choice) - 1
            param_name = list(analyzer.param_ranges.keys())[param_idx]
            print(f"\nRunning sensitivity analysis for: {param_name}")
            analyzer.run_single_parameter_analysis(param_name)
        except (ValueError, IndexError):
            print("Invalid choice!")
            return

    elif choice == "2":
        print("\nRunning complete sensitivity analysis...")
        print("This will take a long time (several hours)!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == "y":
            analyzer.run_complete_sensitivity_analysis()
        else:
            print("Cancelled.")
            return

    elif choice == "3":
        print("\nRunning quick test with learning_rate...")
        analyzer.run_single_parameter_analysis("learning_rate")

    else:
        print("Invalid choice!")
        return

    print(f"\nCompleted at: {datetime.now()}")
    print(f"Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
