#!/usr/bin/env python3
"""
Example script showing how to run REAL training sensitivity analysis
This will take much longer than simulation but provides actual results

Usage:
    python example_real_sensitivity_analysis.py
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sensitivity_analysis_rl import RLSensitivityAnalyzer

def run_single_parameter_real_analysis():
    """
    Example: Run real training sensitivity analysis for a single parameter
    This is recommended to start with to understand the time requirements
    """
    print("=== Single Parameter Real Training Sensitivity Analysis ===")
    print("This will run REAL training for different learning rate values")
    print("Estimated time: 10-30 minutes depending on your hardware")
    
    # Create analyzer with real training enabled
    analyzer = RLSensitivityAnalyzer(
        output_dir="results/sensitivity_analysis_rl_real",
        use_real_training=True
    )
    
    # Run analysis for learning rate only
    results = analyzer.run_single_parameter_analysis("learning_rate")
    
    print("\n=== Results Summary ===")
    for result in results:
        print(f"Learning Rate: {result['value']}")
        print(f"  Final Reward: {result['final_reward']:.2f}")
        print(f"  Convergence Episodes: {result['convergence_episodes']}")
        print(f"  Training Time: {result['training_time']:.1f} minutes")
        print()

def run_complete_real_analysis():
    """
    Example: Run complete real training sensitivity analysis
    WARNING: This will take several hours to complete!
    """
    print("=== Complete Real Training Sensitivity Analysis ===")
    print("WARNING: This will run real training for ALL parameters")
    print("Estimated time: 3-8 hours depending on your hardware")
    
    response = input("Are you sure you want to continue? (yes/no): ").lower().strip()
    if response != 'yes':
        print("Analysis cancelled.")
        return
    
    # Create analyzer with real training enabled
    analyzer = RLSensitivityAnalyzer(
        output_dir="results/sensitivity_analysis_rl_complete",
        use_real_training=True
    )
    
    # Run complete analysis
    analyzer.run_complete_sensitivity_analysis()
    
    print("Complete analysis finished!")

def run_fast_simulation_for_comparison():
    """
    Example: Run simulation-based analysis for quick comparison
    """
    print("=== Fast Simulation Analysis (for comparison) ===")
    print("This uses simulated training data for quick results")
    
    # Create analyzer with simulation
    analyzer = RLSensitivityAnalyzer(
        output_dir="results/sensitivity_analysis_rl_simulation",
        use_real_training=False
    )
    
    # Run complete analysis (fast)
    analyzer.run_complete_sensitivity_analysis()
    
    print("Simulation analysis finished!")

def run_custom_parameter_ranges():
    """
    Example: Run real training with custom parameter ranges
    """
    print("=== Custom Parameter Range Analysis ===")
    
    # Create analyzer
    analyzer = RLSensitivityAnalyzer(
        output_dir="results/sensitivity_analysis_rl_custom",
        use_real_training=True
    )
    
    # Customize parameter ranges (fewer values for faster analysis)
    analyzer.param_ranges = {
        "learning_rate": [0.0005, 0.001],  # Only test 2 values
        "gamma": [0.95, 0.99],             # Only test 2 values
        "batch_size": [64, 128],           # Only test 2 values
    }
    
    # Run analysis
    analyzer.run_complete_sensitivity_analysis()
    
    print("Custom analysis finished!")

if __name__ == "__main__":
    print("Real Training Sensitivity Analysis Examples")
    print("==========================================")
    print("Choose an option:")
    print("1. Single parameter analysis (learning_rate) - ~15-30 min")
    print("2. Complete analysis (all parameters) - ~3-8 hours")
    print("3. Fast simulation analysis (for comparison) - ~2-5 min")
    print("4. Custom parameter ranges - ~30-60 min")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        run_single_parameter_real_analysis()
    elif choice == "2":
        run_complete_real_analysis()
    elif choice == "3":
        run_fast_simulation_for_comparison()
    elif choice == "4":
        run_custom_parameter_ranges()
    elif choice == "5":
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")