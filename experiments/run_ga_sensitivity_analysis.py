#!/usr/bin/env python3
"""
Run GA Sensitivity Analysis
This script performs sensitivity analysis for Genetic Algorithm hyperparameters
Focus: Crossover Rate, Mutation Rate, Population Size, and Number of Generations effects
Updated: No plotting - results saved only for comprehensive analysis
"""

import sys
import os
from datetime import datetime
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sensitivity_analysis_ga import GASensitivityAnalyzer


def run_quick_analysis():
    """
    Run quick analysis for testing (reduced parameter ranges)
    """
    print("🚀 Running Quick GA Sensitivity Analysis...")
    
    analyzer = GASensitivityAnalyzer(output_dir="results/sensitivity_analysis_ga")
    
    # Reduce parameter ranges for quick test
    analyzer.param_ranges = {
        "crossover_rate": [0.7, 0.8, 0.9],
        "mutation_rate": [0.05, 0.1, 0.2],
        "population_size": [50, 100, 150],
        "num_generations": [50, 100],
    }
    
    analyzer.run_complete_sensitivity_analysis()
    analyzer.generate_report()
    
    return analyzer


def run_full_analysis():
    """
    Run comprehensive analysis with all parameter combinations
    """
    print("🔬 Running Full GA Sensitivity Analysis...")
    print("⚠️  This will take significant time!")
    
    analyzer = GASensitivityAnalyzer(output_dir="results/sensitivity_analysis_ga")
    
    # Full parameter ranges (original)
    analyzer.param_ranges = {
        "crossover_rate": [0.6, 0.7, 0.8, 0.9],
        "mutation_rate": [0.01, 0.05, 0.1, 0.2], 
        "population_size": [50, 100, 150, 200],
        "num_generations": [50, 100, 150, 200],
    }
    
    # Show estimated time
    total_combinations = sum(len(values) for values in analyzer.param_ranges.values())
    print(f"📊 Total parameter combinations to test: {total_combinations}")
    print(f"⏱️  Estimated time: {total_combinations * 2:.0f}-{total_combinations * 5:.0f} minutes")
    
    start_time = time.time()
    analyzer.run_complete_sensitivity_analysis()
    analyzer.generate_report()
    
    total_time = time.time() - start_time
    print(f"⏰ Total execution time: {total_time/60:.2f} minutes")
    
    return analyzer


def run_single_parameter_analysis(param_name):
    """
    Run analysis for a single parameter only
    """
    print(f"🎯 Running Single Parameter Analysis: {param_name}")
    
    analyzer = GASensitivityAnalyzer(output_dir="results/sensitivity_analysis_ga")
    
    if param_name not in analyzer.param_ranges:
        print(f"❌ Invalid parameter name: {param_name}")
        print(f"Available parameters: {list(analyzer.param_ranges.keys())}")
        return None
        
    analyzer.run_single_parameter_analysis(param_name)
    analyzer.generate_report()
    
    return analyzer


def show_previous_results():
    """
    Show results from previous analysis if available
    """
    results_dir = "results/sensitivity_analysis_ga"
    
    if not os.path.exists(results_dir):
        print("❌ No previous results found.")
        return
    
    import json
    import pandas as pd
    
    # Check for summary file
    summary_path = os.path.join(results_dir, "sensitivity_summary.csv")
    if os.path.exists(summary_path):
        print("📊 Previous Analysis Summary:")
        df = pd.read_csv(summary_path)
        print(df.to_string(index=False))
        
        # Check for analysis config
        config_path = os.path.join(results_dir, "analysis_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                print(f"\n📅 Analysis Date: {config.get('analysis_timestamp', 'Unknown')}")
                print(f"🔧 Default Parameters: {config.get('default_params', {})}")
    else:
        print("❌ No summary results found.")


def main():
    """
    Main function with menu-driven interface
    """
    print("=" * 60)
    print("🧬 GA SENSITIVITY ANALYSIS FOR ALUMINUM MELTING OPTIMIZATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nIntegrated with normal_ga.py for aluminum melting scheduling")
    print("Results will be saved (no plotting) for comprehensive analysis")
    
    while True:
        print("\n" + "=" * 50)
        print("📋 ANALYSIS OPTIONS:")
        print("1. 🚀 Quick Analysis (reduced parameter ranges)")
        print("2. 🔬 Full Analysis (all parameters - comprehensive)")
        print("3. 🎯 Single Parameter Analysis")
        print("4. 📊 View Previous Results")
        print("5. ❌ Exit")
        print("=" * 50)

        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            print("\n🚀 QUICK ANALYSIS")
            print("-" * 30)
            confirm = input("Run quick analysis? (y/n): ").strip().lower()
            if confirm == "y":
                analyzer = run_quick_analysis()
                show_results_summary(analyzer)
            
        elif choice == "2":
            print("\n🔬 FULL ANALYSIS")
            print("-" * 30)
            print("⚠️  WARNING: This will test all parameter combinations!")
            print("⏱️  Expected time: 30-120 minutes depending on system")
            print("💾 Results will be saved automatically (no plotting)")
            
            confirm = input("Continue with full analysis? (y/n): ").strip().lower()
            if confirm == "y":
                analyzer = run_full_analysis()
                show_results_summary(analyzer)
            else:
                print("❌ Full analysis cancelled.")

        elif choice == "3":
            print("\n🎯 SINGLE PARAMETER ANALYSIS")
            print("-" * 30)
            print("Available parameters:")
            analyzer_temp = GASensitivityAnalyzer()
            for i, param in enumerate(analyzer_temp.param_ranges.keys(), 1):
                values = analyzer_temp.param_ranges[param]
                print(f"  {i}. {param}: {values}")
            
            param_choice = input("Enter parameter name: ").strip()
            if param_choice in analyzer_temp.param_ranges:
                analyzer = run_single_parameter_analysis(param_choice)
                if analyzer:
                    show_results_summary(analyzer)
            else:
                print(f"❌ Invalid parameter: {param_choice}")

        elif choice == "4":
            print("\n📊 PREVIOUS RESULTS")
            print("-" * 30)
            show_previous_results()

        elif choice == "5":
            print("\n👋 Exiting GA Sensitivity Analysis")
            break

        else:
            print("❌ Invalid choice! Please enter 1-5.")


def show_results_summary(analyzer):
    """
    Show summary of analysis results
    """
    if not analyzer or not analyzer.results:
        print("❌ No results to display.")
        return
    
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS RESULTS SUMMARY")
    print("=" * 60)
    
    for param_name, results in analyzer.results.items():
        best_fitness_values = [r["best_fitness"] for r in results]
        convergence_generations = [r["convergence_generation"] for r in results]
        execution_times = [r["execution_time"] for r in results]
        
        best_idx = min(range(len(best_fitness_values)), key=lambda i: best_fitness_values[i])
        best_result = results[best_idx]
        
        print(f"\n🔧 {param_name.upper()}:")
        print(f"  🏆 Optimal Value: {best_result['value']}")
        print(f"  🎯 Best Fitness: {best_result['best_fitness']:.2f}")
        print(f"  ⚡ Convergence: Generation {best_result['convergence_generation']}")
        print(f"  ⏱️  Execution Time: {best_result['execution_time']:.2f}s")
        print(f"  🔄 Diversity: {best_result['diversity_maintenance']:.4f}")
        
        # Show improvement over worst
        worst_fitness = max(best_fitness_values)
        if worst_fitness != best_result['best_fitness'] and worst_fitness != 0:
            improvement = (worst_fitness - best_result['best_fitness']) / worst_fitness * 100
            print(f"  📈 Improvement over worst: {improvement:.1f}%")
    
    print(f"\n📁 All detailed results saved to: {analyzer.output_dir}")
    print("📊 Files created:")
    print("  - sensitivity_summary.csv (summary table)")
    print("  - sensitivity_analysis_report.md (comprehensive report)")
    print("  - detailed_results.json (all raw data)")
    print("  - analysis_config.json (analysis configuration)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user!")
        print("👋 Exiting...")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        print("\n🔍 Full error details:")
        print(traceback.format_exc())