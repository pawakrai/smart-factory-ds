#!/usr/bin/env python3
"""
GA Adapter for Sensitivity Analysis using pymoo framework (normal_ga.py)
This adapts the existing pymoo GA implementation to work with sensitivity analysis
"""

import numpy as np
import sys
import os
from copy import deepcopy
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import pymoo components and normal_ga
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# Import the scheduling cost function from normal_ga
from src.normal_ga import scheduling_cost, greedy_assignment, NUM_BATCHES


class GASensitivityAdapterPymoo:
    """
    Adapter class to make pymoo GA work with sensitivity analysis
    """

    def __init__(
        self,
        population_size=100,
        num_generations=200,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elite_size=20,
        tournament_size=5,
    ):

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size

        # Define the aluminum melting scheduling problem
        self.problem = AluminumMeltingProblem()

        # Configure GA algorithm
        self.algorithm = self._configure_algorithm()

        # Tracking variables
        self.fitness_history = []
        self.diversity_history = []
        self.result = None

    def _configure_algorithm(self):
        """
        Configure pymoo GA algorithm with sensitivity analysis parameters
        """
        # Operators for permutation-based optimization
        sampling = PermutationRandomSampling()
        crossover = OrderCrossover(prob=self.crossover_rate)
        mutation = InversionMutation(prob=self.mutation_rate)

        # Create GA algorithm
        algorithm = GA(
            pop_size=self.population_size,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True,
        )

        return algorithm

    def run_complete_optimization(self):
        """
        Run complete optimization and return results
        """
        # Reset state
        self.fitness_history = []
        self.diversity_history = []

        # Set termination criteria
        termination = get_termination("n_gen", self.num_generations)

        print(
            f"Running pymoo GA optimization for {self.num_generations} generations..."
        )

        # Run optimization
        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=42,
            verbose=False,  # Set to False to reduce output during sensitivity analysis
            save_history=True,
        )

        # Extract fitness history
        if self.result.history:
            self.fitness_history = [
                np.min(gen.opt.get("F")) for gen in self.result.history
            ]

            # Calculate diversity for each generation (simplified)
            for gen in self.result.history:
                pop_fitness = gen.pop.get("F").flatten()
                diversity = np.std(pop_fitness) / (np.mean(pop_fitness) + 1e-8)
                self.diversity_history.append(diversity)

        # Get best individual
        best_individual = self.get_best_individual()

        return {
            "best_individual": best_individual,
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
        }

    def get_best_individual(self):
        """
        Get the best individual found so far
        """
        if self.result is not None and self.result.X is not None:
            return {
                "fitness": float(self.result.F[0]),
                "solution": self.result.X.tolist(),
            }
        else:
            return {"fitness": float("inf"), "solution": None}

    def calculate_population_diversity(self):
        """
        Calculate current population diversity
        """
        if self.diversity_history:
            return self.diversity_history[-1]
        else:
            return 0.0

    def evolve_generation(self):
        """
        Evolve one generation (not directly supported by pymoo, use run_complete_optimization instead)
        """
        # This method is kept for compatibility but pymoo doesn't support step-by-step evolution
        # Users should call run_complete_optimization() instead
        raise NotImplementedError(
            "pymoo GA doesn't support step-by-step evolution. Use run_complete_optimization() instead."
        )


class AluminumMeltingProblem(Problem):
    """
    Aluminum melting scheduling problem for pymoo
    """

    def __init__(self):
        super().__init__(
            n_var=NUM_BATCHES,
            n_obj=1,  # Single objective: minimize cost
            n_constr=0,
            xl=0,
            xu=NUM_BATCHES - 1,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate population of permutations
        """
        results_f = []

        for batch_order_perm in X:
            # Convert permutation to schedule using greedy assignment
            x_schedule_vector = greedy_assignment(
                batch_order_perm, num_batches=self.n_var
            )

            # Calculate cost using scheduling_cost function
            cost, makespan, cost_components = scheduling_cost(x_schedule_vector)

            # Add penalty for failed assignments
            if makespan > 1440 * 1.1:  # If makespan is unusually long
                cost += 1e15

            results_f.append([cost])

        out["F"] = np.array(results_f)


# Test function for the adapter
def test_ga_adapter():
    """
    Test function to verify the GA adapter works correctly
    """
    print("Testing GA Adapter with pymoo...")

    # Create adapter with small parameters for quick test
    adapter = GASensitivityAdapterPymoo(
        population_size=20, num_generations=10, crossover_rate=0.8, mutation_rate=0.1
    )

    # Run optimization
    results = adapter.run_complete_optimization()

    print(f"Test completed!")
    print(f"Best fitness: {results['best_individual']['fitness']:.4f}")
    print(f"Fitness history length: {len(results['fitness_history'])}")
    print(
        f"Final diversity: {results['diversity_history'][-1] if results['diversity_history'] else 0:.4f}"
    )

    return results


if __name__ == "__main__":
    test_ga_adapter()
