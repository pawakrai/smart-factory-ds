#!/usr/bin/env python3
"""
GA Adapter for Sensitivity Analysis
This adapts the existing GA implementation to work with sensitivity analysis
"""

import numpy as np
import sys
import os
from copy import deepcopy

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ga.ga_real_value import GeneticAlgorithm


class GASensitivityAdapter:
    """
    Adapter class to make GA work with sensitivity analysis
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

        # Define optimization problem (Aluminum Melting Scheduling)
        self.problem = self._define_problem()
        self.params = self._define_params()

        # Initialize GA
        self.ga = GeneticAlgorithm(self.problem, self.params)

        # Tracking variables
        self.fitness_history = []
        self.diversity_history = []
        self.population = []

    def _define_problem(self):
        """
        Define the optimization problem for aluminum melting scheduling
        """

        def aluminum_melting_cost_function(x):
            """
            Cost function for aluminum melting scheduling optimization
            x = [power_profile, temperature_targets, timing_parameters]
            """
            # Simplified cost function for demonstration
            # In practice, this would integrate with the aluminum melting environment

            # Power efficiency penalty
            power_penalty = np.sum(np.abs(x)) * 0.1

            # Temperature control penalty
            temp_penalty = np.sum((x - 0.5) ** 2) * 0.2

            # Timing penalty
            timing_penalty = np.sum(np.diff(x) ** 2) * 0.15

            # Energy consumption penalty
            energy_penalty = np.sum(x**2) * 0.05

            total_cost = power_penalty + temp_penalty + timing_penalty + energy_penalty

            return total_cost

        problem = {
            "cost_func": aluminum_melting_cost_function,
            "n_var": 10,  # Number of optimization variables
            "var_min": [0.0] * 10,  # Lower bounds
            "var_max": [1.0] * 10,  # Upper bounds
        }

        return problem

    def _define_params(self):
        """
        Define GA parameters based on sensitivity analysis configuration
        """
        params = {
            "max_iter": self.num_generations,
            "pop_size": self.population_size,
            "beta": 1.0,  # Selection pressure
            "pc": self.crossover_rate,  # Crossover probability
            "gamma": 0.1,  # Crossover parameter
            "mu": self.mutation_rate,  # Mutation probability
            "sigma": 0.1,  # Mutation standard deviation
        }

        return params

    def evolve_generation(self):
        """
        Evolve one generation and track metrics
        """
        if not hasattr(self, "_initialized"):
            # Initialize for first generation
            self.population = self.ga._init_population()
            self.population, self.ga.best_sol = self.ga._evaluate_population(
                self.population
            )
            self._current_generation = 0
            self._initialized = True

        if self._current_generation >= self.num_generations:
            return

        # Run one iteration of GA
        costs = np.array([ind["cost"] for ind in self.population])
        avg_cost = np.mean(costs) if np.mean(costs) != 0 else 1.0
        costs = costs / avg_cost
        probs = np.exp(-self.ga.beta * costs)

        # Crossover & Mutation
        children = []
        nc = int(round(self.ga.pc * self.ga.pop_size / 2) * 2)

        for _ in range(nc // 2):
            from src.ga.operators_real_value import (
                roulette_wheel_selection,
                crossover,
                mutate,
                apply_bound,
            )

            p1 = self.population[roulette_wheel_selection(probs)]
            p2 = self.population[roulette_wheel_selection(probs)]

            c1, c2 = crossover(p1, p2, self.ga.gamma)

            c1 = mutate(c1, self.ga.mu, self.ga.sigma)
            c2 = mutate(c2, self.ga.mu, self.ga.sigma)

            apply_bound(c1, self.ga.var_min, self.ga.var_max)
            apply_bound(c2, self.ga.var_min, self.ga.var_max)

            children.append(c1)
            children.append(c2)

        # Evaluate Children
        for child in children:
            child["cost"] = self.ga.cost_func(child["position"])
            if child["cost"] < self.ga.best_sol["cost"]:
                self.ga.best_sol = deepcopy(child)

        # Merge & Sort
        pop_extended = self.population + children
        pop_extended = sorted(pop_extended, key=lambda x: x["cost"])
        self.population = pop_extended[: self.ga.pop_size]

        # Track metrics
        best_cost = self.ga.best_sol["cost"]
        self.fitness_history.append(best_cost)

        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)

        self._current_generation += 1

    def get_best_individual(self):
        """
        Get the best individual found so far
        """
        if hasattr(self.ga, "best_sol") and self.ga.best_sol is not None:
            return {
                "fitness": self.ga.best_sol["cost"],
                "solution": self.ga.best_sol["position"],
            }
        else:
            return {"fitness": float("inf"), "solution": None}

    def calculate_population_diversity(self):
        """
        Calculate population diversity as average pairwise distance
        """
        if len(self.population) < 2:
            return 0.0

        positions = np.array([ind["position"] for ind in self.population])

        # Calculate pairwise distances
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)

        return np.mean(distances) if distances else 0.0

    def run_complete_optimization(self):
        """
        Run complete optimization and return results
        """
        # Reset state
        self._initialized = False
        self.fitness_history = []
        self.diversity_history = []

        # Run all generations
        for generation in range(self.num_generations):
            self.evolve_generation()

            if generation % 50 == 0:
                best = self.get_best_individual()
                print(f"Generation {generation}: Best Fitness = {best['fitness']:.6f}")

        return {
            "best_individual": self.get_best_individual(),
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
        }
