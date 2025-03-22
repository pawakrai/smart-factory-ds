# โค้ดหลักของ GA (selection, crossover, mutation, fitness evaluation)
# ga/ga.py

import numpy as np
from copy import deepcopy

# Import operators
from src.ga.operators import roulette_wheel_selection, crossover, mutate, apply_bound


class GeneticAlgorithm:
    def __init__(self, problem, params):
        # Unpack Problem
        self.cost_func = problem["cost_func"]
        self.n_var = problem["n_var"]
        self.var_min = np.array(problem["var_min"], dtype=float)
        self.var_max = np.array(problem["var_max"], dtype=float)

        # Unpack Params
        self.max_iter = params["max_iter"]
        self.pop_size = params["pop_size"]
        self.beta = params["beta"]
        self.pc = params["pc"]
        self.gamma = params["gamma"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]

        # Number of children (for crossover)
        self.nc = int(round(self.pc * self.pop_size / 2) * 2)

        # Best Solution
        self.best_sol = None
        self.best_cost_history = []

    def run(self):
        # 1) Initialize Population
        pop = self._init_population()

        # 2) Evaluate cost & find best
        pop, self.best_sol = self._evaluate_population(pop)

        # 3) Main GA Loop
        for it in range(self.max_iter):
            # 3.1) Calculate selection probabilities
            costs = np.array([ind["cost"] for ind in pop])
            avg_cost = np.mean(costs) if np.mean(costs) != 0 else 1.0
            costs = costs / avg_cost
            probs = np.exp(-self.beta * costs)

            # 3.2) Crossover & Mutation
            children = []
            for _ in range(self.nc // 2):
                p1 = pop[roulette_wheel_selection(probs)]
                p2 = pop[roulette_wheel_selection(probs)]

                c1, c2 = crossover(p1, p2, self.gamma)

                c1 = mutate(c1, self.mu, self.sigma)
                c2 = mutate(c2, self.mu, self.sigma)

                apply_bound(c1, self.var_min, self.var_max)
                apply_bound(c2, self.var_min, self.var_max)

                children.append(c1)
                children.append(c2)

            # 3.3) Evaluate Children
            for child in children:
                child["cost"] = self.cost_func(child["position"])
                # Update best
                if child["cost"] < self.best_sol["cost"]:
                    self.best_sol = deepcopy(child)

            # 3.4) Merge & Sort
            pop_extended = pop + children
            pop_extended = sorted(pop_extended, key=lambda x: x["cost"])
            pop = pop_extended[: self.pop_size]

            # 3.5) Store Best Cost
            self.best_cost_history.append(self.best_sol["cost"])

            # 3.6) Print iteration info
            print(f"Iteration {it}: Best Cost = {self.best_sol['cost']:.6f}")

        # Return output
        return {
            "pop": pop,
            "best_sol": self.best_sol,
            "best_cost": self.best_cost_history,
        }

    def _init_population(self):
        """สร้างประชากรเริ่มต้นแบบสุ่ม"""
        pop = []
        for _ in range(self.pop_size):
            # random position
            position = np.random.uniform(self.var_min, self.var_max, self.n_var)
            ind = {"position": position, "cost": None}
            pop.append(ind)
        return pop

    def _evaluate_population(self, pop):
        """Evaluate cost ของประชากร และค้นหา best solution"""
        best_sol_local = None
        for ind in pop:
            ind["cost"] = self.cost_func(ind["position"])
            if (best_sol_local is None) or (ind["cost"] < best_sol_local["cost"]):
                best_sol_local = deepcopy(ind)
        return pop, best_sol_local
