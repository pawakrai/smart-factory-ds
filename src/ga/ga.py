# ga/ga.py

import numpy as np
from copy import deepcopy
from src.ga.operators import roulette_wheel_selection, crossover, mutate, apply_bound


class GeneticAlgorithm:
    def __init__(self, problem, params):
        self.cost_func = problem["cost_func"]
        self.n_var = problem["n_var"]
        self.var_min = np.array(problem["var_min"], dtype=int)
        self.var_max = np.array(problem["var_max"], dtype=int)

        self.max_iter = params["max_iter"]
        self.pop_size = params["pop_size"]
        self.beta = params["beta"]
        self.pc = params["pc"]
        self.gamma = params["gamma"]
        self.mu = params["mu"]
        self.sigma = params["sigma"]

        self.nc = int(round(self.pc * self.pop_size / 2) * 2)
        self.best_sol = None
        self.best_cost_history = []

    def run(self):
        pop = self._init_population()
        pop, self.best_sol = self._evaluate_population(pop)

        for it in range(self.max_iter):
            costs = np.array([ind["cost"] for ind in pop])
            avg_cost = np.mean(costs) if np.mean(costs) != 0 else 1.0
            norm_costs = costs / avg_cost
            probs = np.exp(-self.beta * norm_costs)

            children = []
            for _ in range(self.nc // 2):
                p1 = pop[roulette_wheel_selection(probs)]
                p2 = pop[roulette_wheel_selection(probs)]

                c1, c2 = crossover(p1, p2, self.gamma)
                c1 = mutate(c1, self.mu, self.sigma, self.var_min, self.var_max)
                c2 = mutate(c2, self.mu, self.sigma, self.var_min, self.var_max)

                apply_bound(c1, self.var_min, self.var_max)
                apply_bound(c2, self.var_min, self.var_max)

                children.append(c1)
                children.append(c2)

            for child in children:
                child["cost"] = self.cost_func(child["position"])
                if child["cost"] < self.best_sol["cost"]:
                    self.best_sol = deepcopy(child)

            pop_extended = pop + children
            pop_extended = sorted(pop_extended, key=lambda x: x["cost"])
            pop = pop_extended[: self.pop_size]

            self.best_cost_history.append(self.best_sol["cost"])
            print(f"Iteration {it}: Best Cost = {self.best_sol['cost']:.6f}")

        return {
            "pop": pop,
            "best_sol": self.best_sol,
            "best_cost": self.best_cost_history,
        }

    def _init_population(self):
        pop = []
        n_batches = self.n_var // 2

        while len(pop) < self.pop_size:
            # สร้าง position แบบสุ่มล้วน
            position = np.empty(self.n_var, dtype=int)

            for i in range(n_batches):
                # สุ่ม start_slot (0 ถึง 48) หรืออาจใช้ 48 - (T_LOAD+T_MELT) ถ้าอยากกันขอบ
                start_slot = np.random.randint(0, 49)
                # สุ่ม furnace 0=A, 1=B
                furnace = np.random.randint(0, 2)
                position[2 * i]     = start_slot
                position[2 * i + 1] = furnace

            ind = {"position": position, "cost": None}
            pop.append(ind)

        return pop

    def _evaluate_population(self, pop):
        best_sol_local = None
        for ind in pop:
            ind["cost"] = self.cost_func(ind["position"])
            if best_sol_local is None or ind["cost"] < best_sol_local["cost"]:
                best_sol_local = deepcopy(ind)
        return pop, best_sol_local
