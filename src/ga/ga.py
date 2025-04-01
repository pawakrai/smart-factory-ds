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

        # นำเข้าหรือกำหนดค่า USE_FURNACE_A, USE_FURNACE_B, T_MELT, ฯลฯ
        from src.app import USE_FURNACE_A, USE_FURNACE_B, T_MELT, TOTAL_SLOTS

        # สมมติว่าเราต้องการให้ start_slot กระจายกันไม่ทับ (แบบง่ายๆ)
        # ตัวอย่างเช่น ถ้าเปิด 2 เตา => สลับ A,B และขยับ start_slot ต่อๆ กัน
        # ถ้าเตาเดียว => ไล่เรียงต่อกัน
        # หมายเหตุ: หาก T_MELT=3 เราจะเผื่อให้ batch ต่อไปเริ่มหลังจาก 3 slot ก่อน

        while len(pop) < self.pop_size:
            position = np.empty(self.n_var, dtype=int)

            # ตัวอย่างง่าย: base_start = 0
            base_start = 0

            for i in range(n_batches):
                # กำหนด start_slot:
                #   - ถ้าเตาเดียว => start_slot = base_start + i*T_MELT (เช่น ไล่ต่อกัน)
                #   - ถ้า 2 เตา => start_slot = i*T_MELT (แล้วให้ batch A, B สลับ?)
                #     หรือจะบวก gap อีกสัก 1 slot เพื่อให้สบายขึ้น

                # ตรวจว่าเราเปิดกี่เตา
                if USE_FURNACE_A and USE_FURNACE_B:
                    # เปิด 2 เตา => สลับเตา + ไล่ start slot
                    # เช่น batch i => start = i*(T_MELT), furnace = i%2
                    start_slot = i * (T_MELT)
                    furnace = i % 2  # สลับ 0,1,0,1
                elif USE_FURNACE_A and not USE_FURNACE_B:
                    # เปิด A อย่างเดียว => furnace=0
                    # ไล่ต่อกัน
                    start_slot = i * T_MELT
                    furnace = 0
                elif (not USE_FURNACE_A) and USE_FURNACE_B:
                    # เปิด B อย่างเดียว => furnace=1
                    start_slot = i * T_MELT
                    furnace = 1
                else:
                    # กรณีปิดหมด => fallback (หรือ raise Error)
                    start_slot = 0
                    furnace = 0  # no furnace available

                # กันไม่ให้ start_slot เกิน TOTAL_SLOTS - T_MELT
                start_slot = min(start_slot, TOTAL_SLOTS - T_MELT)

                # ใส่ลงใน position
                position[2 * i] = start_slot
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
