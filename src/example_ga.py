# app.py

import numpy as np
import matplotlib.pyplot as plt

# Import GA class and operators if needed
from src.ga.ga import GeneticAlgorithm


def main():
    # -------------------------------
    # 1) Problem Definition
    # -------------------------------
    def sphere(x):
        """ตัวอย่างฟังก์ชัน cost = sum(x^2)"""
        return np.sum(x**2)

    # ตัวอย่าง: สร้าง dictionary เก็บข้อมูลปัญหา
    problem = {
        "cost_func": sphere,  # ฟังก์ชันคำนวณ cost
        "n_var": 5,  # จำนวนตัวแปร
        "var_min": [-10, -10, -1, -5, 4],
        "var_max": [10, 10, 1, 5, 10],
    }

    # -------------------------------
    # 2) GA Parameters
    # -------------------------------
    params = {
        "max_iter": 100,  # จำนวน iteration
        "pop_size": 50,  # ขนาดประชากร
        "beta": 1.0,  # ควบคุม selection pressure
        "pc": 1.0,  # crossover probability
        "gamma": 0.1,  # parameter สำหรับ crossover
        "mu": 0.01,  # mutation rate
        "sigma": 0.1,  # parameter สำหรับ mutation
    }

    # -------------------------------
    # 3) Run GA
    # -------------------------------
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    # -------------------------------
    # 4) Plot Results
    # -------------------------------
    best_cost = output["best_cost"]
    plt.figure()
    plt.plot(best_cost, "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("Genetic Algorithm")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
