# app.py

import numpy as np
import matplotlib.pyplot as plt
from src.ga.ga import GeneticAlgorithm


# ----- Define the Scheduling Cost Function -----
def scheduling_cost(x):
    """
    x: numpy array of dimension 2*n, where n is the number of batches.
       For each batch i:
         x[2*i]   : start slot (continuous value, rounded to integer)
         x[2*i+1] : furnace assignment (continuous value in [0,1], round to 0 or 1)

    Assumptions:
      - Each batch takes 3 consecutive slots (i.e., 90 minutes, since 1 slot = 30 minutes).
      - Production horizon: 1 day = 1440 minutes => max slot = 1440/30 = 48.
      - Constraint: "No overlap" among all batches (ไม่ให้เตา 2 เตาทำงานพร้อมกัน).
      - If overlap is found, add a large penalty (e.g., 1e6).
    """
    n = int(len(x) / 2)
    schedule = []
    penalty = 0

    # 1) Decode chromosome -> schedule list
    for i in range(n):
        # Round start slot to integer and clamp between 0 and 48
        start_slot = int(round(x[2 * i]))
        start_slot = max(0, min(start_slot, 48))
        # Round furnace assignment: <0.5 -> 0 (Furnace A), >=0.5 -> 1 (Furnace B)
        furnace = 0 if x[2 * i + 1] < 0.5 else 1
        schedule.append((start_slot, furnace, i + 1))

    # 2) Sort schedule by start_slot (ascending)
    schedule.sort(key=lambda gene: gene[0])

    # 3) Check overlap among ALL pairs (i, j)
    #    Each batch uses 3 slots, so finish_slot = start_slot + 3
    for i in range(n):
        start_i = schedule[i][0]
        finish_i = start_i + 3
        for j in range(i + 1, n):
            start_j = schedule[j][0]
            finish_j = start_j + 3
            # if intervals [start_i, finish_i) and [start_j, finish_j) overlap
            if (start_i < finish_j) and (start_j < finish_i):
                penalty += 1e6

    # 4) Makespan = finish time of the last batch (in slots)
    makespan = schedule[-1][0] + 3
    # total_time in minutes
    total_time = makespan * 30
    # assume energy consumption = total_time
    energy_consumption = total_time

    # 5) Overall cost = weighted sum + penalty
    cost = 0.5 * total_time + 0.5 * energy_consumption + penalty
    return cost


# ----- Problem Definition -----
num_batches = 10
problem = {
    "cost_func": scheduling_cost,
    "n_var": num_batches * 2,  # 2 variables per batch
    # For start_slot (indices 0,2,4,...): range [0, 48]
    # For furnace assignment (indices 1,3,5,...): range [0, 1]
    "var_min": [0] * num_batches + [0] * num_batches,
    "var_max": [48] * num_batches + [1] * num_batches,
}

# ----- GA Parameters -----
params = {
    "max_iter": 100,
    "pop_size": 50,
    "beta": 1.0,
    "pc": 0.7,
    "gamma": 0.1,
    "mu": 0.01,
    "sigma": 0.1,
}


# ----- Function to Decode Best Solution into a Schedule -----
def decode_schedule(best_sol):
    """
    Decode best solution chromosome into a schedule list.
    Each gene: (start_slot, furnace_assignment, batch_id)
    """
    x = best_sol["position"]
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start_slot = int(round(x[2 * i]))
        start_slot = max(0, min(start_slot, 48))
        furnace = 0 if x[2 * i + 1] < 0.5 else 1
        schedule.append((start_slot, furnace, i + 1))
    # Sort schedule by start_slot
    schedule.sort(key=lambda gene: gene[0])
    return schedule


# ----- Function to Plot the Schedule as a Gantt Chart -----
def plot_schedule(schedule):
    """
    Plot a Gantt chart for the schedule.
    Assumes:
      - 1 slot = 30 minutes.
      - Each batch duration = 3 slots = 90 minutes.
      - X-axis represents a full day (0 to 1440 minutes).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    # ตำแหน่ง Y สำหรับ Furnace A และ B
    furnace_y = {0: 10, 1: 30}
    height = 8
    for start_slot, furnace, batch_id in schedule:
        start_time = start_slot * 30  # แปลง slot เป็นนาที
        duration = 90  # 3 slots * 30 นาที
        ax.broken_barh(
            [(start_time, duration)],
            (furnace_y[furnace], height),
            facecolors="tab:blue",
        )
        ax.text(
            start_time + duration / 2,
            furnace_y[furnace] + height / 2,
            f"{batch_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Furnace")
    ax.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax.set_yticklabels(["Furnace A", "Furnace B"])
    ax.set_title("Melting Schedule Gantt Chart")
    ax.set_xlim(0, 1440)  # 0 ถึง 1440 นาที = 1 วัน
    ax.grid(True)
    plt.show()


def main():
    # Run GA
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    # Plot best cost history
    plt.figure()
    plt.plot(output["best_cost"], "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("GA for Induction Furnace Scheduling")
    plt.grid(True)
    plt.show()

    # Decode best solution and plot schedule
    best_schedule = decode_schedule(output["best_sol"])
    print("Best Schedule:")
    for gene in best_schedule:
        print(
            f"Batch {gene[2]}: Start Slot = {gene[0]}, Furnace = {'A' if gene[1] == 0 else 'B'}"
        )

    plot_schedule(best_schedule)


if __name__ == "__main__":
    main()
