# app.py
import numpy as np
import matplotlib.pyplot as plt
from src.ga.ga import GeneticAlgorithm

# ----- Define the Scheduling Cost Function -----
def scheduling_cost(x):
    n = len(x) // 2
    slot_usage = [0] * 48
    penalty = 0

    # เติม slot usage
    for i in range(n):
        start_slot = int(x[2 * i])
        start_slot = max(0, min(start_slot, 48 - 3))
        for t in range(start_slot, start_slot + 3):
            if t < 48:
                slot_usage[t] += 1

    # นับ penalty ถ้ามี slot ไหนใช้มากกว่า 1 เตา
    for s in slot_usage:
        if s > 1:
            penalty += (s - 1) * 1e9  # ใช้ penalty ใหญ่แต่ไม่ infinite

    # คำนวณ cost ปกติ
    makespan = max(int(x[2 * i]) + 3 for i in range(n))
    total_time = makespan * 30
    energy_consumption = total_time

    return 0.5 * total_time + 0.5 * energy_consumption + penalty


# ----- Problem Definition -----
num_batches = 14  # เพิ่มจำนวน batch เป็น 14
problem = {
    "cost_func": scheduling_cost,
    "n_var": num_batches * 2,  # 2 variables per batch
    # For start_slot (indices 0,2,4,...): range [0, 48]
    # For furnace assignment (indices 1,3,5,...): range [0, 1]
    "var_min": [0 if i % 2 == 0 else 0 for i in range(num_batches * 2)],
    "var_max": [48 if i % 2 == 0 else 1 for i in range(num_batches * 2)],
}

# ----- GA Parameters -----
params = {
    "max_iter": 3000,
    "pop_size": 100,
    "beta": 0.8,  # ลดแรงกดดันจาก cost
    "pc": 0.9,  # เพิ่ม crossover
    "gamma": 0.2,
    "mu": 0.4,
    "sigma": 3,  # เพิ่ม mutation range
}


def decode_schedule(best_sol):
    """
    Decode best solution chromosome into a schedule list.
    Each gene: (start_slot, end_slot, furnace_assignment, batch_id)
    """
    x = best_sol["position"]
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start_slot = int(round(x[2 * i]))
        start_slot = max(0, min(start_slot, 48 - 3))  # กันหลุดขอบ
        end_slot = start_slot + 3

        # ให้แน่ใจว่า furnace เป็น 0 หรือ 1
        furnace = int(round(x[2 * i + 1]))
        if furnace not in [0, 1]:
            furnace = 0 if x[2 * i + 1] < 0.5 else 1

        schedule.append((start_slot, end_slot, furnace, i + 1))

    # ไม่จำเป็นต้อง sort แล้วก็ได้ แต่ถ้าชอบให้เรียงเวลา:
    schedule.sort(key=lambda gene: gene[0])
    return schedule


def plot_schedule(schedule):
    """
    Plot a Gantt chart for the schedule.
    Assumes:
      - Each schedule entry = (start_slot, end_slot, furnace, batch_id)
      - Each batch takes 3 slots = 90 minutes
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    furnace_y = {0: 10, 1: 30}
    height = 8
    for start_slot, end_slot, furnace, batch_id in schedule:
        # คำนวณเวลาเริ่มในนาที (เริ่มนับจากเที่ยงคืน)
        start_time = start_slot * 30
        duration = (end_slot - start_slot) * 30
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
    ax.set_xlabel("Time (minutes from midnight)")
    ax.set_ylabel("Furnace")
    ax.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax.set_yticklabels(["Furnace A", "Furnace B"])
    ax.set_title("Melting Schedule Gantt Chart")
    ax.set_xlim(0, 1440)
    ax.grid(True)
    plt.show()


def main():
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    plt.figure()
    plt.plot(output["best_cost"], "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("GA for Induction Furnace Scheduling")
    plt.grid(True)
    plt.show()

    best_schedule = decode_schedule(output["best_sol"])
    print("Best Schedule:")
    for gene in best_schedule:
        print(
            f"Batch {gene[3]}: Start Slot = {gene[0]}, Furnace = {'A' if gene[2]==0 else 'B'}"
        )

    plot_schedule(best_schedule)


if __name__ == "__main__":
    main()
