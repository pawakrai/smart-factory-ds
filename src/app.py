import numpy as np
import matplotlib.pyplot as plt

# =============== CONFIG ==================

SLOT_DURATION = 30  # 1 slot = 30 นาที
TOTAL_SLOTS = 1440 // SLOT_DURATION  # 1 วัน = 48 slots ถ้า 30 นาที/slot

T_MELT = 3  # Melting 3 slot = 90 นาที

# กำหนดว่าเตาไหนใช้งานได้บ้าง
USE_FURNACE_A = True
USE_FURNACE_B = True  # ตัวอย่าง: ปิดเตา B

# เตา M&H (โค้ดเดิมไว้ plot)
MH_CAPACITY_MAX = 500.0
MH_CONSUMPTION_RATE = 5.56
REFILL_THRESHOLD = 0.0

SHIFT_START = 9 * 60


def scheduling_cost(x):
    """
    x: chromosome => [start_1, furnace_1, start_2, furnace_2, ...]
    แต่ละ batch ใช้ T_MELT slot (Melting only)

    แนวทางที่ 2:
      - ถ้าเตาไหน "ปิด" → ถ้า chromosome เลือกเตานั้น => penalty สูง
      - ถ้าเตาไหน "เปิด" → ห้ามมี overlap ภายในเตานั้น
      - (ถ้าเปิดทั้ง A และ B) → ตามตัวอย่างนี้ จะยังไม่ลงโทษ cross-furnace
        (แต่คุณสามารถเพิ่มเองได้ง่าย ๆ ถ้าต้องการ)
    """
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        f = int(round(x[2 * i + 1]))

        start = max(0, min(start, TOTAL_SLOTS - T_MELT))
        end = start + T_MELT

        schedule.append((start, end, f, i + 1))

    penalty = 0

    # (1) ลงโทษถ้าเลือกเตาที่ปิด
    for s, e, f, b_id in schedule:
        if f == 0 and not USE_FURNACE_A:  # ถ้า furnace=0 แต่ปิด => penalty
            penalty += 1e12
        if f == 1 and not USE_FURNACE_B:  # ถ้า furnace=1 แต่ปิด => penalty
            penalty += 1e12

    # (2) ห้ามซ้อนภายในเตาที่ "เปิด" โดยใช้ usage_for_furnace
    usage_for_furnace = {0: [0] * TOTAL_SLOTS, 1: [0] * TOTAL_SLOTS}  # A  # B

    for s, e, f, b_id in schedule:
        for t in range(s, e):
            usage_for_furnace[f][t] += 1

    # เตา A เปิด => ถ้าพบ usage_for_furnace[0][t] > 1 => penalize
    if USE_FURNACE_A:
        for t in range(TOTAL_SLOTS):
            if usage_for_furnace[0][t] > 1:
                penalty += (usage_for_furnace[0][t] - 1) * 1e9

    # เตา B เปิด => ถ้าพบ usage_for_furnace[1][t] > 1 => penalize
    if USE_FURNACE_B:
        for t in range(TOTAL_SLOTS):
            if usage_for_furnace[1][t] > 1:
                penalty += (usage_for_furnace[1][t] - 1) * 1e9

    # Global melt check
    global_usage = [0] * TOTAL_SLOTS
    for s, e, f, b_id in schedule:
        for t in range(s, e):
            global_usage[t] += 1

    # ถ้าเปิดทั้ง A,B => ห้าม global_usage[t] > 1
    if USE_FURNACE_A and USE_FURNACE_B:
        for t in range(TOTAL_SLOTS):
            if global_usage[t] > 1:
                penalty += (global_usage[t] - 1) * 1e9

    # makespan
    makespan_slot = max(e for (s, e, f, b_id) in schedule)
    total_time = makespan_slot * SLOT_DURATION
    cost = total_time + penalty

    return cost


def decode_schedule(best_sol):
    x = best_sol["position"]
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        f = int(round(x[2 * i + 1]))
        start = max(0, min(start, TOTAL_SLOTS - T_MELT))
        end = start + T_MELT
        schedule.append((start, end, f, i + 1))
    schedule.sort(key=lambda s: s[0])
    return schedule


# -------------- Plot ส่วนที่เหลือเหมือนเดิม ----------------
def simulate_mh_capacity():
    t = np.arange(0, 1441)
    cap = MH_CAPACITY_MAX
    capacity = np.zeros_like(t, dtype=float)
    for i in range(len(t)):
        capacity[i] = cap
        cap -= MH_CONSUMPTION_RATE
        if cap < 0:
            cap = MH_CAPACITY_MAX
    return t, capacity


def plot_schedule_and_mh(schedule):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))

    furnace_y = {0: 10, 1: 25}
    height = 8

    # ---------------------------
    # พล็อต Gantt chart (ax1)
    # ---------------------------
    for start, end, f, b_id in schedule:
        melt_start_t = SHIFT_START + start * SLOT_DURATION
        melt_dur = (end - start) * SLOT_DURATION
        ax1.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors="tab:blue",
            edgecolor="black",
        )
        ax1.text(
            melt_start_t + melt_dur / 2,
            furnace_y[f] + height / 2,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    ax1.set_xlim(SHIFT_START, SHIFT_START + 1440)

    # สร้าง X-ticks หลัก รายชั่วโมง (60 นาที)
    xticks = np.arange(SHIFT_START, SHIFT_START + 1441, 60)
    ax1.set_xticks(xticks)
    xlabels = []
    for x in xticks:
        hr = (x // 60) % 24
        mn = x % 60
        xlabels.append(f"{hr:02d}:{mn:02d}")
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Time (HH:MM)")

    # ตั้งค่าแกน Y ให้มี Furnace A, B เสมอ
    ax1.set_ylabel("Furnace")
    ax1.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax1.set_yticklabels(["Furnace A", "Furnace B"])
    ax1.set_title("Melting Schedule")

    # เปิด Grid เส้นแนวนอนตามปกติ
    ax1.grid(True, axis="y", linestyle="-", alpha=0.5)

    # ➊ เพิ่มเส้นตั้งทุก slot เพื่อให้เห็นช่วง slot ชัด
    #    สมมติว่ามี TOTAL_SLOTS และ SLOT_DURATION เป็น global หรือ import

    for slot_i in range(TOTAL_SLOTS + 1):
        line_x = SHIFT_START + slot_i * SLOT_DURATION
        ax1.axvline(x=line_x, color="gray", alpha=0.3, linestyle="--")

    # ---------------------------
    # subplot ล่าง (ax2) - M&H consumption
    # ---------------------------
    t, capacity = simulate_mh_capacity()
    t_shifted = t + SHIFT_START
    ax2.plot(t_shifted, capacity, "r-", linewidth=2, label="M&H Capacity")

    ax2.set_xlim(SHIFT_START, SHIFT_START + 1440)
    ax2.set_ylim(0, MH_CAPACITY_MAX * 1.1)
    ax2.set_xlabel("Time (HH:MM)")
    ax2.set_ylabel("Metal in M&H (kg)", color="r")

    # xticks ราย ชม.
    xticks2 = np.arange(SHIFT_START, SHIFT_START + 1441, 60)
    ax2.set_xticks(xticks2)
    xlabels2 = []
    for x in xticks2:
        hr = (x // 60) % 24
        mn = x % 60
        xlabels2.append(f"{hr:02d}:{mn:02d}")
    ax2.set_xticklabels(xlabels2)
    ax2.set_title("M&H Consumption / Capacity")
    ax2.grid(True, which="major", axis="both", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# =========== GA Setup =================
from src.ga.ga import GeneticAlgorithm

num_batches = 14
problem = {
    "cost_func": scheduling_cost,
    "n_var": num_batches * 2,
    "var_min": [0 if i % 2 == 0 else 0 for i in range(num_batches * 2)],
    "var_max": [TOTAL_SLOTS if i % 2 == 0 else 1 for i in range(num_batches * 2)],
}

params = {
    "max_iter": 100,
    "pop_size": 50,
    "beta": 0.6,
    "pc": 0.9,
    "gamma": 0.2,
    "mu": 0.7,
    "sigma": 4,
}


def main():
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    # plot best cost
    plt.figure()
    plt.plot(output["best_cost"], "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("GA for Single/Multiple Furnace Overlap (Approach #2)")
    plt.grid(True)
    plt.show()

    best_schedule = decode_schedule(output["best_sol"])
    print("Best Schedule:")
    for s, e, f, b_id in best_schedule:
        furnace_str = "A" if f == 0 else "B"
        print(f"Batch {b_id}: start={s}, end={e}, Furnace={furnace_str}")

    plot_schedule_and_mh(best_schedule)


if __name__ == "__main__":
    main()
