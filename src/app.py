import numpy as np
import matplotlib.pyplot as plt
from src.ga.ga import GeneticAlgorithm
from matplotlib.patches import Patch


# ----------------------------
# CONFIG
T_LOAD = 1   # Loading phase (1 slot = 30 นาที)
T_MELT = 3   # Melting phase (3 slots = 90 นาที)
# ถ้าอยากมี POURING อีก ก็เพิ่ม T_POUR = 1 ฯลฯ ได้

def scheduling_cost(x):
    """
    x: chromosome ขนาด 2*n
       x[2*i]   = start_slot
       x[2*i+1] = furnace (0=A, 1=B)

    เงื่อนไข:
     - Melting ข้ามเตาไม่ได้พร้อมกัน (global)
     - เตาเดียวกัน ห้าม Loading ทับ Melting
     - Loading + Loading ข้ามเตา หรือข้าม batch ได้
     - Loading + Melting ข้ามเตาได้
    """
    n = len(x) // 2
    penalty = 0

    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        f = int(round(x[2 * i + 1]))
        # กันขอบ
        start = max(0, min(start, 48 - (T_LOAD + T_MELT)))

        load_start = start
        load_end   = start + T_LOAD
        melt_start = load_end
        melt_end   = load_end + T_MELT
        schedule.append((load_start, load_end, melt_start, melt_end, f, i+1))

    # 1) Array ป้องกัน Melting ซ้อนกันข้ามเตา
    global_meltdown_usage = [0]*48

    # 2) Array ป้องกันเตาเดียวกัน Loading+Melting ทับกัน
    usage_for_furnace = {
        0: [0]*48,  # Furnace A
        1: [0]*48   # Furnace B
    }

    for (ls, le, ms, me, furnace, b_id) in schedule:
        # --- Global meltdown check ---
        for t in range(ms, me):
            if 0 <= t < 48:
                global_meltdown_usage[t] += 1

        # --- Furnace-specific usage ---
        # Loading
        for t in range(ls, le):
            if 0 <= t < 48:
                usage_for_furnace[furnace][t] += 1
        # Melting
        for t in range(ms, me):
            if 0 <= t < 48:
                usage_for_furnace[furnace][t] += 1

    # เช็ค penalty:
    # A) Melting ข้ามเตา
    for t in range(48):
        # ถ้า global meltdown usage > 1 => มี 2 เตาละลายพร้อมกัน
        if global_meltdown_usage[t] > 1:
            penalty += (global_meltdown_usage[t] - 1) * 1e9

    # B) Loading + Melting ในเตาเดียวกัน
    for f in [0, 1]:
        for t in range(48):
            if usage_for_furnace[f][t] > 1:
                # มากกว่า 1 แสดงว่าเตานี้มีอย่างน้อย Loading และ Melting ทับกัน
                penalty += (usage_for_furnace[f][t] - 1) * 1e9

    # คำนวณ makespan
    makespan = max(me for (_,_,_,me,_,_) in schedule)
    total_time = makespan * 30
    cost = total_time + penalty
    return cost



# ----- Problem Definition -----
num_batches = 10
problem = {
    "cost_func": scheduling_cost,
    # 2 ตัวแปร/Batch
    "n_var": num_batches * 2,
    # สมมติให้ x[2*i] อยู่ได้ถึง 48 - (T_LOAD+T_MELT) = 48 - 4 = 44 ก็พอ
    # แต่เพื่อความยืดหยุ่น ก็ใส่ 48 ไปเลยแล้วเดี๋ยวใน cost function เช็คขอบ
    "var_min": [0 if i % 2 == 0 else 0 for i in range(num_batches * 2)],
    "var_max": [48 if i % 2 == 0 else 1 for i in range(num_batches * 2)],
}

# ----- GA Parameters -----
params = {
    "max_iter": 1000,
    "pop_size": 200,
    "beta": 0.6,   # ลดแรงกดดันจาก cost
    "pc": 0.9,
    "gamma": 0.2,
    "mu": 0.5,
    "sigma": 4,
}


def decode_schedule(best_sol):
    """
    เปลี่ยนให้สอดคล้องกับ multi-phase
    คืนค่า (load_start, load_end, melt_start, melt_end, furnace, batch_id)
    """
    x = best_sol["position"]
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start = int(round(x[2 * i]))
        furnace = int(round(x[2 * i + 1]))
        start = max(0, min(start, 48 - (T_LOAD + T_MELT)))
        
        load_start = start
        load_end   = start + T_LOAD
        melt_start = load_end
        melt_end   = load_end + T_MELT
        schedule.append((load_start, load_end, melt_start, melt_end, furnace, i+1))

    # เรียงตาม melt_start เพื่อดูง่าย
    schedule.sort(key=lambda s: s[2])  # sort by melt_start
    return schedule

def plot_schedule(schedule):
    """
    schedule: list of tuples (load_start, load_end, melt_start, melt_end, furnace, batch_id)
    1 slot = 30 นาที

    ต้องการแสดงเวลาเริ่มที่ 9:00 (นับเป็นจุดเริ่มของกราฟ)
    และลากยาวจนถึงอีก 24 ชั่วโมง (9:00 ของวันถัดไป) = 1980 นาทีจากเที่ยงคืน
    """

    # นิยามจุดเริ่มที่ 9:00 (540 นาทีจากเที่ยงคืน)
    SHIFT_START = 9 * 60      # 540
    SHIFT_END   = SHIFT_START + 1440  # 540 + 1440 = 1980 (9:00 ของวันถัดไป)

    furnace_y = {0: 10, 1: 30}
    height = 8

    fig, ax = plt.subplots(figsize=(12, 6))

    for (ls, le, ms, me, furnace, b_id) in schedule:
        # คำนวณเวลาเริ่ม-สิ้นสุดในหน่วย "นาทีจากเที่ยงคืน" แต่เราจะ shift ไปเริ่มที่ 9:00
        load_start_time = SHIFT_START + (ls * 30)
        load_dur        = (le - ls) * 30

        melt_start_time = SHIFT_START + (ms * 30)
        melt_dur        = (me - ms) * 30

        # Loading phase (สีเทาอ่อน)
        ax.broken_barh(
            [(load_start_time, load_dur)],
            (furnace_y[furnace], height),
            facecolors="lightgray",
            alpha=0.4,
            edgecolor="black"
        )

        # Melting phase (สีน้ำเงิน)
        ax.broken_barh(
            [(melt_start_time, melt_dur)],
            (furnace_y[furnace], height),
            facecolors="tab:blue",
            alpha=1.0,
            edgecolor="black"
        )

        # Label batch ตรงกลาง bar (Melting)
        ax.text(
            melt_start_time + melt_dur / 2,
            furnace_y[furnace] + height / 2,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    # กำหนดแกน X ให้เริ่มที่ 9:00 (540) จนถึง 9:00 ของวันถัดไป (1980)
    ax.set_xlim(SHIFT_START, SHIFT_END)

    # สร้าง xticks ทุก 60 นาที (1 ชม) ในช่วง 9:00 -> 9:00 ของวันถัดไป
    xticks = np.arange(SHIFT_START, SHIFT_END + 1, 60)
    ax.set_xticks(xticks)

    # แปลงเป็น label "HH:MM" ตามเวลาจริง
    xlabels = []
    for x in xticks:
        # แปลง x (นาทีจากเที่ยงคืน) เป็นชั่วโมง:นาที
        hr = x // 60
        mn = x % 60
        # ex. 540 = 9*60 => hr=9, mn=0 => "09:00"
        # แต่ถ้าเป็น 33:00 => hr=33 => hr%24 = 9 => "09:00" ของวันถัดไป
        hr_mod = hr % 24
        xlabels.append(f"{hr_mod:02d}:{mn:02d}")
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Furnace")

    # ปรับแกน Y ให้เป็นชื่อเตา
    ax.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax.set_yticklabels(["Furnace A", "Furnace B"])

    ax.set_title("Melting Schedule (starting at 09:00)")

    # Grid + Legend
    ax.grid(True)
    legend_elements = [
        Patch(facecolor="lightgray", alpha=0.4, edgecolor="black", label="Loading"),
        Patch(facecolor="tab:blue", alpha=1.0, edgecolor="black", label="Melting"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()


# def plot_schedule(schedule):
#     """
#     Plot Gantt แบบ 2 phase:
#       - Loading (สีจาง)
#       - Melting (สีเข้ม)  # ตัวอย่างเปลี่ยน facecolors ได้ถ้าอยากต่างสี
#     """
#     fig, ax = plt.subplots(figsize=(12, 6))
#     furnace_y = {0: 10, 1: 30}
#     height = 8

#     # วนดูแต่ละ batch
#     for (ls, le, ms, me, furnace, b_id) in schedule:
#         # Loading
#         load_start_time = ls * 30
#         load_dur = (le - ls) * 30
#         ax.broken_barh(
#             [(load_start_time, load_dur)],
#             (furnace_y[furnace], height),
#             facecolors="tab:gray",  # สีเทาอ่อน
#         )
#         # Melting
#         melt_start_time = ms * 30
#         melt_dur = (me - ms) * 30
#         ax.broken_barh(
#             [(melt_start_time, melt_dur)],
#             (furnace_y[furnace], height),
#             facecolors="tab:blue",  # สีฟ้า
#         )
#         # เขียนชื่อ batch ตรงกลางของ Melting
#         ax.text(
#             melt_start_time + melt_dur / 2,
#             furnace_y[furnace] + height / 2,
#             f"{b_id}",
#             ha="center",
#             va="center",
#             color="white",
#             fontsize=10,
#         )
#     ax.set_xlabel("Time (minutes from midnight)")
#     ax.set_ylabel("Furnace")
#     ax.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
#     ax.set_yticklabels(["Furnace A", "Furnace B"])
#     ax.set_title("Melting Schedule (Loading + Melting phases)")
#     ax.set_xlim(0, 1440)
#     ax.grid(True)
#     plt.show()


def main():
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    plt.figure()
    plt.plot(output["best_cost"], "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title("GA for Induction Furnace Scheduling (Multi-phase)")
    plt.grid(True)
    plt.show()

    best_schedule = decode_schedule(output["best_sol"])
    print("Best Schedule:")
    for (ls, le, ms, me, furnace, b_id) in best_schedule:
        print(f"Batch {b_id}: LOAD=({ls}-{le})  MELT=({ms}-{me})  Furnace={'A' if furnace==0 else 'B'}")

    plot_schedule(best_schedule)


if __name__ == "__main__":
    main()
