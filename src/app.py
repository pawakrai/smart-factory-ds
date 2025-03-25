import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.ga.ga import GeneticAlgorithm

# =============== CONFIG ==================

# 1) ตั้ง SLOT_DURATION (นาทีต่อ slot)
SLOT_DURATION = 30  # ปรับได้ 10,20,30 ฯลฯ

# 2) คำนวณ total slots ใน 1 วัน (1440 นาที)
TOTAL_SLOTS = 1440 // SLOT_DURATION  # เช่น ถ้า SLOT_DURATION=20 => TOTAL_SLOTS=72

# 3) กำหนดระยะเวลาของแต่ละ phase (หน่วยเป็น "slot")
T_LOAD = 1
T_MELT = 4
T_POUR = 1

# =========================================


def scheduling_cost(x):
    """
    x: chromosome ขนาด 2*n
       x[2*i]   = start_slot
       x[2*i+1] = furnace (0=A, 1=B)

    เฟส:
      - Loading (T_LOAD slots)
      - Melting (T_MELT slots)
      - Pouring (T_POUR slots)

    เงื่อนไข:
      - เตาเดียวกัน: ห้าม 2 phase ใด ๆ ซ้อนทับกันในเวลาเดียวกัน
      - Melting ระหว่างเตาต่างกัน ห้ามซ้อนเพราะไฟไม่พอ
      - Loading/Pouring คนละเตา ซ้อนได้
    """
    n = len(x) // 2
    penalty = 0

    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        f = int(round(x[2 * i + 1]))

        # กันขอบไม่ให้ start เกินช่วง
        # รวมทั้งหมด T_LOAD + T_MELT + T_POUR
        total_phase = T_LOAD + T_MELT + T_POUR
        start = max(0, min(start, TOTAL_SLOTS - total_phase))

        load_start = start
        load_end = start + T_LOAD

        melt_start = load_end
        melt_end = melt_start + T_MELT

        pour_start = melt_end
        pour_end = pour_start + T_POUR

        schedule.append(
            (load_start, load_end, melt_start, melt_end, pour_start, pour_end, f, i + 1)
        )

    # 1) ตรวจ Melting ข้ามเตา
    meltdown_usage = [0] * TOTAL_SLOTS

    # 2) ตรวจเตาเดียวกัน (loading+melting+pouring)
    usage_for_furnace = {
        0: [0] * TOTAL_SLOTS,  # Furnace A
        1: [0] * TOTAL_SLOTS,  # Furnace B
    }

    # เติมข้อมูล
    for ls, le, ms, me, ps, pe, furnace, b_id in schedule:
        # global meltdown
        for t in range(ms, me):
            if 0 <= t < TOTAL_SLOTS:
                meltdown_usage[t] += 1

        # furnace usage
        for t in range(ls, le):  # Loading
            if 0 <= t < TOTAL_SLOTS:
                usage_for_furnace[furnace][t] += 1
        for t in range(ms, me):  # Melting
            if 0 <= t < TOTAL_SLOTS:
                usage_for_furnace[furnace][t] += 1
        for t in range(ps, pe):  # Pouring
            if 0 <= t < TOTAL_SLOTS:
                usage_for_furnace[furnace][t] += 1

    # ลงโทษ Melting พร้อมกัน >1 เตา
    for t in range(TOTAL_SLOTS):
        if meltdown_usage[t] > 1:
            penalty += (meltdown_usage[t] - 1) * 1e14

    # ลงโทษเตาเดียวกันมีการซ้อน (เฟสไหนก็ได้)
    for f in [0, 1]:
        for t in range(TOTAL_SLOTS):
            if usage_for_furnace[f][t] > 1:
                penalty += (usage_for_furnace[f][t] - 1) * 1e14

    # makespan = slot สิ้นสุดของ Pour
    makespan = 0
    for _, _, _, _, _, pour_end, _, _ in schedule:
        makespan = max(makespan, pour_end)

    # แปลงเป็นนาที (makespan * SLOT_DURATION)
    total_time = makespan * SLOT_DURATION
    cost = total_time + penalty
    return cost


def decode_schedule(best_sol):
    """
    แปลง chromosome -> schedule
    คืน (ls, le, ms, me, ps, pe, furnace, batch_id)
    """
    x = best_sol["position"]
    n = len(x) // 2

    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        f = int(round(x[2 * i + 1]))

        total_phase = T_LOAD + T_MELT + T_POUR
        start = max(0, min(start, TOTAL_SLOTS - total_phase))

        load_start = start
        load_end = start + T_LOAD

        melt_start = load_end
        melt_end = melt_start + T_MELT

        pour_start = melt_end
        pour_end = pour_start + T_POUR

        schedule.append(
            (load_start, load_end, melt_start, melt_end, pour_start, pour_end, f, i + 1)
        )

    # sort ตาม melt_start
    schedule.sort(key=lambda s: s[2])
    return schedule


def plot_schedule(schedule):
    """
    plot 3 เฟส: Loading (เทาอ่อน), Melting (ฟ้า), Pouring (เทาเข้ม)
    X-axis: เริ่ม 9:00 (540) -> 9:00 + 24 ชม = 540 + 1440 = 1980
    แต่จะใช้ TOTAL_SLOTS * SLOT_DURATION => ถ้า slot=72, slot_duration=20 => 72*20=1440
    """
    SHIFT_START = 9 * 60  # เริ่ม 9:00 AM
    # สร้าง x-axis ช่วง 1 วัน = SHIFT_START -> SHIFT_START + 1440
    # แต่จริงๆ ถ้า GA ใช้ไม่ถึงก็ไม่เป็นไร เราเผื่อ 1 วัน

    fig, ax = plt.subplots(figsize=(12, 6))
    height = 8

    furnace_y = {0: 10, 1: 20}

    for ls, le, ms, me, ps, pe, f, b_id in schedule:
        # แปลง slot->นาที
        load_start_t = SHIFT_START + ls * SLOT_DURATION
        load_dur = (le - ls) * SLOT_DURATION

        melt_start_t = SHIFT_START + ms * SLOT_DURATION
        melt_dur = (me - ms) * SLOT_DURATION

        pour_start_t = SHIFT_START + ps * SLOT_DURATION
        pour_dur = (pe - ps) * SLOT_DURATION

        # Loading = lightgray
        ax.broken_barh(
            [(load_start_t, load_dur)],
            (furnace_y[f], height),
            facecolors="lightgray",
            alpha=0.4,
            edgecolor="black",
        )
        # Melting = tab:blue
        ax.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors="tab:blue",
            alpha=1.0,
            edgecolor="black",
        )
        # Pouring = dimgray
        ax.broken_barh(
            [(pour_start_t, pour_dur)],
            (furnace_y[f], height),
            facecolors="dimgray",
            alpha=0.8,
            edgecolor="black",
        )

        # label batch ตรงกลาง Melting
        ax.text(
            melt_start_t + melt_dur / 2,
            furnace_y[f] + height / 2,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    # ตั้งค่า X limit = 9:00 -> 9:00 + 1440 (อีก 24 ชม)
    ax.set_xlim(SHIFT_START, SHIFT_START + 1440)

    # ตั้ง Tick ทุก 60 นาที
    import numpy as np

    xticks = np.arange(SHIFT_START, SHIFT_START + 1440 + 1, 60)
    ax.set_xticks(xticks)
    xlabels = []
    for x in xticks:
        hr = (x // 60) % 24
        mn = x % 60
        xlabels.append(f"{hr:02d}:{mn:02d}")
    ax.set_xticklabels(xlabels)

    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Furnace")

    # Y-axis
    ax.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax.set_yticklabels(["Furnace A", "Furnace B"])

    ax.set_title(f"Melting Schedule (Slot={SLOT_DURATION} min)")

    ax.grid(True)
    legend_elements = [
        Patch(facecolor="lightgray", alpha=0.4, edgecolor="black", label="Loading"),
        Patch(facecolor="tab:blue", alpha=1.0, edgecolor="black", label="Melting"),
        Patch(facecolor="dimgray", alpha=0.8, edgecolor="black", label="Pouring"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.show()


# -------- Problem Definition --------
num_batches = 11
problem = {
    "cost_func": scheduling_cost,
    "n_var": num_batches * 2,
    "var_min": [0 if i % 2 == 0 else 0 for i in range(num_batches * 2)],
    "var_max": [TOTAL_SLOTS if i % 2 == 0 else 1 for i in range(num_batches * 2)],
}


# ------- GA Parameters ------------
params = {
    "max_iter": 2000,
    "pop_size": 200,
    "beta": 0.6,  # Probability of selection
    "pc": 0.9,  # Crossover Probability (ค่านี้ใกล้ 1 หมายถึง แทบทุกคู่ของพ่อแม่ (parents) จะมีการ crossover)
    "gamma": 0.2,  # Crossover Ratio / Blend Parameter
    "mu": 0.7,  # Mutation Rate
    "sigma": 4,  # Mutation Scale (ถ้ามาก กระโดดเยอะ ไม่ stable ถ้าน้อยอาจจะไม่หลุด local min)
}


def main():
    ga_engine = GeneticAlgorithm(problem, params)
    output = ga_engine.run()

    plt.figure()
    plt.plot(output["best_cost"], "b-", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.title(f"GA for Induction Furnace (Slot={SLOT_DURATION} min)")
    plt.grid(True)
    plt.show()

    best_schedule = decode_schedule(output["best_sol"])
    print("Best Schedule:")
    for ls, le, ms, me, ps, pe, f, b_id in best_schedule:
        print(
            f"Batch {b_id}: "
            f"LOAD=({ls}-{le}) MELT=({ms}-{me}) POUR=({ps}-{pe}) "
            f"Furnace={'A' if f==0 else 'B'}"
        )

    plot_schedule(best_schedule)


if __name__ == "__main__":
    main()
