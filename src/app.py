import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

# =============== CONFIG ==================
SLOT_DURATION = 30
TOTAL_SLOTS = 1440 // SLOT_DURATION
T_MELT = 3
USE_FURNACE_A = True
USE_FURNACE_B = True
MH_CAPACITY_MAX = 500.0
MH_CONSUMPTION_RATE = 5.56
SHIFT_START = 9 * 60
NUM_BATCHES = 14


def simulate_mh_consumption(makespan_min, water_events):
    # แค่ mock ไว้ก่อน
    MH_idle_penalty = 0.0
    MH_reheat_penalty = 0.0
    total_energy_mh = 0.0
    return MH_idle_penalty, MH_reheat_penalty, total_energy_mh


def scheduling_cost(x):
    """
    1) สร้างตาราง schedule ของ IF
    2) ตรวจ overlap / penalty IF
    3) ทำ mini-simulation M&H เพื่อคำนวณ Energy + penalty
    4) รวมทั้งหมดเป็น cost
    """

    # ----------------------------
    # 1) สร้างตาราง (start, end, furnace, batch_id)
    # ----------------------------
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        furnace = int(round(x[2 * i + 1]))

        # กันไม่ให้หลุด slot
        start = max(0, min(start, TOTAL_SLOTS - T_MELT))
        end = start + T_MELT
        schedule.append((start, end, furnace, i + 1))

    # ----------------------------
    # 2) ตรวจ Overlap IF -> penalty_if
    # ----------------------------
    # ตัวอย่างเช็คเหมือนเดิม:
    penalty_if = 0.0

    # (2.1) ลงโทษถ้าเตาไหนปิดแต่ x ระบุใช้เตานั้น
    for s, e, f, b_id in schedule:
        if f == 0 and not USE_FURNACE_A:
            penalty_if += 1e9
        if f == 1 and not USE_FURNACE_B:
            penalty_if += 1e9

    # (2.2) ห้ามซ้อนในเตาเดียวกัน
    usage_for_furnace = {0: [0] * TOTAL_SLOTS, 1: [0] * TOTAL_SLOTS}
    for s, e, f, b_id in schedule:
        for t in range(s, e):
            usage_for_furnace[f][t] += 1

    if USE_FURNACE_A:
        for t in range(TOTAL_SLOTS):
            if usage_for_furnace[0][t] > 1:
                penalty_if += (usage_for_furnace[0][t] - 1) * 1e9

    if USE_FURNACE_B:
        for t in range(TOTAL_SLOTS):
            if usage_for_furnace[1][t] > 1:
                penalty_if += (usage_for_furnace[1][t] - 1) * 1e9

    # (2.3) ถ้าเปิดทั้ง A,B => global check
    global_usage = [0] * TOTAL_SLOTS
    if USE_FURNACE_A and USE_FURNACE_B:
        for s, e, f, b_id in schedule:
            for t in range(s, e):
                global_usage[t] += 1
        for t in range(TOTAL_SLOTS):
            if global_usage[t] > 1:
                penalty_if += (global_usage[t] - 1) * 1e9

    # ----------------------------
    # 2.4) คำนวณ makespan slot => makespan (นาที)
    # ----------------------------
    makespan_slot = max(e for (s, e, f, b_id) in schedule) if schedule else 0
    makespan_min = makespan_slot * SLOT_DURATION

    # ----------------------------
    # 3) สร้าง water_events สำหรับ M&H
    # สมมติ 1 batch = 500 kg => มีน้ำหลอม 500 kg ทันทีที่ batch_i finish
    # (ถ้าคุณมี partial batch capacity จริง ๆ ก็ปรับ logic)
    # ----------------------------
    water_events = []  # list of (finish_minute, batch_id)
    for s, e, f, b_id in schedule:
        finish_minute = e * SLOT_DURATION
        water_events.append((finish_minute, b_id))

    water_events.sort(key=lambda w: w[0])  # เรียงตามเวลา

    # ----------------------------
    # 4) เรียก mini-simulation M&H
    # ----------------------------
    MH_idle_penalty, MH_reheat_penalty, total_energy_mh = simulate_mh_consumption(
        makespan_min, water_events
    )

    # ----------------------------
    # 5) รวม cost
    # ----------------------------
    # สมมติ IF กินพลังงาน => 2 kWh/min
    # (ปรับตามจริง)
    if_energy_rate = 2.0
    total_energy_if = if_energy_rate * makespan_min

    # final cost
    cost = 0.0
    cost += total_energy_if  # พลังงาน IF
    cost += total_energy_mh  # พลังงาน M&H
    cost += penalty_if  # overlap/ปิดเตา/ฯลฯ
    cost += MH_idle_penalty * 100  # สมมติ scale penalty idle
    cost += MH_reheat_penalty * 200

    return cost, makespan_min


# ===== NSGA-II Problem Class =====
class MeltingScheduleProblem(ElementwiseProblem):
    def __init__(self, num_batches):
        # Determine bounds based on problem logic (start times and furnace index 0 or 1)
        xl_list = []
        xu_list = []
        for i in range(num_batches):
            xl_list.extend(
                [0, 0]
            )  # Start slot lower bound 0, Furnace index lower bound 0
            # Ensure start time allows for T_MELT duration within TOTAL_SLOTS
            xu_list.extend(
                [TOTAL_SLOTS - T_MELT, 1]
            )  # Start slot upper bound, Furnace index upper bound 1

        super().__init__(
            n_var=num_batches * 2,
            n_obj=2,
            n_constr=0,
            xl=np.array(xl_list),
            xu=np.array(xu_list),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        energy, makespan = scheduling_cost(x)
        out["F"] = [energy, makespan]


# ===== Custom Sampling Class (ปรับปรุง) =====
class CustomInitialSampling(Sampling):
    """
    สร้างประชากรเริ่มต้นตามเตาที่ใช้งานได้:
    - ถ้าใช้ได้ 2 เตา: สุ่มเวลาเริ่มและสลับเตา A/B พยายามไม่ให้ซ้อนในเตาเดียวกัน
    - ถ้าใช้ได้ 1 เตา: สุ่มเวลาเริ่มในเตานั้น พยายามไม่ให้ซ้อนกัน
    """

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0.0)
        num_batches = problem.n_var // 2

        available_furnaces = []
        if USE_FURNACE_A:
            available_furnaces.append(0)
        if USE_FURNACE_B:
            available_furnaces.append(1)

        if not available_furnaces:
            # กรณีไม่มีเตาใช้งานเลย (ไม่ควรเกิด) อาจจะคืนค่าสุ่มหรือ error
            # ในที่นี้จะคืนค่าที่อาจไม่ valid เพื่อให้ penalty จัดการ
            print("Warning: No furnaces enabled for CustomInitialSampling!")
            return FloatRandomSampling()._do(problem, n_samples, **kwargs)

        only_one_furnace = len(available_furnaces) == 1
        furnace_to_use = available_furnaces[0] if only_one_furnace else None

        for i in range(n_samples):
            # เก็บ slot ที่ใช้ไปแล้วสำหรับแต่ละเตา ใน sample ปัจจุบัน
            used_slots = {f: [0] * TOTAL_SLOTS for f in available_furnaces}
            successful_generation = True

            for j in range(num_batches):
                # --- 1. กำหนดเตา (furnace_idx) ---
                if only_one_furnace:
                    furnace_idx = furnace_to_use
                else:
                    # สลับเตา A(0), B(1), A(0), ...
                    furnace_idx = available_furnaces[j % len(available_furnaces)]

                # --- 2. หาวันที่เริ่ม (start_slot) ที่ไม่ซ้อน ---
                found_slot = False
                max_attempts = 50  # จำนวนครั้งสูงสุดในการสุ่มหา slot ว่าง
                min_start_slot = 0
                max_start_slot = TOTAL_SLOTS - T_MELT

                # Ensure furnace_idx is valid before using it as a key
                if furnace_idx not in used_slots:
                    print(
                        f"Error: Invalid furnace index {furnace_idx} generated in sampling. Skipping batch."
                    )
                    # Handle error: skip this batch or assign default values
                    # For simplicity, let's assign defaults here, might lead to invalid solution
                    X[i, 2 * j] = 0
                    X[i, 2 * j + 1] = available_furnaces[
                        0
                    ]  # Assign to the first available furnace
                    successful_generation = False
                    continue  # Move to the next batch

                for attempt in range(max_attempts):
                    start_slot = np.random.randint(min_start_slot, max_start_slot + 1)
                    is_overlap = False
                    for t in range(start_slot, start_slot + T_MELT):
                        if t >= TOTAL_SLOTS or used_slots[furnace_idx][t] == 1:
                            is_overlap = True
                            break
                    if not is_overlap:
                        # เจอ slot ว่าง
                        for t in range(start_slot, start_slot + T_MELT):
                            used_slots[furnace_idx][t] = 1  # ทำเครื่องหมายว่าใช้ slot นี้แล้ว
                        X[i, 2 * j] = start_slot
                        X[i, 2 * j + 1] = furnace_idx
                        found_slot = True
                        break

                if not found_slot:
                    # ถ้าสุ่มหลายครั้งแล้วยังไม่เจอ ลองหาช่องแรกที่ว่าง
                    earliest_slot = -1
                    for s_try in range(min_start_slot, max_start_slot + 1):
                        is_overlap = False
                        for t in range(s_try, s_try + T_MELT):
                            if t >= TOTAL_SLOTS or used_slots[furnace_idx][t] == 1:
                                is_overlap = True
                                break
                        if not is_overlap:
                            earliest_slot = s_try
                            break

                    if earliest_slot != -1:
                        for t in range(earliest_slot, earliest_slot + T_MELT):
                            used_slots[furnace_idx][t] = 1
                        X[i, 2 * j] = earliest_slot
                        X[i, 2 * j + 1] = furnace_idx
                        found_slot = True
                    else:
                        # หา slot ไม่ได้จริงๆ (อาจเกิดถ้า batch เยอะมาก/slot น้อย)
                        print(
                            f"Warning: Could not find non-overlapping slot for sample {i}, batch {j+1} in furnace {furnace_idx}. Filling potentially invalid."
                        )
                        # ใส่ค่า default หรือ random ไปก่อน (อาจไม่ valid)
                        # Assign to earliest possible slot even if overlapping, let penalty handle it
                        X[i, 2 * j] = min_start_slot
                        X[i, 2 * j + 1] = furnace_idx
                        # Mark the slots as used anyway to avoid infinite loops if logic has issues
                        for t in range(
                            min_start_slot, min(min_start_slot + T_MELT, TOTAL_SLOTS)
                        ):
                            used_slots[furnace_idx][t] = 1
                        successful_generation = False  # อาจจะ track sample ที่มีปัญหา

            # อาจจะเพิ่ม logic ถ้า successful_generation เป็น False เช่น สร้าง sample นี้ใหม่

        # Clip ค่าเพื่อให้แน่ใจว่าอยู่ในขอบเขตที่กำหนด (สำคัญ!)
        X = np.clip(X, problem.xl, problem.xu)
        # ตรวจสอบและแก้ไขค่า furnace index ให้เป็น int และอยู่ในช่วง 0, 1
        X[:, 1::2] = np.round(np.clip(X[:, 1::2], 0, 1)).astype(int)
        # ตรวจสอบและแก้ไขค่า start time ให้เป็น int และอยู่ในช่วงขอบเขต
        max_start_time_bound = TOTAL_SLOTS - T_MELT
        X[:, 0::2] = np.round(np.clip(X[:, 0::2], 0, max_start_time_bound)).astype(int)

        return X


def main():
    # ===== Run NSGA-II =====
    problem = MeltingScheduleProblem(NUM_BATCHES)

    # --- ใช้ Custom Sampling ---
    custom_sampling = CustomInitialSampling()

    algorithm = NSGA2(
        pop_size=200,
        sampling=custom_sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", 100)

    result = minimize(problem, algorithm, termination, seed=42, verbose=True)

    # ===== Plot Result =====
    F = result.F
    plt.figure(figsize=(10, 6))
    plt.scatter(F[:, 0], F[:, 1], c="blue", s=40, edgecolors="k")
    plt.xlabel("Total Energy Consumption")
    plt.ylabel("Makespan (minutes)")
    plt.title("NSGA-II Pareto Front for Melting Schedule")
    plt.grid(True)
    plt.show()

    solutions = []

    # แสดงเฉพาะ 5 solution แรก (หรือน้อยกว่าถ้าผลลัพธ์น้อยกว่า 5)
    num_to_show = min(5, len(result.X))
    print(f"\nShowing top {num_to_show} solutions:")
    for i in range(num_to_show):
        print(f"\nSolution #{i+1}")
        print("  Total Energy:", F[i][0])
        print("  Makespan    :", F[i][1])
        print("  Schedule Vec:", result.X[i])
        solutions.append(result.X[i])

    # Plot top solutions
    for i, sol in enumerate(solutions):
        schedule = decode_schedule(sol)
        plot_schedule(schedule, title=f"Melting Schedule - Top {i+1}")


# =====================

## Config
# SLOT_DURATION = 30
# TOTAL_SLOTS = 1440 // SLOT_DURATION
# T_MELT = 3
# SHIFT_START = 9 * 60
# MH_CAPACITY_MAX = 500.0
# MH_CONSUMPTION_RATE = 5.56

# Furnace plot position
furnace_y = {0: 10, 1: 25}
height = 8


def decode_schedule(x, batch_count=None):  # batch_count is unused if derived from x
    schedule = []
    n = len(x) // 2  # Calculate n based on length of x
    num_batches_actual = (
        batch_count if batch_count is not None else n
    )  # Use provided count if available
    for i in range(num_batches_actual):  # Iterate up to actual number of batches
        start = int(x[2 * i])
        furnace = int(round(x[2 * i + 1]))
        # Clip start time based on problem bounds defined in MeltingScheduleProblem
        # Ensure start time allows for T_MELT duration within TOTAL_SLOTS
        start = max(0, min(start, TOTAL_SLOTS - T_MELT))
        end = start + T_MELT
        schedule.append((start, end, furnace, i + 1))  # Batch ID starts from 1
    return schedule


def plot_schedule(schedule, title="Schedule"):
    fig, ax = plt.subplots(figsize=(14, 4))
    for start, end, f, b_id in schedule:
        melt_start_t = SHIFT_START + start * SLOT_DURATION
        melt_dur = (end - start) * SLOT_DURATION

        # Ensure furnace index is valid before accessing furnace_y
        if f not in furnace_y:
            print(
                f"Warning: Invalid furnace index {f} for batch {b_id}. Skipping plot for this entry."
            )
            continue  # Skip this entry if furnace index is invalid

        ax.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors="tab:blue",
            edgecolor="black",
        )
        ax.text(
            melt_start_t + melt_dur / 2,
            furnace_y[f] + height / 2,
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )
    ax.set_xlim(SHIFT_START, SHIFT_START + 1440)
    xticks = np.arange(SHIFT_START, SHIFT_START + 1441, 60)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{(x // 60) % 24:02d}:{x % 60:02d}" for x in xticks])
    # Check if both furnaces are potentially used based on config or schedule
    yticks_pos = []
    yticks_labels = []
    # Use config flags first, then check schedule if needed
    furnaces_in_schedule = set(f for _, _, f, _ in schedule)

    if USE_FURNACE_A or (0 in furnaces_in_schedule):
        yticks_pos.append(furnace_y[0] + height / 2)
        yticks_labels.append("Furnace A")
    if USE_FURNACE_B or (1 in furnaces_in_schedule):
        yticks_pos.append(furnace_y[1] + height / 2)
        yticks_labels.append("Furnace B")

    # Only add labels if positions exist
    if yticks_pos:
        ax.set_yticks(yticks_pos)
        ax.set_yticklabels(yticks_labels)
    else:  # Handle case where no valid furnaces are used/plotted
        ax.set_yticks([])  # No ticks if no furnaces
        ax.set_yticklabels([])

    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Furnace")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
