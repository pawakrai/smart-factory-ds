import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation

# from pymoo.operators.mutation import SwapMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.crossover.sbx import SBX
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2


# =============== CONFIG ==================

SLOT_DURATION = 30  # 1 slot = 30 นาที
TOTAL_SLOTS = 1440 // SLOT_DURATION  # 1 วัน = 48 slots ถ้า 30 นาที/slot

T_MELT = 3  # Melting 3 slot = 90 นาที
NUM_BATCHES = 6  # <<< DEFINE GLOBALLY HERE

# กำหนดว่าเตาไหนใช้งานได้บ้าง
USE_FURNACE_A = True
USE_FURNACE_B = True

# --- New Constants for Advanced Simulation Logic ---
# Penalty rate per minute for an IF holding a batch after melt completion
IF_HOLDING_ENERGY_PENALTY_PER_MINUTE = 50  # Example value, adjust as needed
# Very high penalty for any batch that cannot be poured by the end of simulation
UNPOURED_BATCH_PENALTY = 1e13  # Increased penalty

# เตา M&H (โค้ดเดิมไว้ plot)
MH_MAX_CAPACITY_KG = {"A": 300.0, "B": 300.0}  # ความจุสูงสุดแต่ละเตา (kg)
MH_INITIAL_LEVEL_KG = {"A": 150.0, "B": 150.0}  # ระดับเริ่มต้น (kg)
MH_CONSUMPTION_RATE_KG_PER_MIN = {"A": 2.0, "B": 2.56}  # อัตราการใช้ kg/min แต่ละเตา
MH_EMPTY_THRESHOLD_KG = 50.0  # ระดับต่ำสุดที่ยอมรับได้ (kg)
IF_BATCH_OUTPUT_KG = 500.0  # ปริมาณที่ IF ผลิตต่อ batch (kg)
MH_REFILL_PER_FURNACE_KG = (
    IF_BATCH_OUTPUT_KG / 2
)  # ปริมาณที่เติมให้ M&H แต่ละเตาต่อ batch IF (kg)
POST_POUR_DOWNTIME_MIN = 10  # เวลาหยุดหลังเทเสร็จ (นาที)
MH_IDLE_PENALTY_RATE = 1.0  # Penalty ต่อนาทีที่เตา M&H ว่าง (ต่ำกว่า threshold)
MH_REHEAT_PENALTY_RATE = 2.0  # Placeholder penalty สำหรับ reheat

# กำหนดช่วงเวลาพัก (นาทีตั้งแต่เริ่มวัน 0 - 1440)
# ตัวอย่าง: 9:00-9:15 และ 12:00-13:00
BREAK_TIMES_MINUTES = [
    (9 * 60, 9 * 60 + 15),
    (12 * 60, 13 * 60),
]

# --- Plotting Config ---
FURNACE_COLORS = {"A": "blue", "B": "green"}
MH_FURNACE_COLORS = {"A": "red", "B": "orange"}

# furnace plot position (สำหรับ IF Gantt)
furnace_y = {0: 10, 1: 25}
height = 8

SHIFT_START = 9 * 60


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
    # ตัวอย่างเช็คเช่นเดิม:
    penalty_if = 0.0

    # (2.1) ลงโทษถ้าเตาไหนปิดแต่ x ระบุใช้เตานั้น
    for s, e, f, b_id in schedule:
        if f == 0 and not USE_FURNACE_A:
            penalty_if += 1e12
        if f == 1 and not USE_FURNACE_B:
            penalty_if += 1e12

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

    # (2.3) ถ้าเปิดทั้ง A,B => global check + Penalty if overlap
    global_usage = [0] * TOTAL_SLOTS
    # Constraint: ห้ามซ้อนทับกันเลย ไม่ว่าจะเตาไหน (สำหรับ IF)
    for s, e, f, b_id in schedule:
        for t in range(s, e):
            # เช็คว่า slot นี้ถูกใช้ไปหรือยัง (โดย IF อื่น)
            if global_usage[t] > 0:
                # ซ้อนทับ! ลงโทษหนักๆ
                penalty_if += 1e12  # Penalty สูงมากสำหรับการซ้อนทับของ IF
            global_usage[t] += 1  # นับว่า slot นี้ถูกใช้แล้วโดย IF

    # ----------------------------
    # 2.4) คำนวณ makespan slot => makespan (นาที)
    # ----------------------------
    makespan_slot = max(e for (s, e, f, b_id) in schedule) if schedule else 0
    makespan_min = makespan_slot * SLOT_DURATION

    # ----------------------------
    # 3) สร้าง water_events สำหรับ M&H
    # สมมติ 1 batch = 500 kg => มีน้ำหลอม 500 kg ทันทีที่ batch_i finish
    # (ถ้าคุณมี partial batch capacity จริง ๆ ก็ปรับ logic)
    # water_events will now store (melt_finish_minute, batch_id)
    # ----------------------------
    melt_completion_events = []  # list of (melt_finish_minute, batch_id)
    for s, e, f, b_id in schedule:
        finish_minute = e * SLOT_DURATION
        melt_completion_events.append((finish_minute, b_id))

    melt_completion_events.sort(key=lambda w: w[0])  # เรียงตามเวลา

    # ----------------------------
    # 4) เรียก mini-simulation M&H V2 (Modified)
    # ----------------------------
    # ใช้ makespan_min เพื่อกำหนดช่วงเวลาที่ต้องพิจารณา แต่ simulation อาจรันนานกว่านั้น
    (
        MH_idle_penalty,
        MH_reheat_penalty,  # Still placeholder
        total_energy_mh,  # Still placeholder
        time_points,
        mh_levels,
        actual_pour_events,  # New output
        unpoured_batches_at_end,  # New output
        total_if_holding_minutes,  # New output
        pour_induced_mh_overflow_penalty,  # New output
    ) = simulate_mh_consumption_v2(
        melt_completion_events
    )  # Pass melt_completion_events

    # --- ลบการตรวจสอบ M&H Overflow Penalty แบบเดิมออก ---
    # penalty_mh_overflow = 0.0 (This is now handled by pour_induced_mh_overflow_penalty from sim)

    # ----------------------------
    # 5) รวม cost
    # ----------------------------
    if_energy_rate = 2.0  # สมมติ IF กินพลังงาน (ปรับตามจริง)
    # คำนวณพลังงาน IF จาก *เวลาทำงานจริง* ของเตา ไม่ใช่แค่ makespan
    total_if_melting_slots = len(schedule) * T_MELT
    base_total_energy_if = if_energy_rate * total_if_melting_slots * SLOT_DURATION

    # Add IF holding energy penalty
    if_holding_penalty_cost = (
        total_if_holding_minutes * IF_HOLDING_ENERGY_PENALTY_PER_MINUTE
    )

    # Add penalty for unpoured batches
    unpoured_batches_penalty_cost = (
        len(unpoured_batches_at_end) * UNPOURED_BATCH_PENALTY
    )

    # final cost (Objective 1: Energy/Cost)
    cost = 0.0
    cost += base_total_energy_if  # พลังงาน IF (พื้นฐานจากการหลอม)
    cost += if_holding_penalty_cost  # พลังงาน/ค่าปรับจากการถือครอง IF batch
    cost += total_energy_mh  # พลังงาน M&H (ยังเป็น placeholder)
    cost += penalty_if  # Overlap IF / ปิดเตา (จากการวางแผนเบื้องต้น)
    cost += MH_idle_penalty  # Penalty จาก M&H idle
    cost += MH_reheat_penalty  # Penalty M&H reheat (ยังเป็น placeholder)
    cost += pour_induced_mh_overflow_penalty  # Penalty M&H overflow จากการเทจริง
    cost += unpoured_batches_penalty_cost  # Penalty สำหรับ batch ที่ไม่ได้เท

    # Objective 2: Makespan
    # Recalculate makespan based on actual pour times
    if actual_pour_events:
        makespan_min_actual = max(ape[0] for ape in actual_pour_events)
    elif schedule:  # If schedule exists but no pours happened (all unpoured)
        makespan_min_actual = (
            1440  # Max simulation time, heavily penalized by unpoured_penalty
        )
    else:  # No schedule
        makespan_min_actual = 0

    # If there are unpoured batches, makespan is effectively very bad.
    # The unpoured_batches_penalty_cost already makes the solution undesirable.
    # We can let makespan_min_actual reflect the last pour, or be max sim time if all failed.

    return cost, makespan_min_actual


def simulate_mh_consumption_v2(melt_completion_events):  # Changed input
    """
    จำลองระดับน้ำในเตา M&H A และ B แบบนาทีต่อนาที
    คำนวณ penalties และคืน time series สำหรับ plot
    NEW: Handles pour queue, M&H capacity check before pour, IF holding times, and pour-induced overflow.
    """
    simulation_duration_min = 1440
    time_points = np.arange(simulation_duration_min)
    mh_levels = {
        "A": np.zeros(simulation_duration_min),
        "B": np.zeros(simulation_duration_min),
    }
    mh_status = {"A": "idle", "B": "idle"}

    current_level = MH_INITIAL_LEVEL_KG.copy()
    downtime_remaining = {"A": 0, "B": 0}
    idle_duration = {"A": 0, "B": 0}

    total_idle_penalty = 0.0
    total_reheat_penalty = 0.0  # Placeholder
    pour_induced_mh_overflow_penalty = 0.0  # New

    ready_to_pour_queue = []  # Stores (melt_finish_minute, batch_id)
    actual_pour_events = []  # Stores (actual_pour_minute, batch_id)
    total_if_holding_minutes = 0

    melt_event_idx = 0

    for t in range(simulation_duration_min):
        minute_of_day = t

        # Add newly completed IF batches to ready_to_pour_queue
        while (
            melt_event_idx < len(melt_completion_events)
            and melt_completion_events[melt_event_idx][0] <= minute_of_day
        ):
            if melt_completion_events[melt_event_idx][0] == minute_of_day:
                ready_to_pour_queue.append(melt_completion_events[melt_event_idx])
                # print(f"Minute {t}: Batch {melt_completion_events[melt_event_idx][1]} melt complete, added to pour queue.") # Debug
            melt_event_idx += 1

        # Increment IF holding time for batches in queue
        total_if_holding_minutes += len(ready_to_pour_queue)

        # Attempt to pour batches from the queue
        poured_this_minute_batch_ids = []  # To avoid modifying queue while iterating

        # Sort queue by melt_finish_time (FIFO for pouring attempts) - already sorted by append order if events are sorted
        # ready_to_pour_queue.sort(key=lambda x: x[0]) # Might not be necessary if events added in order

        for i in range(len(ready_to_pour_queue)):
            melt_finish_min_q, batch_id_q = ready_to_pour_queue[i]

            available_capacity_A = MH_MAX_CAPACITY_KG["A"] - current_level["A"]
            available_capacity_B = MH_MAX_CAPACITY_KG["B"] - current_level["B"]
            total_available_mh_capacity = available_capacity_A + available_capacity_B

            if total_available_mh_capacity >= IF_BATCH_OUTPUT_KG:
                # print(f"Minute {t}: Attempting to pour Batch {batch_id_q}. Total M&H available: {total_available_mh_capacity:.1f} kg.") # Debug
                # Sufficient total capacity, proceed with pour

                pour_action_A_kg = MH_REFILL_PER_FURNACE_KG
                pour_action_B_kg = MH_REFILL_PER_FURNACE_KG

                # Check for individual M&H overflow before adding
                if current_level["A"] + pour_action_A_kg > MH_MAX_CAPACITY_KG["A"]:
                    pour_induced_mh_overflow_penalty += (
                        current_level["A"] + pour_action_A_kg - MH_MAX_CAPACITY_KG["A"]
                    ) * 1  # Penalty per kg overflow
                    # print(f"Minute {t}: Batch {batch_id_q} pour caused overflow in M&H A.") # Debug
                if current_level["B"] + pour_action_B_kg > MH_MAX_CAPACITY_KG["B"]:
                    pour_induced_mh_overflow_penalty += (
                        current_level["B"] + pour_action_B_kg - MH_MAX_CAPACITY_KG["B"]
                    ) * 1
                    # print(f"Minute {t}: Batch {batch_id_q} pour caused overflow in M&H B.") # Debug

                current_level["A"] = min(
                    current_level["A"] + pour_action_A_kg, MH_MAX_CAPACITY_KG["A"]
                )
                current_level["B"] = min(
                    current_level["B"] + pour_action_B_kg, MH_MAX_CAPACITY_KG["B"]
                )

                for furnace_id_mh in ["A", "B"]:
                    downtime_remaining[furnace_id_mh] = POST_POUR_DOWNTIME_MIN
                    mh_status[furnace_id_mh] = "downtime"

                actual_pour_events.append((t, batch_id_q))
                poured_this_minute_batch_ids.append(batch_id_q)
                # print(f"Minute {t}: Batch {batch_id_q} poured successfully. M&H A: {current_level['A']:.1f}, B: {current_level['B']:.1f}") # Debug

                # Since one pour happens per minute (conceptual simplification), break from trying other queued batches this minute
                # Or, allow multiple if sim logic handles it. For now, assume one pour action can be processed.
                # To allow multiple pours if capacity allows for subsequent batches in the same minute:
                # we need to update current_level *immediately* and re-check for next in queue.
                # For simplicity, let's assume the check for total_available_mh_capacity uses levels *at the start of the minute*.
                # A single batch pour event is resolved. If other batches are also ready, they wait for next minute's check.
                # This means if multiple batches become ready and M&H has huge capacity, they still pour one per minute.
                # This simplification might be acceptable.
                break  # Only one pour attempt resolution per minute from the queue
            # else: # Debug
            # print(f"Minute {t}: Batch {batch_id_q} cannot pour. Total M&H available: {total_available_mh_capacity:.1f} kg. Needed: {IF_BATCH_OUTPUT_KG} kg.")

        # Remove poured batches from the main queue
        if poured_this_minute_batch_ids:
            ready_to_pour_queue = [
                item
                for item in ready_to_pour_queue
                if item[1] not in poured_this_minute_batch_ids
            ]

        is_break_time = False
        for break_start, break_end in BREAK_TIMES_MINUTES:
            if break_start <= minute_of_day < break_end:
                is_break_time = True
                break

        for furnace_id in ["A", "B"]:
            # If a pour happened *into* this furnace this minute, its level is already set.
            # The consumption logic should apply to the state *after* any pour.
            # No, if a pour happened, it's in downtime.

            if downtime_remaining[furnace_id] > 0:
                # This check happens even if a pour just occurred for this furnace.
                downtime_remaining[furnace_id] -= 1
                mh_status[furnace_id] = "downtime"
                if downtime_remaining[furnace_id] == 0:
                    mh_status[furnace_id] = (
                        "running"
                        if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG
                        else "idle"
                    )

            if is_break_time or mh_status[furnace_id] == "downtime":
                mh_levels[furnace_id][t] = current_level[furnace_id]
                if (
                    mh_status[furnace_id] != "downtime"
                ):  # if it's break time but not downtime
                    idle_duration[furnace_id] = 0  # Reset idle if it's just a break
                continue

            if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                current_level[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[furnace_id]
                current_level[furnace_id] = max(current_level[furnace_id], 0)
                mh_status[furnace_id] = "running"
                idle_duration[furnace_id] = 0

                if current_level[furnace_id] <= MH_EMPTY_THRESHOLD_KG:
                    mh_status[furnace_id] = "idle"
                    idle_duration[furnace_id] = 1
                    total_idle_penalty += MH_IDLE_PENALTY_RATE
            else:
                mh_status[furnace_id] = "idle"
                idle_duration[furnace_id] += 1
                total_idle_penalty += MH_IDLE_PENALTY_RATE

            mh_levels[furnace_id][t] = current_level[furnace_id]

    total_energy_mh = 0.0  # Placeholder
    unpoured_batches_at_end = [
        item[1] for item in ready_to_pour_queue
    ]  # Batches remaining in queue

    return (
        total_idle_penalty,
        total_reheat_penalty,
        total_energy_mh,
        time_points,
        mh_levels,
        actual_pour_events,
        unpoured_batches_at_end,
        total_if_holding_minutes,
        pour_induced_mh_overflow_penalty,
    )


def decode_schedule(x_vector):
    x = x_vector
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

    # def simulate_mh_capacity():
    t = np.arange(0, 1441)
    cap = MH_MAX_CAPACITY_KG["A"]
    capacity = np.zeros_like(t, dtype=float)
    for i in range(len(t)):
        capacity[i] = cap
        cap -= MH_CONSUMPTION_RATE_KG_PER_MIN["A"]
        if cap < 0:
            cap = MH_MAX_CAPACITY_KG["A"]
    return t, capacity

    # def plot_schedule_and_mh(schedule, title="Melting Schedule & M&H"):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9))

    furnace_y = {0: 10, 1: 25}
    height = 8

    # ---------------------------
    # พล็อต Gantt chart (ax1)
    # ---------------------------
    for start, end, f, b_id in schedule:
        # --- Add check for valid furnace index ---
        if f not in furnace_y:
            print(
                f"Warning: Invalid furnace index '{f}' for batch {b_id} in plot_schedule_and_mh. Skipping this batch."
            )
            continue  # Skip to the next item in the schedule
        # --- End check ---

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
    ax1.set_title(title)

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
    ax2.set_ylim(0, MH_MAX_CAPACITY_KG["A"] * 1.1)
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


def plot_schedule_and_mh(
    schedule,
    title="Melting Schedule & M&H",
    simulated_time_points=None,
    simulated_mh_levels=None,
):
    """
    พล็อต Gantt chart ของ IF และ กราฟระดับน้ำใน M&H A, B
    Now accepts pre-simulated M&H data.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9), sharex=True)

    # --- Use pre-simulated M&H levels if provided ---
    if simulated_time_points is not None and simulated_mh_levels is not None:
        time_points_shifted = simulated_time_points + SHIFT_START
        mh_levels_to_plot = simulated_mh_levels
    # --- Fallback to re-simulating if not provided (should be avoided for consistency) ---
    elif (
        schedule
    ):  # Fallback, ideally this branch is not hit from main optimization flow
        print(
            "Warning: Re-simulating M&H for plotting. Use pre-simulated data for consistency."
        )
        makespan_slot = (
            max(e for (s, e, f, b_id) in schedule) if schedule else 0
        )  # Ensure schedule is not empty
        # makespan_min = makespan_slot * SLOT_DURATION # Not directly used by v2 sim for duration

        melt_completion_events_plot = []
        for s, e, f, b_id in schedule:
            finish_minute = e * SLOT_DURATION
            melt_completion_events_plot.append((finish_minute, b_id))
        melt_completion_events_plot.sort(key=lambda w: w[0])

        (
            _,
            _,
            _,  # Penalties not needed for plotting
            time_points_plot,
            mh_levels_plot,
            _,
            _,
            _,
            _,  # Other outputs not needed for basic plot
        ) = simulate_mh_consumption_v2(melt_completion_events_plot)
        time_points_shifted = time_points_plot + SHIFT_START
        mh_levels_to_plot = mh_levels_plot
    else:  # Empty schedule
        time_points_shifted = np.arange(SHIFT_START, SHIFT_START + 1440)
        mh_levels_to_plot = {
            "A": np.full(1440, MH_INITIAL_LEVEL_KG["A"]),
            "B": np.full(1440, MH_INITIAL_LEVEL_KG["B"]),
        }

    # ---------------------------
    # พล็อต Gantt chart (ax1)
    # ---------------------------
    for start, end, f, b_id in schedule:
        if f not in furnace_y:
            print(f"Warning: Invalid IF index '{f}' for batch {b_id}. Skipping plot.")
            continue
        melt_start_t = SHIFT_START + start * SLOT_DURATION
        melt_dur = (end - start) * SLOT_DURATION
        ax1.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors=FURNACE_COLORS.get(f, "gray"),  # ใช้สีจาก config
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
    ax1.set_ylabel("IF Furnace")
    ax1.set_yticks([furnace_y[0] + height / 2, furnace_y[1] + height / 2])
    ax1.set_yticklabels(["Furnace A", "Furnace B"])
    ax1.set_title(title)
    ax1.grid(True, axis="y", linestyle="-", alpha=0.5)

    # เส้นแบ่ง slot
    for slot_i in range(TOTAL_SLOTS + 1):
        line_x = SHIFT_START + slot_i * SLOT_DURATION
        ax1.axvline(x=line_x, color="gray", alpha=0.3, linestyle="--")

    # ---------------------------
    # subplot ล่าง (ax2) - M&H levels
    # ---------------------------
    max_plot_capacity = max(MH_MAX_CAPACITY_KG.values()) * 1.1

    for furnace_id in ["A", "B"]:
        ax2.plot(
            time_points_shifted,
            mh_levels_to_plot[furnace_id],  # Use mh_levels_to_plot
            color=MH_FURNACE_COLORS[furnace_id],
            linewidth=2,
            label=f"M&H {furnace_id} Level",
        )
        # เส้น Max Capacity
        ax2.axhline(
            y=MH_MAX_CAPACITY_KG[furnace_id],
            color=MH_FURNACE_COLORS[furnace_id],
            linestyle=":",
            alpha=0.7,
            label=f"M&H {furnace_id} Max Cap ({MH_MAX_CAPACITY_KG[furnace_id]} kg)",
        )

    # เส้น Empty Threshold
    ax2.axhline(
        y=MH_EMPTY_THRESHOLD_KG,
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Empty Threshold ({MH_EMPTY_THRESHOLD_KG} kg)",
    )

    ax2.set_ylim(0, max_plot_capacity)
    ax2.set_xlabel("Time (HH:MM)")
    ax2.set_ylabel("Metal in M&H (kg)")

    # xticks ราย ชม.
    xticks = np.arange(SHIFT_START, SHIFT_START + 1441, 60)
    ax2.set_xticks(xticks)
    xlabels = [f"{(x // 60) % 24:02d}:{x % 60:02d}" for x in xticks]
    ax2.set_xticklabels(xlabels)
    ax2.set_title("Simulated M&H Levels")
    ax2.grid(True, which="major", axis="both", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# --- ลบฟังก์ชัน simulate_mh_capacity() ที่ไม่ใช้ออก ---
# def simulate_mh_capacity(): ... (ลบออก)
# --- ฟังก์ชัน Greedy Assignment (Level 2) ---
def greedy_assignment(batch_order, num_batches=NUM_BATCHES):
    """
    กำหนด start_slot และ furnace สำหรับลำดับ batch ที่กำหนด
    โดยห้ามมี batch ใดๆ ทำงานซ้อนทับกันเลย (Global Constraint)
    และสลับเตาหากเป็นไปได้
    """
    x = np.zeros(num_batches * 2, dtype=int)
    global_used_slots = np.zeros(
        TOTAL_SLOTS, dtype=int
    )  # Restored for global IF slot tracking

    # available_furnaces_for_scheduling is still relevant for choosing which furnace (A or B)
    available_furnaces_for_assignment = []
    if USE_FURNACE_A:
        available_furnaces_for_assignment.append(0)
    if USE_FURNACE_B:
        available_furnaces_for_assignment.append(1)

    if not available_furnaces_for_assignment:
        print("Error in greedy_assignment: No available furnaces!")
        # Return potentially invalid x, let penalty handle it
        return x.astype(float)  # scheduling_cost expects float/can handle conversion

    furnace_assignment_counter = 0

    for batch_idx in batch_order:
        chosen_furnace_idx = -1
        if len(available_furnaces_for_assignment) == 1:
            chosen_furnace_idx = available_furnaces_for_assignment[0]
        elif len(available_furnaces_for_assignment) > 1:
            chosen_furnace_idx = available_furnaces_for_assignment[
                furnace_assignment_counter % len(available_furnaces_for_assignment)
            ]
            furnace_assignment_counter += 1
        else:
            continue

        # 2. หา Start Slot ที่เร็วที่สุดที่ไม่ซ้อนทับ Globaly (สำหรับ IF)
        found_slot = False
        search_start = 0
        while search_start <= TOTAL_SLOTS - T_MELT:
            is_overlap = False
            # Check overlap using global_used_slots for any IF operation
            for t in range(search_start, search_start + T_MELT):
                if t >= TOTAL_SLOTS or global_used_slots[t] == 1:
                    is_overlap = True
                    break

            if not is_overlap:
                start_slot = search_start
                x[2 * batch_idx] = start_slot
                x[2 * batch_idx + 1] = chosen_furnace_idx  # Assign to chosen furnace
                # อัปเดต global_used_slots เพราะ IF นี้กำลังจะทำงาน
                for t in range(start_slot, start_slot + T_MELT):
                    global_used_slots[t] = 1
                found_slot = True
                break
            search_start += 1

        if not found_slot:
            print(
                f"Warning in greedy_assignment: Could not find GLOBAL slot for batch {batch_idx+1}. Assigning fallback."
            )
            fallback_start = TOTAL_SLOTS - T_MELT
            x[2 * batch_idx] = fallback_start
            x[2 * batch_idx + 1] = chosen_furnace_idx
            # Mark global slots, even if overlapping (will be heavily penalized by scheduling_cost)
            for t in range(fallback_start, fallback_start + T_MELT):
                if t < TOTAL_SLOTS:
                    global_used_slots[t] = 1

    return x.astype(float)


# --- คลาส HGA Problem (Level 1) ---
class HGAProblem(Problem):  # สืบทอดจาก Problem
    def __init__(self, num_batches=NUM_BATCHES):
        super().__init__(
            n_var=num_batches, n_obj=2, n_constr=0, xl=0, xu=num_batches - 1
        )  # Bounds สำหรับ Permutation

    def _evaluate(self, P, out, *args, **kwargs):
        # P คือ population ณ generation ปัจจุบัน (array of permutations)
        results_f = []  # เก็บ objective values [energy, makespan]

        # วนลูปแต่ละ Permutation ใน Population
        for batch_order in P:
            # 1. แปลง Permutation เป็น x vector (schedule) โดย Greedy Assignment
            x = greedy_assignment(batch_order, num_batches=self.n_var)

            # 2. คำนวณ Objectives โดยใช้ scheduling_cost เดิม
            energy, makespan = scheduling_cost(x)

            # เพิ่ม penalty สูงมากถ้า greedy assignment ล้มเหลว (เช่น หา slot ไม่ได้)
            # (ตรวจจาก x ที่ได้ หรือเพิ่ม flag จาก greedy_assignment)
            # ตัวอย่าง: ตรวจสอบว่า makespan ดูสมเหตุสมผลไหม
            if makespan > TOTAL_SLOTS * SLOT_DURATION * 1.1:  # ถ้า makespan ยาวผิดปกติ
                energy += 1e15  # ลงโทษหนักๆ

            results_f.append([energy, makespan])

        # กำหนดค่า Objectives ให้กับ Population
        out["F"] = np.array(results_f)


def main():
    # ===== Run HGA using NSGA-II =====
    problem = HGAProblem(NUM_BATCHES)  # <--- ใช้ HGAProblem

    # --- Operators สำหรับ Permutation ---
    sampling = PermutationRandomSampling()
    crossover = OrderCrossover()
    mutation = InversionMutation()

    algorithm = NSGA2(
        pop_size=50,
        sampling=sampling,  # <--- ใช้ Permutation Sampling
        crossover=crossover,  # <--- ใช้ Permutation Crossover
        mutation=mutation,  # <--- ใช้ Permutation Mutation
        eliminate_duplicates=True,  # อาจจะต้อง custom duplicate detection สำหรับ permutation ถ้าจำเป็น
    )

    termination = get_termination("n_gen", 100)  # หรือเกณฑ์อื่นๆ

    print("Running HGA optimization...")
    result = minimize(
        problem, algorithm, termination, seed=42, verbose=True, save_history=False
    )  # ปิด history ถ้าไม่ได้ใช้ เพื่อประหยัด memory

    print("Optimization finished.")

    # ===== Plot Result (Pareto Front) =====
    F = result.F  # Objective values [energy, makespan]
    plt.figure(figsize=(10, 6))
    plt.scatter(
        F[:, 0], F[:, 1], c="red", s=40, edgecolors="k", label="HGA Pareto Front"
    )  # เปลี่ยนสี/label
    plt.xlabel("Total Energy Consumption")
    plt.ylabel("Makespan (minutes)")
    plt.title("HGA Pareto Front for Melting Schedule")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ===== แสดงผลลัพธ์และ Plot ตารางเวลาสำหรับ Solutions ที่ดีที่สุด =====
    print("\nProcessing results...")
    solutions_x = []  # เก็บ x vector ที่แปลงแล้ว
    solutions_p = result.X  # Permutations ที่ได้จาก HGA

    # แสดงเฉพาะ top solutions (อาจจะต้องเรียงตาม objective หรือเลือกจาก Pareto front)
    # pymoo result อาจจะเรียงมาให้แล้ว หรืออาจจะต้องเลือกเอง
    num_to_show = min(5, len(solutions_p))
    print(f"\nShowing top {num_to_show} solutions from Pareto front:")

    for i in range(num_to_show):
        permutation = solutions_p[i]
        energy = F[i][0]
        makespan = F[i][1]

        print(f"\nSolution #{i+1}")
        print(f"  Batch Order (Permutation): {permutation}")
        print(f"  Total Energy: {energy}")
        print(f"  Makespan    : {makespan}")

        # --- แปลง Permutation กลับเป็น x vector เพื่อ plot ---
        x_vector = greedy_assignment(permutation, num_batches=NUM_BATCHES)
        print(
            f"  Generated Schedule Vec (x): {np.round(x_vector, 2)}"
        )  # แสดง x vector ที่ได้
        solutions_x.append(x_vector)  # เก็บ x vector ไว้ plot

        # --- Plot ตารางเวลา ---
        # We need the mh_levels and time_points from the *actual evaluation* of this x_vector
        # This requires re-calculating or storing them.
        # For now, the plot function can re-simulate, but it's not ideal.
        # Let's pass the results from the main simulation loop if we decide to plot one specific solution.
        # The current loop plots for top N solutions from HGA. Each would have its own sim data.

        # To plot with the correct M&H data, we'd need to store F[i]'s corresponding
        # time_points and mh_levels from its evaluation inside HGAProblem._evaluate or re-run.
        # For simplicity now, plot_schedule_and_mh will re-simulate its own mini-simulation
        # if not provided with data, which is what it does with the fallback.
        # This part of the code for plotting top N might show slightly different M&H if
        # random elements were in sim, but our current sim is deterministic.

        # Ideal way: Store (energy, makespan, time_points_for_plot, mh_levels_for_plot) in HGA results.
        # Quick fix for now: allow plot_schedule_and_mh to re-simulate based on schedule if needed.
        # The refactor of plot_schedule_and_mh already allows passing data.
        # How to get that data here for each of the top N solutions?
        # HGAProblem._evaluate would need to return it, and `minimize` would need to store it.
        # This is a larger change to Pymoo's interaction.

        # Simplest for now: The plot function's fallback re-simulation is acceptable given determinism.
        # To make it use the *optimization-run* data, we would need to:
        # 1. Modify HGAProblem to store/return mh_levels and time_points along with F.
        # 2. Modify the main loop to extract and pass these to plot_schedule_and_mh.

        # For the "Testing with fixed and random batch orders" section, we *do* have the sim results.
        # We can enhance that plotting part later if needed.

        if x_vector is not None:
            try:
                schedule = decode_schedule(x_vector)
                if schedule:
                    # For plotting top HGA solutions, we don't have direct access to the
                    # specific time_points and mh_levels from their original evaluation in problem._evaluate
                    # without significant changes to how results are stored by pymoo.
                    # So, we let plot_schedule_and_mh use its fallback to re-simulate for plotting here.
                    plot_schedule_and_mh(
                        schedule,
                        title=f"HGA Schedule - Solution {i+1} (Energy: {energy:.0f}, Makespan: {makespan:.0f} (from cost func))",
                        # We used makespan_min_actual in the printout, let's use it for title
                        # Note: `makespan` here is F[i][1] which is makespan_min_actual from that specific eval
                    )
                else:
                    print("  Could not decode schedule for plotting.")
            except Exception as e:
                print(f"  Error plotting schedule for solution {i+1}: {e}")
        else:
            print("  Could not generate schedule vector (x) for plotting.")


# ... (ส่วน decode_schedule, plot_schedule คงเดิม) ...

if __name__ == "__main__":
    main()
