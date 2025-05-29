# This is a new file, populated with the content of app_v3.py

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
import random


# =============== CONFIG ==================

HOURS_A_DAY = 24 * 60  # 1440
SLOT_DURATION = 5  # 1 slot = 10 นาที
TOTAL_SLOTS = HOURS_A_DAY // SLOT_DURATION  # 1 วัน = 144 slots ถ้า 10 นาที/slot

T_MELT = 18  # Melting 9 slot = 90 นาที (9 * 10 min/slot)
NUM_BATCHES = 10  # <<< DEFINE GLOBALLY HERE

# กำหนดว่าเตาไหนใช้งานได้บ้าง
USE_FURNACE_A = True
USE_FURNACE_B = True

# --- New Constants for Advanced Simulation Logic ---
# Penalty rate per minute for an IF holding a batch after melt completion
IF_HOLDING_ENERGY_PENALTY_PER_MINUTE = 0  # Example value, adjust as needed
# Very high penalty for any batch that cannot be poured by the end of simulation
UNPOURED_BATCH_PENALTY = 1e13  # Increased penalty
# Penalty for idle time between consecutive batches on the same IF furnace
IF_GAP_TIME_PENALTY_RATE_PER_MINUTE = 0.5  # Example: 0.5 cost unit per minute of gap
# Penalty for IF working during designated break times
IF_WORKING_IN_BREAK_PENALTY_PER_MINUTE = (
    10000  # Example: 100 cost unit per minute of work in break
)
# Average Power Rating for each IF in kW
IF_POWER_RATING_KW = {"A": 550.0, "B": 600.0}  # เตา A = 550 kW, เตา B = 600 kW

# เตา M&H (โค้ดเดิมไว้ plot)
MH_MAX_CAPACITY_KG = {"A": 300.0, "B": 300.0}  # ความจุสูงสุดแต่ละเตา (kg)
MH_INITIAL_LEVEL_KG = {"A": 250.0, "B": 250.0}  # ระดับเริ่มต้น (kg)
MH_CONSUMPTION_RATE_KG_PER_MIN = {"A": 3.0, "B": 2.56}  # อัตราการใช้ kg/min แต่ละเตา
MH_EMPTY_THRESHOLD_KG = 20  # ระดับต่ำสุดที่ยอมรับได้ (kg)
IF_BATCH_OUTPUT_KG = 500.0  # ปริมาณที่ IF ผลิตต่อ batch (kg)
MH_REFILL_PER_FURNACE_KG = (
    IF_BATCH_OUTPUT_KG / 2
)  # ปริมาณที่เติมให้ M&H แต่ละเตาต่อ batch IF (kg)
POST_POUR_DOWNTIME_MIN = 10  # เวลาหยุดหลังเทเสร็จ (นาที)
MH_IDLE_PENALTY_RATE = 0  # Penalty ต่อนาทีที่เตา M&H ว่าง (ต่ำกว่า threshold)
MH_REHEAT_PENALTY_RATE = 20.0  # Placeholder penalty สำหรับ reheat

# กำหนดช่วงเวลาพัก (นาทีตั้งแต่เริ่มวัน 0 - 1440)
# ตัวอย่าง: 9:00-9:15 และ 12:00-13:00
BREAK_TIMES_MINUTES = [
    (0 * 60, 0 * 60 + 40),  # 08:00 - 08:40
    (9 * 60, 9 * 60 + 40),  # 20:00 - 20:40
]

# --- Plotting Config ---
FURNACE_COLORS = {"A": "blue", "B": "green"}
MH_FURNACE_COLORS = {"A": "red", "B": "orange"}

# furnace plot position (สำหรับ IF Gantt)
furnace_y = {0: 10, 1: 25}
height = 8

SHIFT_START = 0 * 60


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
        for t_slot in range(s, e):  # Renamed t to t_slot to avoid conflict
            usage_for_furnace[f][t_slot] += 1

    if USE_FURNACE_A:
        for t_slot in range(TOTAL_SLOTS):
            if usage_for_furnace[0][t_slot] > 1:
                penalty_if += (usage_for_furnace[0][t_slot] - 1) * 1e9

    if USE_FURNACE_B:
        for t_slot in range(TOTAL_SLOTS):
            if usage_for_furnace[1][t_slot] > 1:
                penalty_if += (usage_for_furnace[1][t_slot] - 1) * 1e9

    # (2.3) ถ้าเปิดทั้ง A,B => global check + Penalty if overlap
    global_usage = [0] * TOTAL_SLOTS
    # Constraint: ห้ามซ้อนทับกันเลย ไม่ว่าจะเตาไหน (สำหรับ IF)
    for s, e, f, b_id in schedule:
        for t_slot in range(s, e):
            # เช็คว่า slot นี้ถูกใช้ไปหรือยัง (โดย IF อื่น)
            if global_usage[t_slot] > 0:
                # ซ้อนทับ! ลงโทษหนักๆ
                penalty_if += 1e12  # Penalty สูงมากสำหรับการซ้อนทับของ IF
            global_usage[t_slot] += 1  # นับว่า slot นี้ถูกใช้แล้วโดย IF

    # (2.4) Penalty for Gap Time within the same IF furnace
    if_gap_time_penalty = 0.0
    schedule_by_furnace = {0: [], 1: []}
    for s, e, f, b_id in schedule:
        schedule_by_furnace[f].append((s, e, f, b_id))

    for f_idx in [0, 1]:
        if (f_idx == 0 and not USE_FURNACE_A) or (f_idx == 1 and not USE_FURNACE_B):
            continue

        furnace_schedule = sorted(schedule_by_furnace[f_idx], key=lambda item: item[0])
        last_batch_end_slot = (
            0  # Assuming work can start from slot 0 or after previous batch
        )
        is_first_batch_in_furnace = True
        for s, e, f, b_id in furnace_schedule:
            if not is_first_batch_in_furnace:
                gap_slots = s - last_batch_end_slot
                if gap_slots > 0:
                    gap_minutes = gap_slots * SLOT_DURATION
                    if_gap_time_penalty += (
                        gap_minutes * IF_GAP_TIME_PENALTY_RATE_PER_MINUTE
                    )
            last_batch_end_slot = e
            is_first_batch_in_furnace = False

    penalty_if += if_gap_time_penalty  # Add to existing IF penalties

    # (2.5) Penalty for IF working during break times
    if_working_in_break_penalty = 0.0
    for s, e, f, b_id in schedule:
        start_minute_abs = s * SLOT_DURATION
        end_minute_abs = e * SLOT_DURATION  # exclusive end minute
        for break_start_abs, break_end_abs in BREAK_TIMES_MINUTES:
            # Calculate overlap duration
            overlap_start = max(start_minute_abs, break_start_abs)
            overlap_end = min(end_minute_abs, break_end_abs)
            if overlap_end > overlap_start:
                overlap_duration_minutes = overlap_end - overlap_start
                if_working_in_break_penalty += (
                    overlap_duration_minutes * IF_WORKING_IN_BREAK_PENALTY_PER_MINUTE
                )

    penalty_if += if_working_in_break_penalty  # Add to existing IF penalties

    # ----------------------------
    # 2.6) คำนวณ makespan slot => makespan (นาที)
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
    # --- NEW IF ENERGY CALCULATION (in kWh) ---
    base_total_energy_if_kwh = 0.0
    # Map furnace index (0, 1) to furnace ID ("A", "B") - assuming 0 is A, 1 is B
    # This mapping should be consistent with how USE_FURNACE_A/B is handled or defined
    if_furnace_id_map = {}
    if USE_FURNACE_A:
        if_furnace_id_map[0] = "A"
    if USE_FURNACE_B:
        if (
            len(if_furnace_id_map) == 1 and 0 not in if_furnace_id_map
        ):  # e.g. only B is used, assign 0 to B
            if_furnace_id_map[0] = "B"  # This case for B only and f=0
        elif 1 not in if_furnace_id_map and USE_FURNACE_B:  # e.g. A and B used, B is 1
            if_furnace_id_map[1] = "B"
        elif (
            0 not in if_furnace_id_map and USE_FURNACE_A
        ):  # e.g. A and B used, A is 0 (already handled)
            pass  # Already handled if USE_FURNACE_A is true, 0 is A
        elif len(if_furnace_id_map) == 0:  # Neither A or B is used.
            pass  # Should not happen if schedule has items
        # If only one furnace is active, its index in schedule (0) will map to its ID ("A" or "B")
        # If both are active, 0 maps to "A", 1 maps to "B" (needs careful check if USE_FURNACE_A/B allows gaps)

    # Fallback for furnace mapping if only one furnace is used, assume it's index 0
    # This logic needs to be robust if furnace indices in `schedule` are not always 0 or 1 or if only one is used.
    # For simplicity, let's assume if only one furnace type is active, its ID is used for index 0.
    # And if both are active, 0 is A, 1 is B.

    # Simplified mapping based on active furnaces
    # This mapping is crucial. Let's refine it.
    # if_map_for_power = {0: "A", 1: "B"} # Default if both used
    # if USE_FURNACE_A and not USE_FURNACE_B:
    #     if_map_for_power = {0: "A"}
    # elif not USE_FURNACE_A and USE_FURNACE_B:
    #     if_map_for_power = {0: "B"}
    # elif not USE_FURNACE_A and not USE_FURNACE_B:
    #     if_map_for_power = {}

    for s, e, f, b_id in schedule:
        furnace_idx_in_schedule = f  # f is the furnace index (0 or 1) from schedule
        furnace_id_for_power = None

        if USE_FURNACE_A and USE_FURNACE_B:
            furnace_id_for_power = "A" if furnace_idx_in_schedule == 0 else "B"
        elif USE_FURNACE_A:  # Only Furnace A is active
            furnace_id_for_power = (
                "A"  # All scheduled batches must be on furnace A (index 0)
            )
        elif USE_FURNACE_B:  # Only Furnace B is active
            furnace_id_for_power = (
                "B"  # All scheduled batches must be on furnace B (index 0)
            )

        if furnace_id_for_power and furnace_id_for_power in IF_POWER_RATING_KW:
            power_rating_kw = IF_POWER_RATING_KW[furnace_id_for_power]
            melting_duration_hours = (T_MELT * SLOT_DURATION) / 60.0
            energy_for_batch_kwh = power_rating_kw * melting_duration_hours
            base_total_energy_if_kwh += energy_for_batch_kwh
        # else:
        # This case should ideally not happen if schedule is valid and furnaces are defined
        # print(f"Warning: Could not map furnace index {furnace_idx_in_schedule} to power rating for batch {b_id}")

    # Add IF holding energy penalty
    # IF_HOLDING_ENERGY_PENALTY_PER_MINUTE is currently 50.
    # If this is meant to be an actual energy cost, it needs to be kW * (minutes/60)
    # For now, let's assume it's a separate penalty, not directly kWh.
    # If you want to convert this to kWh based on a standby power:
    # standby_power_A_kw = ... (define this)
    # standby_power_B_kw = ... (define this)
    # holding_energy_kwh = (total_if_holding_minutes_A / 60.0 * standby_power_A_kw) + ...
    if_holding_penalty_cost = (
        total_if_holding_minutes * IF_HOLDING_ENERGY_PENALTY_PER_MINUTE
    )

    # Add penalty for unpoured batches
    unpoured_batches_penalty_cost = (
        len(unpoured_batches_at_end) * UNPOURED_BATCH_PENALTY
    )

    # final cost (Objective 1: Energy/Cost)
    # The 'cost' objective now primarily reflects energy in kWh plus other penalties
    cost = 0.0
    cost += base_total_energy_if_kwh  # Base IF melting energy in kWh
    cost += if_holding_penalty_cost  # This is still a penalty value, not kWh unless redefined
    cost += total_energy_mh  # พลังงาน M&H (ยังเป็น placeholder)
    cost += (
        penalty_if  # Overlap IF / ปิดเตา / Gap Time / IF in Break (จากการวางแผนเบื้องต้น)
    )
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

    # Return detailed cost components as well
    cost_components = {
        "total_cost": cost,
        "makespan_minutes": makespan_min_actual,
        "base_if_energy_kwh": base_total_energy_if_kwh,
        "if_holding_penalty": if_holding_penalty_cost,
        "if_general_penalty": penalty_if,  # Contains overlap, gap, break penalties
        "mh_idle_penalty": MH_idle_penalty,
        "mh_reheat_penalty": MH_reheat_penalty,  # Placeholder
        "mh_overflow_penalty": pour_induced_mh_overflow_penalty,
        "unpoured_batch_penalty": unpoured_batches_penalty_cost,
        "mh_energy_kwh": total_energy_mh,  # Placeholder
    }

    return cost, makespan_min_actual, cost_components


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

        for furnace_id in ["A", "B"]:
            # Handle Downtime
            if downtime_remaining[furnace_id] > 0:
                downtime_remaining[furnace_id] -= 1
                mh_status[furnace_id] = "downtime"
                if downtime_remaining[furnace_id] == 0:
                    mh_status[furnace_id] = (
                        "running"
                        if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG
                        else "idle"
                    )
                mh_levels[furnace_id][t] = current_level[furnace_id]
                continue  # Skip consumption and idle penalty if in downtime

            if mh_status[furnace_id] == "downtime":
                mh_levels[furnace_id][t] = current_level[furnace_id]
                continue

            if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                current_level[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[furnace_id]
                current_level[furnace_id] = max(current_level[furnace_id], 0)
                mh_status[furnace_id] = "running"
                if current_level[furnace_id] <= MH_EMPTY_THRESHOLD_KG:
                    mh_status[furnace_id] = "idle"
                    # idle_duration[furnace_id] = 1 # Start/reset idle duration counter
            else:  # Already at or below threshold
                mh_status[furnace_id] = "idle"
                # if mh_status was already "idle", increment idle_duration
                # idle_duration[furnace_id] += 1

            # Apply idle penalty if the furnace is idle (and not in downtime or break)
            if mh_status[furnace_id] == "idle":
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
        # makespan_slot = (
        #     max(e for (s, e, f, b_id) in schedule) if schedule else 0
        # )  # Ensure schedule is not empty
        # # makespan_min = makespan_slot * SLOT_DURATION # Not directly used by v2 sim for duration

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
        if (
            f not in furnace_y
        ):  # Using global furnace_y here, ensure it's what's intended
            print(f"Warning: Invalid IF index '{f}' for batch {b_id}. Skipping plot.")
            continue
        melt_start_t = SHIFT_START + start * SLOT_DURATION
        melt_dur = (end - start) * SLOT_DURATION
        ax1.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),  # Using global furnace_y and height
            facecolors=FURNACE_COLORS.get(f, "gray"),  # ใช้สีจาก config
            edgecolor="black",
        )
        ax1.text(
            melt_start_t + melt_dur / 2,
            furnace_y[f] + height / 2,  # Using global furnace_y and height
            f"{b_id}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
        )

    ax1.set_xlim(SHIFT_START, SHIFT_START + 1440)
    ax1.set_ylabel("IF Furnace")
    ax1.set_yticks(
        [furnace_y[0] + height / 2, furnace_y[1] + height / 2]
    )  # Using global furnace_y and height
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

    for furnace_id_mh_plot in ["A", "B"]:  # Renamed furnace_id to furnace_id_mh_plot
        ax2.plot(
            time_points_shifted,
            mh_levels_to_plot[furnace_id_mh_plot],  # Use mh_levels_to_plot
            color=MH_FURNACE_COLORS[furnace_id_mh_plot],
            linewidth=2,
            label=f"M&H {furnace_id_mh_plot} Level",
        )
        # เส้น Max Capacity
        ax2.axhline(
            y=MH_MAX_CAPACITY_KG[furnace_id_mh_plot],
            color=MH_FURNACE_COLORS[furnace_id_mh_plot],
            linestyle=":",
            alpha=0.7,
            label=f"M&H {furnace_id_mh_plot} Max Cap ({MH_MAX_CAPACITY_KG[furnace_id_mh_plot]} kg)",
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
    xticks_ax2 = np.arange(SHIFT_START, SHIFT_START + 1441, 60)  # Renamed xticks
    ax2.set_xticks(xticks_ax2)
    xlabels_ax2 = [
        f"{(x_val // 60) % 24:02d}:{x_val % 60:02d}" for x_val in xticks_ax2
    ]  # Renamed xlabels and x
    ax2.set_xticklabels(xlabels_ax2)
    ax2.set_title("Simulated M&H Levels")
    ax2.grid(True, which="major", axis="both", alpha=0.5)
    ax2.legend(loc="upper right")

    # Add shaded regions for break times on both subplots
    for break_start, break_end in BREAK_TIMES_MINUTES:
        # Adjust for SHIFT_START for plotting coordinates
        plot_break_start = break_start  # Break times are absolute day minutes
        plot_break_end = break_end

        # Ensure breaks are within the plotted x-axis range (0 to 1440 relative to day start for sim data,
        # then shifted by SHIFT_START for plot)
        # The time_points_shifted already accounts for SHIFT_START
        # We need to ensure the axvspan uses coordinates relative to the plot's x-axis

        # For ax1 (IF Gantt)
        ax1.axvspan(
            plot_break_start,
            plot_break_end,
            facecolor="grey",
            alpha=0.2,
            zorder=-1,
        )
        # For ax2 (M&H Levels)
        ax2.axvspan(
            plot_break_start,
            plot_break_end,
            facecolor="grey",
            alpha=0.2,
            zorder=-1,
        )

    plt.tight_layout()
    plt.show()


# +++ NEW HELPER FUNCTION for greedy_assignment +++
def _greedy_simulate_mh_state_to_minute(
    target_minute,
    current_levels_kg,  # dict: {"A": level, "B": level}
    downtime_remaining_min,  # dict: {"A": minutes, "B": minutes}
    last_simulated_minute,
):
    """
    Simulates M&H consumption and status from last_simulated_minute up to (but not including) target_minute.
    Returns the predicted M&H levels and downtime state at the BEGINNING of target_minute.
    This is a simplified forward simulation for greedy's lookahead.
    It does NOT handle new pours, only consumption, downtime ticks, and breaks.
    """
    new_levels_kg = current_levels_kg.copy()
    new_downtime_remaining_min = downtime_remaining_min.copy()

    # Simulate minute by minute from (last_simulated_minute + 1) up to target_minute -1
    # The state at the *beginning* of target_minute is what we want.
    for t_sim in range(last_simulated_minute + 1, target_minute):
        minute_of_day_sim = t_sim

        for furnace_id_mh in ["A", "B"]:
            if new_downtime_remaining_min[furnace_id_mh] > 0:
                new_downtime_remaining_min[furnace_id_mh] -= 1
                # Status becomes "running" or "idle" when downtime hits 0, handled by consumption logic below

            if new_downtime_remaining_min[furnace_id_mh] > 0:
                # No consumption during break or active downtime
                continue

            # If not in break and not in downtime, consume
            if new_levels_kg[furnace_id_mh] > MH_EMPTY_THRESHOLD_KG:
                new_levels_kg[furnace_id_mh] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                    furnace_id_mh
                ]
                new_levels_kg[furnace_id_mh] = max(new_levels_kg[furnace_id_mh], 0)
            # Idle penalty logic is not needed for this feasibility check, only levels and downtime.

    return new_levels_kg, new_downtime_remaining_min


# --- ฟังก์ชัน Greedy Assignment (Level 2) ---
def greedy_assignment(batch_order, num_batches=NUM_BATCHES):
    """
    กำหนด start_slot และ furnace สำหรับลำดับ batch ที่กำหนด
    โดยห้ามมี batch ใดๆ ทำงานซ้อนทับกันเลย (Global Constraint for IFs)
    และพยายามจัดตาราง IF ให้สอดคล้องกับความพร้อมของ M&H (M&H-aware).
    NEW: Introduces randomness in choosing among viable M&H-aware slots.
    """
    MAX_VIABLE_SLOTS_TO_CONSIDER = (
        5  # Max number of early viable slots to consider for random choice
    )

    x_schedule_vector = np.full(
        num_batches * 2, -1, dtype=int
    )  # Initialize with -1 (unscheduled)
    global_if_used_slots = np.zeros(TOTAL_SLOTS, dtype=int)

    available_if_furnaces_for_assignment = []
    if USE_FURNACE_A:
        available_if_furnaces_for_assignment.append(0)
    if USE_FURNACE_B:
        available_if_furnaces_for_assignment.append(1)

    if not available_if_furnaces_for_assignment:
        print("Error in greedy_assignment: No available IF furnaces!")
        return x_schedule_vector.astype(float)

    # Internal M&H state for greedy's simulation
    internal_mh_levels_kg = MH_INITIAL_LEVEL_KG.copy()
    internal_mh_downtime_remaining_min = {"A": 0, "B": 0}
    internal_last_mh_sim_minute = -1  # Simulates from minute 0 onwards

    if_furnace_assignment_counter = 0

    MAX_SEARCH_MINUTES = (
        1440 + T_MELT * SLOT_DURATION
    )  # Search a bit beyond one day if needed

    for (
        batch_idx_in_order
    ) in batch_order:  # batch_idx_in_order is the actual batch ID (0 to N-1)
        chosen_if_furnace_idx = -1
        if len(available_if_furnaces_for_assignment) == 1:
            chosen_if_furnace_idx = available_if_furnaces_for_assignment[0]
        elif len(available_if_furnaces_for_assignment) > 1:
            chosen_if_furnace_idx = available_if_furnaces_for_assignment[
                if_furnace_assignment_counter
                % len(available_if_furnaces_for_assignment)
            ]
            # Cycle through furnaces more robustly for subsequent batches
            # if_furnace_assignment_counter is advanced per BATCH, not per furnace chosen
        else:  # Should be caught by earlier check
            continue

        found_slot_for_batch = False
        viable_slots_data = (
            []
        )  # Stores (potential_if_start_slot, mh_levels_before_pour, mh_downtime_before_pour, potential_if_melt_finish_minute)

        # Start searching for an IF start slot from the end of the last IF operation,
        # or from 0 if this is the first batch.
        # More accurately, from the earliest possible time considering M&H readiness.
        # The search for if_start_slot is in SLOTS.
        # The M&H simulation is in MINUTES.

        # Determine the earliest minute this batch *could* start melting based on IF availability
        # This means searching for a free IF slot first.
        # --- REMOVED BREAK TIME CHECK FROM HERE ---
        # The greedy assignment will no longer try to avoid IF operations during breaks;
        # this will be handled by a penalty in scheduling_cost.

        min_possible_if_start_minute_overall = internal_last_mh_sim_minute + 1

        for potential_if_start_slot in range(TOTAL_SLOTS - T_MELT + 1):
            # Constraint 1: Global IF slot availability
            is_if_globally_free = True
            for t_if_check_slot in range(
                potential_if_start_slot, potential_if_start_slot + T_MELT
            ):
                if (
                    t_if_check_slot >= TOTAL_SLOTS
                    or global_if_used_slots[t_if_check_slot] == 1
                ):
                    is_if_globally_free = False
                    break

            if not is_if_globally_free:
                continue  # Try next potential_if_start_slot

            # # NEW Constraint: Check if IF operation falls into any break time
            # if_op_start_minute = potential_if_start_slot * SLOT_DURATION
            # if_op_end_minute = (potential_if_start_slot + T_MELT) * SLOT_DURATION
            # is_in_break = False
            # for break_start_abs, break_end_abs in BREAK_TIMES_MINUTES:
            #     # Check for overlap:
            #     # (IF_start < Break_end) and (IF_end > Break_start)
            #     if (
            #         if_op_start_minute < break_end_abs
            #         and if_op_end_minute > break_start_abs
            #     ):
            #         is_in_break = True
            #         break

            # if is_in_break:
            #     continue  # Skip this slot as it overlaps with a break time

            potential_if_melt_finish_minute = (
                potential_if_start_slot + T_MELT
            ) * SLOT_DURATION

            # Ensure we are not trying to schedule IF to finish in the past relative to M&H sim
            if potential_if_melt_finish_minute < internal_last_mh_sim_minute:
                continue

            # Constraint 2: M&H readiness at potential_if_melt_finish_minute
            # Simulate M&H state up to the point just before this potential pour
            mh_levels_before_pour, mh_downtime_before_pour = (
                _greedy_simulate_mh_state_to_minute(
                    potential_if_melt_finish_minute,
                    internal_mh_levels_kg,
                    internal_mh_downtime_remaining_min,
                    internal_last_mh_sim_minute,
                )
            )

            # --- REMOVED BREAK TIME CHECK FOR POURING FROM HERE ---
            # The original check `is_break_at_pour_minute` is removed from greedy_assignment.
            # If a pour happens during a break, simulate_mh_consumption_v2 will handle M&H downtime correctly.
            # The IF working during break penalty is now the main deterrent for IF activity during breaks.

            # Check if M&H furnaces are in post-pour downtime from a *previous* pour
            # The _greedy_simulate_mh_state_to_minute updates downtimes, so mh_downtime_before_pour reflects this.

            available_capacity_A = MH_MAX_CAPACITY_KG["A"] - mh_levels_before_pour["A"]
            available_capacity_B = MH_MAX_CAPACITY_KG["B"] - mh_levels_before_pour["B"]
            total_available_mh_capacity = available_capacity_A + available_capacity_B

            if total_available_mh_capacity >= IF_BATCH_OUTPUT_KG:
                # This slot is viable for IF and M&H! Store it.
                viable_slots_data.append(
                    (
                        potential_if_start_slot,
                        mh_levels_before_pour.copy(),
                        mh_downtime_before_pour.copy(),
                        potential_if_melt_finish_minute,
                    )
                )
                if len(viable_slots_data) >= MAX_VIABLE_SLOTS_TO_CONSIDER:
                    break  # Found enough viable slots, move to selection

        # After checking potential slots, choose one if any were found
        if viable_slots_data:
            # Randomly select one of the found viable slots
            (
                selected_start_slot,
                selected_mh_levels_before_pour,
                selected_mh_downtime_before_pour,
                selected_melt_finish_minute,
            ) = random.choice(viable_slots_data)

            x_schedule_vector[2 * batch_idx_in_order] = selected_start_slot
            x_schedule_vector[2 * batch_idx_in_order + 1] = chosen_if_furnace_idx

            for t_slot_update in range(
                selected_start_slot, selected_start_slot + T_MELT
            ):
                global_if_used_slots[t_slot_update] = 1

            # Update internal M&H state to reflect this pour
            # Set M&H state to what it was just before this chosen pour
            internal_mh_levels_kg = selected_mh_levels_before_pour
            internal_mh_downtime_remaining_min = selected_mh_downtime_before_pour

            # Perform the pour
            pour_A_kg = MH_REFILL_PER_FURNACE_KG
            pour_B_kg = MH_REFILL_PER_FURNACE_KG

            internal_mh_levels_kg["A"] = min(
                internal_mh_levels_kg["A"] + pour_A_kg, MH_MAX_CAPACITY_KG["A"]
            )
            internal_mh_levels_kg["B"] = min(
                internal_mh_levels_kg["B"] + pour_B_kg, MH_MAX_CAPACITY_KG["B"]
            )

            internal_mh_downtime_remaining_min["A"] = POST_POUR_DOWNTIME_MIN
            internal_mh_downtime_remaining_min["B"] = POST_POUR_DOWNTIME_MIN

            internal_last_mh_sim_minute = selected_melt_finish_minute  # M&H state is now known AT this minute, after pour

            if_furnace_assignment_counter += (
                1  # Advance furnace assignment for next batch
            )
            found_slot_for_batch = True
            # No break here, as we've processed one batch and will continue to the next in batch_order

        if not found_slot_for_batch:
            # If no slot could be found (e.g. M&H never becomes ready or IF slots all conflict)
            # Assign a fallback (e.g. very late, or rely on -1 to be penalized by scheduling_cost)
            # For now, leave as -1, to be heavily penalized by scheduling_cost if it can't handle it.
            # This should ideally be caught by high penalties in scheduling_cost for unplaced batches.
            # print(
            #     f"Warning in greedy_assignment: Could not find M&H-aware slot for batch {batch_idx_in_order + 1}."
            # )
            # As a simple fallback, try to place it like the old greedy, ignoring M&H, but this might be bad.
            # For now, we let it remain -1 if truly no M&H-aware slot is found.
            # This means scheduling_cost MUST be robust to x values of -1 (or handle them).
            # The current scheduling_cost converts x to int, so -1 might cause issues if not bounded.
            # Let's ensure scheduling_cost handles negative start times by clamping them.
            # Or, assign a very late slot that will likely be invalid and heavily penalized.
            x_schedule_vector[2 * batch_idx_in_order] = (
                TOTAL_SLOTS  # Invalid start, will be clamped/penalized
            )
            x_schedule_vector[2 * batch_idx_in_order + 1] = chosen_if_furnace_idx

    # Ensure any -1 are replaced, e.g., with a highly penalized slot or handle in scheduling_cost
    # The current fallback above assigns TOTAL_SLOTS, which scheduling_cost should clamp.
    # If a batch truly couldn't be scheduled, its x entries would be TOTAL_SLOTS.
    # `scheduling_cost` clamps `start = max(0, min(start, TOTAL_SLOTS - T_MELT))`.
    # So `TOTAL_SLOTS` becomes `TOTAL_SLOTS - T_MELT`. This is a valid placement, but likely suboptimal.
    # A batch that is "unscheduled" (still -1) would need special handling in scheduling_cost or a huge penalty.
    # The penalty for unpoured_batches in scheduling_cost (via simulate_mh_consumption_v2) might cover this
    # if unscheduled batches translate to unpoured.

    return x_schedule_vector.astype(float)


# --- คลาส HGA Problem (Level 1) ---
class HGAProblem(Problem):  # สืบทอดจาก Problem
    def __init__(self, num_batches_hga=NUM_BATCHES):  # Renamed num_batches
        super().__init__(
            n_var=num_batches_hga,
            n_obj=2,
            n_constr=0,
            xl=0,
            xu=num_batches_hga - 1,  # Use num_batches_hga
        )  # Bounds สำหรับ Permutation

    def _evaluate(self, P, out, *args, **kwargs):
        # P คือ population ณ generation ปัจจุบัน (array of permutations)
        results_f = []  # เก็บ objective values [energy, makespan]
        evaluated_schedules_x = []  # Store x_schedule_vector for each individual in P
        all_cost_components = []  # Store cost_components for each individual

        # วนลูปแต่ละ Permutation ใน Population
        for batch_order_perm in P:  # Renamed batch_order
            # 1. แปลง Permutation เป็น x_schedule_vector (schedule) โดย Greedy Assignment
            x_sched_vec = greedy_assignment(
                batch_order_perm, num_batches=self.n_var
            )  # Renamed x
            evaluated_schedules_x.append(x_sched_vec)  # Store the generated schedule

            # 2. คำนวณ Objectives โดยใช้ scheduling_cost เดิม
            energy, makespan, cost_components = scheduling_cost(x_sched_vec)
            all_cost_components.append(cost_components)  # Store the detailed components

            # เพิ่ม penalty สูงมากถ้า greedy assignment ล้มเหลว (เช่น หา slot ไม่ได้)
            # (ตรวจจาก x_sched_vec ที่ได้ หรือเพิ่ม flag จาก greedy_assignment)
            # ตัวอย่าง: ตรวจสอบว่า makespan ดูสมเหตุสมผลไหม
            if makespan > TOTAL_SLOTS * SLOT_DURATION * 1.1:  # ถ้า makespan ยาวผิดปกติ
                energy += 1e15  # ลงโทษหนักๆ

            results_f.append([energy, makespan])

        # กำหนดค่า Objectives ให้กับ Population
        out["F"] = np.array(results_f)
        out["schedules"] = np.array(
            evaluated_schedules_x
        )  # Attach all generated schedules to the output
        out["cost_details"] = all_cost_components  # Attach all cost component dicts


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
    F_results = result.F  # Renamed F to F_results Objective values [energy, makespan]
    plt.figure(figsize=(10, 6))
    plt.scatter(
        F_results[:, 0],
        F_results[:, 1],
        c="red",
        s=40,
        edgecolors="k",
        label="HGA Pareto Front",
    )  # เปลี่ยนสี/label
    plt.xlabel("Total Energy Consumption")
    plt.ylabel("Makespan (minutes)")
    plt.title("HGA Pareto Front for Melting Schedule")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ===== แสดงผลลัพธ์และ Plot ตารางเวลาสำหรับ Solutions ที่ดีที่สุด =====
    print("\nProcessing results...")

    opt_population = (
        result.opt
    )  # Population object containing non-dominated individuals

    num_to_show = min(5, len(opt_population) if opt_population is not None else 0)

    if num_to_show == 0:
        print("\nNo solutions found in the Pareto front to display.")
    else:
        print(f"\nShowing top {num_to_show} (up to 5) solutions from the Pareto front:")

    for i in range(num_to_show):
        individual = opt_population[i]
        permutation = individual.X  # The permutation (decision variables)
        energy, makespan = individual.F  # The objective values

        # Retrieve the stored x_vector (schedule) generated during evaluation
        x_vector = individual.get("schedules")
        cost_details = individual.get("cost_details")  # Retrieve cost details

        print(f"\nSolution #{i+1}")
        print(f"  Batch Order (Permutation): {permutation}")
        # print(f"  Total Energy (from F): {energy:.2f}") # This is 'cost' which includes penalties
        # print(f"  Makespan (from F)    : {makespan:.2f}")

        if cost_details:
            print(f"  Total Cost (Objective 1) : {cost_details['total_cost']:.2f}")
            print(
                f"  Makespan (Objective 2)   : {cost_details['makespan_minutes']:.2f} min"
            )
            print(f"  --------------------------------------------------")
            print(f"  Cost Component Details:")
            print(
                f"    Base IF Energy (kWh)     : {cost_details['base_if_energy_kwh']:.2f}"
            )

            num_actual_batches = NUM_BATCHES  # Assumes all batches are processed
            if cost_details["base_if_energy_kwh"] > 0 and num_actual_batches > 0:
                avg_if_energy_per_batch = (
                    cost_details["base_if_energy_kwh"] / num_actual_batches
                )
                print(
                    f"      -> Avg Base IF Energy/Batch (kWh): {avg_if_energy_per_batch:.2f}"
                )

            print(
                f"    IF Holding Penalty       : {cost_details['if_holding_penalty']:.2f}"
            )
            # Note: To make IF Holding an energy component (kWh):
            # 1. Define IF_STANDBY_POWER_KW = {"A": val, "B": val}
            # 2. Modify simulate_mh_consumption_v2 to calculate actual holding time per furnace IF A and B.
            # 3. In scheduling_cost, calculate holding_energy_kwh = (holding_mins_A/60 * standby_A_kw) + (holding_mins_B/60 * standby_B_kw)
            # 4. Replace if_holding_penalty_cost with this holding_energy_kwh in total cost and here.
            # For now, if IF_HOLDING_ENERGY_PENALTY_PER_MINUTE is set to a kW value, and if total_if_holding_minutes is for one furnace type:
            # actual_holding_energy_kwh = (total_if_holding_minutes / 60.0) * IF_HOLDING_ENERGY_PENALTY_PER_MINUTE (if it was kW)
            # Since IF_HOLDING_ENERGY_PENALTY_PER_MINUTE is currently 0, this part is 0.

            print(
                f"    IF General Penalty       : {cost_details['if_general_penalty']:.2f} (Overlaps, Gaps, Breaks)"
            )
            print(
                f"    MH Idle Penalty          : {cost_details['mh_idle_penalty']:.2f}"
            )
            print(
                f"    MH Reheat Penalty        : {cost_details['mh_reheat_penalty']:.2f} (Placeholder)"
            )
            print(
                f"    MH Overflow Penalty      : {cost_details['mh_overflow_penalty']:.2f}"
            )
            print(
                f"    Unpoured Batch Penalty   : {cost_details['unpoured_batch_penalty']:.2f}"
            )
            print(
                f"    MH Energy (kWh)          : {cost_details['mh_energy_kwh']:.2f} (Placeholder)"
            )
            print(f"  --------------------------------------------------")
        else:
            # Fallback if cost_details are not available for some reason
            print(f"  Total Cost (from F)      : {energy:.2f}")
            print(f"  Makespan (from F)        : {makespan:.2f} min")

        # --- Plot ตารางเวลา using the retrieved x_vector ---
        if x_vector is not None:
            print(f"  Retrieved Schedule Vec (x): {np.round(x_vector, 2)}")
            # Check if the schedule vector indicates any valid scheduling attempt
            # (e.g., not all default -1 if greedy_assignment could return that for full failure)
            # The current greedy_assignment assigns TOTAL_SLOTS for fallback.
            # decode_schedule handles clamping.
            try:
                schedule = decode_schedule(x_vector)
                if schedule:  # Ensure schedule is not empty
                    # Use makespan from F_results for the title, as it's the 'evaluated' makespan
                    plot_title = f"HGA Schedule - Sol {i+1} (Energy: {energy:.0f}, Makespan: {makespan:.0f})"

                    plot_schedule_and_mh(
                        schedule,
                        title=plot_title,
                        simulated_time_points=None,  # Trigger re-simulation in plot function for M&H
                        simulated_mh_levels=None,  # Trigger re-simulation in plot function for M&H
                    )
                else:
                    print(
                        "  Could not decode schedule for plotting (empty or invalid schedule returned from evaluation)."
                    )
            except Exception as e:
                print(f"  Error plotting schedule for solution {i+1}: {e}")
                import traceback

                print(traceback.format_exc())
        else:
            print(
                f"  Could not retrieve schedule vector (x) for plotting for solution {i+1} (x_vector is None)."
            )


if __name__ == "__main__":
    main()
