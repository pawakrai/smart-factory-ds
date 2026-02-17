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
from typing import Dict, Tuple


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
IF_HOLDING_ENERGY_PENALTY_PER_MINUTE = 7.5  # Example value, adjust as needed
# Very high penalty for any batch that cannot be poured by the end of simulation
UNPOURED_BATCH_PENALTY = 1e13  # Increased penalty
# Penalty for idle time between consecutive batches on the same IF furnace.
# ค่าตรงนี้แทน \"โอกาสที่สูญเสีย\" เมื่อเตาหลอมถูกปล่อยว่าง ทั้งในแง่พลังงานสูญเปล่า
# (ต้องอุ่นใหม่ / cold start) และกำลังการผลิตที่ไม่ถูกใช้
IF_GAP_TIME_PENALTY_RATE_PER_MINUTE = 1.0  # cost unit per minute of gap
# Penalty for IF working during designated break times
IF_WORKING_IN_BREAK_PENALTY_PER_MINUTE = (
    10000  # Example: 100 cost unit per minute of work in break
)
# Average Power Rating for each IF in kW
IF_POWER_RATING_KW = {"A": 420, "B": 420}

# === New: Per-batch power options & empirical profiles (from plant data) ===
# Options of maximum power that can be used per melting batch (kW)
IF_POWER_OPTIONS = [450.0, 475.0, 500.0]

# Empirical profiles for a "hot" furnace (no significant cooldown between batches).
# Values are approximate means derived from recent plant data in
# `data/สรุปการหลอมทุก Batch new.xlsx`.
# Each entry maps power -> (expected_duration_min_hot, expected_energy_kwh_hot)
POWER_PROFILE: Dict[float, Dict[str, float]] = {
    # 3 samples, Energy ≈ [593, 614, 576], Duration ≈ [94, 92, 88]
    450.0: {"duration_min_hot": 88.0, "energy_kwh_hot": 565.8},
    # 2 samples, Energy ≈ [613, 579], Duration ≈ [94, 90]
    475.0: {"duration_min_hot": 86.0, "energy_kwh_hot": 582.5},
    # 4 samples, Energy ≈ [585, 585, 576, 576], Duration ≈ [84, 84, 84, 86]
    500.0: {"duration_min_hot": 84.0, "energy_kwh_hot": 587.4},
}

# Default power levels that the heuristic will use for each batch type.
# These can be adjusted easily when experimenting with different policies.
IF_POWER_FOR_HOT_BATCH_KW = 450
IF_POWER_FOR_COLD_START_BATCH_KW = 450

# Cool / Cold start modelling (gap-based)
# If the idle gap since the previous batch on the same furnace exceeds this threshold,
# the next batch is treated as a "cold start".
COLD_START_GAP_THRESHOLD_MIN = 180.0  # minutes
# Additional time & energy associated with a cold start, relative to the hot profile.
COLD_START_EXTRA_DURATION_MIN = 8.0
COLD_START_EXTRA_ENERGY_KWH = 30.0

# เตา M&H (โค้ดเดิมไว้ plot)
MH_MAX_CAPACITY_KG = {"A": 400.0, "B": 250.0}  # ความจุสูงสุดแต่ละเตา (kg)
MH_INITIAL_LEVEL_KG = {"A": 400.0, "B": 230.0}  # ระดับเริ่มต้น (kg)
MH_CONSUMPTION_RATE_KG_PER_MIN = {"A": 2.80, "B": 2.50}  # อัตราการใช้ kg/min แต่ละเตา
MH_EMPTY_THRESHOLD_KG = 0  # ระดับต่ำสุดที่ยอมรับได้ (kg)
IF_BATCH_OUTPUT_KG = 500.0  # ปริมาณที่ IF ผลิตต่อ batch (kg)
POST_POUR_DOWNTIME_MIN = 10  # เวลาหยุดหลังเทเสร็จ (นาที)
MH_IDLE_PENALTY_RATE = 1.0  # Penalty ต่อนาทีที่เตา M&H ว่าง (ไม่มีโลหะให้ใช้งาน)
MH_REHEAT_PENALTY_RATE = 20.0  # Placeholder penalty สำหรับ reheat
OVERFLOW_PENALTY_PER_KG = 50.0
DEBUG = False

# ระดับน้ำโลหะที่ถือว่า \"ปลอดภัย\" สำหรับการทำงานของหัวพ่นไฟ (Operational level)
# หากต่ำกว่านี้จะถือว่าเตาทำงานไม่มีประสิทธิภาพ (ต้องใช้พลังงานมากขึ้น / อุ่นได้ช้าลง)
MH_MIN_OPERATIONAL_LEVEL_KG = {
    "A": 0.5 * MH_MAX_CAPACITY_KG["A"],  # 50% ของความจุสูงสุด
    "B": 0.5 * MH_MAX_CAPACITY_KG["B"],
}
# Penalty ต่อนาทีเมื่อระดับน้ำอยู่ในช่วง (0, MH_MIN_OPERATIONAL_LEVEL_KG)
# แทนต้นทุนโอกาสจากกำลังการผลิตที่ลดลงและประสิทธิภาพการถ่ายเทความร้อนที่แย่ลง
MH_LOW_LEVEL_PENALTY_RATE = 200.0


def _select_if_power_kw(is_cold_start: bool) -> float:
    return (
        IF_POWER_FOR_COLD_START_BATCH_KW if is_cold_start else IF_POWER_FOR_HOT_BATCH_KW
    )


def _get_power_profile(power_kw: float) -> Dict[str, float]:
    return POWER_PROFILE.get(power_kw, next(iter(POWER_PROFILE.values())))


def _compute_batch_profile(is_cold_start: bool) -> Dict[str, float]:
    power_kw = _select_if_power_kw(is_cold_start)
    profile = _get_power_profile(power_kw)
    duration_min = profile["duration_min_hot"]
    energy_kwh = profile["energy_kwh_hot"]
    cold_start_energy_kwh = 0.0
    if is_cold_start:
        duration_min += COLD_START_EXTRA_DURATION_MIN
        cold_start_energy_kwh = COLD_START_EXTRA_ENERGY_KWH
        energy_kwh += cold_start_energy_kwh
    return {
        "power_kw": power_kw,
        "duration_min": duration_min,
        "energy_kwh": energy_kwh,
        "is_cold_start": is_cold_start,
        "cold_start_energy_kwh": cold_start_energy_kwh,
    }


# ช่วงเวลาที่ต้องการให้เตา IF ทำงานเพื่อใช้ไฟฟ้าต้นทุนต่ำ (เช่น จาก Solar)
# ระบุเป็นนาทีตั้งแต่เริ่มวัน เช่น 12:00-13:00 => (12*60, 13*60)
SOLAR_PREFERRED_WINDOWS_MIN = [
    (12 * 60, 13 * 60),  # ช่วงเที่ยง – บ่ายหนึ่ง
]
# ตัวคูณส่วนลดต้นทุนพลังงานเมื่ออยู่ในช่วง Solar เต็ม ๆ
# 0.0 = ไม่มีส่วนลด, 1.0 = พลังงานในช่วงนี้ไม่คิด cost เลย
SOLAR_ENERGY_DISCOUNT_FACTOR = 0.5

# กำหนดช่วงเวลาพัก (นาทีตั้งแต่เริ่มวัน 0 - 1440)
# ตัวอย่าง: 9:00-9:15 และ 12:00-13:00
BREAK_TIMES_MINUTES = [
    # (8 * 60, 8 * 60 + 40),  # 08:00 - 08:40
    # (20 * 60, 20 * 60 + 40),  # 20:00 - 20:40
]

# +++ NEW CONFIG FOR MH FILLING PREFERENCE +++
PREFERRED_MH_FURNACE_TO_FILL_FIRST = (
    "B"  # Options: "A", "B", or None for default/other logic
)

# --- Plotting Config ---
FURNACE_COLORS = {"A": "blue", "B": "green"}
MH_FURNACE_COLORS = {"A": "red", "B": "orange"}

# furnace plot position (สำหรับ IF Gantt)
furnace_y = {0: 10, 1: 25}
height = 8

SHIFT_START = 8 * 60


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
    # 2.7) IF energy & cold start accounting
    # ----------------------------
    base_total_energy_if_kwh = 0.0  # พลังงานรวมจริงของ IF (kWh) ไม่คิดราคา
    base_if_energy_kwh_no_reheat = 0.0
    cold_start_energy_kwh = 0.0
    priced_total_energy_if_cost = 0.0  # ต้นทุนพลังงานของ IF หลังคิดส่วนลด Solar
    num_cold_start_events = 0

    def _fraction_overlap_with_windows(
        start_minute: float,
        duration_min: float,
        windows: list[tuple[int, int]],
    ) -> float:
        """
        คำนวณสัดส่วนเวลาของ batch ที่ทับกับช่วงเวลาที่ระบุใน windows.
        """
        if duration_min <= 0 or not windows:
            return 0.0
        end_minute = start_minute + duration_min
        total_overlap = 0.0
        for w_start, w_end in windows:
            overlap_start = max(start_minute, w_start)
            overlap_end = min(end_minute, w_end)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
        return total_overlap / duration_min

    # ----------------------------
    # 3) Apply reheat-aware timeline (batch continues at max power until pour)
    # ----------------------------
    reheat_result = _apply_reheat_and_shift_schedule(schedule)
    batch_timing = reheat_result["batch_timing"]
    batch_profiles = reheat_result["batch_profiles"]
    (
        MH_idle_penalty,
        MH_reheat_penalty,  # Still placeholder
        total_energy_mh,  # Still placeholder
        time_points,
        mh_levels,
        actual_pour_events,
        unpoured_batches_at_end,
        total_if_holding_minutes,
        pour_induced_mh_overflow_penalty,
        MH_low_level_penalty,
        batch_pour_times,
    ) = reheat_result["sim_outputs"]

    # Compute IF energy and cold-start count from reheat-aware timing
    for b_id, profile in batch_profiles.items():
        timing = batch_timing.get(b_id)
        if not timing:
            continue
        base_if_energy_kwh_no_reheat += profile["energy_kwh"]
        base_total_energy_if_kwh += profile["energy_kwh"]
        if profile["is_cold_start"]:
            num_cold_start_events += 1
            cold_start_energy_kwh += profile["cold_start_energy_kwh"]

        solar_frac = _fraction_overlap_with_windows(
            timing["start_min"],
            profile["duration_min"],
            SOLAR_PREFERRED_WINDOWS_MIN,
        )
        energy_cost_kwh = profile["energy_kwh"] * (
            1.0 - SOLAR_ENERGY_DISCOUNT_FACTOR * solar_frac
        )
        priced_total_energy_if_cost += energy_cost_kwh

    # Reheat energy (kWh) at max IF power
    reheat_energy_kwh = 0.0
    for s, e, f, b_id in schedule:
        timing = batch_timing.get(b_id)
        if not timing:
            continue
        reheat_minutes = max(0.0, timing["pour_min"] - timing["melt_finish_min"])
        furnace_key = "A" if f == 0 else "B"
        reheat_energy_kwh += reheat_minutes * IF_POWER_RATING_KW[furnace_key] / 60.0

    base_total_energy_if_kwh += reheat_energy_kwh
    priced_total_energy_if_cost += reheat_energy_kwh

    # --- ลบการตรวจสอบ M&H Overflow Penalty แบบเดิมออก ---
    # penalty_mh_overflow = 0.0 (This is now handled by pour_induced_mh_overflow_penalty from sim)

    # ----------------------------
    # 5) รวม cost
    # ----------------------------
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
    # The 'cost' objective nowสะท้อน \"ต้นทุนพลังงาน\" หลังคิดส่วนลด Solar + penalty อื่น ๆ
    cost = 0.0
    cost += priced_total_energy_if_cost  # IF energy cost after Solar discount
    cost += if_holding_penalty_cost  # This is still a penalty value, not kWh unless redefined
    cost += total_energy_mh  # พลังงาน M&H (ยังเป็น placeholder)
    cost += (
        penalty_if  # Overlap IF / ปิดเตา / Gap Time / IF in Break (จากการวางแผนเบื้องต้น)
    )
    cost += MH_idle_penalty  # Penalty จาก M&H idle
    cost += MH_reheat_penalty  # Penalty M&H reheat (ยังเป็น placeholder)
    cost += MH_low_level_penalty  # Penalty M&H ทำงานที่ระดับน้ำต่ำ
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
        # พลังงานจริงของ IF (kWh) และต้นทุนหลังคิด Solar ส่วนลด
        "base_if_energy_kwh": base_total_energy_if_kwh,
        "base_if_energy_kwh_no_reheat": base_if_energy_kwh_no_reheat,
        "cold_start_energy_kwh": cold_start_energy_kwh,
        "priced_if_energy_cost_kwh": priced_total_energy_if_cost,
        "if_holding_penalty": if_holding_penalty_cost,
        "if_general_penalty": penalty_if,  # Contains overlap, gap, break penalties
        "mh_idle_penalty": MH_idle_penalty,
        "mh_reheat_penalty": MH_reheat_penalty,  # Placeholder
        "mh_overflow_penalty": pour_induced_mh_overflow_penalty,
        "mh_low_level_penalty": MH_low_level_penalty,
        "unpoured_batch_penalty": unpoured_batches_penalty_cost,
        "mh_energy_kwh": total_energy_mh,  # Placeholder
        "num_cold_start_events": num_cold_start_events,
        "reheat_energy_kwh": reheat_energy_kwh,
        "batch_timing": batch_timing,
    }

    return cost, makespan_min_actual, cost_components


def _apply_reheat_and_shift_schedule(schedule, max_iterations: int = 6):
    """
    Shift IF timeline so each batch continues reheating at max power until poured.
    Returns per-batch timing and the final M&H simulation outputs.
    """
    if not schedule:
        sim_outputs = simulate_mh_consumption_v2([])
        return {"batch_timing": {}, "sim_outputs": sim_outputs}

    planned = sorted(schedule, key=lambda item: item[0])
    planned_start_min = {b_id: s * SLOT_DURATION for s, e, f, b_id in planned}
    actual_start_min = dict(planned_start_min)
    sim_outputs = None
    batch_profiles = {}

    def _build_profiles_and_events(start_map, prev_pour_times):
        profiles = {}
        melt_events = []
        last_end_by_furnace = {}

        for s, e, f, b_id in planned:
            start_min = start_map[b_id]
            prev_end = last_end_by_furnace.get(f)
            if prev_end is None:
                is_cold_start = True
            else:
                gap_minutes = start_min - prev_end
                is_cold_start = gap_minutes >= COLD_START_GAP_THRESHOLD_MIN

            profile = _compute_batch_profile(is_cold_start)
            profiles[b_id] = profile
            melt_finish = start_min + profile["duration_min"]
            melt_events.append((melt_finish, b_id))

            last_end_by_furnace[f] = prev_pour_times.get(b_id, melt_finish)

        melt_events.sort(key=lambda w: w[0])
        return profiles, melt_events

    prev_pour_times = {}
    for _ in range(max_iterations):
        batch_profiles, melt_events = _build_profiles_and_events(
            actual_start_min, prev_pour_times
        )
        sim_outputs = simulate_mh_consumption_v2(melt_events)
        batch_pour_times = sim_outputs[-1]

        new_actual_start_min = {}
        prev_end = None
        for s, e, f, b_id in planned:
            planned_start = planned_start_min[b_id]
            start_min = (
                planned_start if prev_end is None else max(planned_start, prev_end)
            )
            new_actual_start_min[b_id] = start_min

            profile = batch_profiles.get(b_id) or _compute_batch_profile(True)
            melt_finish = start_min + profile["duration_min"]
            pour_min = batch_pour_times.get(b_id, melt_finish)
            prev_end = max(prev_end if prev_end is not None else 0.0, pour_min)

        if new_actual_start_min == actual_start_min:
            break
        actual_start_min = new_actual_start_min
        prev_pour_times = batch_pour_times

    batch_timing = {}
    batch_pour_times = sim_outputs[-1] if sim_outputs else {}
    batch_profiles, _ = _build_profiles_and_events(actual_start_min, batch_pour_times)
    for s, e, f, b_id in planned:
        start_min = actual_start_min[b_id]
        profile = batch_profiles.get(b_id) or _compute_batch_profile(True)
        melt_finish = start_min + profile["duration_min"]
        pour_min = batch_pour_times.get(b_id, melt_finish)
        batch_timing[b_id] = {
            "start_min": start_min,
            "melt_finish_min": melt_finish,
            "pour_min": pour_min,
            "furnace": f,
        }

    return {
        "batch_timing": batch_timing,
        "batch_profiles": batch_profiles,
        "sim_outputs": sim_outputs,
    }


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
    total_low_level_penalty = 0.0  # Penalty for operating with low metal level
    pour_induced_mh_overflow_penalty = 0.0  # New

    ready_to_pour_queue = []  # Stores (melt_finish_minute, batch_id)
    actual_pour_events = []  # Stores (actual_pour_minute, batch_id)
    batch_pour_times = {}  # batch_id -> actual pour minute
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

            # อนุญาตให้พยายามเทได้ถ้ามี capacity มากกว่า 0
            # ส่วนที่เหลือจากการเทจะถูกนับเป็น overflow และลงโทษ
            if total_available_mh_capacity > 0:
                # print(f"Minute {t}: Attempting to pour Batch {batch_id_q}. Total M&H available: {total_available_mh_capacity:.1f} kg.") # Debug

                # --- NEW PRIORITIZED POURING LOGIC ---
                poured_amount_A_this_batch = 0
                poured_amount_B_this_batch = 0
                remaining_metal_to_distribute = IF_BATCH_OUTPUT_KG

                furnaces_in_fill_order = []
                preferred_f = PREFERRED_MH_FURNACE_TO_FILL_FIRST

                if preferred_f == "A":
                    furnaces_in_fill_order = ["A", "B"]
                elif preferred_f == "B":
                    furnaces_in_fill_order = ["B", "A"]
                else:
                    # Default behavior if PREFERRED_MH_FURNACE_TO_FILL_FIRST is None or invalid
                    # Fallback: try to fill the one with more space first, then the other.
                    # For now, if not "A" or "B", let's default to A then B as a simple fallback.
                    # print(f"Warning: PREFERRED_MH_FURNACE_TO_FILL_FIRST ('{preferred_f}') is not 'A' or 'B'. Defaulting to A then B.")
                    furnaces_in_fill_order = ["A", "B"]

                for f_id in furnaces_in_fill_order:
                    if remaining_metal_to_distribute <= 0:
                        break

                    space_in_this_furnace = (
                        MH_MAX_CAPACITY_KG[f_id] - current_level[f_id]
                    )
                    amount_to_pour_here = min(
                        remaining_metal_to_distribute, space_in_this_furnace
                    )

                    if amount_to_pour_here > 0:
                        if f_id == "A":
                            poured_amount_A_this_batch = amount_to_pour_here
                        else:  # f_id == "B"
                            poured_amount_B_this_batch = amount_to_pour_here
                        remaining_metal_to_distribute -= amount_to_pour_here

                overflow_kg = max(0.0, remaining_metal_to_distribute)
                pour_induced_mh_overflow_penalty += (
                    overflow_kg * OVERFLOW_PENALTY_PER_KG
                )

                current_level["A"] = min(
                    current_level["A"] + poured_amount_A_this_batch,
                    MH_MAX_CAPACITY_KG["A"],
                )
                current_level["B"] = min(
                    current_level["B"] + poured_amount_B_this_batch,
                    MH_MAX_CAPACITY_KG["B"],
                )
                # --- END NEW PRIORITIZED POURING LOGIC ---

                for furnace_id_mh in ["A", "B"]:
                    downtime_remaining[furnace_id_mh] = POST_POUR_DOWNTIME_MIN
                    mh_status[furnace_id_mh] = "downtime"

                actual_pour_events.append((t, batch_id_q))
                batch_pour_times[batch_id_q] = t
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

            # Apply idle penalty if the furnace is idle (ไม่มีโลหะให้ใช้)
            if mh_status[furnace_id] == "idle":
                total_idle_penalty += MH_IDLE_PENALTY_RATE

            # Apply low-level penalty whenระดับน้ำอยู่ต่ำกว่า operational level แต่ยังไม่ว่าง
            current_level_now = current_level[furnace_id]
            if (
                current_level_now > MH_EMPTY_THRESHOLD_KG
                and current_level_now < MH_MIN_OPERATIONAL_LEVEL_KG[furnace_id]
            ):
                total_low_level_penalty += MH_LOW_LEVEL_PENALTY_RATE

            mh_levels[furnace_id][t] = current_level[furnace_id]

    total_energy_mh = 0.0  # Placeholder
    unpoured_batches_at_end = [
        item[1] for item in ready_to_pour_queue
    ]  # Batches remaining in queue
    for batch_id in unpoured_batches_at_end:
        # If never poured, treat as reheating until end of simulation window
        batch_pour_times.setdefault(batch_id, simulation_duration_min)

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
        total_low_level_penalty,
        batch_pour_times,
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
    batch_timing=None,
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
        if batch_timing:
            for b_id, timing in batch_timing.items():
                melt_completion_events_plot.append((timing["melt_finish_min"], b_id))
        else:
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
            _,  # low level penalty (not needed for plotting)
            _,  # batch_pour_times (not needed for plotting)
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
        if batch_timing and b_id in batch_timing:
            timing = batch_timing[b_id]
            melt_start_t = SHIFT_START + timing["start_min"]
            melt_finish_t = SHIFT_START + timing["melt_finish_min"]
            pour_t = SHIFT_START + timing["pour_min"]
            melt_dur = max(0.0, melt_finish_t - melt_start_t)
            reheat_dur = max(0.0, pour_t - melt_finish_t)

            ax1.broken_barh(
                [(melt_start_t, melt_dur)],
                (furnace_y[f], height),
                facecolors="gray",
                edgecolor="black",
            )
            if reheat_dur > 0:
                ax1.broken_barh(
                    [(melt_finish_t, reheat_dur)],
                    (furnace_y[f], height),
                    facecolors="red",
                    edgecolor="black",
                )
        else:
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


def format_schedule_breakdown(cost_components: dict) -> str:
    if not cost_components:
        return "Cost component details are not available."

    base_if_energy = cost_components.get("base_if_energy_kwh", 0.0)
    base_if_no_reheat = cost_components.get("base_if_energy_kwh_no_reheat", 0.0)
    reheat_energy = cost_components.get("reheat_energy_kwh", 0.0)
    cold_start_energy = cost_components.get("cold_start_energy_kwh", 0.0)

    lines = [
        "Cost Component Details:",
        f"  Base IF Energy (kWh)     : {base_if_energy:.2f}",
        f"    - Melt Energy (kWh)    : {base_if_no_reheat:.2f}",
        f"    - Cold Start Extra     : {cold_start_energy:.2f}",
        f"    - Reheat Energy (kWh)  : {reheat_energy:.2f}",
        f"  IF Holding Penalty       : {cost_components.get('if_holding_penalty', 0.0):.2f}",
        f"  IF General Penalty       : {cost_components.get('if_general_penalty', 0.0):.2f} (Overlaps, Gaps, Breaks)",
        f"  MH Idle Penalty          : {cost_components.get('mh_idle_penalty', 0.0):.2f}",
        f"  MH Reheat Penalty        : {cost_components.get('mh_reheat_penalty', 0.0):.2f} (Placeholder)",
        f"  MH Overflow Penalty      : {cost_components.get('mh_overflow_penalty', 0.0):.2f}",
        f"  MH Low Level Penalty     : {cost_components.get('mh_low_level_penalty', 0.0):.2f}",
        f"  Unpoured Batch Penalty   : {cost_components.get('unpoured_batch_penalty', 0.0):.2f}",
        f"  MH Energy (kWh)          : {cost_components.get('mh_energy_kwh', 0.0):.2f} (Placeholder)",
    ]
    return "\n".join(lines)


def _greedy_fast_forward_mh_state(
    target_minute: int,
    current_levels_kg: dict,  # {"A": level, "B": level}
    downtime_remaining_min: dict,  # {"A": dt, "B": dt}
    last_simulated_minute: int,
):
    """
    O(1) forward for greedy lookahead (no pours in between).
    Returns state at the BEGINNING of target_minute.
    """
    dt = max(0, target_minute - last_simulated_minute - 1)
    if dt == 0:
        return current_levels_kg.copy(), downtime_remaining_min.copy()

    new_levels = current_levels_kg.copy()
    new_down = downtime_remaining_min.copy()

    for f in ["A", "B"]:
        d0 = new_down[f]
        blocked = min(dt, d0)
        run_minutes = dt - blocked

        new_down[f] = max(0, d0 - dt)

        if run_minutes > 0:
            new_levels[f] = max(
                0.0,
                new_levels[f] - run_minutes * MH_CONSUMPTION_RATE_KG_PER_MIN[f],
            )

    return new_levels, new_down


def _score_candidate_slot(
    start_slot: int,
    melt_finish_minute: int,
    mh_levels_before_pour: dict,
    total_available_mh_capacity: float,
    last_if_end_slot_for_furnace: int | None,
) -> tuple[float, float, float, float]:
    """
    Deterministic slot scoring for greedy decoder.
    Designed to reduce noisy fitness and avoid unstable random choices.
    """
    if last_if_end_slot_for_furnace is None:
        predicted_if_gap_minutes = 0.0
    else:
        predicted_if_gap_minutes = max(
            0, (start_slot - last_if_end_slot_for_furnace) * SLOT_DURATION
        )

    low_level_risk = 0.0
    for f_id in ["A", "B"]:
        low_level_risk += max(
            0.0, MH_MIN_OPERATIONAL_LEVEL_KG[f_id] - mh_levels_before_pour[f_id]
        )

    shortfall_kg = max(0.0, IF_BATCH_OUTPUT_KG - total_available_mh_capacity)

    w_makespan = 1.0
    w_low = 30.0
    w_gap = 10.0
    w_shortfall = 3000.0
    score = (
        w_makespan * melt_finish_minute
        + w_low * low_level_risk
        + w_gap * predicted_if_gap_minutes
        + w_shortfall * shortfall_kg
    )
    return score, shortfall_kg, predicted_if_gap_minutes, low_level_risk


def _validate_no_global_if_overlap(x_schedule_vector, num_batches: int) -> bool:
    usage = np.zeros(TOTAL_SLOTS, dtype=int)
    for i in range(num_batches):
        start = int(x_schedule_vector[2 * i])
        if start < 0:
            continue
        end = start + T_MELT
        if start < 0 or end > TOTAL_SLOTS:
            return False
        for t_slot in range(start, end):
            if usage[t_slot] > 0:
                return False
            usage[t_slot] = 1
    return True


# --- ฟังก์ชัน Greedy Assignment (Level 2) ---
def greedy_assignment(batch_order, num_batches=NUM_BATCHES):
    """
    กำหนด start_slot และ furnace สำหรับลำดับ batch ที่กำหนด
    โดยห้ามมี batch ใดๆ ทำงานซ้อนทับกันเลย (Global Constraint for IFs)
    และพยายามจัดตาราง IF ให้สอดคล้องกับความพร้อมของ M&H (M&H-aware).
    Deterministic decoder:
    - no random choice (stable fitness for same permutation)
    - uses slot scoring
    - fallback avoids clamp-induced overlap explosion
    """
    x_schedule_vector = np.full(
        num_batches * 2, -1, dtype=int
    )  # Initialize with -1 (unscheduled)
    global_if_used_slots = np.zeros(TOTAL_SLOTS, dtype=int)
    decoder_feasible = True

    available_if_furnaces_for_assignment = []
    if USE_FURNACE_A:
        available_if_furnaces_for_assignment.append(0)
    if USE_FURNACE_B:
        available_if_furnaces_for_assignment.append(1)

    if not available_if_furnaces_for_assignment:
        print("Error in greedy_assignment: No available IF furnaces!")
        return x_schedule_vector.astype(float), False

    # Internal M&H state for greedy's simulation
    internal_mh_levels_kg = MH_INITIAL_LEVEL_KG.copy()
    internal_mh_downtime_remaining_min = {"A": 0, "B": 0}
    internal_last_mh_sim_minute = -1  # Simulates from minute 0 onwards

    if_furnace_assignment_counter = 0
    last_if_end_slot_by_furnace = {0: None, 1: None}
    candidate_scan_kernel = np.ones(T_MELT, dtype=int)
    max_candidate_starts_to_scan = 60

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
        viable_slots_data = []

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

        free_run = (
            np.convolve(global_if_used_slots, candidate_scan_kernel, mode="valid") == 0
        )
        candidate_starts = np.flatnonzero(free_run)

        used_slots = np.where(global_if_used_slots == 1)[0]
        min_start_slot = int(used_slots.max() + 1) if used_slots.size else 0
        min_start_slot = max(0, min_start_slot - T_MELT)

        candidate_starts = candidate_starts[candidate_starts >= min_start_slot]
        candidate_starts = candidate_starts[:max_candidate_starts_to_scan]

        for potential_if_start_slot in candidate_starts:

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
                _greedy_fast_forward_mh_state(
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
            # The fast-forward helper updates downtime state deterministically.

            available_capacity_A = MH_MAX_CAPACITY_KG["A"] - mh_levels_before_pour["A"]
            available_capacity_B = MH_MAX_CAPACITY_KG["B"] - mh_levels_before_pour["B"]
            total_available_mh_capacity = available_capacity_A + available_capacity_B

            score, shortfall_kg, _, _ = _score_candidate_slot(
                potential_if_start_slot,
                potential_if_melt_finish_minute,
                mh_levels_before_pour,
                total_available_mh_capacity,
                last_if_end_slot_by_furnace[chosen_if_furnace_idx],
            )
            candidate = {
                "start_slot": potential_if_start_slot,
                "mh_levels_before_pour": mh_levels_before_pour.copy(),
                "mh_downtime_before_pour": mh_downtime_before_pour.copy(),
                "melt_finish_minute": potential_if_melt_finish_minute,
                "score": score,
                "shortfall_kg": shortfall_kg,
            }

            # Align with simulator: candidate is viable if there is any free MH capacity.
            if total_available_mh_capacity > 0:
                viable_slots_data.append(candidate)

        selected_candidate = None
        if viable_slots_data:
            # Deterministic tie-break: score -> shortfall -> start slot -> melt finish
            selected_candidate = min(
                viable_slots_data,
                key=lambda c: (
                    c["score"],
                    c["shortfall_kg"],
                    c["start_slot"],
                    c["melt_finish_minute"],
                ),
            )

        # After checking potential slots, choose one if any were found
        if selected_candidate is not None:
            selected_start_slot = selected_candidate["start_slot"]
            selected_mh_levels_before_pour = selected_candidate["mh_levels_before_pour"]
            selected_mh_downtime_before_pour = selected_candidate[
                "mh_downtime_before_pour"
            ]
            selected_melt_finish_minute = selected_candidate["melt_finish_minute"]

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

            # Greedy state update follows simulator rule:
            # pour as much as capacity allows; the remainder is overflow (not added to level).
            g_poured_amount_A = 0
            g_poured_amount_B = 0
            g_remaining_metal = IF_BATCH_OUTPUT_KG

            g_furnaces_in_fill_order = []
            g_preferred_f = PREFERRED_MH_FURNACE_TO_FILL_FIRST

            if g_preferred_f == "A":
                g_furnaces_in_fill_order = ["A", "B"]
            elif g_preferred_f == "B":
                g_furnaces_in_fill_order = ["B", "A"]
            else:  # Fallback
                g_furnaces_in_fill_order = ["A", "B"]

            for g_f_id in g_furnaces_in_fill_order:
                if g_remaining_metal <= 0:
                    break

                g_space_in_furnace = MH_MAX_CAPACITY_KG[g_f_id] - internal_mh_levels_kg[g_f_id]
                g_amount_to_pour = min(g_remaining_metal, g_space_in_furnace)

                if g_amount_to_pour > 0:
                    if g_f_id == "A":
                        g_poured_amount_A = g_amount_to_pour
                    else:  # g_f_id == "B"
                        g_poured_amount_B = g_amount_to_pour
                    g_remaining_metal -= g_amount_to_pour

            g_overflow_kg = max(0.0, g_remaining_metal)
            _ = g_overflow_kg

            # Update internal levels (greedy doesn't track overflow penalty, just levels)
            internal_mh_levels_kg["A"] = min(
                internal_mh_levels_kg["A"] + g_poured_amount_A,
                MH_MAX_CAPACITY_KG["A"],
            )
            internal_mh_levels_kg["B"] = min(
                internal_mh_levels_kg["B"] + g_poured_amount_B,
                MH_MAX_CAPACITY_KG["B"],
            )

            internal_mh_downtime_remaining_min["A"] = POST_POUR_DOWNTIME_MIN
            internal_mh_downtime_remaining_min["B"] = POST_POUR_DOWNTIME_MIN

            # M&H state is now known at this minute.
            internal_last_mh_sim_minute = selected_melt_finish_minute
            last_if_end_slot_by_furnace[chosen_if_furnace_idx] = (
                selected_start_slot + T_MELT
            )

            if_furnace_assignment_counter += (
                1  # Advance furnace assignment for next batch
            )
            found_slot_for_batch = True
            # No break here, as we've processed one batch and will continue to the next in batch_order

        if not found_slot_for_batch:
            # No globally free IF block left: return deterministic infeasible decode.
            decoder_feasible = False
            break

    # Quick guardrail: decoder output should never contain global overlap.
    if not _validate_no_global_if_overlap(x_schedule_vector, num_batches):
        decoder_feasible = False

    # Also mark infeasible if some batches remain unassigned.
    if np.any(x_schedule_vector[::2] < 0):
        decoder_feasible = False

    return x_schedule_vector.astype(float), decoder_feasible


_EVAL_CACHE = {}


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
            cache_key = tuple(batch_order_perm.tolist())
            if cache_key in _EVAL_CACHE:
                energy, makespan, x_cached, cost_cached = _EVAL_CACHE[cache_key]
                x_sched_vec = x_cached.copy()
                cost_components = dict(cost_cached)
            else:
                # 1. แปลง Permutation เป็น x_schedule_vector (schedule) โดย Greedy Assignment
                x_sched_vec, decoder_feasible = greedy_assignment(
                    batch_order_perm, num_batches=self.n_var
                )  # Renamed x

                if decoder_feasible:
                    # 2. คำนวณ Objectives โดยใช้ scheduling_cost เดิม
                    energy, makespan, cost_components = scheduling_cost(x_sched_vec)
                else:
                    # Controlled penalty for decode failure (avoid clamp-induced overlap explosions).
                    energy = 1e15
                    makespan = TOTAL_SLOTS * SLOT_DURATION * 2
                    cost_components = {
                        "total_cost": energy,
                        "makespan_minutes": makespan,
                        "decoder_infeasible_penalty": 1e15,
                    }

                _EVAL_CACHE[cache_key] = (
                    energy,
                    makespan,
                    x_sched_vec.copy(),
                    dict(cost_components),
                )

            evaluated_schedules_x.append(x_sched_vec)  # Store the generated schedule
            all_cost_components.append(cost_components)  # Store the detailed components

            # เพิ่ม penalty สูงมากถ้า greedy assignment ล้มเหลว (เช่น หา slot ไม่ได้)
            # (ตรวจจาก x_sched_vec ที่ได้ หรือเพิ่ม flag จาก greedy_assignment)
            # ตัวอย่าง: ตรวจสอบว่า makespan ดูสมเหตุสมผลไหม
            if makespan > TOTAL_SLOTS * SLOT_DURATION * 1.1:  # ถ้า makespan ยาวผิดปกติ
                energy += 1e15  # ลงโทษหนักๆ

            results_f.append([energy, makespan])

        # กำหนดค่า Objectives ให้กับ Population
        out["F"] = np.array(results_f)
        if DEBUG:
            unique = np.unique(np.round(out["F"], 2), axis=0).shape[0]
            print("DEBUG unique F:", unique)
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
            print(format_schedule_breakdown(cost_details))
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
                    base_if_energy_kwh = cost_details.get("base_if_energy_kwh", 0.0)
                    if_holding_penalty_cost = cost_details.get(
                        "if_holding_penalty", 0.0
                    )
                    actual_if_energy_kwh = base_if_energy_kwh + if_holding_penalty_cost
                    plot_title = f"HGA Schedule - Sol {i+1} (Actual IF Energy: {actual_if_energy_kwh:.0f} kWh, Total Cost: {cost_details['total_cost']:.0f}, Makespan: {cost_details['makespan_minutes']:.0f} min)"

                    plot_schedule_and_mh(
                        schedule,
                        title=plot_title,
                        simulated_time_points=None,  # Trigger re-simulation in plot function for M&H
                        simulated_mh_levels=None,  # Trigger re-simulation in plot function for M&H
                        batch_timing=cost_details.get("batch_timing"),
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
