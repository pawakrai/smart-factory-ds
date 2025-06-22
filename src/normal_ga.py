# Smart Factory Aluminum Melting Optimization with Realistic GA
# Cleaned version - keeping only essential components

import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.soo.nonconvex.ga import GA
import random

# =============== CONFIGURATION ==================

HOURS_A_DAY = 24 * 60  # 1440 minutes
SLOT_DURATION = 5  # 1 slot = 5 minutes
TOTAL_SLOTS = HOURS_A_DAY // SLOT_DURATION  # 288 slots per day

T_MELT = 18  # Melting time: 18 slots = 90 minutes
NUM_BATCHES = 10  # Number of batches to schedule

# Furnace availability
USE_FURNACE_A = True
USE_FURNACE_B = True

# Penalty rates
IF_HOLDING_ENERGY_PENALTY_PER_MINUTE = 7.5
UNPOURED_BATCH_PENALTY = 1e13
IF_GAP_TIME_PENALTY_RATE_PER_MINUTE = 0
IF_WORKING_IN_BREAK_PENALTY_PER_MINUTE = 10000

# Power ratings (kW)
IF_POWER_RATING_KW = {"A": 535.0, "B": 535.0}

# M&H Configuration
MH_MAX_CAPACITY_KG = {"A": 400.0, "B": 250.0}
MH_INITIAL_LEVEL_KG = {"A": 300.0, "B": 200.0}  # ค่าจริงจากหน้างาน - ห้ามปรับ
MH_CONSUMPTION_RATE_KG_PER_MIN = {
    "A": 2.50,
    "B": 2.55,
}  # อัตราการใช้น้ำจริงจากหน้างาน - ห้ามปรับ
MH_EMPTY_THRESHOLD_KG = 0
IF_BATCH_OUTPUT_KG = 500.0  # ค่าจริงจากหน้างาน - ห้ามปรับ
POST_POUR_DOWNTIME_MIN = 10
MH_IDLE_PENALTY_RATE = 0
MH_REHEAT_PENALTY_RATE = 20.0

# Break times (minutes from day start)
BREAK_TIMES_MINUTES = []

# M&H filling preference
PREFERRED_MH_FURNACE_TO_FILL_FIRST = "B"

# Plotting configuration
FURNACE_COLORS = {"A": "blue", "B": "green"}
MH_FURNACE_COLORS = {"A": "red", "B": "orange"}
furnace_y = {0: 10, 1: 25}
height = 8
SHIFT_START = 8 * 60

# Realistic penalty system configuration
MAX_SHIFT_DURATION_HOURS = 16
MAX_SHIFT_DURATION_MINUTES = MAX_SHIFT_DURATION_HOURS * 60
FURNACE_BALANCE_TARGET = 0.5
MH_IDLE_PENALTY_RATE_PER_MINUTE = 100.0
SHIFT_OVERTIME_PENALTY_RATE_PER_MINUTE = 200.0
FURNACE_IMBALANCE_PENALTY_RATE = 10000.0  # เพิ่มจาก 1000 เป็น 10000

# =============== CORE SCHEDULING FUNCTIONS ==================


def scheduling_cost(x):
    """
    Calculate scheduling cost including energy and penalties
    """
    n = len(x) // 2
    schedule = []
    for i in range(n):
        start = int(x[2 * i])
        furnace = int(round(x[2 * i + 1]))
        start = max(0, min(start, TOTAL_SLOTS - T_MELT))
        end = start + T_MELT
        schedule.append((start, end, furnace, i + 1))

    # Check overlaps and constraints
    penalty_if = 0.0

    # Furnace availability penalties
    for s, e, f, b_id in schedule:
        if f == 0 and not USE_FURNACE_A:
            penalty_if += 1e12
        if f == 1 and not USE_FURNACE_B:
            penalty_if += 1e12

    # Same furnace overlap check
    usage_for_furnace = {0: [0] * TOTAL_SLOTS, 1: [0] * TOTAL_SLOTS}
    for s, e, f, b_id in schedule:
        for t_slot in range(s, e):
            usage_for_furnace[f][t_slot] += 1

    for f_idx in [0, 1]:
        if (f_idx == 0 and USE_FURNACE_A) or (f_idx == 1 and USE_FURNACE_B):
            for t_slot in range(TOTAL_SLOTS):
                if usage_for_furnace[f_idx][t_slot] > 1:
                    penalty_if += (usage_for_furnace[f_idx][t_slot] - 1) * 1e9

    # Global overlap check (no simultaneous melting)
    global_usage = [0] * TOTAL_SLOTS
    for s, e, f, b_id in schedule:
        for t_slot in range(s, e):
            if global_usage[t_slot] > 0:
                penalty_if += 1e12
            global_usage[t_slot] += 1

    # Calculate makespan
    makespan_slot = max(e for (s, e, f, b_id) in schedule) if schedule else 0
    makespan_min = makespan_slot * SLOT_DURATION

    # Create melt completion events for M&H simulation
    melt_completion_events = []
    for s, e, f, b_id in schedule:
        finish_minute = e * SLOT_DURATION
        melt_completion_events.append((finish_minute, b_id))
    melt_completion_events.sort(key=lambda w: w[0])

    # Run M&H simulation
    (
        MH_idle_penalty,
        MH_reheat_penalty,
        total_energy_mh,
        time_points,
        mh_levels,
        actual_pour_events,
        unpoured_batches_at_end,
        total_if_holding_minutes,
        pour_induced_mh_overflow_penalty,
    ) = simulate_mh_consumption_v2(melt_completion_events)

    # Calculate IF energy
    base_total_energy_if_kwh = 0.0
    for s, e, f, b_id in schedule:
        furnace_id_for_power = None
        if USE_FURNACE_A and USE_FURNACE_B:
            furnace_id_for_power = "A" if f == 0 else "B"
        elif USE_FURNACE_A:
            furnace_id_for_power = "A"
        elif USE_FURNACE_B:
            furnace_id_for_power = "B"

        if furnace_id_for_power and furnace_id_for_power in IF_POWER_RATING_KW:
            power_rating_kw = IF_POWER_RATING_KW[furnace_id_for_power]
            energy_for_batch_kwh = power_rating_kw
            base_total_energy_if_kwh += energy_for_batch_kwh

    # Calculate penalties
    if_holding_penalty_cost = (
        total_if_holding_minutes * IF_HOLDING_ENERGY_PENALTY_PER_MINUTE
    )
    unpoured_batches_penalty_cost = (
        len(unpoured_batches_at_end) * UNPOURED_BATCH_PENALTY
    )

    # Total cost
    cost = (
        base_total_energy_if_kwh
        + if_holding_penalty_cost
        + total_energy_mh
        + penalty_if
        + MH_idle_penalty
        + MH_reheat_penalty
        + pour_induced_mh_overflow_penalty
        + unpoured_batches_penalty_cost
    )

    # Calculate actual makespan
    if actual_pour_events:
        makespan_min_actual = max(ape[0] for ape in actual_pour_events)
    elif schedule:
        makespan_min_actual = 1440
    else:
        makespan_min_actual = 0

    # Cost components for analysis
    cost_components = {
        "total_cost": cost,
        "makespan_minutes": makespan_min_actual,
        "base_if_energy_kwh": base_total_energy_if_kwh,
        "if_holding_penalty": if_holding_penalty_cost,
        "if_general_penalty": penalty_if,
        "mh_idle_penalty": MH_idle_penalty,
        "mh_reheat_penalty": MH_reheat_penalty,
        "mh_overflow_penalty": pour_induced_mh_overflow_penalty,
        "unpoured_batch_penalty": unpoured_batches_penalty_cost,
        "mh_energy_kwh": total_energy_mh,
    }

    return cost, makespan_min_actual, cost_components


def simulate_mh_consumption_v2(melt_completion_events):
    """
    Simulate M&H consumption with PERFECT SYNC - batch เสร็จ → เททันที
    ใช้ logic เดียวกันกับ calculate_mh_capacity_shortage_penalty
    """
    simulation_duration_min = 1440
    time_points = np.arange(simulation_duration_min)
    mh_levels = {
        "A": np.zeros(simulation_duration_min),
        "B": np.zeros(simulation_duration_min),
    }

    # เรียง events ตามเวลา
    melt_events = sorted(melt_completion_events, key=lambda x: x[0])

    current_level = MH_INITIAL_LEVEL_KG.copy()
    total_idle_penalty = 0.0
    total_reheat_penalty = 0.0
    pour_induced_mh_overflow_penalty = 0.0
    actual_pour_events = []
    unpoured_batches_at_end = []
    total_if_holding_minutes = 0
    last_minute = 0

    # Initialize M&H levels
    for t in range(simulation_duration_min):
        mh_levels["A"][t] = current_level["A"]
        mh_levels["B"][t] = current_level["B"]

    # Process each melt completion event
    for finish_minute, batch_id in melt_events:
        finish_minute = int(finish_minute)

        if finish_minute >= simulation_duration_min:
            unpoured_batches_at_end.append(batch_id)
            continue

        # Simulate M&H consumption จาก last_minute ถึง finish_minute
        for minute in range(last_minute, finish_minute):
            if minute >= simulation_duration_min:
                break

            for furnace_id in ["A", "B"]:
                if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                    current_level[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                        furnace_id
                    ]
                    current_level[furnace_id] = max(current_level[furnace_id], 0)

            # Update levels in array
            mh_levels["A"][minute] = current_level["A"]
            mh_levels["B"][minute] = current_level["B"]

        # ตรวจสอบ M&H capacity ณ เวลาที่ batch เสร็จ
        available_A = MH_MAX_CAPACITY_KG["A"] - current_level["A"]
        available_B = MH_MAX_CAPACITY_KG["B"] - current_level["B"]
        total_available = available_A + available_B

        # เทน้ำทันทีหลังจาก batch เสร็จ (ถ้าเทได้)
        if total_available >= IF_BATCH_OUTPUT_KG:
            # เทได้เต็มจำนวน - PERFECT SYNC
            remaining_metal = IF_BATCH_OUTPUT_KG
            pour_order = (
                ["B", "A"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B" else ["A", "B"]
            )

            for furnace_id in pour_order:
                if remaining_metal <= 0:
                    break
                available_space = (
                    MH_MAX_CAPACITY_KG[furnace_id] - current_level[furnace_id]
                )
                pour_amount = min(remaining_metal, available_space)
                if pour_amount > 0:
                    current_level[furnace_id] += pour_amount
                    remaining_metal -= pour_amount

            actual_pour_events.append((finish_minute, batch_id))

        else:
            # ไม่สามารถเทได้ - เพิ่มใน unpoured list
            unpoured_batches_at_end.append(batch_id)

        # Update levels ณ เวลาที่เท (PERFECT SYNC)
        if finish_minute < simulation_duration_min:
            mh_levels["A"][finish_minute] = current_level["A"]
            mh_levels["B"][finish_minute] = current_level["B"]

        last_minute = finish_minute

    # Simulate remaining time after last event
    for minute in range(last_minute + 1, simulation_duration_min):
        for furnace_id in ["A", "B"]:
            if current_level[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                current_level[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[furnace_id]
                current_level[furnace_id] = max(current_level[furnace_id], 0)

        mh_levels["A"][minute] = current_level["A"]
        mh_levels["B"][minute] = current_level["B"]

    total_energy_mh = 0.0

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
    """Decode schedule vector to readable format"""
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
    Plot Gantt chart and M&H levels
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(18, 9), sharex=True)

    # Use pre-simulated M&H levels if provided
    if simulated_time_points is not None and simulated_mh_levels is not None:
        time_points_shifted = simulated_time_points + SHIFT_START
        mh_levels_to_plot = simulated_mh_levels
    elif schedule:
        print(
            "Warning: Re-simulating M&H for plotting. Use pre-simulated data for consistency."
        )
        melt_completion_events_plot = []
        for s, e, f, b_id in schedule:
            finish_minute = e * SLOT_DURATION
            melt_completion_events_plot.append((finish_minute, b_id))

        melt_completion_events_plot.sort(key=lambda w: w[0])

        (
            _,
            _,
            _,
            time_points_plot,
            mh_levels_plot,
            _,
            _,
            _,
            _,
        ) = simulate_mh_consumption_v2(melt_completion_events_plot)
        time_points_shifted = time_points_plot + SHIFT_START
        mh_levels_to_plot = mh_levels_plot
    else:
        time_points_shifted = np.arange(SHIFT_START, SHIFT_START + 1440)
        mh_levels_to_plot = {
            "A": np.full(1440, MH_INITIAL_LEVEL_KG["A"]),
            "B": np.full(1440, MH_INITIAL_LEVEL_KG["B"]),
        }

    # Plot Gantt chart
    for start, end, f, b_id in schedule:
        if f not in furnace_y:
            print(f"Warning: Invalid IF index '{f}' for batch {b_id}. Skipping plot.")
            continue
        melt_start_t = SHIFT_START + start * SLOT_DURATION
        melt_dur = (end - start) * SLOT_DURATION
        ax1.broken_barh(
            [(melt_start_t, melt_dur)],
            (furnace_y[f], height),
            facecolors=FURNACE_COLORS.get(f, "gray"),
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

    # Plot M&H levels
    max_plot_capacity = max(MH_MAX_CAPACITY_KG.values()) * 1.1

    for furnace_id_mh_plot in ["A", "B"]:
        ax2.plot(
            time_points_shifted,
            mh_levels_to_plot[furnace_id_mh_plot],
            color=MH_FURNACE_COLORS[furnace_id_mh_plot],
            linewidth=2,
            label=f"M&H {furnace_id_mh_plot} Level",
        )
        ax2.axhline(
            y=MH_MAX_CAPACITY_KG[furnace_id_mh_plot],
            color=MH_FURNACE_COLORS[furnace_id_mh_plot],
            linestyle=":",
            alpha=0.7,
            label=f"M&H {furnace_id_mh_plot} Max Cap ({MH_MAX_CAPACITY_KG[furnace_id_mh_plot]} kg)",
        )

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

    xticks_ax2 = np.arange(SHIFT_START, SHIFT_START + 1441, 60)
    ax2.set_xticks(xticks_ax2)
    xlabels_ax2 = [f"{(x_val // 60) % 24:02d}:{x_val % 60:02d}" for x_val in xticks_ax2]
    ax2.set_xticklabels(xlabels_ax2)
    ax2.set_title("Simulated M&H Levels")
    ax2.grid(True, which="major", axis="both", alpha=0.5)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# =================== M&H-AWARE SCHEDULING ===================


def calculate_mh_available_time(target_minute, initial_levels=None):
    """
    คำนวณว่า M&H จะมีพื้นที่ว่างเพียงพอที่นาทีไหน
    """
    if initial_levels is None:
        initial_levels = MH_INITIAL_LEVEL_KG.copy()

    current_levels = initial_levels.copy()

    # Simulate M&H consumption until target_minute
    for minute in range(target_minute):
        for furnace_id in ["A", "B"]:
            if current_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                current_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[furnace_id]
                current_levels[furnace_id] = max(current_levels[furnace_id], 0)

    # Check available capacity at target_minute
    available_A = MH_MAX_CAPACITY_KG["A"] - current_levels["A"]
    available_B = MH_MAX_CAPACITY_KG["B"] - current_levels["B"]
    total_available = available_A + available_B

    return total_available >= IF_BATCH_OUTPUT_KG, current_levels, total_available


def find_earliest_pour_time(melt_finish_minute, existing_pours=None):
    """
    หาเวลาที่เร็วที่สุดที่สามารถเทได้หลังจาก melt เสร็จ
    โดยพิจารณา M&H capacity และ existing pours
    """
    if existing_pours is None:
        existing_pours = []

    # Simulate M&H state with existing pours
    current_levels = MH_INITIAL_LEVEL_KG.copy()

    # Apply existing pours chronologically
    for pour_minute, _ in sorted(existing_pours):
        if pour_minute < melt_finish_minute:
            # Simulate consumption up to pour time
            for minute in range(pour_minute):
                for furnace_id in ["A", "B"]:
                    if current_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                        current_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                            furnace_id
                        ]
                        current_levels[furnace_id] = max(current_levels[furnace_id], 0)

            # Apply pour (simplified - fill B first, then A)
            remaining_metal = IF_BATCH_OUTPUT_KG
            if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B":
                pour_order = ["B", "A"]
            else:
                pour_order = ["A", "B"]

            for furnace_id in pour_order:
                if remaining_metal <= 0:
                    break
                available_space = (
                    MH_MAX_CAPACITY_KG[furnace_id] - current_levels[furnace_id]
                )
                pour_amount = min(remaining_metal, available_space)
                if pour_amount > 0:
                    current_levels[furnace_id] += pour_amount
                    remaining_metal -= pour_amount

    # Now find earliest time after melt_finish_minute when we can pour
    for check_minute in range(melt_finish_minute, 1440):
        # Simulate consumption from last pour to check_minute
        test_levels = current_levels.copy()

        for minute in range(melt_finish_minute, check_minute):
            for furnace_id in ["A", "B"]:
                if test_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                    test_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                        furnace_id
                    ]
                    test_levels[furnace_id] = max(test_levels[furnace_id], 0)

        # Check if we can pour at check_minute
        available_A = MH_MAX_CAPACITY_KG["A"] - test_levels["A"]
        available_B = MH_MAX_CAPACITY_KG["B"] - test_levels["B"]
        total_available = available_A + available_B

        if total_available >= IF_BATCH_OUTPUT_KG:
            return check_minute, test_levels

    return 1440, current_levels  # Fallback - end of day


def optimize_schedule_for_mh_capacity(schedule_vector, n_batches):
    """
    ปรับปรุง schedule ให้เหมาะสมกับ M&H capacity
    """
    if n_batches == 0:
        return schedule_vector

    # Extract schedule information
    batches = []
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        furnace_idx = int(schedule_vector[2 * batch_idx + 1])
        finish_minute = (start_slot + T_MELT) * SLOT_DURATION
        batches.append(
            {
                "batch_id": batch_idx,
                "start_slot": start_slot,
                "furnace_idx": furnace_idx,
                "finish_minute": finish_minute,
                "optimal_pour_time": None,
            }
        )

    # Sort by finish time
    batches.sort(key=lambda x: x["finish_minute"])

    # Calculate optimal pour times considering M&H capacity
    existing_pours = []

    for batch in batches:
        optimal_pour_time, _ = find_earliest_pour_time(
            batch["finish_minute"], existing_pours
        )
        batch["optimal_pour_time"] = optimal_pour_time
        existing_pours.append((optimal_pour_time, batch["batch_id"]))

        # If pour time is significantly delayed, try to delay the melt start
        pour_delay = optimal_pour_time - batch["finish_minute"]
        if pour_delay > 30:  # If delayed more than 30 minutes
            # Calculate new start time to minimize delay
            new_finish_minute = max(
                batch["finish_minute"], optimal_pour_time - 15
            )  # Allow 15 min buffer
            new_start_slot = max(
                0, (new_finish_minute - T_MELT * SLOT_DURATION) // SLOT_DURATION
            )

            # Update schedule vector
            schedule_vector[2 * batch["batch_id"]] = new_start_slot
            batch["start_slot"] = new_start_slot
            batch["finish_minute"] = (
                new_start_slot * SLOT_DURATION + T_MELT * SLOT_DURATION
            )

    return schedule_vector


# =================== SMART SCHEDULING LOGIC ===================


def calculate_pour_timing_penalty(schedule_vector, n_batches):
    """
    คำนวณ penalty สำหรับ timing ที่ไม่เหมาะสมระหว่าง melting และ M&H capacity
    ใช้ข้อมูลจริงจากหน้างาน: IF_BATCH_OUTPUT_KG, MH_INITIAL_LEVEL_KG, MH_CONSUMPTION_RATE_KG_PER_MIN
    """
    if n_batches == 0:
        return 0.0, []

    # สร้าง melt completion events
    melt_events = []
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        finish_minute = (start_slot + T_MELT) * SLOT_DURATION
        melt_events.append((finish_minute, batch_idx))

    melt_events.sort(key=lambda x: x[0])

    # Simulate M&H state และคำนวณ penalty
    current_levels = MH_INITIAL_LEVEL_KG.copy()
    total_penalty = 0.0
    pour_timing_details = []
    last_simulated_minute = 0

    for finish_minute, batch_idx in melt_events:
        # Simulate M&H consumption จาก last_simulated_minute ถึง finish_minute
        for minute in range(last_simulated_minute, finish_minute):
            for furnace_id in ["A", "B"]:
                if current_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                    current_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                        furnace_id
                    ]
                    current_levels[furnace_id] = max(current_levels[furnace_id], 0)

        # ตรวจสอบ M&H capacity ณ เวลาที่ batch เสร็จ
        available_A = MH_MAX_CAPACITY_KG["A"] - current_levels["A"]
        available_B = MH_MAX_CAPACITY_KG["B"] - current_levels["B"]
        total_available = available_A + available_B

        # คำนวณเวลาที่ต้องรอ (ถ้ามี)
        wait_time = 0
        if total_available < IF_BATCH_OUTPUT_KG:
            # คำนวณเวลาที่ต้องรอให้ M&H มีพื้นที่เพียงพอ
            needed_space = IF_BATCH_OUTPUT_KG - total_available

            # คำนวณเวลาที่ต้องรอโดยใช้ consumption rate
            total_consumption_rate = sum(MH_CONSUMPTION_RATE_KG_PER_MIN.values())
            wait_time = needed_space / total_consumption_rate

            # Simulate consumption ระหว่างรอ
            for minute in range(int(wait_time) + 1):
                for furnace_id in ["A", "B"]:
                    if current_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                        current_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                            furnace_id
                        ]
                        current_levels[furnace_id] = max(current_levels[furnace_id], 0)

        # คำนวณ penalty ตามเวลาที่ต้องรอ
        if wait_time > 0:
            # Penalty สำหรับการรอ: ยิ่งรอนาน penalty ยิ่งสูง
            wait_penalty = wait_time * 100.0  # 100 units per minute

            # Extra penalty สำหรับการรอนานเกินไป
            if wait_time > 30:  # รอเกิน 30 นาที
                wait_penalty += (wait_time - 30) * 200.0

            total_penalty += wait_penalty

        # บันทึกข้อมูลสำหรับ debug
        pour_timing_details.append(
            {
                "batch_id": batch_idx,
                "finish_minute": finish_minute,
                "wait_time": wait_time,
                "available_space": total_available,
                "needed_space": IF_BATCH_OUTPUT_KG,
                "penalty": wait_time * 100.0 if wait_time > 0 else 0,
            }
        )

        # Pour metal ลง M&H (simulate การเท)
        remaining_metal = IF_BATCH_OUTPUT_KG
        pour_order = (
            ["B", "A"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B" else ["A", "B"]
        )

        for furnace_id in pour_order:
            if remaining_metal <= 0:
                break
            available_space = (
                MH_MAX_CAPACITY_KG[furnace_id] - current_levels[furnace_id]
            )
            pour_amount = min(remaining_metal, available_space)
            if pour_amount > 0:
                current_levels[furnace_id] += pour_amount
                remaining_metal -= pour_amount

        # อัปเดต last_simulated_minute
        actual_pour_time = finish_minute + wait_time
        last_simulated_minute = int(actual_pour_time)

    return total_penalty, pour_timing_details


def optimize_melting_timing(schedule_vector, n_batches):
    """
    ปรับ timing การหลอมให้เหมาะสมกับ M&H capacity
    โดยไม่เปลี่ยนค่า config จริงจากหน้างาน
    """
    if n_batches == 0:
        return schedule_vector

    # คำนวณ penalty ของ schedule ปัจจุบัน
    current_penalty, timing_details = calculate_pour_timing_penalty(
        schedule_vector, n_batches
    )

    if current_penalty == 0:
        return schedule_vector  # Schedule ดีอยู่แล้ว

    # ปรับ schedule โดยการ delay batches ที่มี wait time สูง
    improved_schedule = schedule_vector.copy()

    for detail in timing_details:
        if detail["wait_time"] > 15:  # ถ้ารอเกิน 15 นาที
            batch_idx = detail["batch_id"]
            current_start_slot = int(improved_schedule[2 * batch_idx])

            # Delay การเริ่มหลอมเพื่อให้เสร็จตอนที่ M&H พร้อม
            delay_minutes = min(detail["wait_time"] * 0.7, 60)  # Delay สูงสุด 60 นาที
            delay_slots = int(delay_minutes / SLOT_DURATION)

            new_start_slot = current_start_slot + delay_slots
            new_start_slot = min(new_start_slot, TOTAL_SLOTS - T_MELT)  # ไม่เกินขอบเขต

            improved_schedule[2 * batch_idx] = new_start_slot

    return improved_schedule


def optimize_schedule_for_mh_readiness(schedule_vector, n_batches):
    """
    ปรับ schedule ให้ batch เสร็จตอนที่ M&H มีพื้นที่เพียงพอ
    ตามหลักความเป็นจริง: เมื่อหลอมเสร็จ ต้องเทได้ทันที
    """
    if n_batches == 0:
        return schedule_vector

    # คำนวณ shortage penalty ของ schedule ปัจจุบัน
    current_penalty, shortage_details = calculate_mh_capacity_shortage_penalty(
        schedule_vector, n_batches
    )

    if current_penalty == 0:
        return schedule_vector  # Schedule ดีอยู่แล้ว

    # ปรับ schedule โดยการ delay batches ที่มี shortage
    improved_schedule = schedule_vector.copy()

    # เรียงตาม shortage amount (แก้ไขตัวที่มีปัญหามากก่อน)
    shortage_details.sort(key=lambda x: x["shortage_amount"], reverse=True)

    for detail in shortage_details:
        if detail["shortage_amount"] > 0:  # มี shortage
            batch_idx = detail["batch_id"]
            current_start_slot = int(improved_schedule[2 * batch_idx])

            # คำนวณเวลาที่ต้อง delay เพื่อให้ M&H มีพื้นที่เพียงพอ
            shortage_amount = detail["shortage_amount"]

            # ใช้ consumption rate เพื่อคำนวณเวลาที่ต้องรอ
            total_consumption_rate = sum(MH_CONSUMPTION_RATE_KG_PER_MIN.values())
            delay_minutes = shortage_amount / total_consumption_rate

            # เพิ่ม buffer 10% เพื่อความปลอดภัย
            delay_minutes *= 1.1

            # แปลงเป็น slots และ delay
            delay_slots = max(1, int(delay_minutes / SLOT_DURATION))
            new_start_slot = current_start_slot + delay_slots

            # ตรวจสอบขอบเขต
            new_start_slot = min(new_start_slot, TOTAL_SLOTS - T_MELT)

            improved_schedule[2 * batch_idx] = new_start_slot

    return improved_schedule


# =================== REALISTIC PENALTY SYSTEM ===================


def calculate_realistic_penalties(schedule_vector, cost_components):
    """
    Calculate realistic penalties based on real-world constraints
    """
    penalties = {
        "overlap_penalty": 0.0,
        "mh_idle_penalty": 0.0,
        "shift_overtime_penalty": 0.0,
        "furnace_imbalance_penalty": 0.0,
        "mh_capacity_mismatch_penalty": 0.0,
        "total_realistic_penalty": 0.0,
    }

    n_batches = len(schedule_vector) // 2

    # 1. Overlap penalty (same and different furnaces)
    penalties["overlap_penalty"] = calculate_overlap_penalty(schedule_vector, n_batches)

    # 2. M&H idle penalty
    penalties["mh_idle_penalty"] = calculate_mh_idle_penalty(schedule_vector)

    # 3. Shift overtime penalty
    penalties["shift_overtime_penalty"] = calculate_shift_overtime_penalty(
        schedule_vector, n_batches
    )

    # 4. Furnace imbalance penalty
    penalties["furnace_imbalance_penalty"] = calculate_furnace_imbalance_penalty(
        schedule_vector, n_batches
    )

    # 5. M&H capacity shortage penalty (ใช้ข้อมูลจริงจากหน้างาน)
    shortage_penalty, _ = calculate_mh_capacity_shortage_penalty(
        schedule_vector, n_batches
    )
    penalties["mh_capacity_mismatch_penalty"] = shortage_penalty

    # Total penalties
    penalties["total_realistic_penalty"] = (
        penalties["overlap_penalty"]
        + penalties["mh_idle_penalty"]
        + penalties["shift_overtime_penalty"]
        + penalties["furnace_imbalance_penalty"]
        + penalties["mh_capacity_mismatch_penalty"]
    )

    return penalties


def calculate_overlap_penalty(schedule_vector, n_batches):
    """
    Penalty for overlapping melting operations
    """
    overlap_penalty = 0.0

    # Create time intervals
    intervals = []
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        furnace_idx = int(schedule_vector[2 * batch_idx + 1])

        start_minute = start_slot * SLOT_DURATION
        end_minute = (start_slot + T_MELT) * SLOT_DURATION

        intervals.append(
            {
                "batch": batch_idx,
                "furnace": furnace_idx,
                "start": start_minute,
                "end": end_minute,
            }
        )

    # Check for overlaps
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            interval1 = intervals[i]
            interval2 = intervals[j]

            overlap_start = max(interval1["start"], interval2["start"])
            overlap_end = min(interval1["end"], interval2["end"])

            if overlap_end > overlap_start:  # Overlap exists
                overlap_duration = overlap_end - overlap_start

                if interval1["furnace"] == interval2["furnace"]:
                    # Same furnace overlap - very high penalty
                    overlap_penalty += overlap_duration * 1000.0
                else:
                    # Different furnace overlap - medium penalty
                    overlap_penalty += overlap_duration * 500.0

    return overlap_penalty


def calculate_mh_idle_penalty(schedule_vector):
    """
    Penalty for M&H being idle (unable to pour due to empty state)
    """
    n_batches = len(schedule_vector) // 2

    # Create melt completion events
    melt_events = []
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        finish_minute = (start_slot + T_MELT) * SLOT_DURATION
        melt_events.append((finish_minute, batch_idx))

    melt_events.sort(key=lambda x: x[0])

    # Simulate M&H consumption
    mh_levels = MH_INITIAL_LEVEL_KG.copy()
    mh_downtime = {"A": 0, "B": 0}

    total_idle_minutes = 0
    simulation_duration = min(1440, max([event[0] for event in melt_events] + [1440]))

    melt_event_idx = 0

    for minute in range(simulation_duration):
        # Process melt completions
        while (
            melt_event_idx < len(melt_events)
            and melt_events[melt_event_idx][0] <= minute
        ):
            if melt_events[melt_event_idx][0] == minute:
                pour_metal_to_mh(mh_levels, mh_downtime)
            melt_event_idx += 1

        # Update M&H states
        for furnace_id in ["A", "B"]:
            if mh_downtime[furnace_id] > 0:
                mh_downtime[furnace_id] -= 1
            elif mh_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                mh_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[furnace_id]
                mh_levels[furnace_id] = max(mh_levels[furnace_id], 0)

            # Check if idle (cannot pour)
            if (
                mh_levels[furnace_id] <= MH_EMPTY_THRESHOLD_KG
                and mh_downtime[furnace_id] == 0
            ):
                total_idle_minutes += 1

    return total_idle_minutes * MH_IDLE_PENALTY_RATE_PER_MINUTE


def pour_metal_to_mh(mh_levels, mh_downtime):
    """
    Simulate pouring metal to M&H
    """
    remaining_metal = IF_BATCH_OUTPUT_KG

    # Choose furnace order
    if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "A":
        furnace_order = ["A", "B"]
    elif PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B":
        furnace_order = ["B", "A"]
    else:
        furnace_order = ["A", "B"]

    for furnace_id in furnace_order:
        if remaining_metal <= 0:
            break

        available_space = MH_MAX_CAPACITY_KG[furnace_id] - mh_levels[furnace_id]
        pour_amount = min(remaining_metal, available_space)

        if pour_amount > 0:
            mh_levels[furnace_id] += pour_amount
            remaining_metal -= pour_amount

    # Set downtime
    mh_downtime["A"] = POST_POUR_DOWNTIME_MIN
    mh_downtime["B"] = POST_POUR_DOWNTIME_MIN


def calculate_shift_overtime_penalty(schedule_vector, n_batches):
    """
    Penalty for working beyond 16 hours
    """
    if n_batches == 0:
        return 0.0

    # Find makespan
    max_end_minute = 0
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        end_minute = (start_slot + T_MELT) * SLOT_DURATION
        max_end_minute = max(max_end_minute, end_minute)

    # Calculate overtime
    if max_end_minute > MAX_SHIFT_DURATION_MINUTES:
        overtime_minutes = max_end_minute - MAX_SHIFT_DURATION_MINUTES
        return overtime_minutes * SHIFT_OVERTIME_PENALTY_RATE_PER_MINUTE

    return 0.0


def calculate_furnace_imbalance_penalty(schedule_vector, n_batches):
    """
    Penalty for unbalanced furnace usage
    """
    if n_batches == 0:
        return 0.0

    # Count batches per furnace
    furnace_usage = {"A": 0, "B": 0}

    for batch_idx in range(n_batches):
        furnace_idx = int(schedule_vector[2 * batch_idx + 1])
        furnace_key = "A" if furnace_idx == 0 else "B"
        furnace_usage[furnace_key] += 1

    # Calculate imbalance
    total_batches = furnace_usage["A"] + furnace_usage["B"]
    if total_batches == 0:
        return 0.0

    furnace_a_ratio = furnace_usage["A"] / total_batches
    furnace_b_ratio = furnace_usage["B"] / total_batches

    # Calculate deviation from target (50-50)
    a_deviation = abs(furnace_a_ratio - FURNACE_BALANCE_TARGET)
    b_deviation = abs(furnace_b_ratio - FURNACE_BALANCE_TARGET)

    total_deviation = a_deviation + b_deviation

    return total_deviation * FURNACE_IMBALANCE_PENALTY_RATE


def calculate_mh_capacity_shortage_penalty(schedule_vector, n_batches):
    """
    คำนวณ penalty สำหรับการที่ IF หลอมเสร็จแต่ M&H ไม่มีพื้นที่เพียงพอรับน้ำ
    ตามหลักความเป็นจริง: เมื่อ batch เสร็จ ต้องเทได้ทันที
    """
    if n_batches == 0:
        return 0.0, []

    # สร้าง melt completion events
    melt_events = []
    for batch_idx in range(n_batches):
        start_slot = int(schedule_vector[2 * batch_idx])
        finish_minute = (start_slot + T_MELT) * SLOT_DURATION
        melt_events.append((finish_minute, batch_idx))

    melt_events.sort(key=lambda x: x[0])

    # Simulate M&H state และคำนวณ penalty
    current_levels = MH_INITIAL_LEVEL_KG.copy()
    total_penalty = 0.0
    shortage_details = []
    last_minute = 0

    for finish_minute, batch_idx in melt_events:
        # Simulate M&H consumption จาก last_minute ถึง finish_minute
        for minute in range(last_minute, finish_minute):
            for furnace_id in ["A", "B"]:
                if current_levels[furnace_id] > MH_EMPTY_THRESHOLD_KG:
                    current_levels[furnace_id] -= MH_CONSUMPTION_RATE_KG_PER_MIN[
                        furnace_id
                    ]
                    current_levels[furnace_id] = max(current_levels[furnace_id], 0)

        # ตรวจสอบ M&H capacity ณ เวลาที่ batch เสร็จ
        available_A = MH_MAX_CAPACITY_KG["A"] - current_levels["A"]
        available_B = MH_MAX_CAPACITY_KG["B"] - current_levels["B"]
        total_available = available_A + available_B

        # คำนวณ shortage penalty
        shortage_penalty = 0.0
        if total_available < IF_BATCH_OUTPUT_KG:
            shortage_amount = IF_BATCH_OUTPUT_KG - total_available

            # Penalty แบบ exponential: ยิ่งขาดมาก penalty ยิ่งสูง
            shortage_penalty = (shortage_amount**1.5) * 10.0  # Exponential penalty

            # Extra penalty สำหรับ shortage มาก
            if shortage_amount > 200:  # ขาดเกิน 200 kg
                shortage_penalty += (shortage_amount - 200) * 50.0

            total_penalty += shortage_penalty

        # บันทึกข้อมูลสำหรับ debug
        shortage_details.append(
            {
                "batch_id": batch_idx,
                "finish_minute": finish_minute,
                "available_space": total_available,
                "needed_space": IF_BATCH_OUTPUT_KG,
                "shortage_amount": max(0, IF_BATCH_OUTPUT_KG - total_available),
                "penalty": shortage_penalty,
                "mh_levels_before": current_levels.copy(),
            }
        )

        # เทน้ำ (ถ้าเทได้)
        if total_available >= IF_BATCH_OUTPUT_KG:
            # เทได้เต็มจำนวน
            remaining_metal = IF_BATCH_OUTPUT_KG
            pour_order = (
                ["B", "A"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B" else ["A", "B"]
            )

            for furnace_id in pour_order:
                if remaining_metal <= 0:
                    break
                available_space = (
                    MH_MAX_CAPACITY_KG[furnace_id] - current_levels[furnace_id]
                )
                pour_amount = min(remaining_metal, available_space)
                if pour_amount > 0:
                    current_levels[furnace_id] += pour_amount
                    remaining_metal -= pour_amount
        else:
            # เทได้เท่าที่มีพื้นที่ (partial pour)
            remaining_space = total_available
            pour_order = (
                ["B", "A"] if PREFERRED_MH_FURNACE_TO_FILL_FIRST == "B" else ["A", "B"]
            )

            for furnace_id in pour_order:
                if remaining_space <= 0:
                    break
                available_space = (
                    MH_MAX_CAPACITY_KG[furnace_id] - current_levels[furnace_id]
                )
                pour_amount = min(remaining_space, available_space)
                if pour_amount > 0:
                    current_levels[furnace_id] += pour_amount
                    remaining_space -= pour_amount

        last_minute = finish_minute

    return total_penalty, shortage_details


# =================== REALISTIC GA COMPONENTS ===================


class RealisticDirectSchedulingProblem(Problem):
    """
    Direct scheduling problem with realistic penalty system
    """

    def __init__(self, num_batches=NUM_BATCHES):
        super().__init__(
            n_var=num_batches * 2,
            n_obj=1,
            n_constr=0,
            xl=np.zeros(num_batches * 2),
            xu=np.array(
                [
                    TOTAL_SLOTS - T_MELT if i % 2 == 0 else 1
                    for i in range(num_batches * 2)
                ]
            ),
        )
        self.num_batches = num_batches

    def _evaluate(self, X, out, **kwargs):
        results_f = []

        for individual in X:
            # Calculate original cost
            original_cost, makespan, cost_components = scheduling_cost(individual)

            # Calculate realistic penalties
            realistic_penalties = calculate_realistic_penalties(
                individual, cost_components
            )

            # Create final cost
            base_cost = cost_components.get("base_if_energy_kwh", 5000)
            holding_penalty = cost_components.get("if_holding_penalty", 0)

            # Use realistic penalties with extra weight on furnace balance
            furnace_balance_penalty = (
                realistic_penalties["furnace_imbalance_penalty"] * 2.0
            )  # Double weight
            other_penalties = (
                realistic_penalties["overlap_penalty"]
                + realistic_penalties["mh_idle_penalty"]
                + realistic_penalties["shift_overtime_penalty"]
            )

            final_cost = (
                base_cost
                + holding_penalty
                + other_penalties
                + furnace_balance_penalty  # Extra weighted furnace balance
                + (makespan * 0.05)
            )

            results_f.append([final_cost])

        out["F"] = np.array(results_f)


class BalancedDirectSchedulingSampling(Sampling):
    """
    Sampling that creates balanced schedules between furnaces
    """

    def _do(self, problem, n_samples, **kwargs):
        n_batches = problem.n_var // 2
        X = np.zeros((n_samples, problem.n_var))

        for i in range(n_samples):
            X[i] = self._create_balanced_schedule(n_batches)

        return X

    def _create_balanced_schedule(self, n_batches):
        """
        Create balanced and compact schedule between furnaces A and B
        """
        schedule = np.zeros(n_batches * 2)

        # Create STRICTLY balanced furnace assignment
        furnace_assignments = []

        # Force exact 50-50 split
        half_batches = n_batches // 2

        # Add equal numbers of each furnace
        for i in range(half_batches):
            furnace_assignments.append(0)  # Furnace A
            furnace_assignments.append(1)  # Furnace B

        # Handle odd number of batches
        if n_batches % 2 == 1:
            # Randomly assign the extra batch
            furnace_assignments.append(np.random.choice([0, 1]))

        # Limited shuffle to maintain balance but add some diversity
        # Only swap pairs to maintain balance
        for _ in range(n_batches // 4):  # Limited swaps
            if len(furnace_assignments) >= 2:
                idx1, idx2 = np.random.choice(
                    len(furnace_assignments), 2, replace=False
                )
                furnace_assignments[idx1], furnace_assignments[idx2] = (
                    furnace_assignments[idx2],
                    furnace_assignments[idx1],
                )

        # Create compact schedule using parallel melting
        furnace_next_available = {"A": 0, "B": 0}

        for batch_idx in range(n_batches):
            furnace_idx = furnace_assignments[batch_idx]
            furnace_key = "A" if furnace_idx == 0 else "B"

            # Find appropriate start time
            start_slot = furnace_next_available[furnace_key]

            # Add small randomness
            random_delay = np.random.randint(0, 3)
            start_slot += random_delay

            # Check bounds
            start_slot = max(0, min(start_slot, TOTAL_SLOTS - T_MELT))

            schedule[2 * batch_idx] = start_slot
            schedule[2 * batch_idx + 1] = furnace_idx

            # Update furnace availability
            furnace_next_available[furnace_key] = start_slot + T_MELT + 1

        return schedule


class ScheduleRepair(Repair):
    """
    Repair invalid schedules (overlaps, out of bounds)
    """

    def _do(self, problem, X, **kwargs):
        n_batches = problem.n_var // 2

        for i in range(len(X)):
            X[i] = self._repair_individual(X[i], n_batches)

        return X

    def _repair_individual(self, x, n_batches):
        """
        Repair individual with problems including M&H capacity constraints
        """
        schedule_data = []

        # Extract data as (start_slot, furnace, batch_id)
        for batch_idx in range(n_batches):
            start_slot = int(x[2 * batch_idx])
            furnace_idx = int(round(x[2 * batch_idx + 1]))

            # Fix bounds
            start_slot = max(0, min(start_slot, TOTAL_SLOTS - T_MELT))
            furnace_idx = max(0, min(furnace_idx, 1))

            schedule_data.append([start_slot, furnace_idx, batch_idx])

        # Sort by start time
        schedule_data.sort(key=lambda item: item[0])

        # Fix overlaps
        furnace_last_end = {"A": 0, "B": 0}

        for item in schedule_data:
            start_slot, furnace_idx, batch_idx = item
            furnace_key = "A" if furnace_idx == 0 else "B"

            # Check overlap
            if start_slot < furnace_last_end[furnace_key]:
                start_slot = furnace_last_end[furnace_key]
                start_slot = min(start_slot, TOTAL_SLOTS - T_MELT)
                item[0] = start_slot

            # Update last end time
            furnace_last_end[furnace_key] = start_slot + T_MELT

        # Rebuild x first
        repaired_x = np.zeros_like(x)
        for start_slot, furnace_idx, batch_idx in schedule_data:
            repaired_x[2 * batch_idx] = start_slot
            repaired_x[2 * batch_idx + 1] = furnace_idx

        # Apply Smart Scheduling for M&H Readiness (ใช้ข้อมูลจริงจากหน้างาน)
        try:
            repaired_x = optimize_schedule_for_mh_readiness(repaired_x, n_batches)
        except:
            # If smart scheduling fails, use the basic repaired version
            pass

        return repaired_x


# =================== TESTING AND EXECUTION ===================


def test_realistic_system():
    """
    Test realistic penalty system
    """
    print("=== Testing Realistic Penalty System ===")

    # Create test schedule
    test_schedule = np.array(
        [
            0,
            0,  # Batch 0: Furnace A, Time 0-90
            18,
            1,  # Batch 1: Furnace B, Time 90-180
            36,
            0,  # Batch 2: Furnace A, Time 180-270
            54,
            1,  # Batch 3: Furnace B, Time 270-360
            72,
            0,  # Batch 4: Furnace A, Time 360-450
            90,
            1,  # Batch 5: Furnace B, Time 450-540
            108,
            0,  # Batch 6: Furnace A, Time 540-630
            126,
            1,  # Batch 7: Furnace B, Time 630-720
            144,
            0,  # Batch 8: Furnace A, Time 720-810
            162,
            1,  # Batch 9: Furnace B, Time 810-900
        ]
    )

    print("Test Schedule (Balanced):")
    for i in range(NUM_BATCHES):
        start_slot = int(test_schedule[2 * i])
        furnace = int(test_schedule[2 * i + 1])
        start_time = start_slot * SLOT_DURATION
        end_time = (start_slot + T_MELT) * SLOT_DURATION
        furnace_name = "A" if furnace == 0 else "B"
        print(f"  Batch {i}: Furnace {furnace_name}, Time {start_time}-{end_time} min")

    # Calculate penalties
    original_cost, makespan, cost_components = scheduling_cost(test_schedule)
    realistic_penalties = calculate_realistic_penalties(test_schedule, cost_components)

    print(f"\nOriginal Cost: {original_cost:.2f}")
    print(f"Makespan: {makespan:.2f} min")
    print(f"\nRealistic Penalties:")
    for penalty_name, penalty_value in realistic_penalties.items():
        print(f"  {penalty_name}: {penalty_value:.2f}")

    # Test imbalanced schedule
    print(f"\n--- Testing Imbalanced Schedule ---")
    imbalanced_schedule = test_schedule.copy()
    # Change all batches to use Furnace B
    for i in range(NUM_BATCHES):
        imbalanced_schedule[2 * i + 1] = 1

    imbalanced_penalties = calculate_realistic_penalties(
        imbalanced_schedule, cost_components
    )
    print(f"Imbalanced Penalties:")
    for penalty_name, penalty_value in imbalanced_penalties.items():
        print(f"  {penalty_name}: {penalty_value:.2f}")

    return True


def run_realistic_ga():
    """
    Run GA with realistic penalty system
    """
    print("=== Running Realistic GA ===")

    # Test system first
    if not test_realistic_system():
        print("❌ Realistic system test failed.")
        return None

    print("\n=== Starting Realistic GA Optimization ===")

    # Use realistic problem and balanced sampling
    problem = RealisticDirectSchedulingProblem(NUM_BATCHES)
    sampling = BalancedDirectSchedulingSampling()

    algorithm = GA(
        pop_size=60,
        sampling=sampling,
        crossover=SBX(eta=15, prob=0.8),
        mutation=PolynomialMutation(eta=20, prob=0.2),
        repair=ScheduleRepair(),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", 40)

    print("Starting optimization...")
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
        save_history=True,
    )

    print("Optimization finished.")

    # Plot convergence
    if result.history:
        best_cost_per_gen = [np.min(gen.opt.get("F")) for gen in result.history]

        plt.figure(figsize=(12, 6))
        plt.plot(
            np.arange(len(best_cost_per_gen)),
            best_cost_per_gen,
            marker="o",
            linestyle="-",
        )
        plt.title("Realistic GA Cost Convergence")
        plt.xlabel("Generation")
        plt.ylabel("Best Cost Found")
        plt.grid(True)
        plt.show()

    # Show results
    print("\nProcessing results...")
    if result.X is None:
        print("\nNo solution found.")
    else:
        print(f"\nFound Best Solution:")
        best_schedule = result.X
        best_cost_value = result.F[0]

        original_cost, final_makespan, cost_details = scheduling_cost(best_schedule)
        realistic_penalties = calculate_realistic_penalties(best_schedule, cost_details)

        print(f"  GA Cost (Realistic): {best_cost_value:.2f}")
        print(f"  Original Cost: {original_cost:.2f}")
        print(f"  Makespan: {final_makespan:.2f} min ({final_makespan/60:.1f} hours)")

        print(f"\n  Realistic Penalties:")
        for penalty_name, penalty_value in realistic_penalties.items():
            if penalty_value > 0:
                print(f"    {penalty_name}: {penalty_value:.2f}")

        # Analyze furnace usage
        furnace_usage = {"A": 0, "B": 0}
        print(f"\n  Schedule Details:")
        for batch_idx in range(NUM_BATCHES):
            start_slot = int(best_schedule[2 * batch_idx])
            furnace = int(best_schedule[2 * batch_idx + 1])
            start_time = start_slot * SLOT_DURATION
            end_time = (start_slot + T_MELT) * SLOT_DURATION
            furnace_name = "A" if furnace == 0 else "B"
            furnace_usage[furnace_name] += 1
            print(
                f"    Batch {batch_idx}: Furnace {furnace_name}, Time {start_time}-{end_time} min"
            )

        print(f"\n  Furnace Usage Balance:")
        total_batches = furnace_usage["A"] + furnace_usage["B"]
        a_percentage = (
            (furnace_usage["A"] / total_batches) * 100 if total_batches > 0 else 0
        )
        b_percentage = (
            (furnace_usage["B"] / total_batches) * 100 if total_batches > 0 else 0
        )
        print(f"    Furnace A: {furnace_usage['A']} batches ({a_percentage:.1f}%)")
        print(f"    Furnace B: {furnace_usage['B']} batches ({b_percentage:.1f}%)")

        # Plot schedule
        try:
            schedule = decode_schedule(best_schedule)
            if schedule:
                plot_title = (
                    f"Realistic GA - Best Schedule\n"
                    f"Cost: {best_cost_value:.0f}, Makespan: {final_makespan:.0f} min, "
                    f"A: {furnace_usage['A']}, B: {furnace_usage['B']}"
                )
                plot_schedule_and_mh(schedule, title=plot_title)
        except Exception as e:
            print(f"  Error plotting schedule: {e}")

    return result


if __name__ == "__main__":
    # Run Realistic GA
    run_realistic_ga()
