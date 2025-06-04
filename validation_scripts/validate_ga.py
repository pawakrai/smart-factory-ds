# This file will be populated with the content of src/app_v5.py
# for GA validation purposes.

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
import sys
import os


# Adjust path to import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root should be the parent of 'validation_scripts' and 'src'
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Attempt to import components from src.app_v5
# We will import specific functions and constants as needed for validation
try:
    from src.app_v5 import (
        # Configuration Constants (example, add more as needed)
        SLOT_DURATION,
        TOTAL_SLOTS,
        T_MELT,
        NUM_BATCHES,
        IF_BATCH_OUTPUT_KG,
        MH_MAX_CAPACITY_KG,
        MH_INITIAL_LEVEL_KG,
        MH_CONSUMPTION_RATE_KG_PER_MIN,
        MH_EMPTY_THRESHOLD_KG,
        POST_POUR_DOWNTIME_MIN,
        BREAK_TIMES_MINUTES,
        IF_POWER_RATING_KW,
        USE_FURNACE_A,
        USE_FURNACE_B,
        # Core Functions
        scheduling_cost,
        simulate_mh_consumption_v2,
        greedy_assignment,
        decode_schedule,
        plot_schedule_and_mh,
        # Potentially HGAProblem if we want to test its _evaluate method directly
        # HGAProblem
    )

    print("Successfully imported components from src/app_v5.py")
except ImportError as e:
    print(f"Error importing from src.app_v5: {e}")
    print(
        f"Please ensure src/app_v5.py exists and src is a package (contains __init__.py)."
    )
    # Exit if imports fail, as validation depends on it
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


# --- Baseline Greedy Assignment (No M&H Awareness) ---
def greedy_assignment_baseline(batch_order, num_batches_to_schedule):
    """
    Greedy assignment that schedules IFs sequentially without M&H awareness.
    Respects Global No-Overlap for IFs.
    Alternates between available furnaces.
    """
    x_schedule_vector = np.full(num_batches_to_schedule * 2, -1, dtype=int)
    # Uses TOTAL_SLOTS and T_MELT from the imported app_v5 constants
    global_if_used_slots = np.zeros(TOTAL_SLOTS, dtype=int)

    active_if_furnaces = []
    if USE_FURNACE_A:
        active_if_furnaces.append(0)  # Furnace A is index 0
    if USE_FURNACE_B:
        active_if_furnaces.append(1)  # Furnace B is index 1

    if not active_if_furnaces:
        print(
            "Error in greedy_assignment_baseline: No active IF furnaces defined in app_v5 config!"
        )
        return x_schedule_vector.astype(float)

    furnace_assignment_idx_counter = 0

    # Keep track of the last finish slot for each physical furnace to allow them to work in parallel
    # if global no-overlap was not a strict requirement.
    # However, with strict global no-overlap, we primarily need to track the overall last busy slot.
    last_system_busy_slot = 0

    for i in range(num_batches_to_schedule):
        batch_id_from_order = batch_order[
            i
        ]  # This is the actual batch index (0 to N-1)

        chosen_physical_furnace_idx = active_if_furnaces[
            furnace_assignment_idx_counter % len(active_if_furnaces)
        ]

        found_slot_for_batch = False
        # Start searching for a slot from the time the entire system was last busy
        for potential_start_slot in range(
            last_system_busy_slot, TOTAL_SLOTS - T_MELT + 1
        ):
            is_globally_free = True
            for t_check_slot in range(
                potential_start_slot, potential_start_slot + T_MELT
            ):
                if (
                    t_check_slot >= TOTAL_SLOTS
                    or global_if_used_slots[t_check_slot] == 1
                ):
                    is_globally_free = False
                    break

            if is_globally_free:
                # Assign batch to this slot and furnace
                # The batch_id_from_order determines *which* batch's schedule is being set in x_schedule_vector
                x_schedule_vector[2 * batch_id_from_order] = potential_start_slot
                x_schedule_vector[2 * batch_id_from_order + 1] = (
                    chosen_physical_furnace_idx
                )

                # Mark these slots as used globally
                for t_update_slot in range(
                    potential_start_slot, potential_start_slot + T_MELT
                ):
                    global_if_used_slots[t_update_slot] = 1

                last_system_busy_slot = (
                    potential_start_slot + T_MELT
                )  # Update the overall system busy time
                furnace_assignment_idx_counter += 1
                found_slot_for_batch = True
                break

        if not found_slot_for_batch:
            # This batch could not be scheduled (e.g., ran out of TOTAL_SLOTS)
            # Assign a highly penalized slot (or rely on -1 to be penalized by scheduling_cost)
            # For consistency with how original greedy handles failure (though it tries to avoid -1)
            # print(f"Baseline: Could not schedule batch_id {batch_id_from_order} (original index in batch_order: {i}).")
            x_schedule_vector[2 * batch_id_from_order] = (
                TOTAL_SLOTS  # Will be clamped by scheduling_cost
            )
            x_schedule_vector[2 * batch_id_from_order + 1] = (
                chosen_physical_furnace_idx  # Assign to a furnace anyway
            )

    return x_schedule_vector.astype(float)


# --- Validation Test Functions Will Go Here ---


# Example: Test function for simulate_mh_consumption_v2
def test_mh_simulation_scenario_1():
    print("\n--- Testing M&H Simulation: Scenario 1 (Simple Pour) ---")
    # Define a simple melt completion event
    # Batch 1 finishes at minute 100
    melt_events = [(100, 1)]

    (
        idle_penalty,
        reheat_penalty,
        energy_mh,
        time_points,
        mh_levels,
        actual_pour_events,
        unpoured_batches,
        holding_minutes,
        overflow_penalty,
    ) = simulate_mh_consumption_v2(melt_events)

    print(f"Melt Events: {melt_events}")
    print(f"Actual Pour Events: {actual_pour_events}")
    print(f"Total IF Holding Minutes: {holding_minutes}")
    print(f"Unpoured Batches at End: {unpoured_batches}")
    print(f"M&H Idle Penalty: {idle_penalty}")
    print(f"M&H Overflow Penalty: {overflow_penalty}")

    # Basic assertions (can be more detailed)
    if (
        actual_pour_events and actual_pour_events[0][0] == 100
    ):  # Assumes pour happens at melt completion if capacity allows
        print("Test Scenario 1 PASSED (Basic pour occurred as expected)")
    else:
        print(
            "Test Scenario 1 FAILED or pour was delayed (Check M&H initial capacity and logic)"
        )

    # Optionally, plot this specific scenario
    # For plotting, we'd need a dummy 'schedule' that produced these melt_events
    # Or adapt plot_schedule_and_mh to take melt_events directly if useful for isolated M&H testing

    # For now, just print levels at a few key time points
    if time_points is not None and mh_levels is not None:
        # Ensure indices are within bounds before accessing
        idx_99 = 99
        idx_100 = 100
        idx_100_plus_downtime = 100 + POST_POUR_DOWNTIME_MIN
        len_mh_a = len(mh_levels["A"])

        print(
            f"M&H A level at minute {idx_99}: {mh_levels['A'][idx_99] if idx_99 < len_mh_a else 'N/A'}"
        )
        print(
            f"M&H A level at minute {idx_100} (after pour): {mh_levels['A'][idx_100]if idx_100 < len_mh_a else 'N/A'}"
        )
        print(
            f"M&H A level at minute {idx_100_plus_downtime}: {mh_levels['A'][idx_100_plus_downtime] if idx_100_plus_downtime < len_mh_a else 'N/A'}"
        )


# Example: Test function for M&H-aware greedy_assignment
def test_greedy_assignment_mhawake():
    print("\n--- Testing M&H-Aware Greedy Assignment (Original) ---")
    num_b_test = min(NUM_BATCHES, 3)
    batch_order_test = np.arange(num_b_test)

    print(f"Test with batch order: {batch_order_test} for {num_b_test} batches.")
    x_schedule_vector = greedy_assignment(batch_order_test, num_batches=num_b_test)

    print(f"Generated x_schedule_vector (M&H-aware): {x_schedule_vector}")

    if np.all(x_schedule_vector != -1):
        schedule = decode_schedule(x_schedule_vector)
        print("Decoded Schedule (M&H-aware):")
        for item in schedule:
            print(
                f"  Batch {item[3]}: Start Slot {item[0]}, End Slot {item[1]}, Furnace {item[2]}"
            )
        print("M&H-Aware Greedy Assignment Test: Check output for plausibility.")
    else:
        print(
            "M&H-Aware Greedy Assignment Test FAILED: Not all batches were scheduled."
        )


# --- Main Validation Execution ---
if __name__ == "__main__":
    print("Starting GA Model Validation...")

    # --- Sanity Checks / Component Tests ---
    # You can call your specific test functions here
    test_mh_simulation_scenario_1()
    test_greedy_assignment_mhawake()

    # --- Evaluate Baseline Scenario ---
    print("\n--- Evaluating Baseline Scenario (No M&H Awareness) ---")
    # Use all NUM_BATCHES defined in app_v5 for baseline
    baseline_batch_order = np.arange(NUM_BATCHES)

    baseline_x_schedule = greedy_assignment_baseline(
        baseline_batch_order, num_batches_to_schedule=NUM_BATCHES
    )

    # Check if scheduling was successful (all batches got a slot other than the fallback TOTAL_SLOTS)
    # A more robust check for failure in baseline_x_schedule might be needed if TOTAL_SLOTS is a valid assignment
    # For now, assume if it's not -1, scheduling_cost will process it.
    # The real indicator of failure will be very high penalties from scheduling_cost.

    print(f"Baseline x_schedule_vector: {baseline_x_schedule}")
    baseline_cost, baseline_makespan, baseline_cost_components = scheduling_cost(
        baseline_x_schedule
    )

    print(
        f"Baseline - Total Cost (Objective 1) : {baseline_cost_components['total_cost']:.2f}"
    )
    print(
        f"Baseline - Makespan (Objective 2)   : {baseline_cost_components['makespan_minutes']:.2f} min"
    )
    print(f"Baseline - Cost Component Details:")
    print(
        f"  Base IF Energy (kWh)     : {baseline_cost_components['base_if_energy_kwh']:.2f}"
    )
    print(
        f"  IF Holding Penalty       : {baseline_cost_components['if_holding_penalty']:.2f}"
    )
    print(
        f"  IF General Penalty       : {baseline_cost_components['if_general_penalty']:.2f} (Overlaps, Gaps, Breaks)"
    )
    print(
        f"  MH Idle Penalty          : {baseline_cost_components['mh_idle_penalty']:.2f}"
    )
    print(
        f"  MH Reheat Penalty        : {baseline_cost_components['mh_reheat_penalty']:.2f} (Placeholder)"
    )
    print(
        f"  MH Overflow Penalty      : {baseline_cost_components['mh_overflow_penalty']:.2f}"
    )
    print(
        f"  Unpoured Batch Penalty   : {baseline_cost_components['unpoured_batch_penalty']:.2f}"
    )
    print(
        f"  MH Energy (kWh)          : {baseline_cost_components['mh_energy_kwh']:.2f} (Placeholder)"
    )

    baseline_schedule_decoded = decode_schedule(baseline_x_schedule)

    if baseline_schedule_decoded:  # Ensure schedule is not empty
        melt_events_baseline = []
        for s, e, f, b_id in baseline_schedule_decoded:
            finish_minute = e * SLOT_DURATION
            melt_events_baseline.append(
                (finish_minute, b_id)
            )  # b_id from decoded schedule is 1-indexed
        melt_events_baseline.sort(key=lambda w: w[0])

        (
            _,
            _,
            _,
            sim_time_points_base,
            sim_mh_levels_base,
            actual_pours_base,
            unpoured_base,
            _,
            _,
        ) = simulate_mh_consumption_v2(melt_events_baseline)

        print(f"Baseline - Actual Pour Events: {actual_pours_base}")
        print(f"Baseline - Unpoured Batches at End: {unpoured_base}")

        plot_schedule_and_mh(
            baseline_schedule_decoded,
            title=f"Baseline Schedule (Energy: {baseline_cost_components['total_cost']:.0f}, Makespan: {baseline_makespan:.0f})",
            simulated_time_points=sim_time_points_base,
            simulated_mh_levels=sim_mh_levels_base,
        )
        print("Displayed baseline schedule plot.")
    else:
        print("Baseline schedule could not be decoded for plotting or was empty.")

    # --- Plotting Example for M&H-aware greedy (Illustrative, using its own test data) ---
    print("\n--- Plotting M&H-Aware Greedy Test Schedule ---")
    num_plot_batches_mha = min(NUM_BATCHES, 3)
    mha_batch_order_plot = np.arange(num_plot_batches_mha)
    mha_x_plot = greedy_assignment(
        mha_batch_order_plot, num_batches=num_plot_batches_mha
    )

    if np.all(mha_x_plot != -1):
        mha_schedule_for_plot = decode_schedule(mha_x_plot)
        print(f"M&H-aware schedule for plotting: {mha_schedule_for_plot}")

        melt_events_mha_plot = []
        for s, e, f, b_id in mha_schedule_for_plot:
            finish_minute = e * SLOT_DURATION
            melt_events_mha_plot.append((finish_minute, b_id))
        melt_events_mha_plot.sort(key=lambda w: w[0])

        (
            cost_mha_plot,
            makespan_mha_plot,
            components_mha_plot,  # Get cost for title
        ) = scheduling_cost(mha_x_plot)

        (_, _, _, sim_time_points_mha, sim_mh_levels_mha, _, _, _, _) = (
            simulate_mh_consumption_v2(melt_events_mha_plot)
        )

        plot_schedule_and_mh(
            mha_schedule_for_plot,
            title=f"M&H-Aware Greedy (Test) (Energy: {components_mha_plot['total_cost']:.0f}, Makespan: {makespan_mha_plot:.0f})",
            simulated_time_points=sim_time_points_mha,
            simulated_mh_levels=sim_mh_levels_mha,
        )
        print("Displayed M&H-aware greedy (test) schedule plot.")
    else:
        print("Could not generate a valid M&H-aware schedule for plotting.")

    print("\nGA Model Validation Script Finished.")

# More validation functions to be added:
# - Test scheduling_cost with handcrafted schedules
# - Test Pareto front evolution (requires running optimizer)
# - Test objective improvement over generations (requires running optimizer)
# - Test penalty analysis for solutions
# - Test sensitivity to parameter changes
