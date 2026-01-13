from __future__ import annotations

"""
Core scheduling service wrapper around the HGA implementation in src/app_v5.py.

Entry point:
    run_hga_schedule(config: Optional[dict]) -> dict
"""

from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from src import app_v5


def _apply_config_overrides(cfg: Dict[str, Any]) -> None:
    """Apply simple config overrides into the global variables of app_v5."""

    # Number of batches
    num_batches = cfg.get("num_batches")
    if isinstance(num_batches, int) and num_batches > 0:
        app_v5.NUM_BATCHES = num_batches

    # Enable/disable furnaces
    if_cfg = cfg.get("if", {})
    use_a = if_cfg.get("use_furnace_a")
    use_b = if_cfg.get("use_furnace_b")
    if isinstance(use_a, bool):
        app_v5.USE_FURNACE_A = use_a
    if isinstance(use_b, bool):
        app_v5.USE_FURNACE_B = use_b

    # M&H parameters
    mh_cfg = cfg.get("mh", {})
    max_cap = mh_cfg.get("max_capacity")
    if isinstance(max_cap, dict):
        app_v5.MH_MAX_CAPACITY_KG.update(
            {k: float(v) for k, v in max_cap.items() if k in app_v5.MH_MAX_CAPACITY_KG}
        )

    init_level = mh_cfg.get("initial_level")
    if isinstance(init_level, dict):
        app_v5.MH_INITIAL_LEVEL_KG.update(
            {
                k: float(v)
                for k, v in init_level.items()
                if k in app_v5.MH_INITIAL_LEVEL_KG
            }
        )

    cons_rate = mh_cfg.get("consumption_rate")
    if isinstance(cons_rate, dict):
        app_v5.MH_CONSUMPTION_RATE_KG_PER_MIN.update(
            {
                k: float(v)
                for k, v in cons_rate.items()
                if k in app_v5.MH_CONSUMPTION_RATE_KG_PER_MIN
            }
        )

    # Solar configuration
    solar_cfg = cfg.get("solar", {})
    windows = solar_cfg.get("windows")
    if isinstance(windows, list):
        parsed: List[Tuple[int, int]] = []
        for w in windows:
            if (
                isinstance(w, (list, tuple))
                and len(w) == 2
                and isinstance(w[0], (int, float))
                and isinstance(w[1], (int, float))
            ):
                parsed.append((int(w[0]), int(w[1])))
        if parsed:
            app_v5.SOLAR_PREFERRED_WINDOWS_MIN = parsed

    discount = solar_cfg.get("discount_factor")
    if isinstance(discount, (int, float)):
        app_v5.SOLAR_ENERGY_DISCOUNT_FACTOR = float(discount)


def run_hga_schedule(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the HGA optimisation once and return a JSON‑serialisable result.
    """

    cfg = config or {}
    _apply_config_overrides(cfg)

    # --- Configure GA / NSGA-II ---
    ga_cfg = cfg.get("ga", {})
    pop_size = int(ga_cfg.get("pop_size", 50))
    n_gen = int(ga_cfg.get("n_gen", 100))
    seed = int(ga_cfg.get("seed", 42))

    problem = app_v5.HGAProblem(app_v5.NUM_BATCHES)

    sampling = app_v5.PermutationRandomSampling()
    crossover = app_v5.OrderCrossover()
    mutation = app_v5.InversionMutation()

    algorithm = app_v5.NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )

    termination = app_v5.get_termination("n_gen", n_gen)

    result = app_v5.minimize(
        problem, algorithm, termination, seed=seed, verbose=False, save_history=False
    )

    opt_population = result.opt
    if opt_population is None or len(opt_population) == 0:
        return {
            "status": "error",
            "message": "No feasible schedule found by HGA.",
        }

    individual = opt_population[0]
    energy_obj, makespan_obj = individual.F
    x_vector = individual.get("schedules")
    cost_details = individual.get("cost_details") or {}

    schedule_list: List[Dict[str, Any]] = []
    if x_vector is not None:
        sched = app_v5.decode_schedule(x_vector)
        for start_slot, end_slot, furnace_idx, batch_id in sched:
            schedule_list.append(
                {
                    "batch_id": int(batch_id),
                    "furnace": "A" if furnace_idx == 0 else "B",
                    "start_min": int(
                        start_slot * app_v5.SLOT_DURATION + app_v5.SHIFT_START
                    ),
                    "end_min": int(
                        end_slot * app_v5.SLOT_DURATION + app_v5.SHIFT_START
                    ),
                }
            )

    response: Dict[str, Any] = {
        "status": "ok",
        "objectives": {
            "total_cost": float(cost_details.get("total_cost", energy_obj)),
            "energy_cost_kwh_equiv": float(
                cost_details.get(
                    "priced_if_energy_cost_kwh",
                    cost_details.get("base_if_energy_kwh", energy_obj),
                )
            ),
            "makespan_min": float(cost_details.get("makespan_minutes", makespan_obj)),
        },
        "summary": {
            "num_cold_starts": int(cost_details.get("num_cold_start_events", 0)),
            "mh_low_level_penalty": float(
                cost_details.get("mh_low_level_penalty", 0.0)
            ),
            "mh_idle_penalty": float(cost_details.get("mh_idle_penalty", 0.0)),
            "mh_overflow_penalty": float(cost_details.get("mh_overflow_penalty", 0.0)),
            "if_general_penalty": float(cost_details.get("if_general_penalty", 0.0)),
            "if_holding_penalty": float(cost_details.get("if_holding_penalty", 0.0)),
        },
        "config_used": {
            "num_batches": app_v5.NUM_BATCHES,
            "use_furnace_a": app_v5.USE_FURNACE_A,
            "use_furnace_b": app_v5.USE_FURNACE_B,
            "solar_windows": app_v5.SOLAR_PREFERRED_WINDOWS_MIN,
            "solar_discount_factor": app_v5.SOLAR_ENERGY_DISCOUNT_FACTOR,
        },
        "schedule": sorted(
            schedule_list, key=lambda x: (x["start_min"], x["batch_id"])
        ),
    }

    return response
