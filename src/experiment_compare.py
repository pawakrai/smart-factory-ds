import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from . import app_v9 as sim
    from .policies_baseline import (
        continuous_melting_controller,
        make_rule_based_controller,
    )
else:
    import app_v9 as sim
    from policies_baseline import continuous_melting_controller, make_rule_based_controller
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination


def _default_policy_vector():
    n = 12 if sim.SIMPLE_POLICY_MODE else 21
    return np.zeros(n, dtype=float)


def _details_from_sim(sim_result, name, opt_mode="baseline"):
    m = sim_result["metrics"]
    details = {
        "policy_name": name,
        "opt_mode": opt_mode,
        "policy": sim_result["policy"],
        "schedule": sim_result["schedule"],
        "mh_levels": sim_result["mh_levels"],
        "baseline_kw": sim_result["baseline_kw"],
        "if_kw": sim_result["if_kw"],
        "tou_raw_price": sim_result["tou_raw_price"],
        "tou_effective_price": sim_result["tou_effective_price"],
        "total_plant_kw": sim_result["total_plant_kw"],
        "objective_total_cost": np.nan,
        "energy_mode_total_cost": np.nan,
        "total_if_kwh": float(m.get("total_if_kwh", 0.0)),
        "melt_kwh": float(m.get("melt_kwh", 0.0)),
        "reheat_kwh": float(m.get("reheat_kwh", 0.0)),
        "total_energy_cost": float(m.get("total_energy_cost", 0.0)),
        "reheat_energy_cost": float(m.get("reheat_energy_cost", 0.0)),
        "peak_kw": float(m.get("peak_kw", 0.0)),
        "demand_penalty": float(m.get("demand_penalty", 0.0)),
        "holding_minutes_total": float(m.get("holding_minutes_total", 0.0)),
        "overflow_kg_total": float(m.get("overflow_kg_total", 0.0)),
        "poured_batches_count": int(m.get("poured_batches_count", 0)),
        "missing_batches": int(m.get("missing_batches", 0)),
        "zero_minutes_total": float(m.get("zero_minutes_total", 0.0)),
        "eps_deficit_area": float(m.get("eps_deficit_area", 0.0)),
        "mh_empty_minutes_A": float(m.get("mh_empty_minutes", {}).get("A", 0.0)),
        "mh_empty_minutes_B": float(m.get("mh_empty_minutes", {}).get("B", 0.0)),
        "mh_low_level_minutes_A": float(m.get("mh_low_level_minutes", {}).get("A", 0.0)),
        "mh_low_level_minutes_B": float(m.get("mh_low_level_minutes", {}).get("B", 0.0)),
        "min_level_A": float(m.get("min_level_reached", {}).get("A", 0.0)),
        "min_level_B": float(m.get("min_level_reached", {}).get("B", 0.0)),
        "solar_cost_saving": float(m.get("solar_cost_saving", 0.0)),
        "delay_reason_counts": m.get("delay_reason_counts", {}),
        "controller_name": m.get("controller_name", "baseline"),
    }
    return details


def _run_ga_mode(opt_mode, seed=42, n_gen=100, pop_size=80):
    sim.OPT_MODE = opt_mode
    sim._EVAL_CACHE.clear()
    np.random.seed(seed)

    problem = sim.PolicyProblem()
    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.92, eta=8),
        mutation=PolynomialMutation(prob=0.45, eta=8),
        eliminate_duplicates=sim.PolicyDuplicateElimination(),
    )
    termination = get_termination("n_gen", n_gen)
    callback = sim.StagnationEarlyStopCallback(
        patience_gens=sim.EARLY_STOP_PATIENCE_GENS,
        delta_obj=sim.EARLY_STOP_DELTA_OBJ1,
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=True,
        save_history=False,
        callback=callback,
    )

    if result is not None and result.X is not None:
        best_x = np.asarray(result.X, dtype=float).ravel()
    else:
        pop = result.pop if (result is not None and getattr(result, "pop", None) is not None) else algorithm.pop
        pop_x = pop.get("X")
        pop_f = pop.get("F")
        pop_g = pop.get("G")
        if pop_g is None:
            cv = np.zeros(len(pop_x), dtype=float)
        else:
            g = np.asarray(pop_g, dtype=float)
            if g.ndim == 1:
                g = g.reshape(-1, 1)
            cv = np.sum(np.maximum(0.0, g), axis=1)
        fvals = np.asarray(pop_f, dtype=float).reshape(-1) if pop_f is not None else np.full(len(pop_x), np.inf)
        candidate_idxs = np.where(cv == np.min(cv))[0]
        if len(candidate_idxs) > 1:
            idx = int(candidate_idxs[np.argmin(fvals[candidate_idxs])])
        else:
            idx = int(candidate_idxs[0])
        best_x = np.asarray(pop_x[idx], dtype=float).ravel()

    _, _, details = sim.evaluate_policy(best_x)
    details["policy_name"] = f"ga_{opt_mode}"
    details["controller_name"] = "ga_policy"
    return details


def _comparison_rows(details_list):
    rows = []
    for d in details_list:
        rows.append(
            {
                "policy": d.get("policy_name", "-"),
                "opt_mode": d.get("opt_mode", "-"),
                "objective_total_cost": round(float(d.get("objective_total_cost", np.nan)), 3),
                "energy_cost": round(float(d.get("total_energy_cost", 0.0)), 3),
                "reheat_kwh": round(float(d.get("reheat_kwh", 0.0)), 3),
                "reheat_cost": round(float(d.get("reheat_energy_cost", 0.0)), 3),
                "holding_min": round(float(d.get("holding_minutes_total", 0.0)), 3),
                "poured_batches": int(d.get("poured_batches_count", 0)),
                "missing_batches": int(d.get("missing_batches", 0)),
                "overflow_kg": round(float(d.get("overflow_kg_total", 0.0)), 3),
                "zero_minutes": round(float(d.get("zero_minutes_total", 0.0)), 3),
                "eps_deficit_area": round(float(d.get("eps_deficit_area", 0.0)), 3),
                "min_level_A": round(float(d.get("min_level_A", 0.0)), 3),
                "min_level_B": round(float(d.get("min_level_B", 0.0)), 3),
                "peak_kw": round(float(d.get("peak_kw", 0.0)), 3),
                "solar_saving": round(float(d.get("solar_cost_saving", 0.0)), 3),
            }
        )
    return rows


def _print_table(rows):
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print("\n=== Experiment Comparison ===")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _write_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_overlay_total_kw(details_list):
    plt.figure(figsize=(16, 5))
    t = np.arange(sim.SIM_DURATION_MIN) + sim.SHIFT_START
    for d in details_list:
        y = np.asarray(d.get("total_plant_kw"), dtype=float)
        plt.plot(t, y, linewidth=1.8, label=d.get("policy_name", "policy"))
    plt.axhline(sim.CONTRACT_DEMAND_KW, linestyle="--", color="black", label="Contract kW")
    plt.title("Total Plant kW Overlay Across Policies")
    plt.ylabel("kW")
    plt.xlabel("Time (HH:MM)")
    xticks = np.arange(sim.SHIFT_START, sim.SHIFT_START + sim.SIM_DURATION_MIN + 1, 60)
    plt.xticks(xticks, [f"{(x // 60) % 24:02d}:{x % 60:02d}" for x in xticks], rotation=0)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main(seed=42):
    np.random.seed(seed)
    sanity = sim.quick_capacity_sanity_check()
    sim.print_capacity_sanity_report(sanity)
    if not sanity["is_feasible_coarse"]:
        raise ValueError("Scenario fails coarse capacity sanity check.")

    base_x = _default_policy_vector()
    rb_controller = make_rule_based_controller()

    res_cont = _details_from_sim(
        sim.simulate_policy_day(base_x, controller=continuous_melting_controller),
        name="continuous_baseline",
        opt_mode="baseline",
    )
    res_rule = _details_from_sim(
        sim.simulate_policy_day(base_x, controller=rb_controller),
        name="rule_based",
        opt_mode="baseline",
    )
    res_ga_energy = _run_ga_mode("energy", seed=seed)
    res_ga_service = _run_ga_mode("service", seed=seed)

    all_results = [res_cont, res_rule, res_ga_energy, res_ga_service]
    rows = _comparison_rows(all_results)
    _print_table(rows)

    csv_path = Path(__file__).resolve().parent / "experiment_compare_results.csv"
    _write_csv(rows, csv_path)
    print(f"\nSaved comparison CSV: {csv_path}")

    for d in all_results:
        sim.plot_policy_result(d, title_prefix=d.get("policy_name", "policy"))

    _plot_overlay_total_kw(all_results)


if __name__ == "__main__":
    main(seed=42)

