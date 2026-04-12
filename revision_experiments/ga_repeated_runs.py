"""
Script E — GA Repeated-Run Statistics (Reviewer Item E)
========================================================
Runs the GA day scheduler across 20 independent random seeds to provide
proper repeated-run statistics for the paper (mean, SD, best, median).

TWO-TIER APPROACH:
  - Reduced-budget runs (n_gen=50, pop_size=50): 20 seeds × 2 modes = 40 runs
    → used for mean/SD/best/median statistics
  - Bridge comparison (n_gen=100, pop_size=80, the original paper config):
    3 seeds × 2 modes = 6 runs → validates that reduced-budget results are
    consistent with the original configuration; MUST be reported in paper

The bridge comparison ensures reviewers can connect these repeatability
statistics back to the main paper results.

Metrics collected per run:
  energy_cost, reheat_kwh, holding_min, poured_batches,
  missing_batches, peak_kw, solar_saving

Outputs (outputs/revision_phase1/ga_repeated/):
  ga_20runs_energy.csv
  ga_20runs_service.csv
  ga_bridge_energy.csv
  ga_bridge_service.csv
  ga_statistics_summary.csv
  ga_distribution_plots.png
  ga_repeated_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  PYTHONPATH=. python revision_experiments/ga_repeated_runs.py
"""

import os
import sys
import json
import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # required for import app_v9 as sim

# Mirror the import pattern from src/experiment_compare.py
import src.app_v9 as sim
from src.experiment_compare import _run_ga_mode

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "ga_repeated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
# Reduced-budget runs (repeatability experiment)
REDUCED_N_GEN = 50
REDUCED_POP_SIZE = 50
REDUCED_SEEDS = list(range(20))  # seeds 0-19
OPT_MODES = ["energy", "service"]

# Bridge comparison runs (original paper config)
BRIDGE_N_GEN = 100
BRIDGE_POP_SIZE = 80
BRIDGE_SEEDS = [42, 0, 1]   # seed 42 = original paper run

METRIC_KEYS = [
    "energy_cost",
    "reheat_kwh",
    "holding_min",
    "poured_batches",
    "missing_batches",
    "peak_kw",
    "solar_saving",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _extract_metrics(details: dict) -> dict:
    """Extract the key metric values from a _run_ga_mode() result dict."""
    return {
        "energy_cost":    float(details.get("total_energy_cost", 0.0)),
        "reheat_kwh":     float(details.get("reheat_kwh", 0.0)),
        "holding_min":    float(details.get("holding_minutes_total", 0.0)),
        "poured_batches": int(details.get("poured_batches_count", 0)),
        "missing_batches":int(details.get("missing_batches", 0)),
        "peak_kw":        float(details.get("peak_kw", 0.0)),
        "solar_saving":   float(details.get("solar_cost_saving", 0.0)),
    }


def _run_batch(seeds: list[int], mode: str, n_gen: int, pop_size: int, label: str) -> list[dict]:
    """Run GA for each seed and return list of result dicts."""
    results = []
    total = len(seeds)
    for i, seed in enumerate(seeds):
        t0 = time.perf_counter()
        print(f"  [{label}/{mode}] Seed {seed} ({i+1}/{total})...", end=" ", flush=True)
        try:
            details = _run_ga_mode(mode, seed=seed, n_gen=n_gen, pop_size=pop_size)
            metrics = _extract_metrics(details)
            metrics.update({"seed": seed, "mode": mode, "n_gen": n_gen,
                            "pop_size": pop_size, "tier": label,
                            "wall_time_sec": round(time.perf_counter() - t0, 2)})
            results.append(metrics)
            print(f"done ({metrics['wall_time_sec']:.1f}s) | "
                  f"energy_cost={metrics['energy_cost']:.1f} | "
                  f"poured={metrics['poured_batches']}")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"seed": seed, "mode": mode, "n_gen": n_gen,
                            "pop_size": pop_size, "tier": label,
                            "error": str(e)})
    return results


def _compute_stats(rows: list[dict], mode: str) -> dict:
    """Compute mean, SD, best (min energy_cost), median, worst per metric."""
    clean = [r for r in rows if "error" not in r and r["mode"] == mode]
    if not clean:
        return {"mode": mode, "n_runs": 0, "note": "all runs errored"}

    stats = {"mode": mode, "n_runs": len(clean)}
    for key in METRIC_KEYS:
        vals = np.array([r[key] for r in clean], dtype=float)
        is_higher_better = key in ("poured_batches", "solar_saving")
        stats[f"{key}_mean"]   = float(vals.mean())
        stats[f"{key}_sd"]     = float(vals.std())
        stats[f"{key}_median"] = float(np.median(vals))
        stats[f"{key}_best"]   = float(vals.max() if is_higher_better else vals.min())
        stats[f"{key}_worst"]  = float(vals.min() if is_higher_better else vals.max())
    return stats


def _save_csv(rows: list[dict], path: Path) -> None:
    clean = [r for r in rows if "error" not in r]
    if not clean:
        return
    all_keys = list(clean[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(clean)


def _save_summary(all_stats: list[dict], path: Path) -> None:
    if not all_stats:
        return
    all_keys = list(all_stats[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_stats)


def _plot_distributions(reduced_rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        plot_metrics = ["energy_cost", "reheat_kwh", "holding_min",
                        "poured_batches", "peak_kw", "solar_saving"]
        colors = {"energy": "#E3000F", "service": "#3B82F6"}

        for ax, metric in zip(axes.flatten(), plot_metrics):
            for mode in OPT_MODES:
                vals = [r[metric] for r in reduced_rows
                        if "error" not in r and r["mode"] == mode]
                if vals:
                    ax.hist(vals, bins=8, alpha=0.6, label=mode,
                            color=colors.get(mode, "gray"))
            ax.set_title(metric)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle(
            f"GA Repeated Runs Distribution\n"
            f"(n_gen={REDUCED_N_GEN}, pop_size={REDUCED_POP_SIZE}, 20 seeds each mode)",
            fontsize=11,
        )
        plt.tight_layout()
        path = OUTPUT_DIR / "ga_distribution_plots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {path.relative_to(PROJECT_ROOT)}")
    except ImportError:
        print("  [SKIP] matplotlib not available.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ga_repeated_runs.py — Reviewer Item E")
    print("=" * 65)
    print(f"Reduced-budget : n_gen={REDUCED_N_GEN}, pop_size={REDUCED_POP_SIZE}, seeds={REDUCED_SEEDS}")
    print(f"Bridge         : n_gen={BRIDGE_N_GEN}, pop_size={BRIDGE_POP_SIZE}, seeds={BRIDGE_SEEDS}")
    print(f"Modes          : {OPT_MODES}")
    print(f"Total runs     : {len(REDUCED_SEEDS)*len(OPT_MODES)} (reduced) + "
          f"{len(BRIDGE_SEEDS)*len(OPT_MODES)} (bridge) = "
          f"{len(REDUCED_SEEDS)*len(OPT_MODES) + len(BRIDGE_SEEDS)*len(OPT_MODES)} total")

    # ── Reduced-budget runs ───────────────────────────────────────────────────
    all_reduced = []
    for mode in OPT_MODES:
        print(f"\n=== Reduced-budget: mode={mode} ===")
        rows = _run_batch(REDUCED_SEEDS, mode, REDUCED_N_GEN, REDUCED_POP_SIZE, "reduced")
        all_reduced.extend(rows)

    # ── Bridge comparison runs ────────────────────────────────────────────────
    all_bridge = []
    for mode in OPT_MODES:
        print(f"\n=== Bridge comparison: mode={mode} (original config) ===")
        rows = _run_batch(BRIDGE_SEEDS, mode, BRIDGE_N_GEN, BRIDGE_POP_SIZE, "bridge")
        all_bridge.extend(rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    for mode in OPT_MODES:
        _save_csv(
            [r for r in all_reduced if r.get("mode") == mode],
            OUTPUT_DIR / f"ga_20runs_{mode}.csv",
        )
        _save_csv(
            [r for r in all_bridge if r.get("mode") == mode],
            OUTPUT_DIR / f"ga_bridge_{mode}.csv",
        )

    # ── Statistics ────────────────────────────────────────────────────────────
    all_stats = []
    print("\n=== Summary Statistics ===")
    for tier, rows in [("reduced", all_reduced), ("bridge", all_bridge)]:
        for mode in OPT_MODES:
            stats = _compute_stats(rows, mode)
            stats["tier"] = tier
            all_stats.append(stats)
            print(f"\n  {tier}/{mode} (N={stats.get('n_runs', 0)} runs):")
            for key in METRIC_KEYS:
                if f"{key}_mean" in stats:
                    print(f"    {key}: {stats[f'{key}_mean']:.2f} ± {stats[f'{key}_sd']:.2f}  "
                          f"(best={stats[f'{key}_best']:.2f}, median={stats[f'{key}_median']:.2f})")

    _save_summary(all_stats, OUTPUT_DIR / "ga_statistics_summary.csv")

    # ── Bridge consistency check ──────────────────────────────────────────────
    print("\n=== Bridge Consistency Check ===")
    for mode in OPT_MODES:
        red = _compute_stats(all_reduced, mode)
        brd = _compute_stats(all_bridge, mode)
        if "energy_cost_mean" in red and "energy_cost_mean" in brd:
            diff_pct = abs(red["energy_cost_mean"] - brd["energy_cost_mean"]) / max(1, brd["energy_cost_mean"]) * 100
            print(f"  {mode} energy_cost: reduced={red['energy_cost_mean']:.1f}, "
                  f"bridge={brd['energy_cost_mean']:.1f}, diff={diff_pct:.1f}%")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_distributions(all_reduced)

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "script": "ga_repeated_runs.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "reduced_budget": {
            "n_gen": REDUCED_N_GEN,
            "pop_size": REDUCED_POP_SIZE,
            "seeds": REDUCED_SEEDS,
            "n_modes": len(OPT_MODES),
            "total_runs": len(REDUCED_SEEDS) * len(OPT_MODES),
        },
        "bridge": {
            "n_gen": BRIDGE_N_GEN,
            "pop_size": BRIDGE_POP_SIZE,
            "seeds": BRIDGE_SEEDS,
            "n_modes": len(OPT_MODES),
            "total_runs": len(BRIDGE_SEEDS) * len(OPT_MODES),
        },
        "opt_modes": OPT_MODES,
        "metric_keys": METRIC_KEYS,
        "early_stop_config": {
            "patience_gens": sim.EARLY_STOP_PATIENCE_GENS,
            "delta_obj": sim.EARLY_STOP_DELTA_OBJ1,
        },
        "filters_applied": [],
        "note": (
            "Reduced-budget (n_gen=50, pop_size=50) used for repeatability statistics. "
            "Bridge runs (n_gen=100, pop_size=80) reproduce original paper config on 3 seeds "
            "to validate consistency. Seed 42 = original paper experiment."
        ),
    }
    with open(OUTPUT_DIR / "ga_repeated_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Saved summary : {(OUTPUT_DIR / 'ga_statistics_summary.csv').relative_to(PROJECT_ROOT)}")
    print(f"  Saved manifest: {(OUTPUT_DIR / 'ga_repeated_manifest.json').relative_to(PROJECT_ROOT)}")
    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/ga_repeated/")
    print("=" * 65)


if __name__ == "__main__":
    main()
