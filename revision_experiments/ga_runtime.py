"""
Script H — GA Computational Cost / Runtime (Reviewer Item H)
=============================================================
Measures the wall-clock time for one 24-hour schedule optimization
to support the paper's claim that daily re-optimization is feasible
in a production deployment.

Method:
  - Run _run_ga_mode("energy") 5 times with seeds 0-4
  - Use original paper configuration: n_gen=100, pop_size=80
  - Record wall-clock time per run with time.perf_counter()
  - Collect machine info: platform, processor, CPU count

Outputs (outputs/revision_phase1/ga_runtime/):
  timing_results.json    — per-run wall times
  machine_info.json      — hardware details
  ga_runtime_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  PYTHONPATH=. python revision_experiments/ga_runtime.py
"""

import os
import sys
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.experiment_compare import _run_ga_mode
import src.app_v9 as sim

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "ga_runtime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_TIMING_RUNS = 5
TIMING_SEEDS = list(range(N_TIMING_RUNS))   # seeds 0-4
OPT_MODE = "energy"
N_GEN = 100      # original paper config
POP_SIZE = 80    # original paper config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_machine_info() -> dict:
    info = {
        "platform": platform.platform(),
        "system": platform.system(),
        "node": platform.node(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
    }
    # Try psutil for more detail
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["memory_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        info["psutil_available"] = False
    return info


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ga_runtime.py — Reviewer Item H (GA Computational Cost)")
    print("=" * 65)
    print(f"Config: mode={OPT_MODE}, n_gen={N_GEN}, pop_size={POP_SIZE}")
    print(f"Seeds : {TIMING_SEEDS}")

    # Machine info
    machine_info = _get_machine_info()
    print(f"\nHardware:")
    print(f"  Platform  : {machine_info['platform']}")
    print(f"  Processor : {machine_info['processor']}")
    print(f"  CPU cores : {machine_info['cpu_count_logical']}")

    machine_path = OUTPUT_DIR / "machine_info.json"
    with open(machine_path, "w") as f:
        json.dump(machine_info, f, indent=2)
    print(f"  Saved: {machine_path.relative_to(PROJECT_ROOT)}")

    # Timing runs
    print(f"\nRunning {N_TIMING_RUNS} timing runs...")
    timing_results = []

    for i, seed in enumerate(TIMING_SEEDS):
        print(f"\n  Run {i+1}/{N_TIMING_RUNS} (seed={seed})...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            details = _run_ga_mode(OPT_MODE, seed=seed, n_gen=N_GEN, pop_size=POP_SIZE)
            elapsed = time.perf_counter() - t0
            result = {
                "run": i + 1,
                "seed": seed,
                "opt_mode": OPT_MODE,
                "n_gen": N_GEN,
                "pop_size": POP_SIZE,
                "wall_time_sec": round(elapsed, 3),
                "poured_batches": int(details.get("poured_batches_count", 0)),
                "missing_batches": int(details.get("missing_batches", 0)),
                "energy_cost": round(float(details.get("total_energy_cost", 0.0)), 2),
                "early_stop_note": (
                    f"Early stopping configured: patience={sim.EARLY_STOP_PATIENCE_GENS} gens, "
                    f"delta={sim.EARLY_STOP_DELTA_OBJ1} THB"
                ),
                "error": None,
            }
            print(f"done in {elapsed:.1f}s | "
                  f"poured={result['poured_batches']} | "
                  f"energy_cost={result['energy_cost']:.1f} THB")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            result = {
                "run": i + 1, "seed": seed, "opt_mode": OPT_MODE,
                "n_gen": N_GEN, "pop_size": POP_SIZE,
                "wall_time_sec": round(elapsed, 3),
                "error": str(e),
            }
            print(f"ERROR: {e}")
        timing_results.append(result)

    # Summary statistics
    valid = [r for r in timing_results if r.get("error") is None]
    if valid:
        times = [r["wall_time_sec"] for r in valid]
        import numpy as np
        summary = {
            "n_successful_runs": len(valid),
            "wall_time_mean_sec": round(float(np.mean(times)), 3),
            "wall_time_sd_sec": round(float(np.std(times)), 3),
            "wall_time_min_sec": round(float(np.min(times)), 3),
            "wall_time_max_sec": round(float(np.max(times)), 3),
            "wall_time_mean_min": round(float(np.mean(times)) / 60.0, 2),
        }
        print(f"\n  === Timing Summary ===")
        print(f"  Mean  : {summary['wall_time_mean_sec']:.1f}s ({summary['wall_time_mean_min']:.2f} min)")
        print(f"  SD    : {summary['wall_time_sd_sec']:.1f}s")
        print(f"  Range : [{summary['wall_time_min_sec']:.1f}s, {summary['wall_time_max_sec']:.1f}s]")
        print(f"\n  Interpretation: GA optimization for one 24-hour schedule takes "
              f"~{summary['wall_time_mean_min']:.1f} min on this hardware, "
              f"feasible for daily re-optimization.")
    else:
        summary = {"n_successful_runs": 0, "error": "all runs failed"}
        print("  [ERROR] No successful timing runs.")

    # Save timing results
    timing_path = OUTPUT_DIR / "timing_results.json"
    with open(timing_path, "w") as f:
        json.dump({
            "config": {
                "opt_mode": OPT_MODE, "n_gen": N_GEN, "pop_size": POP_SIZE,
                "n_timing_runs": N_TIMING_RUNS, "seeds": TIMING_SEEDS,
                "early_stop_patience_gens": sim.EARLY_STOP_PATIENCE_GENS,
                "early_stop_delta": sim.EARLY_STOP_DELTA_OBJ1,
            },
            "summary": summary,
            "per_run_results": timing_results,
        }, f, indent=2)
    print(f"\n  Saved: {timing_path.relative_to(PROJECT_ROOT)}")

    # Manifest
    manifest = {
        "script": "ga_runtime.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "timing_seeds": TIMING_SEEDS,
        "opt_mode": OPT_MODE,
        "n_gen": N_GEN,
        "pop_size": POP_SIZE,
        "early_stop_config": {
            "patience_gens": sim.EARLY_STOP_PATIENCE_GENS,
            "delta_obj": sim.EARLY_STOP_DELTA_OBJ1,
        },
        "machine_info": machine_info,
        "summary": summary,
        "filters_applied": [],
    }
    with open(OUTPUT_DIR / "ga_runtime_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/ga_runtime/")
    print("=" * 65)


if __name__ == "__main__":
    main()
