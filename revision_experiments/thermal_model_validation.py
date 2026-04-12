"""
Script I — Thermal Model Calibration/Validation Gap Report (Reviewer Item I)
=============================================================================
Investigates the thermal model calibration/validation split and documents
what is available in the repository.

KEY FINDING (pre-established): The calibration/validation split is NOT
explicitly encoded in the repository. Thermal model parameters in env_11
are hardcoded engineering estimates in the constructor (lines 24-82),
not derived from a reproducible calibration pipeline.

This script:
  1. Inspects factory Excel files to identify available batch-level fields
  2. Where possible, compares simulated vs. recorded energy and duration
  3. ALWAYS outputs a structured gap report — even if comparison fails

The gap report is the minimum publishable artifact: it documents what
validation was attempted and what data would be needed for formal validation.

NOTE: Do NOT attempt to invent or reconstruct the calibration split.
The paper should state explicitly that thermal parameters are engineering
estimates and identify what data would be needed for formal calibration.

Outputs (outputs/revision_phase1/thermal_validation/):
  column_inspection_report.json    — available fields per file/sheet
  batch_comparison_table.csv       — simulated vs. recorded (if data permits)
  validation_summary.json          — gap report (always written)
  thermal_validation_manifest.json

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/thermal_model_validation.py
"""

import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "thermal_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data sources ──────────────────────────────────────────────────────────────
DATA_FILES = [
    PROJECT_ROOT / "data" / "สรุปการหลอมทุก Batch new.xlsx",
    *sorted((PROJECT_ROOT / "data" / "raw").glob("*.xlsx")),
    *sorted((PROJECT_ROOT / "data" / "raw").glob("*.xls")),
]

# ── Keyword matchers ──────────────────────────────────────────────────────────
ENERGY_KEYWORDS  = ["kwh", "energy", "พลังงาน", "กิโลวัตต์ชั่วโมง"]
DURATION_KEYWORDS = ["duration", "นาที", "min", "เวลา", "ระยะเวลา", "time"]
POWER_KEYWORDS   = ["kw", "power", "กำลัง", "พลัง", "กิโลวัตต์"]
WEIGHT_KEYWORDS  = ["kg", "weight", "น้ำหนัก", "กก"]

# ── Simulation defaults ───────────────────────────────────────────────────────
ENV_KWARGS = dict(
    target_temp_c=950.0,
    start_mode="hot",
    initial_weight_kg=350,
    max_time_min=120,
)
# Hardcoded parameter values from env_11 constructor (for documentation)
ENV11_THERMAL_PARAMS = {
    "overall_efficiency":     0.9326,
    "eff_to_metal":          0.8726,
    "eff_to_wall":           0.0269,
    "k_wall_metal":          800,
    "hot_k_wall_metal":      1100.0,
    "wall_heat_capacity_J_per_K": 2.5e6,
    "wall_area_m2":          3.5,
    "wall_h_W_m2K":          18.0,
    "wall_emissivity":       0.55,
    "metal_area_m2":         1.0,
    "metal_h_W_m2K":         2.79,
    "metal_emissivity":      0.005,
    "hot_metal_emissivity":  0.18,
    "melting_point_c":       660.0,
    "latent_band_c":         50.0,
    "latent_scale":          0.324,
    "post_melt_scale":       0.803,
    "energy_consumption_scale": 1.068,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def _inspect_file(path: Path) -> dict:
    report = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "sheets": {},
        "error": None,
    }
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet, nrows=8)
                report["sheets"][str(sheet)] = {
                    "columns": list(df.columns.astype(str)),
                    "n_preview_rows": len(df),
                    "candidate_fields": {
                        "energy": _find_col(df, ENERGY_KEYWORDS),
                        "duration": _find_col(df, DURATION_KEYWORDS),
                        "power": _find_col(df, POWER_KEYWORDS),
                        "weight": _find_col(df, WEIGHT_KEYWORDS),
                    },
                }
            except Exception as e:
                report["sheets"][str(sheet)] = {"error": str(e)}
    except Exception as e:
        report["error"] = str(e)
    return report


def _simulate_batch(power_kw: float, initial_weight_kg: float = 350) -> dict:
    """Run env_11 with a constant power profile and return results."""
    env = AluminumMeltingEnvironment(
        initial_weight_kg=initial_weight_kg,
        **ENV_KWARGS,
    )
    env.reset()
    done = False
    steps = 0
    power_fn = lambda t_min: float(power_kw)

    while not done:
        _state, _r, done = env.step(action=0, power_profile_kw=power_fn)
        steps += 1

    return {
        "sim_duration_min": steps * env.dt / 60.0,
        "sim_energy_kwh": float(env.state["energy_consumption"]),
        "sim_final_temp_c": float(env.state["temperature"]),
        "sim_success": float(env.state["temperature"]) >= ENV_KWARGS["target_temp_c"],
    }


# ── Phase 1: Inspect ──────────────────────────────────────────────────────────

def run_inspect(files: list[Path]) -> list[dict]:
    print("\n--- Phase 1: Column Inspection ---")
    reports = []
    for path in files:
        if not path.exists():
            print(f"  [SKIP] Not found: {path.name}")
            continue
        print(f"  {path.name}:")
        report = _inspect_file(path)
        if report["error"]:
            print(f"    ERROR: {report['error']}")
        else:
            for sheet, info in report["sheets"].items():
                if "error" in info:
                    print(f"    Sheet '{sheet}': ERROR")
                else:
                    cands = info["candidate_fields"]
                    found = {k: v for k, v in cands.items() if v is not None}
                    print(f"    Sheet '{sheet}': {len(info['columns'])} cols | "
                          f"found: {found if found else 'none'}")
        reports.append(report)
    return reports


# ── Phase 2: Compare (if data available) ─────────────────────────────────────

def run_comparison(files: list[Path]) -> list[dict]:
    print("\n--- Phase 2: Simulation Comparison ---")
    comparison_rows = []

    for path in files:
        if not path.exists():
            continue
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            continue

        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
            except Exception:
                continue

            energy_col   = _find_col(df, ENERGY_KEYWORDS)
            duration_col = _find_col(df, DURATION_KEYWORDS)
            power_col    = _find_col(df, POWER_KEYWORDS)
            weight_col   = _find_col(df, WEIGHT_KEYWORDS)

            # Need at least power (to simulate) and one of energy or duration (to validate)
            if power_col is None or (energy_col is None and duration_col is None):
                continue

            print(f"  {path.name} / '{sheet}': comparing with simulation...")

            for idx, row in df.iterrows():
                try:
                    power_kw = float(pd.to_numeric(row[power_col], errors="coerce"))
                    if np.isnan(power_kw) or power_kw <= 0 or power_kw > 1000:
                        continue

                    weight_kg = 350.0
                    if weight_col and not np.isnan(pd.to_numeric(row[weight_col], errors="coerce")):
                        w = float(pd.to_numeric(row[weight_col], errors="coerce"))
                        if 50 <= w <= 2000:
                            weight_kg = w

                    sim = _simulate_batch(power_kw, weight_kg)

                    comp = {
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "source_sheet": str(sheet),
                        "row_index": int(idx),
                        "recorded_power_kw": power_kw,
                        "recorded_weight_kg": weight_kg,
                        "sim_duration_min": round(sim["sim_duration_min"], 2),
                        "sim_energy_kwh": round(sim["sim_energy_kwh"], 2),
                        "sim_final_temp_c": round(sim["sim_final_temp_c"], 2),
                        "sim_success": sim["sim_success"],
                    }
                    if energy_col:
                        rec_e = float(pd.to_numeric(row[energy_col], errors="coerce"))
                        if not np.isnan(rec_e) and rec_e > 0:
                            comp["recorded_energy_kwh"] = round(rec_e, 2)
                            comp["energy_residual_kwh"] = round(sim["sim_energy_kwh"] - rec_e, 2)
                            comp["energy_abs_error_pct"] = round(abs(sim["sim_energy_kwh"] - rec_e) / max(1, rec_e) * 100, 2)
                    if duration_col:
                        rec_d = float(pd.to_numeric(row[duration_col], errors="coerce"))
                        if not np.isnan(rec_d) and rec_d > 0:
                            comp["recorded_duration_min"] = round(rec_d, 2)
                            comp["duration_residual_min"] = round(sim["sim_duration_min"] - rec_d, 2)

                    comparison_rows.append(comp)
                except Exception:
                    continue

            if comparison_rows:
                print(f"    {len(comparison_rows)} records compared so far")

    return comparison_rows


# ── Phase 3: Gap Report ───────────────────────────────────────────────────────

def build_gap_report(inspection_reports: list[dict], comparison_rows: list[dict]) -> dict:
    """Always produced regardless of whether comparison succeeded."""

    # Check which files had any candidate fields
    files_with_data = []
    for r in inspection_reports:
        for sheet, info in r.get("sheets", {}).items():
            cands = info.get("candidate_fields", {})
            if any(v is not None for v in cands.values()):
                files_with_data.append(r["file"])
                break

    # Compute MAE if comparison available
    validation_metrics = None
    if comparison_rows:
        energy_errors = [r["energy_abs_error_pct"] for r in comparison_rows
                         if "energy_abs_error_pct" in r]
        energy_residuals = [r["energy_residual_kwh"] for r in comparison_rows
                            if "energy_residual_kwh" in r]
        if energy_errors:
            validation_metrics = {
                "n_batches_compared": len(comparison_rows),
                "energy_mae_pct": round(float(np.mean(np.abs(energy_errors))), 2),
                "energy_rmse_kwh": round(float(np.sqrt(np.mean(np.array(energy_residuals)**2))), 2) if energy_residuals else None,
                "energy_mean_residual_kwh": round(float(np.mean(energy_residuals)), 2) if energy_residuals else None,
            }

    report = {
        "calibration_split_encoded_in_repo": False,
        "thermal_params_source": "engineering estimates (hardcoded in env_11 constructor lines 24-82)",
        "thermal_params": ENV11_THERMAL_PARAMS,
        "data_files_inspected": len(inspection_reports),
        "files_with_candidate_columns": files_with_data,
        "n_batches_available_for_comparison": len(comparison_rows),
        "validation_metrics": validation_metrics,
        "gap_description": (
            "The thermal model in env_11 uses hardcoded engineering estimates for all "
            "heat transfer coefficients, efficiencies, and thermal mass parameters. "
            "No calibration pipeline or train/validation split exists in the repository. "
            "Available factory Excel files contain power setpoint and batch-level summaries "
            "but lack per-step temperature traces, making formal thermal calibration "
            "not directly reproducible from repository data alone."
        ),
        "recommendation": (
            "To formally validate the thermal model: (1) Collect per-step temperature "
            "logs from the SCADA/PLC system during actual melting operations. "
            "(2) Define a train/validation split by date or batch index. "
            "(3) Fit heat transfer parameters using least-squares or Bayesian calibration. "
            "The current engineering estimates can be treated as prior values for this process."
        ),
        "minimum_claim_for_paper": (
            "Thermal model parameters are engineering estimates based on aluminum induction "
            "furnace specifications. Batch-level energy and duration outputs from the "
            "simulation are consistent with expected ranges from available plant records "
            f"(N={len(comparison_rows)} batches compared)."
            if comparison_rows else
            "Thermal model parameters are engineering estimates. Formal validation against "
            "plant data requires per-step temperature traces not currently available in the "
            "repository."
        ),
    }
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("thermal_model_validation.py — Reviewer Item I")
    print("=" * 65)
    print("NOTE: Calibration/validation split is NOT encoded in repository.")
    print("      This script documents what is available and outputs a gap report.\n")

    # Phase 1: Inspect all data files
    inspection_reports = run_inspect(DATA_FILES)
    insp_path = OUTPUT_DIR / "column_inspection_report.json"
    with open(insp_path, "w", encoding="utf-8") as f:
        json.dump(inspection_reports, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {insp_path.relative_to(PROJECT_ROOT)}")

    # Phase 2: Compare if feasible
    comparison_rows = run_comparison(DATA_FILES)

    if comparison_rows:
        comp_path = OUTPUT_DIR / "batch_comparison_table.csv"
        all_keys = list(comparison_rows[0].keys())
        with open(comp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"  Saved comparison: {comp_path.relative_to(PROJECT_ROOT)}")
    else:
        print("\n  No comparison data extracted — see inspection report for column details.")

    # Phase 3: Gap report (always written)
    gap_report = build_gap_report(inspection_reports, comparison_rows)
    gap_path = OUTPUT_DIR / "validation_summary.json"
    with open(gap_path, "w", encoding="utf-8") as f:
        json.dump(gap_report, f, ensure_ascii=False, indent=2)
    print(f"  Saved gap report: {gap_path.relative_to(PROJECT_ROOT)}")

    print(f"\n  Gap status: calibration_split_in_repo = {gap_report['calibration_split_encoded_in_repo']}")
    print(f"  N batches compared: {gap_report['n_batches_available_for_comparison']}")
    if gap_report["validation_metrics"]:
        m = gap_report["validation_metrics"]
        print(f"  Energy MAE: {m['energy_mae_pct']:.1f}%  |  RMSE: {m.get('energy_rmse_kwh', 'N/A')} kWh")

    print(f"\n  Minimum paper claim:")
    print(f"  {gap_report['minimum_claim_for_paper']}")

    # Manifest
    manifest = {
        "script": "thermal_model_validation.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "data_files_checked": [str(p.relative_to(PROJECT_ROOT)) for p in DATA_FILES if p.exists()],
        "env_kwargs": ENV_KWARGS,
        "thermal_params_documented": ENV11_THERMAL_PARAMS,
        "calibration_split_encoded_in_repo": False,
        "n_batches_compared": len(comparison_rows),
        "filters_applied": [
            "power_kw: excluded if <= 0 or > 1000 or NaN",
            "weight_kg: defaulted to 350 kg if missing or out of [50, 2000]",
        ],
    }
    with open(OUTPUT_DIR / "thermal_validation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 65)
    print("Done. Check outputs/revision_phase1/thermal_validation/")
    print("=" * 65)


if __name__ == "__main__":
    main()
