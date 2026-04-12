"""
Script A0 — Extract Historical Plant Baseline Statistics
=========================================================
Reviewer item A: Replace vague ">600 kWh" claims with precise statistics derived
exclusively from real factory Excel logs.

IMPORTANT: This script is the ONLY valid source for historical baseline statistics.
Simulated fixed-power policies must NOT be used as stand-ins for real plant data.

Sources:
  data/สรุปการหลอมทุก Batch new.xlsx   — per-batch summary (primary)
    Structure: each sheet named "{power_kw}_{batch_number}" (e.g. '450_1', '475_2')
    - Power (kW) encoded in sheet name: int(sheet_name.split('_')[0])
    - Energy (kWh) embedded in 4th column header:
        e.g. "Energy consumption for Batch 1: 572.0 kWh\\nEnergy consumption per batch: ..."
    - Start time in row 0, column index 1 (datetime.time object)
    - End time in row 1, column index 1
    - Weight (kg) in row with label matching 'ingot' or 'pure', column index 1
  data/raw/*.xlsx                       — raw induction furnace logs (secondary)

Outputs (all in outputs/revision_phase1/historical_baseline/):
  column_inspection_report.json   — always written; lists columns in each file
  per_batch_records.csv           — cleaned per-batch rows (if extractable)
  plant_batch_stats.csv           — mean/SD/median/min/max/N per metric
  extract_manifest.json           — reproducibility sidecar

Usage:
  cd /path/to/smart-factory-ds
  python revision_experiments/extract_historical_baseline.py
"""

import os
import re
import sys
import json
import csv
import hashlib
import subprocess
from datetime import datetime, time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "revision_phase1" / "historical_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data sources ──────────────────────────────────────────────────────────────
# Ordered by priority: summary sheet first, then raw logs
DATA_FILES = [
    PROJECT_ROOT / "data" / "สรุปการหลอมทุก Batch new.xlsx",
    *sorted((PROJECT_ROOT / "data" / "raw").glob("*.xlsx")),
    *sorted((PROJECT_ROOT / "data" / "raw").glob("*.xls")),
    *sorted((PROJECT_ROOT / "data" / "raw" / "seperate_controls").glob("*.xlsx")),
]

# ── Column search terms (Thai + English variants) ─────────────────────────────
# We attempt fuzzy matching against known field names from the domain.
ENERGY_KEYWORDS = ["kwh", "energy", "พลังงาน", "กิโลวัตต์", "กิโลวัตต์ชั่วโมง"]
DURATION_KEYWORDS = ["duration", "time", "นาที", "min", "เวลา", "ระยะเวลา"]
WEIGHT_KEYWORDS = ["kg", "weight", "น้ำหนัก", "กก", "กิโลกรัม"]
POWER_KEYWORDS = ["kw", "power", "กำลัง", "พลัง"]

# ── Validity filters ──────────────────────────────────────────────────────────
# Batches outside these ranges are flagged as suspicious and excluded
ENERGY_VALID_RANGE = (50.0, 2000.0)    # kWh
DURATION_VALID_RANGE = (10.0, 300.0)   # minutes
WEIGHT_VALID_RANGE = (50.0, 2000.0)    # kg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _find_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return the first column whose lowercased name contains any keyword."""
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in keywords):
            return col
    return None


def _inspect_file(path: Path) -> dict:
    """Read a file and return a structured inspection report."""
    report = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "sha256_prefix": _file_sha256(path),
        "sheets": {},
        "error": None,
    }
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet, nrows=5)
                report["sheets"][str(sheet)] = {
                    "columns": list(df.columns.astype(str)),
                    "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
                    "n_preview_rows": len(df),
                }
            except Exception as e:
                report["sheets"][str(sheet)] = {"error": str(e)}
    except Exception as e:
        report["error"] = str(e)
    return report


def _extract_from_sheet(path: Path, sheet_name: str) -> pd.DataFrame | None:
    """
    Attempt to extract energy/duration/weight columns from a single sheet.
    Returns a cleaned DataFrame or None if no usable columns found.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return None

    if df.empty or len(df.columns) == 0:
        return None

    energy_col = _find_column(df, ENERGY_KEYWORDS)
    duration_col = _find_column(df, DURATION_KEYWORDS)
    weight_col = _find_column(df, WEIGHT_KEYWORDS)
    power_col = _find_column(df, POWER_KEYWORDS)

    found = {k: v for k, v in [
        ("energy_kwh", energy_col),
        ("duration_min", duration_col),
        ("weight_kg", weight_col),
        ("power_kw", power_col),
    ] if v is not None}

    if not found:
        return None

    records = pd.DataFrame()
    records["source_file"] = str(path.relative_to(PROJECT_ROOT))
    records["source_sheet"] = str(sheet_name)
    records["row_index"] = range(len(df))

    for metric, col in found.items():
        records[metric] = pd.to_numeric(df[col], errors="coerce")

    return records


# ── Name of the primary summary file (for targeted parsing) ──────────────────
SUMMARY_FILE_NAME = "สรุปการหลอมทุก Batch new.xlsx"


def _parse_time_value(val) -> dt_time | None:
    """Convert a cell value to a time object (handles datetime.time and strings)."""
    if isinstance(val, dt_time):
        return val
    if hasattr(val, "time"):   # pandas Timestamp / datetime
        return val.time()
    if isinstance(val, str):
        for fmt in ("%H:%M", "%H:%M:%S"):
            try:
                return datetime.strptime(val.strip(), fmt).time()
            except ValueError:
                pass
    return None


def _extract_summary_file(path: Path) -> list[dict]:
    """
    Targeted parser for 'สรุปการหลอมทุก Batch new.xlsx'.

    Each sheet = one batch:
      - Sheet name format: "{power_kw}_{batch_num}" (e.g. '450_1')
      - 4th column header contains energy string:
          "Energy consumption for Batch N: <value> kWh\\n..."
      - Row 0: ['Start', <time>, nan, nan]   → start time
      - Row 1: ['End',   <time>, nan, nan]   → end time
      - Row 2+: scan for row where col-0 matches /ingot|pure/i → weight in col-1
    """
    records = []
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"    [ERROR] Cannot open summary file: {e}")
        return records

    for sheet in xl.sheet_names:
        # Skip non-batch sheets (e.g. 'Summary', 'Sheet1')
        if not re.match(r"^\d+_\d+$", str(sheet).strip()):
            continue

        try:
            df = xl.parse(sheet, header=0)
        except Exception:
            continue

        if df.empty or len(df.columns) < 2:
            continue

        # ── Power from sheet name ────────────────────────────────────────
        try:
            power_kw = float(str(sheet).split("_")[0])
        except ValueError:
            power_kw = float("nan")

        # ── Energy from 4th column header (index 3) ──────────────────────
        energy_kwh = float("nan")
        if len(df.columns) >= 4:
            col_header = str(df.columns[3])
            m = re.search(r"Batch\s+\d+[:\s]+([\d.]+)\s*kWh", col_header, re.IGNORECASE)
            if not m:
                # Fallback: grab first numeric value followed by kWh in header
                m = re.search(r"([\d.]+)\s*kWh", col_header, re.IGNORECASE)
            if m:
                try:
                    energy_kwh = float(m.group(1))
                except ValueError:
                    pass

        # ── Start / End times → duration ─────────────────────────────────
        duration_min = float("nan")
        start_time = None
        end_time = None

        for _, row in df.iterrows():
            label = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
            val = row.iloc[1] if len(row) > 1 else None

            if label in ("start", "start time", "เริ่ม", "เริ่มต้น") and start_time is None:
                start_time = _parse_time_value(val)
            elif label in ("end", "end time", "สิ้นสุด", "สิ้น") and end_time is None:
                end_time = _parse_time_value(val)

        if start_time and end_time:
            # Compute duration, handling midnight rollover
            start_min = start_time.hour * 60 + start_time.minute + start_time.second / 60
            end_min   = end_time.hour   * 60 + end_time.minute   + end_time.second   / 60
            if end_min < start_min:
                end_min += 24 * 60   # crossed midnight
            duration_min = end_min - start_min

        # ── Weight from row labelled 'ingot' or 'pure ingot' ─────────────
        weight_kg = float("nan")
        for _, row in df.iterrows():
            label = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
            if "ingot" in label or "pure" in label:
                try:
                    w = float(pd.to_numeric(row.iloc[1], errors="coerce"))
                    if not pd.isna(w):
                        weight_kg = w
                except Exception:
                    pass
                break

        # ── Validate and store ────────────────────────────────────────────
        if not any([
            ENERGY_VALID_RANGE[0]   <= energy_kwh   <= ENERGY_VALID_RANGE[1],
            DURATION_VALID_RANGE[0] <= duration_min <= DURATION_VALID_RANGE[1],
            WEIGHT_VALID_RANGE[0]   <= weight_kg    <= WEIGHT_VALID_RANGE[1],
        ]):
            continue   # skip entirely uninterpretable rows

        records.append({
            "source_file": str(path.relative_to(PROJECT_ROOT)),
            "source_sheet": str(sheet),
            "power_kw":     power_kw,
            "energy_kwh":   energy_kwh if ENERGY_VALID_RANGE[0] <= energy_kwh <= ENERGY_VALID_RANGE[1] else float("nan"),
            "duration_min": duration_min if DURATION_VALID_RANGE[0] <= duration_min <= DURATION_VALID_RANGE[1] else float("nan"),
            "weight_kg":    weight_kg if WEIGHT_VALID_RANGE[0] <= weight_kg <= WEIGHT_VALID_RANGE[1] else float("nan"),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Inspect all files
# ─────────────────────────────────────────────────────────────────────────────

def run_inspect_phase(files: list[Path]) -> list[dict]:
    print("\n" + "=" * 60)
    print("PHASE 1 — Column Inspection")
    print("=" * 60)
    inspection_reports = []
    for path in files:
        if not path.exists():
            print(f"  [SKIP] Not found: {path.name}")
            continue
        print(f"\n  Inspecting: {path.name}")
        report = _inspect_file(path)
        if report["error"]:
            print(f"    ERROR: {report['error']}")
        else:
            for sheet, info in report["sheets"].items():
                if "error" in info:
                    print(f"    Sheet '{sheet}': ERROR — {info['error']}")
                else:
                    print(f"    Sheet '{sheet}': {len(info['columns'])} columns")
                    print(f"      {info['columns']}")
        inspection_reports.append(report)

    out_path = OUTPUT_DIR / "column_inspection_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inspection_reports, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path.relative_to(PROJECT_ROOT)}")
    return inspection_reports


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Extract batch records
# ─────────────────────────────────────────────────────────────────────────────

def run_extract_phase(files: list[Path]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PHASE 2 — Extract Batch Records")
    print("=" * 60)
    all_records: list[pd.DataFrame] = []
    filters_log: list[dict] = []

    for path in files:
        if not path.exists():
            continue

        # ── Targeted parser for the primary summary file ──────────────────
        if path.name == SUMMARY_FILE_NAME:
            print(f"\n  [TARGETED PARSE] {path.name}")
            summary_records = _extract_summary_file(path)
            if summary_records:
                df_summary = pd.DataFrame(summary_records)
                print(f"    Extracted {len(df_summary)} batch records")
                for col in ["energy_kwh", "duration_min", "weight_kg", "power_kw"]:
                    n_valid = df_summary[col].notna().sum() if col in df_summary.columns else 0
                    print(f"      {col}: {n_valid} valid values")
                all_records.append(df_summary)
                filters_log.append({
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "parser": "targeted_summary_parser",
                    "n_sheets_parsed": len(summary_records),
                    "n_after_filter": len(summary_records),
                    "filters_applied": [
                        f"energy_kwh: [{ENERGY_VALID_RANGE[0]}, {ENERGY_VALID_RANGE[1]}] kWh or NaN",
                        f"duration_min: [{DURATION_VALID_RANGE[0]}, {DURATION_VALID_RANGE[1]}] min or NaN",
                        f"weight_kg: [{WEIGHT_VALID_RANGE[0]}, {WEIGHT_VALID_RANGE[1]}] kg or NaN",
                    ],
                })
            else:
                print("    No parseable batch records found in summary file.")
            continue  # don't fall through to generic parser

        # ── Generic keyword-based parser for raw files ────────────────────
        try:
            xl = pd.ExcelFile(path)
        except Exception as e:
            print(f"  [ERROR] Cannot open {path.name}: {e}")
            continue

        for sheet in xl.sheet_names:
            df_raw = _extract_from_sheet(path, sheet)
            if df_raw is None:
                continue

            n_before = len(df_raw)
            df_clean = df_raw.dropna(how="all", subset=[c for c in ["energy_kwh", "duration_min", "weight_kg"] if c in df_raw.columns])

            # Apply validity filters
            filt_notes = []
            if "energy_kwh" in df_clean.columns:
                lo, hi = ENERGY_VALID_RANGE
                mask = df_clean["energy_kwh"].between(lo, hi) | df_clean["energy_kwh"].isna()
                n_dropped = (~mask).sum()
                if n_dropped:
                    filt_notes.append(f"dropped {n_dropped} rows with energy outside [{lo},{hi}] kWh")
                df_clean = df_clean[mask]
            if "duration_min" in df_clean.columns:
                lo, hi = DURATION_VALID_RANGE
                mask = df_clean["duration_min"].between(lo, hi) | df_clean["duration_min"].isna()
                n_dropped = (~mask).sum()
                if n_dropped:
                    filt_notes.append(f"dropped {n_dropped} rows with duration outside [{lo},{hi}] min")
                df_clean = df_clean[mask]
            if "weight_kg" in df_clean.columns:
                lo, hi = WEIGHT_VALID_RANGE
                mask = df_clean["weight_kg"].between(lo, hi) | df_clean["weight_kg"].isna()
                n_dropped = (~mask).sum()
                if n_dropped:
                    filt_notes.append(f"dropped {n_dropped} rows with weight outside [{lo},{hi}] kg")
                df_clean = df_clean[mask]

            n_after = len(df_clean)
            if n_after > 0:
                print(f"  {path.name} / '{sheet}': {n_after} usable rows (from {n_before})")
                if filt_notes:
                    for note in filt_notes:
                        print(f"    Filter: {note}")
                all_records.append(df_clean)
                filters_log.append({
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "sheet": str(sheet),
                    "n_before_filter": int(n_before),
                    "n_after_filter": int(n_after),
                    "filters_applied": filt_notes,
                })

    if not all_records:
        print("\n  [RESULT] No usable batch-level columns found across all files.")
        print("  Outputting documentation gap report.")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "per_batch_records.csv", index=False)
    print(f"\n  Saved: outputs/revision_phase1/historical_baseline/per_batch_records.csv")
    print(f"  Total usable batch records: {len(combined)}")

    # Save filter log alongside
    with open(OUTPUT_DIR / "filter_log.json", "w", encoding="utf-8") as f:
        json.dump(filters_log, f, ensure_ascii=False, indent=2)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Compute statistics
# ─────────────────────────────────────────────────────────────────────────────

def run_stats_phase(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("PHASE 3 — Compute Batch-Level Statistics")
    print("=" * 60)

    metrics = ["energy_kwh", "duration_min", "weight_kg", "power_kw"]
    stats: dict[str, dict] = {}

    if df.empty:
        print("  [SKIP] No data available — outputting gap report only.")
        gap_report = {
            "data_available": False,
            "reason": "No usable batch-level columns found in any data file.",
            "recommendation": (
                "Reviewer item A cannot be computed from available repository data. "
                "The column inspection report lists all available fields per file. "
                "To supply historical baseline statistics, provide raw batch records "
                "with at least one of: energy_kwh, duration_min, or weight_kg columns."
            ),
        }
        with open(OUTPUT_DIR / "plant_batch_stats_gap_report.json", "w", encoding="utf-8") as f:
            json.dump(gap_report, f, ensure_ascii=False, indent=2)
        return gap_report

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        col = df[metric].dropna()
        if len(col) == 0:
            continue
        row = {
            "metric": metric,
            "N": int(len(col)),
            "mean": float(col.mean()),
            "sd": float(col.std()),
            "median": float(col.median()),
            "min": float(col.min()),
            "max": float(col.max()),
            "p25": float(col.quantile(0.25)),
            "p75": float(col.quantile(0.75)),
        }
        stats[metric] = row
        rows.append(row)
        print(f"\n  {metric}:")
        print(f"    N={row['N']}  mean={row['mean']:.2f}  SD={row['sd']:.2f}  "
              f"median={row['median']:.2f}  min={row['min']:.2f}  max={row['max']:.2f}")

    if rows:
        with open(OUTPUT_DIR / "plant_batch_stats.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Saved: outputs/revision_phase1/historical_baseline/plant_batch_stats.csv")

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(files: list[Path], stats: dict) -> None:
    manifest = {
        "script": "extract_historical_baseline.py",
        "run_timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "data_sources": [
            {
                "file": str(p.relative_to(PROJECT_ROOT)),
                "exists": p.exists(),
                "sha256_prefix": _file_sha256(p) if p.exists() else None,
            }
            for p in files
        ],
        "validity_filters": {
            "energy_kwh": list(ENERGY_VALID_RANGE),
            "duration_min": list(DURATION_VALID_RANGE),
            "weight_kg": list(WEIGHT_VALID_RANGE),
        },
        "keyword_matchers": {
            "energy": ENERGY_KEYWORDS,
            "duration": DURATION_KEYWORDS,
            "weight": WEIGHT_KEYWORDS,
            "power": POWER_KEYWORDS,
        },
        "output_stats": stats,
    }
    out = OUTPUT_DIR / "extract_manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n  Manifest saved: {out.relative_to(PROJECT_ROOT)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("extract_historical_baseline.py — Reviewer Item A")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Data files   : {len(DATA_FILES)} candidates")

    # Phase 1: Inspect all files regardless of what is found
    inspection_reports = run_inspect_phase(DATA_FILES)

    # Phase 2: Extract usable batch records
    df = run_extract_phase(DATA_FILES)

    # Phase 3: Compute statistics (gracefully handles empty df)
    stats = run_stats_phase(df)

    # Always write manifest
    write_manifest(DATA_FILES, stats)

    print("\n" + "=" * 60)
    print("Done. Check outputs/revision_phase1/historical_baseline/")
    print("=" * 60)


if __name__ == "__main__":
    main()
