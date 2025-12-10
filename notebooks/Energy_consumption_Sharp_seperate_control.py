# %%
# Configuration & Setup
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Absolute data directory (adjust here if your data moves)
DATA_DIR = "/Users/kraiwitchpawadee/Personal/project/smart-factory-ds/data/raw/seperate_controls"

# Map of Control/Furnace combinations to their source files
FILE_GROUPS: Dict[str, List[str]] = {
    "Control A, Furnace A": [
        "MDB6 (INDUCTION)_20250602.xlsx",
        "MDB6 (INDUCTION)_20250606.xlsx",
        "MDB6 (INDUCTION)_20250625.xlsx",
    ],
    "Control B, Furnace B": [
        "MDB6 (INDUCTION)_20250702.xlsx",
    ],
    "Control A, Furnace B": [
        "MDB6 (INDUCTION)_20250730.xlsx",
        "MDB6 (INDUCTION)_20250926.xlsx",
    ],
}

# Batch detection and cleaning parameters
KW_THRESHOLD = 30  # kW threshold to consider furnace active
TIME_THRESHOLD_MINUTES = (
    3  # gap (minutes) to close a batch once kW falls below threshold
)
EXTEND_START_MINUTES = 5  # extend window before detected start
EXTEND_END_MINUTES = 2  # extend window after detected end

# Manual overrides
USE_MANUAL_IF_AVAILABLE = (
    True  # if manual ranges exist for a file, use them instead of auto-detection
)
EXTEND_MANUAL_WINDOWS = False  # if True, also extend manual windows by EXTEND_* minutes
MANUAL_BATCHES: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "Control A, Furnace A": {
        "MDB6 (INDUCTION)_20250625.xlsx": [
            (
                "2025-06-25 22:15:00",
                "2025-06-25 23:59:00",
            ),
        ],
    },
    "Control B, Furnace B": {
        "MDB6 (INDUCTION)_20250702.xlsx": [
            (
                "2025-07-02 08:57:00",
                "2025-07-02 11:07:00",
            ),
        ],
    },
}
# How to apply manual ranges when provided: "replace" (manual only) or "merge" (manual + auto, remove overlaps)
MANUAL_MODE = "merge"
# When merging, drop auto-detected batches that overlap any manual range
DEDUPLICATE_OVERLAPS = True

# Simple quality filters (tune as needed)
MIN_DURATION_MIN = 5  # exclude tiny operations (minutes)
MAX_DURATION_MIN = 180  # exclude excessively long operations (minutes)
MIN_ENERGY_KWH = 1  # exclude near-zero or negative energy

# Output & plotting controls
SHOW_FILE_LOGS = True  # show scrollable time-series logs per file
HIGHLIGHT_BATCHES_ON_LOGS = True  # draw shaded regions for batches on logs
SHOW_CUMULATIVE_KWH = False  # show cumulative kWh line on logs
SAVE_FIGS = False
FIGS_DIR = "/Users/kraiwitchpawadee/Personal/project/smart-factory-ds/results/usability_results"

# Ensure output directory exists if saving
if SAVE_FIGS:
    os.makedirs(FIGS_DIR, exist_ok=True)


# %%
# Utilities: loading and cleaning data


def load_and_prepare_dataframe(file_path: str) -> pd.DataFrame:
    """Load a log Excel file and return a cleaned DataFrame.

    Expected columns: 'Date Time', 'kW', 'kWh'.
    The file typically has multiple header rows; we read with header=4 and then drop first two rows.
    """
    df = pd.read_excel(file_path, header=4)

    # Drop the first two rows if present (dataset-specific artifact)
    rows_to_drop = [idx for idx in [0, 1] if idx in df.index]
    if rows_to_drop:
        df = df.drop(index=rows_to_drop)

    # Standardize column types
    df["Date Time"] = pd.to_datetime(df["Date Time"], errors="coerce", dayfirst=True)
    df["kW"] = pd.to_numeric(df["kW"], errors="coerce")
    df["kWh"] = pd.to_numeric(df["kWh"], errors="coerce")

    # Basic cleaning
    df = df.dropna(subset=["Date Time", "kW", "kWh"]).copy()
    df = (
        df.sort_values("Date Time")
        .drop_duplicates(subset=["Date Time"])
        .reset_index(drop=True)
    )
    return df


# %%
# Utilities: batch detection and metrics


def detect_batches_with_time_threshold(
    df: pd.DataFrame,
    kw_threshold: float = KW_THRESHOLD,
    time_threshold_minutes: int = TIME_THRESHOLD_MINUTES,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Identify batches using kW threshold and a time gap threshold to close batches.

    Returns a list of (start_time, end_time) based on the last time above threshold.
    """
    batches: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_batch = False
    start_time: pd.Timestamp | None = None
    last_above_threshold_time: pd.Timestamp | None = None

    for i in range(len(df)):
        current_time = df["Date Time"].iloc[i]
        current_kw = df["kW"].iloc[i]

        if current_kw > kw_threshold:
            if not in_batch:
                start_time = current_time
                in_batch = True
            last_above_threshold_time = current_time
        elif in_batch:
            if (
                last_above_threshold_time is not None
                and (current_time - last_above_threshold_time).total_seconds()
                > time_threshold_minutes * 60
            ):
                end_time = last_above_threshold_time
                batches.append((start_time, end_time))
                in_batch = False
                start_time = None
                last_above_threshold_time = None

    # If data ends while still in a batch, close at last_above_threshold_time
    if in_batch and last_above_threshold_time is not None and start_time is not None:
        batches.append((start_time, last_above_threshold_time))

    return batches


def compute_batch_metrics(
    df: pd.DataFrame,
    batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
    extend_start_minutes: int = EXTEND_START_MINUTES,
    extend_end_minutes: int = EXTEND_END_MINUTES,
) -> pd.DataFrame:
    """Compute kWh usage and duration (minutes) for each batch with window extension."""
    results = []
    for batch_index, (start_time, end_time) in enumerate(batches, start=1):
        extended_start = start_time - pd.Timedelta(minutes=extend_start_minutes)
        extended_end = end_time + pd.Timedelta(minutes=extend_end_minutes)

        df_window = df[
            (df["Date Time"] >= extended_start) & (df["Date Time"] <= extended_end)
        ]
        if df_window.empty:
            continue

        energy_kwh = float(df_window["kWh"].iloc[-1] - df_window["kWh"].iloc[0])
        duration_min = float((extended_end - extended_start).total_seconds() / 60.0)

        results.append(
            {
                "batch_in_file": batch_index,
                "start": extended_start,
                "end": extended_end,
                "duration_min": duration_min,
                "energy_kwh": energy_kwh,
            }
        )

    return pd.DataFrame(results)


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove batches outside basic quality bounds."""
    df = df[
        (df["duration_min"] >= MIN_DURATION_MIN)
        & (df["duration_min"] <= MAX_DURATION_MIN)
    ].copy()
    df = df[df["energy_kwh"] >= MIN_ENERGY_KWH].copy()
    return df.reset_index(drop=True)


def parse_manual_batches(
    group_name: str, file_name: str
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse manual ranges from MANUAL_BATCHES, return list of (start, end) timestamps."""
    group_dict = MANUAL_BATCHES.get(group_name)
    if not group_dict:
        return []
    ranges = group_dict.get(file_name, [])
    parsed: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for start_str, end_str in ranges:
        start_ts = pd.to_datetime(start_str)
        end_ts = pd.to_datetime(end_str)
        if pd.isna(start_ts) or pd.isna(end_ts) or end_ts <= start_ts:
            continue
        parsed.append((start_ts, end_ts))
    return parsed


# Helpers for merging manual and auto batches


def intervals_overlap(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def filter_auto_overlaps(
    auto_batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
    manual_batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not manual_batches:
        return auto_batches
    filtered: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for a_start, a_end in auto_batches:
        if any(
            intervals_overlap(a_start, a_end, m_start, m_end)
            for m_start, m_end in manual_batches
        ):
            continue
        filtered.append((a_start, a_end))
    return filtered


def combine_batches(
    auto_batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
    manual_batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
    deduplicate_overlaps: bool = True,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not manual_batches:
        return auto_batches
    auto_kept = (
        filter_auto_overlaps(auto_batches, manual_batches)
        if deduplicate_overlaps
        else auto_batches
    )
    combined = list(manual_batches) + list(auto_kept)
    combined.sort(key=lambda x: x[0])
    return combined


# %%
# Session: Visualization helpers (logs and scatter)


def plot_time_series_with_batches(
    df: pd.DataFrame,
    batches: List[Tuple[pd.Timestamp, pd.Timestamp]] | None,
    title: str,
) -> go.Figure:
    fig = go.Figure()

    # kW trace (left axis)
    fig.add_trace(go.Scatter(x=df["Date Time"], y=df["kW"], mode="lines", name="kW"))

    # kWh trace (right axis) - optional
    if SHOW_CUMULATIVE_KWH:
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=df["kWh"],
                mode="lines",
                name="kWh (cumulative)",
                yaxis="y2",
            )
        )

    shapes = []
    if HIGHLIGHT_BATCHES_ON_LOGS and batches:
        for start, end in batches:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=start,
                    x1=end,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(255, 165, 0, 0.15)",
                    line=dict(color="rgba(255, 165, 0, 0.5)", width=1),
                )
            )

    layout_kwargs = {
        "title": title,
        "xaxis": dict(rangeslider=dict(visible=True), type="date"),
        "yaxis": dict(title="kW"),
        "legend": dict(title="Signals"),
        "shapes": shapes,
        "template": "plotly_white",
    }
    if SHOW_CUMULATIVE_KWH:
        layout_kwargs["yaxis2"] = dict(
            title="kWh (cumulative)", overlaying="y", side="right"
        )

    fig.update_layout(**layout_kwargs)

    return fig


def plot_group_scatter(df_group: pd.DataFrame, group_name: str) -> go.Figure:
    title = f"Energy vs Melt Duration — {group_name}"
    fig = px.scatter(
        df_group,
        x="duration_min",
        y="energy_kwh",
        color="file",
        hover_name="batch_label",
        labels={
            "duration_min": "Time Duration (minutes)",
            "energy_kwh": "Energy Consumption (kWh)",
        },
        template="plotly_white",
        title=title,
    )

    fig.update_layout(legend=dict(title="Source File"))
    return fig


# %%
# Session: Build dataset for each Control/Furnace group


def build_group_metrics(group_name: str, file_list: List[str]) -> pd.DataFrame:
    group_rows: List[pd.DataFrame] = []

    for file_name in file_list:
        file_path = os.path.join(DATA_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"[WARN] Missing file: {file_path}")
            continue

        df = load_and_prepare_dataframe(file_path)

        # Auto-detect
        auto_batches = detect_batches_with_time_threshold(df)

        # Manual overrides if available
        manual_ranges = parse_manual_batches(group_name, file_name)
        use_manual = USE_MANUAL_IF_AVAILABLE and len(manual_ranges) > 0

        if use_manual and MANUAL_MODE == "replace":
            highlight_batches = manual_ranges
            extend_start = EXTEND_START_MINUTES if EXTEND_MANUAL_WINDOWS else 0
            extend_end = EXTEND_END_MINUTES if EXTEND_MANUAL_WINDOWS else 0
            metrics = compute_batch_metrics(
                df,
                manual_ranges,
                extend_start_minutes=extend_start,
                extend_end_minutes=extend_end,
            )
        elif use_manual and MANUAL_MODE == "merge":
            auto_kept = combine_batches(auto_batches, [], deduplicate_overlaps=False)
            # drop overlapping autos if requested
            if DEDUPLICATE_OVERLAPS:
                auto_kept = filter_auto_overlaps(auto_kept, manual_ranges)
            # compute metrics separately so manual can have different extension
            metrics_auto = compute_batch_metrics(df, auto_kept)
            extend_start = EXTEND_START_MINUTES if EXTEND_MANUAL_WINDOWS else 0
            extend_end = EXTEND_END_MINUTES if EXTEND_MANUAL_WINDOWS else 0
            metrics_manual = compute_batch_metrics(
                df,
                manual_ranges,
                extend_start_minutes=extend_start,
                extend_end_minutes=extend_end,
            )
            metrics = pd.concat([metrics_auto, metrics_manual], ignore_index=True)
            highlight_batches = combine_batches(
                auto_kept, manual_ranges, DEDUPLICATE_OVERLAPS
            )
        else:
            metrics = compute_batch_metrics(df, auto_batches)
            highlight_batches = auto_batches

        metrics = apply_quality_filters(metrics)

        # Optional: show scrollable time-series with highlighted batches
        if SHOW_FILE_LOGS:
            title = f"{group_name} — {file_name}: kW & kWh Log"
            fig_log = plot_time_series_with_batches(df, highlight_batches, title)
            fig_log.show()

        if metrics.empty:
            continue

        metrics["group"] = group_name
        metrics["file"] = file_name
        parts = [s.strip() for s in group_name.split(",")]
        control_name = next((p for p in parts if p.lower().startswith("control")), None)
        furnace_name = next((p for p in parts if p.lower().startswith("furnace")), None)
        metrics["control"] = control_name if control_name else "Unknown Control"
        metrics["furnace"] = furnace_name if furnace_name else "Unknown Furnace"
        metrics["batch_label"] = (
            metrics["file"] + " | B" + metrics["batch_in_file"].astype(str)
        )
        group_rows.append(metrics)

    if not group_rows:
        return pd.DataFrame(
            columns=[
                "group",
                "file",
                "batch_in_file",
                "batch_label",
                "start",
                "end",
                "duration_min",
                "energy_kwh",
            ]
        )

    return pd.concat(group_rows, ignore_index=True)


# %%
# Session: Run end-to-end for all groups and visualize
if __name__ == "__main__":
    all_group_metrics: Dict[str, pd.DataFrame] = {}

    for group_name, files in FILE_GROUPS.items():
        print(f"\n=== Processing {group_name} ===")
        metrics_df = build_group_metrics(group_name, files)
        all_group_metrics[group_name] = metrics_df

        if metrics_df.empty:
            print("No valid batches found after filtering.")
            continue

        print(
            f"Batches: {len(metrics_df)} | Duration(min) mean={metrics_df['duration_min'].mean():.1f}, "
            f"Energy(kWh) mean={metrics_df['energy_kwh'].mean():.2f}"
        )

        fig = plot_group_scatter(metrics_df, group_name)
        fig.show()

        if SAVE_FIGS:
            safe_name = group_name.replace(",", "").replace(" ", "_")
            out_path = os.path.join(
                FIGS_DIR, f"scatter_energy_vs_duration_{safe_name}.html"
            )
            fig.write_html(out_path)
            print(f"Saved figure to: {out_path}")

    # Optional: combined view across all groups
    combined = (
        pd.concat(
            [df for df in all_group_metrics.values() if not df.empty], ignore_index=True
        )
        if any(not df.empty for df in all_group_metrics.values())
        else pd.DataFrame()
    )

    if not combined.empty:
        fig_all = px.scatter(
            combined,
            x="duration_min",
            y="energy_kwh",
            color="group",
            hover_name="batch_label",
            labels={
                "duration_min": "Time Duration (minutes)",
                "energy_kwh": "Energy Consumption (kWh)",
            },
            template="plotly_white",
            title="Energy vs Melt Duration — All Groups",
        )
        fig_all.update_layout(legend=dict(title="Group"))
        fig_all.show()

        # Additional combined plot: color by Control (A=Blue, B=Red), symbol by Furnace (A=Triangle, B=Circle)
        fig_all_fc = px.scatter(
            combined,
            x="duration_min",
            y="energy_kwh",
            color="control",
            symbol="furnace",
            category_orders={
                "control": ["Control A", "Control B"],
                "furnace": ["Furnace A", "Furnace B"],
            },
            color_discrete_map={
                "Control A": "blue",
                "Control B": "red",
            },
            symbol_sequence=["triangle-up", "circle"],
            hover_name="batch_label",
            labels={
                "duration_min": "Time Duration (minutes)",
                "energy_kwh": "Energy Consumption (kWh)",
            },
            template="plotly_white",
            title="Energy vs Melt Duration — All Groups",
        )
        fig_all_fc.update_layout(
            legend=dict(title="Legend (Color: Control, Shape: Furnace)")
        )
        fig_all_fc.update_traces(
            marker=dict(size=8, line=dict(width=0.5, color="black"))
        )

        # Mean points by Furnace (shape-based)
        mean_by_furnace = (
            combined.groupby("furnace")[["duration_min", "energy_kwh"]]
            .mean()
            .reset_index()
        )
        for _, row in mean_by_furnace.iterrows():
            furnace_name = row["furnace"]
            symbol_map = {"Furnace A": "triangle-up", "Furnace B": "circle"}
            furnace_symbol = symbol_map.get(furnace_name, "circle")
            fig_all_fc.add_trace(
                go.Scatter(
                    x=[row["duration_min"]],
                    y=[row["energy_kwh"]],
                    mode="markers",
                    name=f"Mean {furnace_name}",
                    marker=dict(
                        symbol=furnace_symbol,
                        size=16,
                        color="black",
                        line=dict(width=2, color="white"),
                    ),
                    hovertemplate=(
                        f"Mean {furnace_name}<br>"
                        + "Time Duration=%{x:.1f} min"
                        + "<br>Energy=%{y:.2f} kWh<extra></extra>"
                    ),
                )
            )

        # Mean points by Control (color-based)
        mean_by_control = (
            combined.groupby("control")[["duration_min", "energy_kwh"]]
            .mean()
            .reset_index()
        )
        for _, row in mean_by_control.iterrows():
            control_name = row["control"]
            color_map = {"Control A": "blue", "Control B": "red"}
            control_color = color_map.get(control_name, "gray")
            fig_all_fc.add_trace(
                go.Scatter(
                    x=[row["duration_min"]],
                    y=[row["energy_kwh"]],
                    mode="markers",
                    name=f"Mean {control_name}",
                    marker=dict(
                        symbol="diamond",
                        size=12,
                        color=control_color,
                        line=dict(width=2, color="black"),
                    ),
                    hovertemplate=(
                        f"Mean {control_name}<br>"
                        + "Time Duration=%{x:.1f} min"
                        + "<br>Energy=%{y:.2f} kWh<extra></extra>"
                    ),
                )
            )

        fig_all_fc.show()

        if SAVE_FIGS:
            out_path_fc = os.path.join(
                FIGS_DIR, "scatter_energy_vs_duration_furnace_color_control_symbol.html"
            )
            fig_all_fc.write_html(out_path_fc)
            print(f"Saved figure to: {out_path_fc}")
