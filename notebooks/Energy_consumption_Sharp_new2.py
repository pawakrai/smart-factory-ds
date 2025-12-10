import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import timedelta

df = pd.read_excel("../data/raw/MDB6 (INDUCTION)_20251110.xlsx", header=4)  # new
# df = pd.read_excel("../data/raw/MDB6 (INDUCTION)_20251111.xlsx", header=4)  # new

df_cleaned = df.drop(index=[0, 1])
df_cleaned.head()

df_cleaned["Date Time"] = pd.to_datetime(
    df_cleaned["Date Time"], format="%d/%m/%Y %H:%M:%S"
)

fig = px.line(df_cleaned, x="Date Time", y="kW", title="Hourly kW Usage Over Time")
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(
    xaxis_title="Date Time",
    yaxis_title="kW Usage",
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    ),
)

fig.show()

df_day = df_cleaned[
    df_cleaned["Date Time"].dt.date == pd.to_datetime("2025-11-10").date()
]

fig = px.line(df_day, x="Date Time", y="kW", title="kW Usage on 2025-11-10")
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(
    xaxis_title="Date Time",
    yaxis_title="kW",
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    ),
)


fig.show()

# Calculate kWh/ton
batch_1_start = "2025-11-10 12:27:00"
batch_1_end = "2025-11-10 13:57:00"

df_batch_1 = df_cleaned[
    (df_cleaned["Date Time"] >= batch_1_start)
    & (df_cleaned["Date Time"] <= batch_1_end)
]

# Calculate energy consumption for each batch by subtracting initial kWh from final kWh
energy_batch_1 = df_batch_1["kWh"].iloc[-1] - df_batch_1["kWh"].iloc[0]

print(f"Energy consumption for Batch 1: {energy_batch_1} kWh")

# Calculate energy consumtion per ton

ton_per_batch = 0.5  # 500 kg per batch

print(f"Energy consumption per batch: {energy_batch_1/ton_per_batch} kWh/ton")


# Step 1: Function to identify batches with time threshold
def identify_batches_with_time_threshold(df, kw_threshold=20, time_threshold_minutes=3):
    batches = []
    in_batch = False
    start_time = None
    last_above_threshold_time = None

    for i in range(len(df)):
        current_time = df["Date Time"].iloc[i]
        current_kw = df["kW"].iloc[i]

        if current_kw > kw_threshold:
            if not in_batch:
                start_time = current_time  # Start new batch
                in_batch = True
            last_above_threshold_time = current_time  # Update last time above threshold

        elif in_batch:
            # Check how long it's been since kW was above threshold
            if last_above_threshold_time and (
                current_time - last_above_threshold_time
            ) > timedelta(minutes=time_threshold_minutes):
                end_time = last_above_threshold_time  # End batch at last valid time above threshold
                batches.append((start_time, end_time))
                in_batch = False  # Reset for next batch

    return batches


# Step 2: Identify batches using both kW and time thresholds
batches = identify_batches_with_time_threshold(
    df_cleaned, kw_threshold=30, time_threshold_minutes=3
)


# Step 3: Calculate energy consumption for each batch
def calculate_energy(df, batches):
    energy_consumption = []
    for start, end in batches:
        start_index = df.index[df["Date Time"] == start][0]
        end_index = df.index[df["Date Time"] == end][0]
        energy_used = df["kWh"].iloc[end_index] - df["kWh"].iloc[start_index]
        energy_consumption.append(energy_used)
    return energy_consumption


energy_consumption = calculate_energy(df_cleaned, batches)


# Step 4: Calculate duration of each batch
def calculate_durations(batches):
    durations = []
    for start, end in batches:
        duration = (end - start).total_seconds() / 3600  # Convert to hours
        durations.append(duration)
    return durations


batch_durations = calculate_durations(batches)

# Step 5: Visualize results

# Plot energy consumption per batch
batch_names = [f"Batch {i+1}" for i in range(len(batches))]

plt.figure(figsize=(10, 6))
plt.bar(batch_names, energy_consumption, color="skyblue")
plt.xlabel("Batch")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Energy Consumption per Batch")
plt.show()

# Plot batch durations
plt.figure(figsize=(10, 6))
plt.bar(batch_names, batch_durations, color="salmon")
plt.xlabel("Batch")
plt.ylabel("Duration (hours)")
plt.title("Duration of Each Batch")
plt.show()


# # fix start time batch 95 to '2024-10-25 16:10:00'
# batches[94][0] = batches[94][0] - timedelta(minutes=30)
# batches[94] = (batches[94][0] - timedelta(minutes=30), batches[94][1])
# batches[94]


import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go

# Assuming df_cleaned is your DataFrame with 'Date Time', 'kW', and 'kWh' columns
# And batches is a list of tuples with (start_time, end_time) for each batch

# Step 1: Extend each batch by 5 minutes before and 2 minutes after
extended_batches = [
    (start - timedelta(minutes=5), end + timedelta(minutes=2)) for start, end in batches
]

# Step 2: Create an empty DataFrame to store the melt profile for each batch
df_melt_profile = pd.DataFrame(
    columns=["batch", "start_time", "end_time", "kWh_usage", "time_duration"]
)

# Step 3: Iterate over each batch and calculate kWh usage and time duration
for i, (start_time, end_time) in enumerate(extended_batches):
    # Filter data for the current batch
    df_batch = df_cleaned[
        (df_cleaned["Date Time"] >= start_time) & (df_cleaned["Date Time"] <= end_time)
    ]

    if not df_batch.empty:
        # Calculate kWh usage: last kWh - first kWh
        kWh_start = df_batch["kWh"].iloc[0]
        kWh_end = df_batch["kWh"].iloc[-1]
        kWh_usage = kWh_end - kWh_start

        # Calculate time duration in minutes
        time_duration = (
            end_time - start_time
        ).total_seconds() / 60  # Convert to minutes

        # Create a new row as a DataFrame and concatenate it to df_melt_profile using pd.concat()
        new_row = pd.DataFrame(
            {
                "batch": [i + 1],
                "start_time": [start_time],
                "end_time": [end_time],
                "kWh_usage": [kWh_usage],
                "time_duration": [time_duration],
            }
        )

        df_melt_profile = pd.concat([df_melt_profile, new_row], ignore_index=True)

        # Print results for each batch
        print(f"Batch {i+1}:")
        print(f"Start Time: {start_time}")
        print(f"End Time: {end_time}")
        print(f"kWh Usage: {kWh_usage:.2f} kWh")
        print(f"Time Duration: {time_duration:.2f} minutes")
        print("-" * 40)
    else:
        print(f"Batch {i+1}: No data available after extending by 5 minutes.")

    # Step 4: Create a figure with two y-axes (kW and cumulative kWh)
    fig = go.Figure()

    # Plot kW usage on the first y-axis
    fig.add_trace(
        go.Scatter(
            x=df_batch["Date Time"], y=df_batch["kW"], mode="lines", name="kW Usage"
        )
    )

    # Plot cumulative kWh usage on the second y-axis
    fig.add_trace(
        go.Scatter(
            x=df_batch["Date Time"],
            y=df_batch["kWh"],
            mode="lines",
            name="Cumulative kWh Usage",
            yaxis="y2",
        )
    )

    # Step 5: Update layout to include two y-axes
    fig.update_layout(
        title=f"Batch {i+1}: kW and Cumulative kWh Usage",
        xaxis_title="Time",
        yaxis_title="kW Usage",
        yaxis2=dict(title="Cumulative kWh Usage", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(rangeslider=dict(visible=True)),
    )

    # Step 6: Show the plot for this batch
    fig.show()


# Step 7: Filter out B21 and B94 from analysis
print(f"\nOriginal number of batches: {len(df_melt_profile)}")
df_melt_profile_filtered = df_melt_profile[
    ~df_melt_profile["batch"].isin([2, 3, 75, 76, 77, 78, 85, 86, 89])
].copy()
print(
    f"Number of batches after filtering (excluding small batch): {len(df_melt_profile_filtered)}"
)
print(f"Excluded batches: small batch\n")

# Step 8: Create scatter plot showing relationship between time duration and energy consumption for filtered batches
plt.figure(figsize=(12, 8))
plt.scatter(
    df_melt_profile_filtered["time_duration"],
    df_melt_profile_filtered["kWh_usage"],
    c=range(len(df_melt_profile_filtered)),
    cmap="viridis",
    s=100,
    alpha=0.7,
)

# Add batch numbers as labels on each point
for i, (duration, energy, batch_num) in enumerate(
    zip(
        df_melt_profile_filtered["time_duration"],
        df_melt_profile_filtered["kWh_usage"],
        df_melt_profile_filtered["batch"],
    )
):
    plt.annotate(
        f"B{int(batch_num)}",
        (duration, energy),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        alpha=0.8,
    )

plt.xlabel("Time Duration (minutes)")
plt.ylabel("Energy Consumption (kWh)")
plt.title("Energy Consumption vs Time Duration")
plt.grid(True, alpha=0.3)
plt.colorbar(label="Batch Sequence")

# Add trend line
z = np.polyfit(
    df_melt_profile_filtered["time_duration"], df_melt_profile_filtered["kWh_usage"], 1
)
p = np.poly1d(z)
plt.plot(
    df_melt_profile_filtered["time_duration"],
    p(df_melt_profile_filtered["time_duration"]),
    "r--",
    alpha=0.8,
    label=f'Trend Line (R² = {np.corrcoef(df_melt_profile_filtered["time_duration"], df_melt_profile_filtered["kWh_usage"])[0,1]**2:.3f})',
)
plt.legend()

plt.tight_layout()
plt.show()

# Alternative interactive scatter plot using Plotly (filtered data)
fig_scatter = go.Figure()

fig_scatter.add_trace(
    go.Scatter(
        x=df_melt_profile_filtered["time_duration"],
        y=df_melt_profile_filtered["kWh_usage"],
        mode="markers+text",
        text=[f"B {int(batch)}" for batch in df_melt_profile_filtered["batch"]],
        textposition="top center",
        marker=dict(
            size=10,
            color=df_melt_profile_filtered["batch"],
            colorscale="viridis",
            showscale=True,
            colorbar=dict(title="Batch Number"),
        ),
        name="Batches",
        hovertemplate="<b>Batch %{text}</b><br>"
        + "Time Duration: %{x:.1f} minutes<br>"
        + "Energy Consumption: %{y:.2f} kWh<br>"
        + "<extra></extra>",
    )
)

# Add trend line
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_melt_profile_filtered["time_duration"], df_melt_profile_filtered["kWh_usage"]
)
line_x = [
    df_melt_profile_filtered["time_duration"].min(),
    df_melt_profile_filtered["time_duration"].max(),
]
line_y = [slope * x + intercept for x in line_x]

fig_scatter.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name=f"Trend Line (R² = {r_value**2:.3f})",
        line=dict(color="red", dash="dash"),
    )
)

fig_scatter.update_layout(
    title="Energy Consumption vs Time Duration (Interactive, Excluding small batch)",
    xaxis_title="Time Duration (minutes)",
    yaxis_title="Energy Consumption (kWh)",
    hovermode="closest",
    showlegend=True,
)

fig_scatter.show()

# Print summary statistics (filtered data)
print("\n" + "=" * 60)
print("BATCH ANALYSIS SUMMARY (Excluding small batch)")
print("=" * 60)
print(f"Total number of batches (filtered): {len(df_melt_profile_filtered)}")
print(
    f"Average time duration: {df_melt_profile_filtered['time_duration'].mean():.1f} minutes"
)
print(
    f"Average energy consumption: {df_melt_profile_filtered['kWh_usage'].mean():.2f} kWh"
)
print(
    f"Energy efficiency range: {df_melt_profile_filtered['kWh_usage'].min():.2f} - {df_melt_profile_filtered['kWh_usage'].max():.2f} kWh"
)
print(
    f"Time duration range: {df_melt_profile_filtered['time_duration'].min():.1f} - {df_melt_profile_filtered['time_duration'].max():.1f} minutes"
)
print(
    f"Correlation coefficient (time vs energy): {np.corrcoef(df_melt_profile_filtered['time_duration'], df_melt_profile_filtered['kWh_usage'])[0,1]:.3f}"
)
print("=" * 60)
