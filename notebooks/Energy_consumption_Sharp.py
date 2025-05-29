import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

df = pd.read_excel("../data/raw/MDB6 (INDUCTION)_20241028_111546.xlsx", header=4)

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
    df_cleaned["Date Time"].dt.date == pd.to_datetime("2024-10-25").date()
]

fig = px.line(df_day, x="Date Time", y="kW", title="kW Usage on 2024-10-25")
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
batch_1_start = "2024-10-20 01:00:00"
batch_1_end = "2024-10-20 03:00:00"

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
batches[94] = (batches[94][0] - timedelta(minutes=30), batches[94][1])
batches[94]


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
