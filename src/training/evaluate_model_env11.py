"""
Evaluate a trained DQN model (dqn_final_model_env11.pth) over 50 episodes.

Metrics collected per episode:
  - duration_min : number of steps taken (each step = dt_sec=60s → 1 minute)
  - total_energy  : cumulative energy consumption (kWh) at episode end

Initial conditions are slightly randomised each episode (via idle_time_min)
so that the 50 evaluation runs differ from one another and produce a
meaningful Standard Deviation.
"""

import sys
import os

# ------------------------------------------------------------------
# Make sure the project root is on sys.path regardless of where this
# script is executed from.
# ------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from src.environment.aluminum_melting_env_11 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_PATH = os.path.join(project_root, "models", "dqn_final_model_env11.pth")
NUM_EPISODES = 50
STATE_DIM = 8  # [temp, weight, time, power, status, energy_kwh, scrap_added, wall_temp]
ACTION_DIM = 10  # 0 kW → 450 kW in 10 equal steps
TARGET_TEMP_C = 950.0

# Range for randomising idle_time_min at the start of each episode.
# A longer idle time → lower initial wall/metal temperature, introducing
# realistic variability across the 50 evaluation runs.
IDLE_TIME_MIN_RANGE = (0, 30)  # minutes


# ---------------------------------------------------------------
# 1. Load the trained model
# ---------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
state_dict = torch.load(MODEL_PATH, map_location=device)
agent.model.load_state_dict(state_dict)
agent.model.to(device)
agent.model.eval()  # evaluation mode — BatchNorm / Dropout behave differently
agent.epsilon = 0.0  # pure greedy — no random exploration

print(f"Model loaded from : {MODEL_PATH}")
print(f"Running on device : {device}")
print(f"Episodes          : {NUM_EPISODES}")
print("-" * 55)

# ---------------------------------------------------------------
# 2. Evaluation loop
# ---------------------------------------------------------------
durations_min = []  # steps × dt_sec / 60  →  duration in minutes
total_energies = []  # kWh at end of episode

rng = np.random.default_rng()  # for sampling idle_time_min per episode

for ep in range(NUM_EPISODES):
    # --- Create a fresh environment with slightly randomised idle time ---
    idle_time = float(rng.uniform(*IDLE_TIME_MIN_RANGE))
    env = AluminumMeltingEnvironment(
        target_temp_c=TARGET_TEMP_C,
        start_mode="hot",
        idle_time_min=idle_time,
        # No fixed seed → scrap amounts are randomised by NumPy's global RNG
    )

    state = env.reset()
    done = False
    steps = 0

    while not done:
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = int(torch.argmax(agent.model(s_tensor)).item())

        state, _reward, done = env.step(action)
        steps += 1

    # dt_sec defaults to 60 s → steps == duration in minutes
    duration_min = steps * env.dt / 60.0
    total_energy = float(env.state["energy_consumption"])

    durations_min.append(duration_min)
    total_energies.append(total_energy)

    print(
        f"Episode {ep + 1:>3d}/{NUM_EPISODES} | "
        f"Duration: {duration_min:6.1f} min | "
        f"Energy: {total_energy:7.2f} kWh | "
        f"Final Temp: {env.state['temperature']:.1f} °C | "
        f"idle_time_min: {idle_time:.1f}"
    )

# ---------------------------------------------------------------
# 3. Summary statistics
# ---------------------------------------------------------------
durations_arr = np.array(durations_min)
energies_arr = np.array(total_energies)

print("\n" + "=" * 55)
print("EVALUATION SUMMARY")
print("=" * 55)
print(f"{'Metric':<25} {'Mean':>10} {'SD':>10}")
print("-" * 55)
print(
    f"{'Duration (min)':<25} "
    f"{np.mean(durations_arr):>10.2f} "
    f"{np.std(durations_arr):>10.2f}"
)
print(
    f"{'Total Energy (kWh)':<25} "
    f"{np.mean(energies_arr):>10.2f} "
    f"{np.std(energies_arr):>10.2f}"
)
print("=" * 55)
