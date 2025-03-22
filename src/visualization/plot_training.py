import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(
    episode_rewards, episode_lengths, episode_energies, episode_losses=None
):
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    # Plot rewards
    axes[0].plot(episode_rewards, label="Total Reward", color="b")
    axes[0].set_title("Episode Rewards over Time")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True)

    # Plot moving average of rewards
    window_size = 100
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0].plot(
            range(window_size - 1, len(episode_rewards)),
            moving_avg,
            "r--",
            label="Moving Average",
        )
    axes[0].legend()

    # Plot episode lengths
    axes[1].plot(episode_lengths, label="Episode Length", color="g")
    axes[1].set_title("Episode Lengths over Time")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Number of Steps")
    axes[1].grid(True)
    axes[1].legend()

    # ปรับปรุงการจัดการ `episode_energies`
    if isinstance(episode_energies, list) and len(episode_energies) > 0:
        # แบนข้อมูลในกรณีที่เป็น list of lists
        episode_energies = np.hstack(episode_energies).astype(np.float32)
    else:
        episode_energies = np.array([0], dtype=np.float32)

    axes[2].plot(episode_energies, color="r", label="Energy Consumption")
    axes[2].set_title("Energy Consumption over Time")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Energy (kWh)")
    axes[2].grid(True)
    axes[2].legend()

    # Plot training loss if available
    if episode_losses is not None and len(episode_losses) > 0:
        axes[3].plot(episode_losses, label="Training Loss", color="m")
        axes[3].set_title("Training Loss over Time")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Loss")
        axes[3].grid(True)
        axes[3].legend()

    plt.tight_layout()
    plt.show()


def plot_episode_details(temperatures, powers, energies, rewards):
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Temperature plot
    axes[0].plot(temperatures, "b-", label="Temperature")
    axes[0].axhline(y=850, color="r", linestyle="--", label="Target Temperature")
    axes[0].set_title("Temperature vs Time")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True)
    axes[0].legend()

    # Power plot
    axes[1].plot(powers, "g-", label="Power")
    axes[1].set_title("Power vs Time")
    axes[1].set_xlabel("Time (minutes)")
    axes[1].set_ylabel("Power (kW)")
    axes[1].grid(True)
    axes[1].legend()

    # Energy consumption plot
    axes[2].plot(energies, "r-", label="Energy Consumption")
    axes[2].set_title("Energy Consumption vs Time")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Energy (kWh)")
    axes[2].grid(True)
    axes[2].legend()

    # Rewards plot
    axes[3].plot(rewards, "m-", label="Reward")
    axes[3].set_title("Rewards vs Time")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_ylabel("Reward")
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def replay_episode(env, agent):
    state = env.reset()
    done = False
    total_reward = 0

    # Lists to store episode data
    temperatures = []
    powers = []
    energies = []
    rewards = []
    actions = []
    times = []

    print("\nStarting Episode Replay:")
    print("Initial state:", state)

    while not done:
        # Get and record action
        action = agent.select_action(state)
        actions.append(env.action_space[action])

        # Take step and record data
        next_state, reward, done = env.step(action)

        # Store data; energy_consumption อยู่ที่ index 5 (state dimension ใหม่)
        temperatures.append(next_state[0])
        powers.append(next_state[3])
        energies.append(next_state[5])
        rewards.append(reward)
        times.append(next_state[2])

        # Print step details
        print(f"\nTime: {next_state[2]:.0f}s")
        print(f"Action taken: {env.action_space[action]}")
        print(f"Temperature: {next_state[0]:.1f}°C")
        print(f"Power: {next_state[3]:.0f} kW")
        print(f"Energy Consumed: {next_state[5]:.2f} kWh")
        print(f"Reward: {reward:.2f}")

        state = next_state
        total_reward += reward

    # Print episode summary
    print("\nEpisode Summary:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final temperature: {temperatures[-1]:.1f}°C")
    print(f"Average power used: {np.mean(powers):.1f} kW")
    print(f"Total energy consumption: {energies[-1]:.2f} kWh")
    print(f"Total time: {times[-1]:.0f} s")

    # Calculate action distribution
    action_counts = {}
    for act in actions:
        action_counts[act] = action_counts.get(act, 0) + 1

    print("\nAction Distribution:")
    for act, count in action_counts.items():
        percentage = (count / len(actions)) * 100
        print(f"{act}: {percentage:.1f}%")

    # Plot episode details
    plot_episode_details(temperatures, powers, energies, rewards)
