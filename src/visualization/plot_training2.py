import matplotlib.pyplot as plt
import numpy as np


def _moving_average(x, window_size):
    x = np.asarray(x, dtype=float).ravel()
    w = int(max(1, window_size))
    if len(x) < w:
        return None
    # ใช้ valid mode เพื่อไม่ให้กราฟมีขอบแปลกๆ
    return np.convolve(x, np.ones(w, dtype=float) / float(w), mode="valid")


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
        moving_avg = _moving_average(episode_rewards, window_size)
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

    # จัดการ `episode_energies`
    if isinstance(episode_energies, list) and len(episode_energies) > 0:
        episode_energies = np.hstack(episode_energies).astype(np.float32)
    else:
        episode_energies = np.array([0], dtype=np.float32)

    axes[2].plot(episode_energies, color="r", label="Energy Consumption")
    axes[2].set_title("Energy Consumption over Time")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Energy (kWh)")
    axes[2].grid(True)

    # Moving average of energy
    moving_avg_energy = _moving_average(episode_energies, window_size)
    if moving_avg_energy is not None:
        axes[2].plot(
            range(window_size - 1, len(episode_energies)),
            moving_avg_energy,
            "k--",
            linewidth=1.6,
            label=f"Moving Average ({window_size})",
        )
    axes[2].legend()

    # Plot training loss if available
    if episode_losses is not None and len(episode_losses) > 0:
        axes[3].plot(episode_losses, label="Training Loss", color="m")
        axes[3].set_title("Training Loss over Time")
        axes[3].set_xlabel("Episode")
        axes[3].set_ylabel("Loss")
        axes[3].grid(True)
        # Moving average of loss
        moving_avg_loss = _moving_average(episode_losses, window_size)
        if moving_avg_loss is not None:
            axes[3].plot(
                range(window_size - 1, len(episode_losses)),
                moving_avg_loss,
                "k--",
                linewidth=1.6,
                label=f"Moving Average ({window_size})",
            )
        axes[3].legend()

    plt.tight_layout()
    plt.show()


def plot_episode_details(
    temperatures, powers, energies, rewards, targetTemp, expert_powers=None
):
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    # Temperature plot
    axes[0].plot(temperatures, "b-", label="Agent Temperature")
    axes[0].axhline(
        y=targetTemp, color="r", linestyle="--", label="Target Temperature (950°C)"
    )
    axes[0].set_title("Temperature vs Time")
    axes[0].set_xlabel("Time (minutes)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].grid(True)
    axes[0].legend()

    # Power plot (เพิ่มเส้น Expert Profile)
    axes[1].plot(powers, "g-", linewidth=2.0, label="Agent Power Decision")
    if expert_powers is not None:
        # พล็อตกราฟเส้นประสีแดง เพื่อเทียบกับสิ่งที่ Agent ตัดสินใจ
        axes[1].plot(
            expert_powers,
            "r--",
            linewidth=2.0,
            alpha=0.7,
            label="Expert Profile (Target)",
        )

    axes[1].set_title("Power vs Time (Agent vs Expert)")
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
    axes[3].plot(rewards, "m-", label="Step Reward")
    axes[3].set_title("Rewards vs Time")
    axes[3].set_xlabel("Time (minutes)")
    axes[3].set_ylabel("Reward")
    axes[3].grid(True)
    axes[3].legend()

    plt.tight_layout()
    plt.show()


def replay_episode(env, agent, debug_q_values: bool = False, debug_steps: int = 5):
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
    expert_powers = []  # เก็บข้อมูลของ Expert เพื่อนำไป plot

    targetTemp = env.target_temp

    print("\nStarting Episode Replay:")
    print("Initial state:", state)

    while not done:
        # Get and record action
        if (
            debug_q_values
            and hasattr(agent, "get_q_values")
            and len(actions) < int(debug_steps)
        ):
            try:
                q = agent.get_q_values(state)
                top = int(np.argmax(q)) if len(q) else -1
                print(
                    f"Q(step={len(actions)}): {np.round(q, 3)} | argmax={top} ({env.action_space.get(top, 0):.0f} kW)"
                )
            except Exception:
                pass

        action = agent.select_action(state, explore=False)
        actions.append(env.action_space[action])

        # เก็บค่า Expert Profile ณ นาทีปัจจุบันไว้สำหรับ Plot เทียบ
        current_minute = state[2] / 60.0
        expert_pwr = env._expert_power_profile(current_minute)
        expert_powers.append(expert_pwr)

        # Take step and record data
        next_state, reward, done = env.step(action)

        temperatures.append(next_state[0])
        powers.append(next_state[3])
        energies.append(next_state[5])
        rewards.append(reward)
        times.append(next_state[2])

        # Print step details
        print(f"\nTime: {next_state[2]:.0f}s")
        print(f"Action taken: {env.action_space[action]:.0f} kW")  # ปรับการแสดงผลให้มี kW
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
        print(f"{act:.0f} kW: {percentage:.1f}%")  # ปรับการแสดงผล Distribution

    # Plot episode details ส่ง expert_powers เข้าไปด้วย
    plot_episode_details(
        temperatures, powers, energies, rewards, targetTemp, expert_powers
    )
