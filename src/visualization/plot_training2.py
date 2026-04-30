import os

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(x, window_size):
    x = np.asarray(x, dtype=float).ravel()
    w = int(max(1, window_size))
    if len(x) < w:
        return None
    # ใช้ valid mode เพื่อไม่ให้กราฟมีขอบแปลกๆ
    return np.convolve(x, np.ones(w, dtype=float) / float(w), mode="valid")


def _style_publication_axes(ax):
    """Apply large, readable fonts and clean styling for publication."""
    ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight="bold", labelpad=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=16, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=13, length=5, width=1.0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=7))
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def _save_and_show(fig, save_dir, filename, show):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        png_path = os.path.join(save_dir, f"{filename}.png")
        pdf_path = os.path.join(save_dir, f"{filename}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {png_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_results(
    episode_rewards,
    episode_lengths,
    episode_energies,
    episode_losses=None,
    save_dir="figures/training",
    show=True,
):
    """Render four publication-ready training figures (one per metric).

    Each figure is saved as a 300 dpi PNG and a vector PDF inside ``save_dir``.
    Pass ``save_dir=None`` to skip saving (display only).
    """
    window_size = 100

    # 1. Episode Rewards
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episode_rewards, color="#1f4e79", linewidth=1.2, label="Total Reward")
    if len(episode_rewards) >= window_size:
        moving_avg = _moving_average(episode_rewards, window_size)
        ax.plot(
            range(window_size - 1, len(episode_rewards)),
            moving_avg,
            color="#c00000",
            linestyle="--",
            linewidth=2.2,
            label=f"Moving Average ({window_size})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "training_rewards", show)

    # 2. Episode Lengths
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episode_lengths, color="#2e7d32", linewidth=1.2, label="Episode Length")
    if len(episode_lengths) >= window_size:
        moving_avg_len = _moving_average(episode_lengths, window_size)
        ax.plot(
            range(window_size - 1, len(episode_lengths)),
            moving_avg_len,
            color="#000000",
            linestyle="--",
            linewidth=2.0,
            label=f"Moving Average ({window_size})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Steps")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "training_episode_lengths", show)

    # 3. Energy Consumption
    if isinstance(episode_energies, list) and len(episode_energies) > 0:
        episode_energies = np.hstack(episode_energies).astype(np.float32)
    else:
        episode_energies = np.asarray(episode_energies, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        episode_energies, color="#c00000", linewidth=1.2, label="Energy Consumption"
    )
    moving_avg_energy = _moving_average(episode_energies, window_size)
    if moving_avg_energy is not None:
        ax.plot(
            range(window_size - 1, len(episode_energies)),
            moving_avg_energy,
            color="#000000",
            linestyle="--",
            linewidth=2.0,
            label=f"Moving Average ({window_size})",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Energy (kWh)")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "training_energy", show)

    # 4. Training Loss
    if episode_losses is not None and len(episode_losses) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(episode_losses, color="#7b2d8e", linewidth=1.2, label="Training Loss")
        moving_avg_loss = _moving_average(episode_losses, window_size)
        if moving_avg_loss is not None:
            ax.plot(
                range(window_size - 1, len(episode_losses)),
                moving_avg_loss,
                color="#000000",
                linestyle="--",
                linewidth=2.0,
                label=f"Moving Average ({window_size})",
            )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Loss")
        _style_publication_axes(ax)
        ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
        fig.tight_layout()
        _save_and_show(fig, save_dir, "training_loss", show)


def plot_episode_details(
    temperatures,
    powers,
    energies,
    rewards,
    targetTemp,
    expert_powers=None,
    save_dir="figures/episode",
    show=True,
):
    """Render four publication-ready episode figures (one per metric).

    Each figure is saved as a 300 dpi PNG and a vector PDF inside ``save_dir``.
    Pass ``save_dir=None`` to skip saving (display only).
    """
    minutes = np.arange(len(temperatures))

    # 1. Temperature vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(minutes, temperatures, color="#1f4e79", linewidth=2.0, label="Agent Temperature")
    ax.axhline(
        y=targetTemp,
        color="#c00000",
        linestyle="--",
        linewidth=2.0,
        label=f"Target Temperature ({targetTemp:.0f}°C)",
    )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Temperature (°C)")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "episode_temperature", show)

    # 2. Power vs Time (Agent vs Expert)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        minutes[: len(powers)],
        powers,
        color="#2e7d32",
        linewidth=2.2,
        label="Agent Power Decision",
    )
    if expert_powers is not None:
        ax.plot(
            np.arange(len(expert_powers)),
            expert_powers,
            color="#c00000",
            linestyle="--",
            linewidth=2.2,
            alpha=0.85,
            label="Expert Profile (Target)",
        )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Power (kW)")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "episode_power_agent_vs_expert", show)

    # 3. Energy Consumption vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        minutes[: len(energies)],
        energies,
        color="#c00000",
        linewidth=2.0,
        label="Energy Consumption",
    )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Energy (kWh)")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "episode_energy", show)

    # 4. Step Reward vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        minutes[: len(rewards)],
        rewards,
        color="#7b2d8e",
        linewidth=1.6,
        label="Step Reward",
    )
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Reward")
    _style_publication_axes(ax)
    ax.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)
    fig.tight_layout()
    _save_and_show(fig, save_dir, "episode_reward", show)

    # 5. Combined: Temperature (top) + Power Agent vs Expert (bottom), shared time axis
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"hspace": 0.12}
    )

    ax_top.plot(
        minutes,
        temperatures,
        color="#1f4e79",
        linewidth=2.0,
        label="Agent Temperature",
    )
    ax_top.axhline(
        y=targetTemp,
        color="#c00000",
        linestyle="--",
        linewidth=2.0,
        label=f"Target Temperature ({targetTemp:.0f}°C)",
    )
    ax_top.set_ylabel("Temperature (°C)")
    _style_publication_axes(ax_top)
    ax_top.set_xlabel("")  # hidden by sharex; only bottom shows the x-axis label
    ax_top.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)

    ax_bot.plot(
        minutes[: len(powers)],
        powers,
        color="#2e7d32",
        linewidth=2.2,
        label="Agent Power Decision",
    )
    if expert_powers is not None:
        ax_bot.plot(
            np.arange(len(expert_powers)),
            expert_powers,
            color="#c00000",
            linestyle="--",
            linewidth=2.2,
            alpha=0.85,
            label="Expert Profile (Target)",
        )
    ax_bot.set_xlabel("Time (minutes)")
    ax_bot.set_ylabel("Power (kW)")
    _style_publication_axes(ax_bot)
    ax_bot.legend(fontsize=12, loc="best", frameon=True, framealpha=0.95)

    fig.tight_layout()
    _save_and_show(fig, save_dir, "episode_temperature_and_power", show)


def replay_episode(
    env,
    agent,
    debug_q_values: bool = False,
    debug_steps: int = 5,
    save_dir: str = "figures/episode",
    show: bool = True,
):
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
        temperatures,
        powers,
        energies,
        rewards,
        targetTemp,
        expert_powers,
        save_dir=save_dir,
        show=show,
    )
