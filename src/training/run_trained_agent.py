import matplotlib.pyplot as plt


def run_episode_and_plot(env, agent, render=False):
    """
    ให้ agent รัน episode โดยใช้ greedy policy (epsilon=0)
    และเก็บข้อมูลสำหรับ plotting episode detail
    """
    state = env.reset()
    done = False

    # Lists for storing metrics
    times = []
    temperatures = []
    powers = []
    energy_consumptions = []
    rewards = []

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # ปิด epsilon-greedy เพื่อทดสอบความแม่นยำ

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        # เก็บข้อมูล (แปลงเวลาเป็นนาที)
        times.append(env.state["time"] / 60.0)
        temperatures.append(env.state["temperature"])
        powers.append(env.state["power"])
        energy_consumptions.append(env.state["energy_consumption"])
        rewards.append(reward)

        state = next_state

        if render:
            print(
                f"Time: {env.state['time']/60:.1f} min, "
                f"Temperature: {env.state['temperature']:.1f}°C, "
                f"Power: {env.state['power']} kW, "
                f"Energy Consumption: {env.state['energy_consumption']:.1f} kWh, "
                f"Reward: {reward:.2f}"
            )

    agent.epsilon = original_epsilon  # คืนค่า epsilon เดิม

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Plot Temperature vs Time
    axes[0].plot(times, temperatures, "b-", label="Temperature (°C)")
    axes[0].axhline(
        y=env.target_temp, color="r", linestyle="--", label="Target Temperature"
    )
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Temperature vs Time")
    axes[0].grid(True)
    axes[0].legend()

    # Plot Power and Energy Consumption vs Time (สองแกน Y)
    color_power = "g"
    color_energy = "m"
    ax_power = axes[1]
    ax_power.plot(times, powers, color=color_power, label="Power (kW)")
    ax_power.set_ylabel("Power (kW)", color=color_power)
    ax_power.tick_params(axis="y", labelcolor=color_power)
    ax_power.grid(True)
    ax_power.set_title("Power & Energy Consumption vs Time")

    ax_energy = ax_power.twinx()
    ax_energy.plot(
        times, energy_consumptions, color=color_energy, label="Energy Consumption (kWh)"
    )
    ax_energy.set_ylabel("Energy Consumption (kWh)", color=color_energy)
    ax_energy.tick_params(axis="y", labelcolor=color_energy)

    # รวม legend ของกราฟที่มีแกน Y สองแกน
    lines1, labels1 = ax_power.get_legend_handles_labels()
    lines2, labels2 = ax_energy.get_legend_handles_labels()
    ax_power.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Plot Reward vs Time
    axes[2].plot(times, rewards, "c-", label="Reward")
    axes[2].set_xlabel("Time (minutes)")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Reward vs Time")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Return collected metrics ifต้องการใช้งานต่อ
    return times, temperatures, powers, energy_consumptions, rewards


if __name__ == "__main__":
    import torch
    from src.environment.aluminum_melting_env_7 import AluminumMeltingEnvironment
    from src.agents.agent2 import DQNAgent  # สมมุติว่า Agent ถูกนิยามในไฟล์นี้

    # สร้าง Environment และ Agent
    env = AluminumMeltingEnvironment(initial_weight_kg=500, target_temp_c=900)
    checkpoint_path = "models/dqn_final_model_10.pth"
    agent = DQNAgent.load_checkpoint(checkpoint_path, state_dim=6, action_dim=5)

    # โหลด model จาก checkpoint และตั้งให้เป็น evaluation mode
    # state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    # agent.model.load_state_dict(state_dict)
    # agent.model.eval()

    # รัน episode และ plot ผลลัพธ์
    run_episode_and_plot(env, agent, render=True)
