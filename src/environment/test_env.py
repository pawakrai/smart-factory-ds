import matplotlib.pyplot as plt
from src.environment.aluminum_melting_env_7 import (
    AluminumMeltingEnvironment,
)  # Import environment


def test_environment():
    env = AluminumMeltingEnvironment()
    state = env.reset()

    temperatures = []
    powers = []
    times = []
    energy_consumption = []
    actions_taken = []

    # กำหนด series of actions ตามที่ออกแบบไว้
    action_sequence = [
        # (1, "increase_power_mild", 1),
        # (2, "maintain", 5),
        # (0, "increase_power_strong", 1),
        # (2, "maintain", 5),
        # (0, "increase_power_strong", 1),
        # (2, "maintain", 5),
        # (0, "increase_power_strong", 1),
        # (2, "maintain", 15),
        # (4, "decrease_power_strong", 4),
        (0, "increase_power_strong", 60),
        # (1, "increase_power_mild", 1),
        (2, "maintain", 60),
    ]

    print("Starting Environment Test")
    done = False

    for action_code, action_name, steps in action_sequence:
        print(f"\nExecuting '{action_name}' for {steps} steps")
        for _ in range(steps):
            next_state, reward, done = env.step(action_code)

            temperatures.append(next_state[0])
            powers.append(next_state[3])
            times.append(next_state[2] / 60.0)
            energy_consumption.append(next_state[5])
            actions_taken.append(action_name)

            print(
                f"Time: {next_state[2]/60:.1f} min, Temp: {next_state[0]:.1f}°C, "
                f"Power: {next_state[3]} kW, Energy Consumption: {next_state[5]:.1f} kWh"
            )

            if done:
                break
        if done:
            break

    # สร้างกราฟ 2 แผนที่
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # กราฟ Temperature vs Time
    ax1.plot(times, temperatures, "b-", label="Temperature")
    ax1.axhline(
        y=env.target_temp, color="r", linestyle="--", label="Target Temperature"
    )
    ax1.set_xlabel("Time (minutes)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature vs Time")
    ax1.grid(True)
    ax1.legend()

    # กราฟเดียวกันสำหรับ Power vs Time และ Energy Consumption vs Time โดยใช้ 2 แกน Y
    color_power = "g"
    color_energy = "m"

    # Plot Power บนแกน Y ด้านซ้าย
    ax2.plot(times, powers, color=color_power, label="Power (kW)")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Power (kW)", color=color_power)
    ax2.tick_params(axis="y", labelcolor=color_power)
    ax2.grid(True)
    ax2.set_title("Power & Energy Consumption vs Time")

    # สร้างแกน Y ที่สองสำหรับ Energy Consumption
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        times, energy_consumption, color=color_energy, label="Energy Consumption (kWh)"
    )
    ax2_twin.set_ylabel("Energy Consumption (kWh)", color=color_energy)
    ax2_twin.tick_params(axis="y", labelcolor=color_energy)

    # รวม legend จากทั้งสองแกน (วิธีหนึ่งคือใช้ combine handles)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()

    total_energy = energy_consumption[-1]  # Total energy consumption in kWh
    weight_kg = env.state["weight"]
    weight_ton = weight_kg / 1000.0  # แปลงเป็นตัน (1 ตัน = 1000 kg)
    kwh_per_ton = total_energy / weight_ton if weight_ton > 0 else 0

    print("\nFinal Energy Statistics:")
    print(f"Total Energy Consumption: {total_energy:.1f} kWh")
    avg_power = sum(powers) / len(powers) if powers else 0.0
    print(f"Average Power Usage: {avg_power:.1f} kW")
    print(f"Energy Consumption: {kwh_per_ton:.1f} kWh/ton")

    return temperatures, powers, times, actions_taken, energy_consumption


if __name__ == "__main__":
    test_environment()
