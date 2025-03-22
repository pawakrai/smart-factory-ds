import numpy as np


class AluminumMeltingEnvironment:
    def __init__(self):
        # Physical constants for aluminum
        self.specific_heat = 900  # J/kg·K
        self.mass = 500  # kg
        self.ambient_temp = 25  # °C
        self.max_temp = 800  # °C
        self.target_temp = 750  # Target temperature

        self.state = {
            "temperature": self.ambient_temp,
            "weight": 500,
            "time": 0,
            "power": 0,
            "status": 0,
            "cumulative_energy": 0,  # Total energy used in kW
            "energy_consumption": 0,  # Energy consumption in kWh
        }

        self.action_space = {0: "increase_power", 1: "decrease_power"}
        # Time settings
        self.dt = 60  # Change time step to 60 seconds (1 minute)
        self.max_time = 90 * 60  # 90 minutes in seconds

    def reset(self):
        self.state = {
            "temperature": self.ambient_temp,
            "weight": 500,
            "time": 0,
            "power": 0,
            "status": 1,  # Start with furnace on
            "cumulative_energy": 0,  # Reset total energy used
            "energy_consumption": 0,  # Reset energy consumption in kWh
        }

        return np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
                self.state["cumulative_energy"],
                self.state["energy_consumption"],
            ],
            dtype=np.float32,
        )

    def calculate_temperature_change(self):
        """
        Calculate temperature change based on power input and heat losses
        Using the equation: ΔT = (Q * Δt)/(m * c)
        Where:
        - Q is power in Watts
        - Δt is time step in seconds
        - m is mass in kg
        - c is specific heat capacity in J/kg·K
        """
        # Heat input from power (reduce efficiency to 60%)
        power_input = self.state["power"] * 1000 * 0.6

        # Increase heat loss coefficient significantly
        heat_loss_coefficient = 350
        heat_loss = heat_loss_coefficient * (
            self.state["temperature"] - self.ambient_temp
        )

        # Add phase change factor when near melting point (660°C for aluminum)
        melting_point = 660
        if abs(self.state["temperature"] - melting_point) < 50:
            # Reduce effective heating rate near melting point
            power_input *= 0.9

        # Net heat
        net_heat = power_input - heat_loss

        # Temperature change for the time step
        delta_T = (net_heat * self.dt) / (self.mass * self.specific_heat)

        return delta_T

    def step(self, action):
        # Convert numeric action to string
        action_type = self.action_space[action]

        if action_type == "stop":
            self.state["status"] = 0
            self.state["power"] = 0

        # Update power based on action
        if (
            self.state["status"] == 1
        ):  # Only allow power changes when furnace is running
            if action == 0:  # increase_power
                self.state["power"] = min(500, self.state["power"] + 50)
            else:  # decrease_power
                self.state["power"] = max(0, self.state["power"] - 50)

        # Update temperature only if status is 1 (running)
        if self.state["status"] == 1:
            delta_T = self.calculate_temperature_change()
            self.state["temperature"] += delta_T
            self.state["temperature"] = min(
                self.max_temp, max(self.ambient_temp, self.state["temperature"])
            )

        # Update time
        self.state["time"] += self.dt

        # Update energy metrics
        self.state["cumulative_energy"] += self.state["power"]
        self.state["energy_consumption"] = (
            self.state["cumulative_energy"] * self.dt
        ) / 3600  # kWh

        # ตรวจสอบว่า episode จบหรือยัง
        done = False
        if (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["status"] == 0  # stop
            or self.state["temperature"] >= self.target_temp
        ):
            done = True

        # คำนวณ reward
        reward = self.calculate_reward() if done else 0.0

        # Convert state to numpy array
        state_array = np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
                self.state["cumulative_energy"],
                self.state["energy_consumption"],
            ],
            dtype=np.float32,
        )

        return state_array, reward, done

    def calculate_quality(self):
        """Calculate melt quality based on temperature control"""
        temp_diff = abs(self.state["temperature"] - self.target_temp)

        if temp_diff < 10:
            return 1.0
        elif temp_diff < 30:
            return 0.8
        elif temp_diff < 50:
            return 0.6
        elif temp_diff < 70:
            return 0.4
        elif temp_diff < 90:
            return 0.2
        else:
            return 0.0

    def calculate_reward(self):
        """
        Reward Function ให้ความสำคัญกับ:
          1. การเข้าถึง target temperature
          2. การประหยัดพลังงาน (energy efficiency)

        - หากอุณหภูมิถึง target หรือเกิน target: temp_component = 1.0
        - หากยังไม่ถึง target: ให้ temp_component ลดลงเชิงเส้นจาก -1 (ที่อุณหภูมิ ambient) ถึง +1 (ที่ target)

        - Energy efficiency คำนวณจาก weight/energy_consumption และ clip ให้มีค่าสูงสุด 1.0

        Reward รวม = w_temp * temp_component + w_energy * norm_efficiency
        โดยที่ w_temp > w_energy
        """
        # --- Temperature Component ---
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            # เมื่อ temperature = ambient จะได้ -1, เมื่อ temperature = target จะได้ +1
            temp_component = -1.0 + 2.0 * (
                self.state["temperature"] - self.ambient_temp
            ) / (self.target_temp - self.ambient_temp)

        # --- Energy Efficiency Component ---
        if self.state["energy_consumption"] > 0:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0
        # ปรับให้มีค่าสูงสุดไม่เกิน 1
        norm_efficiency = min(efficiency, 1.0)

        # --- กำหนดน้ำหนักให้กับแต่ละองค์ประกอบ ---
        w_temp = 0.7  # ให้ความสำคัญกับ target temperature มากกว่า
        w_energy = 0.3

        reward = w_temp * temp_component + w_energy * norm_efficiency
        return reward
