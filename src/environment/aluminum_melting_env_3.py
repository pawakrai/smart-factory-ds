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

        # คำนวณ reward เฉพาะเมื่อ episode จบแล้ว (done == True)
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
        Reward Function โดยรวมปัจจัย:
        1. Energy Efficiency: kg/kWh (normalize โดยสมมุติ maximum ที่คาดหวังคือ 1.0)
        2. Product Quality: จากฟังก์ชัน calculate_quality() (ค่าระหว่าง 0 ถึง 1)
        3. Production Time: Penalize เมื่อใช้เวลานาน (normalize กับเวลาสูงสุด)
        4. Temperature Targeting: รางวัลจากการควบคุมอุณหภูมิให้ใกล้ target (มี tolerance)
        """

        # 1. Energy Efficiency
        if self.state["energy_consumption"] > 0:
            energy_efficiency = (
                self.state["weight"] / self.state["energy_consumption"]
            )  # kg/kWh
        else:
            energy_efficiency = 0
        max_energy_efficiency = 1.0  # ค่าที่คาดหวังสูงสุด (สามารถปรับได้)
        norm_energy = np.clip(energy_efficiency / max_energy_efficiency, 0, 1)

        # 2. Product Quality
        quality = self.calculate_quality()  # คืนค่า 0 ถึง 1
        norm_quality = quality

        # 3. Production Time Penalty
        time_minutes = self.state["time"] / 60.0
        max_time_minutes = self.max_time / 60.0
        norm_time_penalty = time_minutes / max_time_minutes  # ค่าระหว่าง 0 ถึง 1

        # 4. Temperature Targeting Reward
        temp_diff = abs(self.state["temperature"] - self.target_temp)
        tolerance = 20  # กำหนด tolerance 20°C
        if temp_diff <= tolerance:
            temp_reward = 1.0
        else:
            # ลด reward เมื่อ temp_diff เกิน tolerance โดยใช้ฟังก์ชัน inverse
            temp_reward = np.clip(
                1 - (temp_diff - tolerance) / (self.max_temp - self.target_temp), 0, 1
            )

        # กำหนดน้ำหนักให้กับแต่ละองค์ประกอบ
        w_energy = 0.0
        w_quality = 0.2
        w_time = 0.4  # ค่าลบเพื่อเป็น penalty เมื่อใช้เวลานาน
        w_temp = 0.4

        # รวม reward โดยให้ reward สูงขึ้นเมื่อ:
        # - Energy Efficiency สูง (norm_energy ใกล้ 1)
        # - Product Quality ดี (norm_quality ใกล้ 1)
        # - Temperature ใกล้ target (temp_reward ใกล้ 1)
        # และลดลงเมื่อ Production Time ยาวนาน (norm_time_penalty สูง)
        reward = (
            w_energy * norm_energy
            + w_quality * norm_quality
            + w_temp * temp_reward
            - w_time * norm_time_penalty
        )

        return reward
