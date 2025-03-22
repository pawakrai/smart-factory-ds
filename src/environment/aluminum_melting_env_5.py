import numpy as np


class AluminumMeltingEnvironment:
    def __init__(self):
        # Physical constants for aluminum
        self.specific_heat = 900  # J/kg·K
        self.mass = 500  # kg
        self.ambient_temp = 25  # °C
        self.max_temp = 900  # °C
        self.target_temp = 850  # Target temperature

        # กำหนด state โดยไม่รวม cumulative_energy
        self.state = {
            "temperature": self.ambient_temp,
            "weight": 500,
            "time": 0,
            "power": 0,
            "status": 0,
            "energy_consumption": 0,  # Energy consumption in kWh
        }

        self.action_space = {
            0: "increase_power_strong",
            1: "increase_power_mild",
            2: "maintain",
            3: "decrease_power_mild",
            4: "decrease_power_strong",
        }
        # Time settings
        self.dt = 60  # 60 seconds (1 minute)
        self.max_time = 90 * 60  # 90 minutes in seconds

    def reset(self):
        self.state = {
            "temperature": self.ambient_temp,
            "weight": 500,
            "time": 0,
            "power": 0,
            "status": 1,  # Start with furnace on
            "energy_consumption": 0,  # Reset energy consumption in kWh
        }

        return np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
                self.state["energy_consumption"],
            ],
            dtype=np.float32,
        )

    def calculate_temperature_change(self):
        """
        ปรับปรุงการคำนวณการเปลี่ยนแปลงของอุณหภูมิให้สมจริงขึ้น โดยพิจารณา:
          1. พลังงานเข้า (Q_input) จาก power input โดยใช้ efficiency factor ที่สูงขึ้น
          2. การสูญเสียความร้อนแบบคอนเวคทีฟ (Q_conv) โดยลดค่าของ h และ A
          3. การสูญเสียความร้อนแบบรังสี (Q_rad)
          4. ผลของ latent heat ในช่วงหลอม (melting range)

        สูตร:
          net_heat = Q_input - (Q_conv + Q_rad)
          ΔT = (net_heat × dt) / (m × c)    (โดยปรับลดอัตราเพิ่มอุณหภูมิในช่วงหลอม)
        """
        # 1. คำนวณพลังงานเข้า (Q_input)
        efficiency_factor = 0.6  # ใช้ efficiency factor ที่ตั้งไว้
        Q_input = self.state["power"] * 1000 * efficiency_factor  # (W)

        # 2. คำนวณการสูญเสียความร้อนแบบคอนเวคทีฟ
        h = 20.0  # W/m²·K
        A = 4.0  # m²
        Q_conv = h * A * (self.state["temperature"] - self.ambient_temp)

        # 3. คำนวณการสูญเสียความร้อนแบบรังสี
        epsilon = 0.6
        sigma = 5.67e-8  # W/m²·K⁴
        T_current_K = self.state["temperature"] + 273.15
        T_ambient_K = self.ambient_temp + 273.15
        Q_rad = epsilon * sigma * A * (T_current_K**4 - T_ambient_K**4)

        # 4. คำนวณ net heat
        net_heat = Q_input - (Q_conv + Q_rad)

        # 5. คำนวณ ΔT เบื้องต้น
        m = self.mass  # kg
        basic_delta_T = (net_heat * self.dt) / (m * self.specific_heat)

        # 6. Incorporate latent heat effect ในช่วงหลอม
        melting_point = 660.0  # °C
        if self.state["temperature"] < melting_point:
            delta_T = basic_delta_T
        elif self.state["temperature"] < self.target_temp:
            # เมื่ออยู่ในช่วงหลอม ให้ส่วนใหญ่ของพลังงานไปเปลี่ยน phase
            # สมมุติว่าแค่ 50% ของ net heat ส่งผลให้เพิ่มอุณหภูมิได้
            delta_T = basic_delta_T * 0.5
        else:
            delta_T = basic_delta_T

        return delta_T

    def step(self, action):
        # แปลง action จากตัวเลขเป็นข้อความ
        action_type = self.action_space.get(action, "maintain")

        if self.state["status"] != 1:
            self.state["power"] = 0
        else:
            if action_type == "increase_power_strong":
                self.state["power"] = min(500, self.state["power"] + 100)
            elif action_type == "increase_power_mild":
                self.state["power"] = min(500, self.state["power"] + 50)
            elif action_type == "maintain":
                # คงค่า power เดิม
                self.state["power"] = self.state["power"]
            elif action_type == "decrease_power_mild":
                self.state["power"] = max(0, self.state["power"] - 50)
            elif action_type == "decrease_power_strong":
                self.state["power"] = max(0, self.state["power"] - 100)

        # อัปเดตอุณหภูมิ
        if self.state["status"] == 1:
            delta_T = self.calculate_temperature_change()
            self.state["temperature"] += delta_T
            self.state["temperature"] = min(
                self.max_temp, max(self.ambient_temp, self.state["temperature"])
            )

        # อัปเดตเวลา
        self.state["time"] += self.dt

        # อัปเดตการใช้พลังงาน (คำนวณโดยตรงโดยเพิ่มพลังงานที่ใช้ในแต่ละ time step)
        # Energy consumption (kWh) = (power (kW) * dt (sec)) / 3600
        self.state["energy_consumption"] += (self.state["power"] * self.dt) / 3600

        # ตรวจสอบเงื่อนไขสิ้นสุด episode
        done = False
        if (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["temperature"] >= self.target_temp
            or self.state["status"] == 0
        ):
            done = True

        # คำนวณ reward เฉพาะเมื่อ episode จบ
        reward = self.calculate_reward() if done else 0.0
        # reward = self.calculate_reward()

        # ส่งกลับ state โดยไม่รวม cumulative_energy
        state_array = np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
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
        Reward Function ใหม่ โดยให้ความสำคัญกับ:
        1. การเข้าถึง target temperature
        2. ประสิทธิภาพการใช้พลังงาน (energy efficiency)
        3. ใช้เวลาน้อยที่สุด

        Components:
        - Temperature: ถ้า temperature >= target: 1.0, ถ้ายังไม่ถึง target ลดลงเชิงเส้นจาก -1 ถึง +1
        - Energy Efficiency: normalized จาก efficiency = weight/energy_consumption เปรียบเทียบกับ optimal_efficiency
        - Time Efficiency: norm_time = 1 - (current_time / max_time)
        """
        # --- Temperature Component ---
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            temp_component = -1.0 + 2.0 * (
                self.state["temperature"] - self.ambient_temp
            ) / (self.target_temp - self.ambient_temp)

        # --- Energy Efficiency Component ---
        if self.state["energy_consumption"] > 300:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0
        # กำหนด optimal efficiency (ค่าอ้างอิงที่คาดหวัง) เช่น 2.0 kg/kWh
        optimal_efficiency = 2
        norm_efficiency = min(efficiency / optimal_efficiency, 1.0)

        # --- Time Efficiency Component ---
        norm_time = 1.0 - (self.state["time"] / self.max_time)
        norm_time = max(min(norm_time, 1.0), 0.0)

        # --- กำหนดน้ำหนักสำหรับแต่ละองค์ประกอบ ---
        w_temp = 0.7
        w_energy = 0.3
        w_time = 0.0

        reward = (
            w_temp * temp_component + w_energy * norm_efficiency + w_time * norm_time
        )
        return reward
