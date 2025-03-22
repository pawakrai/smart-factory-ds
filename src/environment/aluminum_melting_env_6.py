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
            "spark_penalty": 0.0,  # Penalty accumulated due to spark events
        }

        self.action_space = {
            0: "increase_power_strong",
            1: "increase_power_mild",
            2: "maintain",
            3: "decrease_power_mild",
            4: "decrease_power_strong",
        }
        # Time settings: ตั้งให้ max_time เป็น 120 นาที (7200 วินาที)
        self.dt = 60  # 60 seconds (1 minute)
        self.max_time = 120 * 60  # 120 minutes in seconds

        # Ideal melting time (สำหรับ reward time efficiency)
        self.ideal_time = 120 * 60  # 7200 seconds

        # กำหนด threshold สำหรับ ΔT ที่ถือว่า “เร่งไฟเกิน” (spark)
        self.deltaT_threshold = 3.0  # °C per step (ตัวอย่าง)
        # กำหนด factor สำหรับคำนวณ spark penalty (ยิ่ง ΔT เกิน threshold มาก penalty มาก)
        self.spark_penalty_factor = 0.1

    def reset(self):
        self.state = {
            "temperature": self.ambient_temp,
            "weight": 500,
            "time": 0,
            "power": 0,
            "status": 1,  # Start with furnace on
            "energy_consumption": 0,  # Reset energy consumption in kWh
            "spark_penalty": 0.0,
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
        ปรับปรุงการคำนวณการเปลี่ยนแปลงของอุณหภูมิให้สมจริงขึ้น
          1. คำนวณ Q_input จาก power input
          2. คำนวณการสูญเสียความร้อนแบบคอนเวคทีฟ (Q_conv)
          3. คำนวณการสูญเสียความร้อนแบบรังสี (Q_rad)
          4. Incorporate latent heat effect ในช่วงหลอม (melting range)
        """
        # 1. คำนวณพลังงานเข้า (Q_input)
        efficiency_factor = 0.6  # ใช้ efficiency factor ที่ตั้งไว้
        Q_input = self.state["power"] * 1000 * efficiency_factor  # (W)

        # 2. การสูญเสียความร้อนแบบคอนเวคทีฟ
        h = 20.0  # W/m²·K
        A = 4.0  # m²
        Q_conv = h * A * (self.state["temperature"] - self.ambient_temp)

        # 3. การสูญเสียความร้อนแบบรังสี
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

        # 6. Incorporate latent heat effect ในช่วงหลอม (melting_point = 660°C)
        melting_point = 660.0  # °C
        if self.state["temperature"] < melting_point:
            delta_T = basic_delta_T
        elif self.state["temperature"] < self.target_temp:
            # ในช่วงหลอม: มี latent heat ทำให้ ΔT ลดลง
            delta_T = basic_delta_T * 0.5
        else:
            delta_T = basic_delta_T

        return delta_T

    def step(self, action):
        # แปลง action จากตัวเลขเป็นข้อความ
        action_type = self.action_space.get(action, "maintain")

        # บันทึกค่า temperature ก่อนอัปเดต (สำหรับคำนวณ spark)
        prev_temp = self.state["temperature"]

        if self.state["status"] != 1:
            self.state["power"] = 0
        else:
            if action_type == "increase_power_strong":
                self.state["power"] = min(500, self.state["power"] + 100)
            elif action_type == "increase_power_mild":
                self.state["power"] = min(500, self.state["power"] + 50)
            elif action_type == "maintain":
                pass  # ไม่เปลี่ยนแปลง power
            elif action_type == "decrease_power_mild":
                self.state["power"] = max(0, self.state["power"] - 50)
            elif action_type == "decrease_power_strong":
                self.state["power"] = max(0, self.state["power"] - 100)

        # อัปเดตอุณหภูมิ
        if self.state["status"] == 1:
            delta_T = self.calculate_temperature_change()
            self.state["temperature"] += delta_T
            # จำกัดอุณหภูมิให้อยู่ในช่วง [ambient, max_temp]
            self.state["temperature"] = min(
                self.max_temp, max(self.ambient_temp, self.state["temperature"])
            )

            # ตรวจสอบ spark: ถ้า ΔT เกิน threshold (และ power สูง) ให้เพิ่ม penalty
            if (delta_T > self.deltaT_threshold) and (self.state["power"] >= 500):
                # spark penalty เพิ่มขึ้นตาม (delta_T - threshold)
                spark_penalty = (
                    delta_T - self.deltaT_threshold
                ) * self.spark_penalty_factor
                self.state["spark_penalty"] += spark_penalty

        # อัปเดตเวลา
        self.state["time"] += self.dt

        # อัปเดตการใช้พลังงาน (kWh)
        self.state["energy_consumption"] += (self.state["power"] * self.dt) / 3600

        # ตรวจสอบเงื่อนไขสิ้นสุด episode
        done = False
        # หมดเวลา, ถึง max_temp, หรือถึง target_temp (หรือเครื่องปิด)
        if (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["status"] == 0
            or self.state["temperature"] >= self.target_temp
        ):
            done = True

        # คำนวณ reward เฉพาะเมื่อ episode จบ
        reward = self.calculate_reward() if done else 0.0

        # ส่งกลับ state โดยไม่รวม spark_penalty (หรืออาจรวมไว้เพื่อ debug)
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

    def calculate_reward(self):
        """
        Reward Function ใหม่ โดยให้ความสำคัญกับ:
        1. การเข้าถึง target temperature
        2. ประสิทธิภาพการใช้พลังงาน (หรือไม่คำนึงในบางทดลอง)
        3. ใช้เวลาน้อยที่สุด (ideal คือ 120 นาที)
        4. หักคะแนนเมื่อเกิด spark (การเร่งไฟเกิน)

        Components:
        - Temperature Component: ถ้า temperature >= target: 1.0,
          ถ้ายังไม่ถึง target ลดลงเชิงเส้นจาก -1 ถึง +1
        - Energy Efficiency Component: ในการทดลองนี้อาจจะไม่คำนึงก็ได้ (หรือปรับน้ำหนักต่ำ)
        - Time Efficiency Component: ใช้เวลาน้อยที่สุด (norm_time = 1 - abs(time - ideal_time)/ideal_time)
        - Spark Penalty: หักคะแนนตามค่า spark_penalty ที่สะสมไว้
        """
        # --- Temperature Component ---
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            temp_component = -1.0 + 2.0 * (
                self.state["temperature"] - self.ambient_temp
            ) / (self.target_temp - self.ambient_temp)

        # --- Energy Efficiency Component ---
        if self.state["energy_consumption"] > 0:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0
        optimal_efficiency = 2.0
        norm_efficiency = min(efficiency / optimal_efficiency, 1.0)

        # --- Time Efficiency Component ---
        # กำหนด ideal_time = 120 นาที (7200 s)
        time_diff = abs(self.state["time"] - self.ideal_time)
        norm_time = max(0.0, 1.0 - (time_diff / self.ideal_time))

        # --- รวม Components ---
        # กำหนดน้ำหนักให้กับแต่ละองค์ประกอบ
        w_temp = 0.5
        w_energy = 0.0
        w_time = 0.3

        base_reward = (
            w_temp * temp_component + w_energy * norm_efficiency + w_time * norm_time
        )

        # หัก spark penalty (สมมุติให้ penalty ตรงๆ จาก spark_penalty ที่สะสม)
        reward = base_reward - self.state["spark_penalty"]

        return reward
