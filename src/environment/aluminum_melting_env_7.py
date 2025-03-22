import numpy as np


class AluminumMeltingEnvironment:
    def __init__(self):
        # Physical constants for aluminum
        self.specific_heat = 900  # J/kg·K
        self.mass = 500  # kg
        self.ambient_temp = 25  # °C
        self.max_temp = 900  # °C
        self.target_temp = 900  # Target temperature

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
        self.max_time = 120 * 60  # 120 minutes in seconds

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
        h = 14.0  # W/m²·K
        A = 4.0  # m²
        Q_conv = h * A * (self.state["temperature"] - self.ambient_temp)

        # 3. คำนวณการสูญเสียความร้อนแบบรังสี
        epsilon = 0.5
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
            # สมมุติว่าแค่ 70% ของ net heat ส่งผลให้เพิ่มอุณหภูมิได้
            delta_T = basic_delta_T * 0.8
        else:
            delta_T = basic_delta_T

        return delta_T

    def step(self, action):
        action_type = self.action_space.get(action, "maintain")

        # ถ้าเตาไม่ได้ทำงาน ให้ power = 0
        if self.state["status"] != 1:
            self.state["power"] = 0
        else:
            # อัปเดต power ตาม action ที่เลือก
            if action_type == "increase_power_strong":
                self.state["power"] = min(500, self.state["power"] + 100)
            elif action_type == "increase_power_mild":
                self.state["power"] = min(500, self.state["power"] + 50)
            elif action_type == "maintain":
                pass  # คงค่า power เดิมไว้
            elif action_type == "decrease_power_mild":
                self.state["power"] = max(0, self.state["power"] - 50)
            elif action_type == "decrease_power_strong":
                self.state["power"] = max(0, self.state["power"] - 100)

        # --------------------------------------------------
        # Constraint: บังคับให้ power อยู่ในช่วงที่กำหนดตามเวลา
        # --------------------------------------------------
        current_minute = self.state["time"] / 60

        if 0 <= current_minute < 5:
            max_power = 50
        elif 5 <= current_minute < 10:
            max_power = 150
        elif 10 <= current_minute < 15:
            max_power = 250
        elif 15 <= current_minute < 30:
            max_power = 350
        elif 30 <= current_minute < 32:
            max_power = 0  # ปิดไฟเพื่อเติม Dose [Si + Fe]
        elif 32 <= current_minute < 40:
            max_power = 400
        else:
            max_power = 450
            
        # ควร Fix ว่าเติมอะไรบางอย่างเข้าไป ถ้าไม่เติมจะโดนหักคะแนน

        # ปรับค่า power ไม่ให้เกินข้อจำกัดของแต่ละ Phase
        self.state["power"] = min(self.state["power"], max_power)
        self.state["power"] = max(0, self.state["power"])  # ให้แน่ใจว่าไม่ติดลบ

        # --------------------------------------------------
        # อัปเดตอุณหภูมิ
        # --------------------------------------------------
        if self.state["status"] == 1:
            delta_T = self.calculate_temperature_change()
            self.state["temperature"] += delta_T
            self.state["temperature"] = min(
                self.max_temp, max(self.ambient_temp, self.state["temperature"])
            )

        # อัปเดตเวลา
        self.state["time"] += self.dt

        # อัปเดตการใช้พลังงาน (kWh) สำหรับ time step นี้
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

        # ส่งกลับ state โดยไม่รวมค่า spark_penalty (state vectorมี 6 ตัว)
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
        1. การเข้าถึง target temperature (ideal คือ 850°C)
        2. ประสิทธิภาพการใช้พลังงาน (ลดการใช้พลังงานให้น้อยที่สุด)
        3. ใช้เวลาน้อยที่สุด (ในบางทดลองอาจให้ความสำคัญน้อยลง)

        - Temperature Component: ถ้า temperature >= target ให้ได้ 1.0,
        ถ้ายังไม่ถึง target ให้ลดลงเชิงเส้นจาก -1 ถึง +1
        - Energy Efficiency Component: คำนวณจาก weight/energy_consumption เปรียบเทียบกับ optimal efficiency
        - Time Efficiency Component: norm_time = 1 - (current_time/max_time)
        (ในตัวอย่างนี้น้ำหนักของ Time Component ตั้งเป็น 0)
        """
        # --- Temperature Component ---
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            temp_component = -1.0 + 2.0 * (
                self.state["temperature"] - self.ambient_temp
            ) / (self.target_temp - self.ambient_temp)

        # --- Energy Efficiency Component ---
        # หาก energy_consumption มากกว่า 300 kWh ให้คำนวณ efficiency (สมมุติว่า optimal efficiency คือ 2.0 kg/kWh)
        if self.state["energy_consumption"] > 300:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0
        optimal_efficiency = 2.0
        norm_efficiency = min(efficiency / optimal_efficiency, 1.0)

        # --- Time Efficiency Component ---
        norm_time = 1.0 - (self.state["time"] / self.max_time)
        norm_time = max(min(norm_time, 1.0), 0.0)

        # --- กำหนดน้ำหนักสำหรับแต่ละองค์ประกอบ ---
        w_temp = 0.7
        w_energy = 0.3
        w_time = 0.0  # ในตัวอย่างนี้ให้ความสำคัญกับอุณหภูมิและพลังงาน

        reward = (
            w_temp * temp_component + w_energy * norm_efficiency + w_time * norm_time
        )

        return reward
