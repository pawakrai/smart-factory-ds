import numpy as np


class AluminumMeltingEnvironment:
    def __init__(
        self,
        initial_weight_kg=350,
        target_temp_c=900,
        scrap_addition_start=60 * 60,
        scrap_addition_end=75 * 60,
        start_mode="cold",  # "cold" or "hot"
        idle_time_min=0,  # idle time from previous batch
        wall_temp_c=None,  # preset wall temperature if known
        seed=None,
        use_fixed_scrap_schedule=False,
        fixed_scrap_schedule=None,  # list of (time_sec, weight_kg)
    ):
        # Physical constants for aluminum
        self.specific_heat = 900  # J/kg·K
        self.initial_mass = initial_weight_kg  # kg - เริ่มต้นด้วย Al ingot 350 kg
        self.current_mass = initial_weight_kg  # kg - น้ำหนักปัจจุบัน
        self.max_capacity = 500  # kg - ความจุสูงสุด
        self.ambient_temp = 25  # °C
        self.max_temp = 1000  # °C
        self.target_temp = target_temp_c  # Target temperature

        # Scrap addition parameters
        self.scrap_addition_start = scrap_addition_start  # 60 minutes in seconds
        self.scrap_addition_end = scrap_addition_end  # 75 minutes in seconds
        self.scrap_additions = []  # List to track scrap additions
        self.total_scrap_added = 0  # Track total scrap added

        # Start condition / furnace thermal state
        self.start_mode = start_mode
        self.idle_time_min = idle_time_min
        self.rng = np.random.default_rng(seed)

        if wall_temp_c is not None:
            self.wall_temp_c0 = wall_temp_c
        else:
            if start_mode == "cold":
                self.wall_temp_c0 = self.ambient_temp
            else:
                decay = np.exp(-idle_time_min / 120.0)  # simple cooling heuristic
                self.wall_temp_c0 = self.ambient_temp + 250 * decay

        # Optional deterministic scrap schedule
        self.use_fixed_scrap_schedule = use_fixed_scrap_schedule
        self.fixed_scrap_schedule = fixed_scrap_schedule or []

        # กำหนด state โดยเพิ่มการติดตาม scrap
        self.state = {
            "temperature": float(self.ambient_temp),
            "furnace_wall_temp": float(self.wall_temp_c0),
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,
            "status": 0,
            "energy_consumption": 0.0,  # Energy consumption in kWh
            "scrap_added": 0.0,  # Total scrap added so far
            "last_scrap_time": 0.0,  # Time of last scrap addition
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
        self.current_mass = self.initial_mass
        self.scrap_additions = []
        self.total_scrap_added = 0

        self.state = {
            "temperature": float(self.ambient_temp),
            "furnace_wall_temp": float(self.wall_temp_c0),
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,
            "status": 1,  # Start with furnace on
            "energy_consumption": 0.0,  # Reset energy consumption in kWh
            "scrap_added": 0.0,
            "last_scrap_time": 0.0,
        }

        return np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
                self.state["energy_consumption"],
                self.state["scrap_added"],
                self.state["furnace_wall_temp"],
            ],
            dtype=np.float32,
        )

    def add_scrap(self):
        """
        เพิ่ม scrap ในช่วงเวลา 60-75 นาที
        จะเพิ่ม 2-3 รอบ รอบละ 50-100 kg
        """
        current_time = self.state["time"]
        current_minute = current_time / 60

        if self.use_fixed_scrap_schedule:
            for t_sec, w_kg in self.fixed_scrap_schedule:
                already = any(
                    abs(s["time"] - t_sec) < 1e-6 for s in self.scrap_additions
                )
                if (not already) and abs(current_time - t_sec) <= self.dt / 2:
                    return self._apply_scrap(scrap_weight=w_kg, scrap_time=t_sec)
            return False

        # ตรวจสอบว่าอยู่ในช่วงเวลาเพิ่ม scrap หรือไม่
        if (
            self.scrap_addition_start <= current_time <= self.scrap_addition_end
            and len(self.scrap_additions) < 3
        ):

            # ตรวจสอบว่าผ่านไปแล้ว 5 นาทีจากการเพิ่ม scrap ครั้งล่าสุดหรือไม่
            time_since_last_scrap = current_time - self.state["last_scrap_time"]

            if len(self.scrap_additions) == 0 or time_since_last_scrap >= 5 * 60:
                # คำนวณน้ำหนัก scrap ที่จะเพิ่ม
                if len(self.scrap_additions) == 0:
                    scrap_weight = self.rng.uniform(40, 70)  # รอบแรก
                elif len(self.scrap_additions) == 1:
                    scrap_weight = self.rng.uniform(50, 80)  # รอบสอง
                else:
                    # รอบสาม - เพิ่มให้ใกล้เคียง 500 kg
                    remaining_capacity = self.max_capacity - self.current_mass
                    scrap_weight = min(remaining_capacity, self.rng.uniform(40, 70))

                return self._apply_scrap(scrap_weight, current_time)
        return False

    def _apply_scrap(self, scrap_weight: float, scrap_time: float):
        """Apply scrap addition consistently across random and fixed schedules."""
        if self.current_mass + scrap_weight > self.max_capacity:
            return False

        self.current_mass += scrap_weight
        self.total_scrap_added += scrap_weight
        self.scrap_additions.append(
            {
                "time": scrap_time,
                "weight": scrap_weight,
                "temperature_drop": scrap_weight * 0.5,
            }
        )
        self.state["scrap_added"] = float(self.total_scrap_added)
        self.state["last_scrap_time"] = float(scrap_time)
        self.state["weight"] = float(self.current_mass)

        # Scrap ที่เพิ่มเข้ามาจะทำให้อุณหภูมิลดลง
        temp_drop = (
            (scrap_weight / self.current_mass)
            * (self.state["temperature"] - self.ambient_temp)
            * 0.3
        )
        self.state["temperature"] -= temp_drop
        return True

    def calculate_temperature_change(self):
        """
        ปรับปรุงการคำนวณการเปลี่ยนแปลงของอุณหภูมิให้สอดคล้องกับข้อมูลจริง
        โดยปรับค่า parameters ให้เหมาะสมกับกระบวนการหลอมจริง

        ใช้ข้อมูลจริงจากตารางเพื่อปรับปรุงการคำนวณ:
        - ช่วงเริ่มต้น (0-30 นาที): อุณหภูมิเพิ่มขึ้นช้า
        - ช่วงกลาง (30-60 นาที): อุณหภูมิเพิ่มขึ้นเร็วขึ้น
        - ช่วงหลัง (60+ นาที): อุณหภูมิเพิ่มขึ้นช้าลงเนื่องจากมี scrap
        """
        current_minute = self.state["time"] / 60
        hotness = (self.state["furnace_wall_temp"] - self.ambient_temp) / 300.0
        hotness = float(np.clip(hotness, 0.0, 1.0))

        # 1. คำนวณพลังงานเข้า (Q_input) - ปรับ efficiency ตามช่วงเวลา
        if current_minute < 30:
            base_eff = 0.10  # ช่วงแรกประสิทธิภาพต่ำ
        elif current_minute < 40:
            base_eff = 0.20  # ช่วงกลางประสิทธิภาพดี
        elif current_minute < 64:
            base_eff = 0.35  # ช่วงกลางประสิทธิภาพดี
        elif current_minute < 70:
            base_eff = 0.10  # วงหลังลดลงเพราะมี scrap
        elif current_minute < 85:
            base_eff = 0.9  # วงหลังลดลงเพราะมี scrap
        else:
            base_eff = 0.9  # ช่วงท้ายระสิทธิภาพดีมาก

        efficiency_factor = float(np.clip(base_eff + 0.1 * hotness, 0.05, 0.90))

        Q_input = self.state["power"] * 1000 * efficiency_factor  # (W)

        # 2. คำนวณการสูญเสียความร้อนแบบคอนเวคทีฟ - ปรับค่าตามข้อมูลจริง
        # ปรับค่า h และ A ให้สอดคล้องกับข้อมูลจริง
        if self.state["temperature"] < 500:
            h_base = 12.0  # W/m²·K - ค่าต่ำในช่วงแรก
        elif self.state["temperature"] < 700:
            h_base = 18.0  # W/m²·K - เพิ่มขึ้นเมื่ออุณหภูมิสูง
        else:
            h_base = 20.0  # W/m²·K - สูงสุดในช่วงอุณหภูมิสูง

        h = h_base * (1.0 + 0.25 * (1.0 - hotness))  # cold start สูญเสียมากกว่า

        A = 3.5  # m² - ปรับลดพื้นที่ผิว
        T_wall = self.state["furnace_wall_temp"]
        Q_conv = h * A * (self.state["temperature"] - self.ambient_temp)

        # 3. คำนวณการสูญเสียความร้อนแบบรังสี - ปรับค่า emissivity
        eps_base = 0.4  # ลดค่า emissivity
        epsilon = eps_base * (1.0 + 0.10 * (1.0 - hotness))
        sigma = 5.67e-8  # W/m²·K⁴
        T_current_K = self.state["temperature"] + 273.15
        T_ambient_K = self.ambient_temp + 273.15
        Q_rad = epsilon * sigma * A * (T_current_K**4 - T_ambient_K**4)

        # 4. คำนวณ net heat
        net_heat = Q_input - (Q_conv + Q_rad)

        # 5. ปรับปรุงการคำนวณ ΔT ให้สอดคล้องกับข้อมูลจริง
        m = self.current_mass  # ใช้น้ำหนักปัจจุบัน

        # ปรับ specific heat ตามช่วงอุณหภูมิ
        # if self.state["temperature"] < 300:
        #     effective_specific_heat = 900  # J/kg·K
        # elif self.state["temperature"] < 600:
        #     effective_specific_heat = 1000  # เพิ่มขึ้นเมื่ออุณหภูมิสูง
        # elif self.state["temperature"] < 660:  # ใกล้จุดหลอม
        #     effective_specific_heat = 1200  # เพิ่มขึ้นมากเนื่องจาก latent heat
        # else:
        #     effective_specific_heat = 900  # ลดลงหลังจากหลอมแล้ว

        effective_specific_heat = 900
        basic_delta_T = (net_heat * self.dt) / (m * effective_specific_heat)

        # 6. ปรับปรุงการจัดการ latent heat ในช่วงหลอม
        melting_point = 660.0  # °C

        if self.state["temperature"] < melting_point - 50:
            # ช่วงก่อนหลอม
            delta_T = basic_delta_T
        elif self.state["temperature"] < melting_point + 50:
            # ช่วงหลอม - ใช้พลังงานส่วนใหญ่ในการเปลี่ยน phase
            delta_T = basic_delta_T * 0.25  # ลดการเพิ่มอุณหภูมิลงมาก
        else:
            # หลังหลอมแล้ว
            delta_T = basic_delta_T * 0.80

        # 7. เพิ่มผลกระทบจากการเพิ่ม scrap
        if len(self.scrap_additions) > 0:
            # ตรวจสอบว่ามีการเพิ่ม scrap ใหม่ในรอบนี้
            recent_scrap = [
                s
                for s in self.scrap_additions
                if self.state["time"] - s["time"] < 5 * 60
            ]  # 5 นาทีหลังเพิ่ม scrap
            if recent_scrap:
                delta_T *= 0.6  # ลดการเพิ่มอุณหภูมิลงเนื่องจาก scrap เย็น

        return delta_T

    def step(self, action):
        # เพิ่ม scrap ถ้าอยู่ในช่วงเวลาที่เหมาะสม
        scrap_added = self.add_scrap()

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
        elif 30 <= current_minute < 31:
            max_power = 0  # ปิดไฟเพื่อเติม Dose [Si + Fe]
        elif 31 <= current_minute < 35:
            max_power = 400
        else:
            max_power = 450

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

        # อัปเดตอุณหภูมิผนังเตา (thermal inertia)
        wall_heat_gain = 0.04 * self.state["power"]
        wall_cool = 0.03 * (self.state["furnace_wall_temp"] - self.ambient_temp)
        self.state["furnace_wall_temp"] = max(
            self.ambient_temp,
            self.state["furnace_wall_temp"] + wall_heat_gain - wall_cool,
        )

        # อัปเดตการใช้พลังงาน (kWh) สำหรับ time step นี้
        self.state["energy_consumption"] += float(
            (self.state["power"] * self.dt) / 3600
        )

        # อัปเดตน้ำหนัก
        self.state["weight"] = float(self.current_mass)

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

        # ส่งกลับ state รวมข้อมูล scrap และอุณหภูมิผนังเตา (state vector 8 ตัว)
        state_array = np.array(
            [
                self.state["temperature"],
                self.state["weight"],
                self.state["time"],
                self.state["power"],
                self.state["status"],
                self.state["energy_consumption"],
                self.state["scrap_added"],
                self.state["furnace_wall_temp"],
            ],
            dtype=np.float32,
        )

        return state_array, reward, done

    def calculate_reward(self):
        """
        ปรับปรุง Reward Function ให้พิจารณาการจัดการ scrap ด้วย
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
        optimal_efficiency = 1.8  # ปรับลดเพราะมีการเพิ่ม scrap
        norm_efficiency = min(efficiency / optimal_efficiency, 1.0)

        # --- Scrap Management Component ---
        # ให้ reward เพิ่มถ้าจัดการ scrap ได้ดี (ใกล้เคียง max capacity)
        target_scrap = self.max_capacity - self.initial_mass  # 150 kg
        if self.total_scrap_added > 0:
            scrap_ratio = min(self.total_scrap_added / target_scrap, 1.0)
            scrap_component = scrap_ratio
        else:
            scrap_component = 0.0

        # --- Time Efficiency Component ---
        norm_time = 1.0 - (self.state["time"] / self.max_time)
        norm_time = max(min(norm_time, 1.0), 0.0)

        # --- กำหนดน้ำหนักสำหรับแต่ละองค์ประกอบ ---
        w_temp = 0.7
        w_energy = 0.30
        w_scrap = 0
        w_time = 0

        reward = (
            w_temp * temp_component
            + w_energy * norm_efficiency
            + w_scrap * scrap_component
            + w_time * norm_time
        )

        return reward

    def get_scrap_info(self):
        """
        ฟังก์ชันสำหรับดูข้อมูลการเพิ่ม scrap
        """
        return {
            "total_scrap_added": self.total_scrap_added,
            "current_mass": self.current_mass,
            "scrap_additions": self.scrap_additions,
            "capacity_utilization": self.current_mass / self.max_capacity,
        }
