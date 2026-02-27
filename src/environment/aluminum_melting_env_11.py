import numpy as np


class AluminumMeltingEnvironment:
    """
    Environment สำหรับการหลอมอลูมิเนียม พร้อมระบบ Reward Function แบบ Expert Guided
    Action Space: Discrete 10 ระดับ (0, 50, 100, 150, 200, 250, 300, 350, 400, 450 kW)
    """

    def __init__(
        self,
        initial_weight_kg=350,
        target_temp_c=950,
        scrap_addition_start=60 * 60,
        scrap_addition_end=75 * 60,
        start_mode="hot",
        idle_time_min=0,
        wall_temp_c=None,
        initial_metal_temp_c=None,
        seed=None,
        use_fixed_scrap_schedule=False,
        fixed_scrap_schedule=None,
        # ---------- power handling ----------
        overall_efficiency=0.9326,
        eff_to_metal=0.8726,
        eff_to_wall=0.0269,
        use_time_efficiency=True,
        time_efficiency_schedule=[
            (0, 0.80),
            (30, 0.80),
            (30, 0.90),
            (40, 0.90),
            (40, 0.95),
            (60, 0.95),
            (60, 0.50),
            (70, 0.50),
            (70, 0.98),
            (90, 0.98),
        ],
        # ---------- wall-metal coupling ----------
        k_wall_metal=800,
        k_dT_alpha=0.0003,
        hot_k_wall_metal=1100.0,
        hot_k_early_factor=0.80,
        hot_early_minutes=100,
        wall_heat_capacity_J_per_K=2.5e6,
        # ---------- wall losses ----------
        wall_area_m2=3.5,
        wall_h_W_m2K=18.0,
        wall_emissivity=0.55,
        # ---------- metal losses ----------
        metal_area_m2=1.0,
        metal_h_W_m2K=2.79,
        metal_emissivity=0.005,
        hot_metal_emissivity=0.18,
        # ---------- latent heat handling ----------
        melting_point_c=660.0,
        latent_band_c=50.0,
        latent_scale=0.324,
        post_melt_scale=0.803,
        # ---------- scrap effects ----------
        scrap_temp_drop_scale=0.50,
        scrap_recent_scale=0.80,
        scrap_recent_window_min=5,
        k_reduce_when_recent_scrap=0.75,
        # ---------- start condition heuristic ----------
        hot_wall_delta_c=200.0,
        hot_wall_tau_min=60.0,
        hot_metal_fraction_of_wall=0.70,
        # ---------- simulation timing ----------
        dt_sec=60,
        max_time_min=120,
        max_capacity_kg=500,
        ambient_temp_c=25,
        max_temp_c=1000,
        # ---------- energy metering ----------
        energy_consumption_scale=1.068,
        auxiliary_power_kw=0.0,
        # ---------- high-power efficiency drop ----------
        high_power_eff_start_kw=450.0,
        high_power_eff_end_kw=500.0,
        high_power_eff_min=0.95,
    ):
        self.rng = np.random.default_rng(seed)

        # constants / settings
        self.sigma = 5.67e-8
        self.specific_heat = 900.0

        self.ambient_temp = float(ambient_temp_c)
        self.max_temp = float(max_temp_c)
        self.target_temp = float(target_temp_c)

        self.initial_mass = float(initial_weight_kg)
        self.current_mass = float(initial_weight_kg)
        self.max_capacity = float(max_capacity_kg)

        self.dt = int(dt_sec)
        self.max_time = int(max_time_min * 60)

        # scrap config
        self.scrap_addition_start = float(scrap_addition_start)
        self.scrap_addition_end = float(scrap_addition_end)
        self.use_fixed_scrap_schedule = bool(use_fixed_scrap_schedule)
        self.fixed_scrap_schedule = fixed_scrap_schedule or []
        self.scrap_additions = []
        self.total_scrap_added = 0.0
        self.scrap_temp_drop_scale = float(scrap_temp_drop_scale)
        self.scrap_recent_scale = float(scrap_recent_scale)
        self.scrap_recent_window_sec = int(scrap_recent_window_min * 60)

        # start mode
        self.start_mode = str(start_mode).lower()
        self.idle_time_min = float(idle_time_min)

        # power handling
        self.overall_eff = float(np.clip(overall_efficiency, 0.05, 0.98))
        self.use_time_efficiency = bool(use_time_efficiency)
        self.time_efficiency_schedule = (
            list(time_efficiency_schedule) if time_efficiency_schedule else []
        )

        w_m = max(0.0, float(eff_to_metal))
        w_w = max(0.0, float(eff_to_wall))
        w_sum = max(w_m + w_w, 1e-9)
        self.split_metal = w_m / w_sum
        self.split_wall = w_w / w_sum

        # coupling params
        self.k_wall_metal_cold = float(k_wall_metal)
        self.k_wall_metal_hot = float(hot_k_wall_metal)
        self.k_dT_alpha = float(k_dT_alpha)
        self.hot_k_early_factor = float(hot_k_early_factor)
        self.hot_early_minutes = int(hot_early_minutes)
        self.k_reduce_when_recent_scrap = float(k_reduce_when_recent_scrap)
        self.C_wall = float(wall_heat_capacity_J_per_K)

        # losses
        self.wall_area = float(wall_area_m2)
        self.wall_h = float(wall_h_W_m2K)
        self.wall_eps = float(wall_emissivity)
        self.metal_area = float(metal_area_m2)
        self.metal_h = float(metal_h_W_m2K)
        self.metal_eps_cold = float(metal_emissivity)
        self.metal_eps_hot = float(hot_metal_emissivity)

        # latent
        self.melting_point = float(melting_point_c)
        self.latent_band = float(latent_band_c)
        self.latent_scale = float(latent_scale)
        self.post_melt_scale = float(post_melt_scale)

        # start heuristics
        if wall_temp_c is not None:
            self.wall_temp_c0 = float(wall_temp_c)
        else:
            if self.start_mode == "cold":
                self.wall_temp_c0 = self.ambient_temp
            else:
                decay = np.exp(-self.idle_time_min / max(hot_wall_tau_min, 1e-6))
                self.wall_temp_c0 = self.ambient_temp + float(hot_wall_delta_c) * decay

        if initial_metal_temp_c is not None:
            self.metal_temp_c0 = float(initial_metal_temp_c)
        else:
            if self.start_mode == "cold":
                self.metal_temp_c0 = self.ambient_temp
            else:
                self.metal_temp_c0 = max(
                    self.ambient_temp,
                    self.ambient_temp
                    + (self.wall_temp_c0 - self.ambient_temp)
                    * float(hot_metal_fraction_of_wall),
                )

        # --- NEW ACTION SPACE ---
        # ซอยกำลังไฟเป็น 10 ระดับ ตั้งแต่ 0 ถึง 450 kW (0, 50, 100, ..., 450)
        self.max_power_kw = 450.0
        power_levels = np.linspace(0, self.max_power_kw, 10)
        self.action_space = {i: float(power_levels[i]) for i in range(10)}

        # state
        self.state = {
            "temperature": float(self.metal_temp_c0),
            "furnace_wall_temp": float(self.wall_temp_c0),
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,
            "status": 0,
            "energy_consumption": 0.0,
            "scrap_added": 0.0,
            "last_scrap_time": 0.0,
        }

        self._dT_wall = 0.0
        self.energy_consumption_scale = float(max(0.0, energy_consumption_scale))
        self.auxiliary_power_kw = float(max(0.0, auxiliary_power_kw))
        self.high_power_eff_start_kw = float(max(0.0, high_power_eff_start_kw))
        self.high_power_eff_end_kw = float(
            max(self.high_power_eff_start_kw, high_power_eff_end_kw)
        )
        self.high_power_eff_min = float(np.clip(high_power_eff_min, 0.3, 1.0))

        self._prev_progress = 0.0

    def _progress(self):
        denom = max(1e-9, float(self.target_temp - self.ambient_temp))
        p = (float(self.state["temperature"]) - float(self.ambient_temp)) / denom
        return float(np.clip(p, 0.0, 1.0))

    def reset(self):
        self.current_mass = float(self.initial_mass)
        self.scrap_additions = []
        self.total_scrap_added = 0.0

        self.state = {
            "temperature": float(self.metal_temp_c0),
            "furnace_wall_temp": float(self.wall_temp_c0),
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,
            "status": 1,
            "energy_consumption": 0.0,
            "scrap_added": 0.0,
            "last_scrap_time": 0.0,
        }

        self._prev_progress = self._progress()

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

    # ----------------------------
    # EXPERT PROFILE GUIDANCE
    # ----------------------------
    def _expert_power_profile(self, t_min: float):
        """จำลอง Profile ในอุดมคติจากประสบการณ์ 30 ปี"""
        if 0 <= t_min < 5:
            return 50.0
        elif 5 <= t_min < 10:
            return 150.0
        elif 10 <= t_min < 15:
            return 250.0
        elif 15 <= t_min < 30:
            return 350.0
        elif 30 <= t_min < 31:  # ช่วงใส่ Si+Fe (นาทีที่ 30-31)
            return 0.0
        elif 31 <= t_min < 35:
            return 400.0
        else:
            return self.max_power_kw

    # ----------------------------
    # Scrap & Thermal Physics (เหมือนเดิม 100%)
    # ----------------------------
    def add_scrap(self):
        current_time = float(self.state["time"])
        if self.use_fixed_scrap_schedule:
            for t_sec, w_kg in self.fixed_scrap_schedule:
                already = any(
                    abs(s["time"] - float(t_sec)) < 1e-6 for s in self.scrap_additions
                )
                if (not already) and abs(current_time - float(t_sec)) <= self.dt / 2:
                    return self._apply_scrap(float(w_kg), float(t_sec))
            return False

        if (
            self.scrap_addition_start <= current_time <= self.scrap_addition_end
            and len(self.scrap_additions) < 3
        ):
            time_since_last = current_time - float(self.state["last_scrap_time"])
            if len(self.scrap_additions) == 0 or time_since_last >= 5 * 60:
                if len(self.scrap_additions) == 0:
                    scrap_weight = float(self.rng.uniform(40, 70))
                elif len(self.scrap_additions) == 1:
                    scrap_weight = float(self.rng.uniform(50, 80))
                else:
                    remaining = self.max_capacity - self.current_mass
                    scrap_weight = float(min(remaining, self.rng.uniform(40, 70)))
                return self._apply_scrap(scrap_weight, current_time)
        return False

    def _apply_scrap(self, scrap_weight: float, scrap_time: float):
        if self.current_mass + scrap_weight > self.max_capacity:
            return False
        self.current_mass += scrap_weight
        self.total_scrap_added += scrap_weight
        self.scrap_additions.append({"time": scrap_time, "weight": scrap_weight})
        self.state["scrap_added"] = float(self.total_scrap_added)
        self.state["last_scrap_time"] = float(scrap_time)
        self.state["weight"] = float(self.current_mass)

        Tm = float(self.state["temperature"])
        temp_drop = (
            (scrap_weight / self.current_mass)
            * (Tm - self.ambient_temp)
            * self.scrap_temp_drop_scale
        )
        self.state["temperature"] = float(max(self.ambient_temp, Tm - temp_drop))
        return True

    def _wall_losses(self, Tw: float):
        TwK, TaK = Tw + 273.15, self.ambient_temp + 273.15
        Q_conv = self.wall_h * self.wall_area * max(0.0, (Tw - self.ambient_temp))
        Q_rad = (
            self.wall_eps * self.sigma * self.wall_area * max(0.0, (TwK**4 - TaK**4))
        )
        return Q_conv + Q_rad

    def _metal_losses(self, Tm: float):
        eps = self.metal_eps_hot if self.start_mode == "hot" else self.metal_eps_cold
        TmK, TaK = Tm + 273.15, self.ambient_temp + 273.15
        Q_conv = self.metal_h * self.metal_area * max(0.0, (Tm - self.ambient_temp))
        Q_rad = eps * self.sigma * self.metal_area * max(0.0, (TmK**4 - TaK**4))
        return Q_conv + Q_rad

    def _k_wall_metal_eff(self, Tw: float, Tm: float, t_min: float):
        k0 = (
            self.k_wall_metal_hot
            if self.start_mode == "hot"
            else self.k_wall_metal_cold
        )
        if self.start_mode == "hot" and t_min < self.hot_early_minutes:
            k0 *= self.hot_k_early_factor
        k_eff = k0 / (1.0 + self.k_dT_alpha * abs(Tw - Tm))
        if self.scrap_additions:
            recent = [
                s
                for s in self.scrap_additions
                if float(self.state["time"]) - s["time"] < self.scrap_recent_window_sec
            ]
            if recent:
                k_eff *= self.k_reduce_when_recent_scrap
        return float(max(0.0, k_eff))

    def _get_overall_efficiency(self, t_min: float):
        if not self.use_time_efficiency or not self.time_efficiency_schedule:
            return float(self.overall_eff)
        schedule = sorted(self.time_efficiency_schedule, key=lambda x: x[0])
        if t_min <= schedule[0][0]:
            return float(np.clip(schedule[0][1], 0.05, 0.98))
        if t_min >= schedule[-1][0]:
            return float(np.clip(schedule[-1][1], 0.05, 0.98))
        for (t0, e0), (t1, e1) in zip(schedule[:-1], schedule[1:]):
            if t0 <= t_min <= t1:
                eff = e1 if t1 == t0 else e0 + ((t_min - t0) / (t1 - t0)) * (e1 - e0)
                return float(np.clip(eff, 0.05, 0.98))
        return float(self.overall_eff)

    def _power_efficiency_factor(self, power_kw: float):
        if power_kw <= self.high_power_eff_start_kw:
            return 1.0
        if self.high_power_eff_end_kw <= self.high_power_eff_start_kw:
            return float(self.high_power_eff_min)
        alpha = np.clip(
            (power_kw - self.high_power_eff_start_kw)
            / (self.high_power_eff_end_kw - self.high_power_eff_start_kw),
            0.0,
            1.0,
        )
        return float(1.0 + alpha * (self.high_power_eff_min - 1.0))

    def calculate_temperature_change(self):
        dt, Tm, Tw = (
            float(self.dt),
            float(self.state["temperature"]),
            float(self.state["furnace_wall_temp"]),
        )
        t_min = float(self.state["time"]) / 60.0
        P_watt = float(self.state["power"]) * 1000.0

        eff_total = self._get_overall_efficiency(t_min) * self._power_efficiency_factor(
            self.state["power"]
        )
        Q_total = P_watt * eff_total

        split_m, split_w = self.split_metal, self.split_wall
        if self.start_mode == "hot" and t_min < self.hot_early_minutes:
            split_m = max(0.0, split_m - 0.08)
            split_w = 1.0 - split_m

        Q_to_m, Q_to_w = Q_total * split_m, Q_total * split_w
        Q_wm = self._k_wall_metal_eff(Tw, Tm, t_min) * (Tw - Tm)

        Q_loss_w, Q_loss_m = self._wall_losses(Tw), self._metal_losses(Tm)

        net_m = Q_to_m + Q_wm - Q_loss_m
        dT_m = (net_m * dt) / (float(self.current_mass) * float(self.specific_heat))

        if (
            (self.melting_point - self.latent_band)
            < Tm
            < (self.melting_point + self.latent_band)
        ):
            dT_m *= self.latent_scale
        elif Tm >= self.melting_point + self.latent_band:
            dT_m *= self.post_melt_scale

        if self.scrap_additions and [
            s
            for s in self.scrap_additions
            if float(self.state["time"]) - s["time"] < self.scrap_recent_window_sec
        ]:
            dT_m *= self.scrap_recent_scale

        net_w = Q_to_w - Q_wm - Q_loss_w
        self._dT_wall = float((net_w * dt) / self.C_wall)

        return float(dT_m)

    # ----------------------------
    # New Step & Reward Function
    # ----------------------------
    def step(self, action, power_profile_kw=None):
        _ = self.add_scrap()
        current_minute = float(self.state["time"]) / 60.0

        # 1. รับค่า Action ที่เป็นตัวเลขกำลังไฟโดยตรง
        if self.state["status"] != 1:
            target_power = 0.0
        else:
            if power_profile_kw is not None:
                target_power = float(power_profile_kw(current_minute))
            else:
                # แปลง action index (0-9) ให้กลายเป็นค่าไฟ (0-450 kW)
                target_power = self.action_space.get(int(action), 0.0)

        # 2. ปรับกำลังไฟเข้า State
        is_alloying_time = 30.0 <= current_minute < 31.0  # ช่วงเวลาใส่ Si+Fe

        if is_alloying_time:
            self.state["power"] = 0.0  # Safety Override: บังคับตัดไฟ
        else:
            self.state["power"] = float(np.clip(target_power, 0.0, self.max_power_kw))

        # 3. อัปเดตอุณหภูมิและเวลาตามหลักฟิสิกส์
        if self.state["status"] == 1:
            dT_m = self.calculate_temperature_change()
            self.state["temperature"] = float(
                np.clip(
                    self.state["temperature"] + dT_m, self.ambient_temp, self.max_temp
                )
            )
            self.state["furnace_wall_temp"] = float(
                max(self.ambient_temp, self.state["furnace_wall_temp"] + self._dT_wall)
            )

        self.state["time"] += float(self.dt)
        metered_power_kw = float(self.state["power"]) + self.auxiliary_power_kw
        energy_used_kwh = (
            float((metered_power_kw * self.dt) / 3600.0) * self.energy_consumption_scale
        )
        self.state["energy_consumption"] += energy_used_kwh
        self.state["weight"] = float(self.current_mass)

        # 4. เช็คสถานะการจบ Episode
        done = False
        success = False
        if self.state["temperature"] >= self.target_temp:
            done = True
            success = True
        elif (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["status"] == 0
        ):
            done = True

        # ==========================================================
        # 5. คำนวณ Reward แบบผสม (Expert Profile + Efficiency)
        # ==========================================================
        reward = 0.0

        # 5.1 Expert Tracking Penalty (หักลบคะแนนหากจ่ายไฟไม่เหมือนผู้เชี่ยวชาญ)
        expert_pwr = self._expert_power_profile(current_minute)
        power_error = abs(target_power - expert_pwr)
        tracking_penalty = -(power_error / self.max_power_kw) * 0.5
        reward += tracking_penalty

        # 5.2 Safety Penalty (ลงโทษหนักมากหากจ่ายไฟตอนคนงานใส่ส่วนผสม)
        if is_alloying_time and target_power > 0.0:
            reward -= 5.0

        # 5.3 Progress Reward (ให้คะแนนตามระดับความร้อนที่ไต่ขึ้นไป)
        prog = self._progress()
        dprog = prog - float(self._prev_progress)
        self._prev_progress = prog
        reward += dprog * 2.0

        # 5.4 Energy Efficiency Penalty (หักคะแนนตามพลังงานที่ใช้จริงเพื่อบังคับให้ประหยัด)
        reward -= energy_used_kwh * 0.02  # ปรับตัวคูณเพื่อเร่งให้ประหยัดขึ้น

        # 5.5 Time Penalty (นิดหน่อย เพื่อกระตุ้นให้จบงาน)
        reward -= 0.001

        # 5.6 Terminal Reward (โบนัส/บทลงโทษตอนจบ)
        if done:
            if success:
                # คำนวณประสิทธิภาพ กิโลกรัม/หน่วยไฟ (ยิ่งมากยิ่งดี)
                efficiency = self.state["weight"] / max(
                    1e-6, self.state["energy_consumption"]
                )
                optimal_efficiency = 1.8  # ตัวเลขเป้าหมายคร่าวๆ จากโค้ดเดิม
                norm_eff = min(efficiency / optimal_efficiency, 1.0)

                # โบนัสก้อนใหญ่เมื่อสำเร็จ + โบนัสความประหยัด
                reward += 10.0 + (norm_eff * 5.0)
            else:
                reward -= 5.0  # หักคะแนนถ้าหมดเวลาแล้วยังไม่ถึง 950 องศา

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

        return state_array, float(reward), done

    def get_scrap_info(self):
        return {
            "total_scrap_added": float(self.total_scrap_added),
            "current_mass": float(self.current_mass),
            "scrap_additions": list(self.scrap_additions),
            "capacity_utilization": float(self.current_mass / self.max_capacity),
        }
