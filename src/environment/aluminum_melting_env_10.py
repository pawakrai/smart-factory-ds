import numpy as np


class AluminumMeltingEnvironment:
    """
    2-node thermal model (Metal Tm + Wall Tw) with better cold/hot behavior.

    Key changes (to fix HOT start being too fast):
    1) Prevent "power double counting":
        - eff_to_metal / eff_to_wall are treated as SPLIT WEIGHTS (not efficiencies),
          then normalized so total power split <= 100%.
        - overall_efficiency controls how much electrical power becomes useful heat.
    2) Hot-start tuned dynamics:
        - k_wall_metal is reduced for hot start (and made state-dependent so it doesn't
          transfer unrealistically large heat when Tw>>Tm).
        - early hot-start coupling can be further reduced in first minutes.
        - metal losses are slightly higher in hot start (radiation grows fast in reality).
    3) Optional time-dependent effective metal-coupling / power split for hot start:
        - early period may have lower coupling (ramp/hold / non-ideal induction coupling).
    4) Optional: reduce wall residual heat estimate (Tw0) with faster decay / smaller delta.

    NOTE:
      - For data-fitting, use_fixed_scrap_schedule=True and a fixed power_profile_kw.
      - For RL, omit power_profile_kw and use actions.
    """

    def __init__(
        self,
        initial_weight_kg=350,
        target_temp_c=950,
        scrap_addition_start=60 * 60,
        scrap_addition_end=75 * 60,
        start_mode="hot",  # "cold" or "hot"
        idle_time_min=0,  # idle time from previous batch
        wall_temp_c=None,  # preset wall temperature if known
        initial_metal_temp_c=None,  # preset initial metal temp if known
        seed=None,
        use_fixed_scrap_schedule=False,
        fixed_scrap_schedule=None,  # list of (time_sec, weight_kg)
        # ---------- power handling ----------
        overall_efficiency=0.9326,  # fraction of electrical power that becomes useful heat (0..1)
        eff_to_metal=0.8726,  # SPLIT WEIGHT to metal (will be normalized)
        eff_to_wall=0.0269,  # SPLIT WEIGHT to wall  (will be normalized)
        use_time_efficiency=True,
        time_efficiency_schedule=[
            (0, 0.80),
            (30, 0.80),  # จบช่วงแรก
            (30, 0.90),  # เริ่มช่วงใหม่แบบ step
            (40, 0.90),
            (40, 0.95),
            (60, 0.95),
            (60, 0.50),
            (70, 0.50),
            (70, 0.98),
            # (70, 0.98),
            (90, 0.98),  # ต่อไปยาวๆ
        ],
        # ---------- wall-metal coupling ----------
        k_wall_metal=800,  # W/K base coupling (cold). hot override below
        k_dT_alpha=0.0003,  # state-dependent reduction: k_eff = k / (1 + alpha*|Tw-Tm|)
        hot_k_wall_metal=1100.0,  # W/K base coupling in hot start (main hot-start fix)
        hot_k_early_factor=0.80,  # additional factor for first hot minutes (e.g., 0-10 min)
        hot_early_minutes=100,  # minutes
        wall_heat_capacity_J_per_K=2.5e6,  # J/K effective wall inertia
        # ---------- wall losses ----------
        wall_area_m2=3.5,
        wall_h_W_m2K=18.0,
        wall_emissivity=0.55,
        # ---------- metal losses (keep modest to avoid double counting) ----------
        metal_area_m2=1.0,
        metal_h_W_m2K=2.79,
        metal_emissivity=0.005,  # cold default
        hot_metal_emissivity=0.18,  # slightly higher in hot start (helps slow hot curve)
        # ---------- latent heat handling ----------
        melting_point_c=660.0,
        latent_band_c=50.0,
        latent_scale=0.324,
        post_melt_scale=0.803,
        # ---------- scrap effects ----------
        scrap_temp_drop_scale=0.50,
        scrap_recent_scale=0.80,
        scrap_recent_window_min=5,
        k_reduce_when_recent_scrap=0.75,  # reduce wall-metal coupling briefly after scrap
        # ---------- start condition heuristic ----------
        hot_wall_delta_c=200.0,  # how much above ambient wall could be at idle=0
        hot_wall_tau_min=60.0,  # faster cooling than 120 min (reduces residual heat)
        hot_metal_fraction_of_wall=0.70,  # metal initial temp relative to wall in hot start
        # ---------- simulation timing ----------
        dt_sec=60,
        max_time_min=120,
        max_capacity_kg=500,
        ambient_temp_c=25,
        max_temp_c=1000,
        # ---------- energy metering ----------
        energy_consumption_scale=1.068,  # scale factor for electrical kWh meter
        auxiliary_power_kw=0.0,  # constant auxiliary load (e.g., fans, pumps)
        # ---------- high-power efficiency drop ----------
        high_power_eff_start_kw=450.0,  # start reducing thermal efficiency above this
        high_power_eff_end_kw=500.0,  # efficiency reaches min at this power
        high_power_eff_min=0.95,  # minimum multiplier for thermal efficiency
        # ---------- RL reward shaping ----------
        dense_reward=True,  # if True, emit step rewards (default keeps legacy terminal-only)
        dense_reward_w_progress=2.0,
        dense_reward_w_power=0.05,
        dense_reward_w_time=0.002,
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

        # treat eff_to_metal/eff_to_wall as weights and normalize
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
                # faster decay + smaller delta -> less aggressive hot boost
                decay = np.exp(-self.idle_time_min / max(hot_wall_tau_min, 1e-6))
                self.wall_temp_c0 = self.ambient_temp + float(hot_wall_delta_c) * decay

        if initial_metal_temp_c is not None:
            self.metal_temp_c0 = float(initial_metal_temp_c)
        else:
            if self.start_mode == "cold":
                self.metal_temp_c0 = self.ambient_temp
            else:
                # metal starts warmer than ambient, but not too high
                self.metal_temp_c0 = max(
                    self.ambient_temp,
                    self.ambient_temp
                    + (self.wall_temp_c0 - self.ambient_temp)
                    * float(hot_metal_fraction_of_wall),
                )

        # RL action space
        self.action_space = {
            0: "increase_power_strong",
            1: "increase_power_mild",
            2: "maintain",
            3: "decrease_power_mild",
            4: "decrease_power_strong",
        }

        # state
        self.state = {
            "temperature": float(self.metal_temp_c0),
            "furnace_wall_temp": float(self.wall_temp_c0),
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,  # kW
            "status": 0,
            "energy_consumption": 0.0,  # kWh
            "scrap_added": 0.0,
            "last_scrap_time": 0.0,
        }

        self._dT_wall = 0.0

        # energy metering
        self.energy_consumption_scale = float(max(0.0, energy_consumption_scale))
        self.auxiliary_power_kw = float(max(0.0, auxiliary_power_kw))

        # high-power efficiency drop
        self.high_power_eff_start_kw = float(max(0.0, high_power_eff_start_kw))
        self.high_power_eff_end_kw = float(
            max(self.high_power_eff_start_kw, high_power_eff_end_kw)
        )
        self.high_power_eff_min = float(np.clip(high_power_eff_min, 0.3, 1.0))

        # reward shaping
        self.dense_reward = bool(dense_reward)
        self.dense_reward_w_progress = float(dense_reward_w_progress)
        self.dense_reward_w_power = float(dense_reward_w_power)
        self.dense_reward_w_time = float(dense_reward_w_time)
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

        # Initialize progress tracker for dense reward shaping.
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
    # Scrap
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

        # simple mixing/cooling
        Tm = float(self.state["temperature"])
        temp_drop = (
            (scrap_weight / self.current_mass)
            * (Tm - self.ambient_temp)
            * self.scrap_temp_drop_scale
        )
        self.state["temperature"] = float(max(self.ambient_temp, Tm - temp_drop))
        return True

    # ----------------------------
    # Loss models
    # ----------------------------
    def _wall_losses(self, Tw: float):
        Ta = self.ambient_temp
        Aw = self.wall_area
        hw = self.wall_h
        eps = self.wall_eps

        TwK = Tw + 273.15
        TaK = Ta + 273.15
        Q_conv = hw * Aw * max(0.0, (Tw - Ta))
        Q_rad = eps * self.sigma * Aw * max(0.0, (TwK**4 - TaK**4))
        return Q_conv + Q_rad  # W

    def _metal_losses(self, Tm: float):
        Ta = self.ambient_temp
        Am = self.metal_area
        hm = self.metal_h

        eps = self.metal_eps_hot if self.start_mode == "hot" else self.metal_eps_cold

        TmK = Tm + 273.15
        TaK = Ta + 273.15
        Q_conv = hm * Am * max(0.0, (Tm - Ta))
        Q_rad = eps * self.sigma * Am * max(0.0, (TmK**4 - TaK**4))
        return Q_conv + Q_rad  # W

    # ----------------------------
    # Coupling: k_wall_metal effective
    # ----------------------------
    def _k_wall_metal_eff(self, Tw: float, Tm: float, t_min: float):
        # base by mode
        k0 = (
            self.k_wall_metal_hot
            if self.start_mode == "hot"
            else self.k_wall_metal_cold
        )

        # early hot-start reduction (helps prevent unrealistically fast rise)
        if self.start_mode == "hot" and t_min < self.hot_early_minutes:
            k0 *= self.hot_k_early_factor

        # state-dependent reduction when |Tw-Tm| is large
        dT = abs(Tw - Tm)
        k_eff = k0 / (1.0 + self.k_dT_alpha * dT)

        # if just added scrap, coupling effectively worse for a short period
        if self.scrap_additions:
            recent = [
                s
                for s in self.scrap_additions
                if float(self.state["time"]) - s["time"] < self.scrap_recent_window_sec
            ]
            if recent:
                k_eff *= self.k_reduce_when_recent_scrap

        return float(max(0.0, k_eff))

    # ----------------------------
    # Thermal update
    # ----------------------------
    def calculate_temperature_change(self):
        dt = float(self.dt)
        Tm = float(self.state["temperature"])
        Tw = float(self.state["furnace_wall_temp"])
        t_min = float(self.state["time"]) / 60.0

        P_watt = float(self.state["power"]) * 1000.0

        # (1) total usable heat from electrical power
        eff_total = self._get_overall_efficiency(t_min)
        eff_total *= self._power_efficiency_factor(self.state["power"])
        Q_total = P_watt * eff_total

        # (2) split to metal/wall (normalized weights)
        # Optional: in hot start early minutes, metal split can be reduced slightly
        split_m = self.split_metal
        split_w = self.split_wall

        if self.start_mode == "hot" and t_min < self.hot_early_minutes:
            # reduce metal split a bit early; keep sum=1
            split_m = max(0.0, split_m - 0.08)
            split_w = 1.0 - split_m

        Q_to_m = Q_total * split_m
        Q_to_w = Q_total * split_w

        # (3) coupling wall <-> metal
        k_eff = self._k_wall_metal_eff(Tw, Tm, t_min)
        Q_wm = k_eff * (Tw - Tm)  # + means wall heats metal

        # (4) losses
        Q_loss_w = self._wall_losses(Tw)
        Q_loss_m = self._metal_losses(Tm)

        # (5) metal balance
        m = float(self.current_mass)
        cp = float(self.specific_heat)

        net_m = Q_to_m + Q_wm - Q_loss_m
        dT_m = (net_m * dt) / (m * cp)

        # latent handling
        if (Tm > self.melting_point - self.latent_band) and (
            Tm < self.melting_point + self.latent_band
        ):
            dT_m *= self.latent_scale
        elif Tm >= self.melting_point + self.latent_band:
            dT_m *= self.post_melt_scale

        # recent scrap slows heating for a short window
        if self.scrap_additions:
            recent = [
                s
                for s in self.scrap_additions
                if float(self.state["time"]) - s["time"] < self.scrap_recent_window_sec
            ]
            if recent:
                dT_m *= self.scrap_recent_scale

        # (6) wall balance
        net_w = Q_to_w - Q_wm - Q_loss_w
        dT_w = (net_w * dt) / self.C_wall
        self._dT_wall = float(dT_w)

        return float(dT_m)

    # ----------------------------
    # Efficiency (optional schedule)
    # ----------------------------
    def _get_overall_efficiency(self, t_min: float):
        """Return effective efficiency, optionally from a time schedule."""
        if not self.use_time_efficiency or not self.time_efficiency_schedule:
            return float(self.overall_eff)

        schedule = sorted(self.time_efficiency_schedule, key=lambda x: x[0])
        if t_min <= schedule[0][0]:
            return float(np.clip(schedule[0][1], 0.05, 0.98))
        if t_min >= schedule[-1][0]:
            return float(np.clip(schedule[-1][1], 0.05, 0.98))

        for (t0, e0), (t1, e1) in zip(schedule[:-1], schedule[1:]):
            if t0 <= t_min <= t1:
                if t1 == t0:
                    eff = e1
                else:
                    alpha = (t_min - t0) / (t1 - t0)
                    eff = e0 + alpha * (e1 - e0)
                return float(np.clip(eff, 0.05, 0.98))

        return float(self.overall_eff)

    def _power_efficiency_factor(self, power_kw: float):
        """Extra efficiency drop at high electrical power."""
        if power_kw <= self.high_power_eff_start_kw:
            return 1.0
        if self.high_power_eff_end_kw <= self.high_power_eff_start_kw:
            return float(self.high_power_eff_min)

        alpha = (power_kw - self.high_power_eff_start_kw) / (
            self.high_power_eff_end_kw - self.high_power_eff_start_kw
        )
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(1.0 + alpha * (self.high_power_eff_min - 1.0))

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, action, power_profile_kw=None):
        # scrap
        _ = self.add_scrap()

        # power update
        if self.state["status"] != 1:
            self.state["power"] = 0.0
        else:
            if power_profile_kw is not None:
                t_min = float(self.state["time"]) / 60.0
                self.state["power"] = float(power_profile_kw(t_min))
            else:
                action_type = self.action_space.get(action, "maintain")
                if action_type == "increase_power_strong":
                    self.state["power"] = min(500.0, float(self.state["power"]) + 100.0)
                elif action_type == "increase_power_mild":
                    self.state["power"] = min(500.0, float(self.state["power"]) + 50.0)
                elif action_type == "maintain":
                    pass
                elif action_type == "decrease_power_mild":
                    self.state["power"] = max(0.0, float(self.state["power"]) - 50.0)
                elif action_type == "decrease_power_strong":
                    self.state["power"] = max(0.0, float(self.state["power"]) - 100.0)

        # phase constraints (keep consistent with your plant logic)
        current_minute = float(self.state["time"]) / 60.0
        if 0 <= current_minute < 5:
            max_power = 50
        elif 5 <= current_minute < 10:
            max_power = 150
        elif 10 <= current_minute < 15:
            max_power = 250
        elif 15 <= current_minute < 30:
            max_power = 350
        elif 30 <= current_minute < 31:
            max_power = 0
        elif 31 <= current_minute < 35:
            max_power = 400
        else:
            max_power = 450

        self.state["power"] = float(np.clip(self.state["power"], 0.0, float(max_power)))

        # update temperatures
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

        # time update
        self.state["time"] = float(self.state["time"]) + float(self.dt)

        # energy consumption
        metered_power_kw = float(self.state["power"]) + self.auxiliary_power_kw
        self.state["energy_consumption"] += (
            float((metered_power_kw * self.dt) / 3600.0) * self.energy_consumption_scale
        )

        # weight update
        self.state["weight"] = float(self.current_mass)

        # done
        done = False
        if (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["temperature"] >= self.target_temp
            or self.state["status"] == 0
        ):
            done = True

        shaped = 0.0
        if self.dense_reward:
            prog = self._progress()
            dprog = prog - float(self._prev_progress)
            self._prev_progress = prog
            power_pen = float(np.clip(metered_power_kw / 500.0, 0.0, 1.5))
            shaped = (
                self.dense_reward_w_progress * dprog
                - self.dense_reward_w_power * power_pen
                - self.dense_reward_w_time
            )
        reward = float(self.calculate_reward()) if done else 0.0
        reward = float(reward + shaped)

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

    # ----------------------------
    # Reward
    # ----------------------------
    def calculate_reward(self):
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            temp_component = -1.0 + 2.0 * (
                (self.state["temperature"] - self.ambient_temp)
                / (self.target_temp - self.ambient_temp + 1e-9)
            )

        if self.state["energy_consumption"] > 1e-6:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0

        optimal_efficiency = 1.8
        norm_eff = min(efficiency / optimal_efficiency, 1.0)

        w_temp = 0.70
        w_energy = 0.30
        return float(w_temp * temp_component + w_energy * norm_eff)

    def get_scrap_info(self):
        return {
            "total_scrap_added": float(self.total_scrap_added),
            "current_mass": float(self.current_mass),
            "scrap_additions": list(self.scrap_additions),
            "capacity_utilization": float(self.current_mass / self.max_capacity),
        }
