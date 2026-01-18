import numpy as np


class AluminumMeltingEnvironment:
    """
    Improved thermal simulation with a 2-node model:
      - Metal/bath temperature (T_m)
      - Furnace wall/lining temperature (T_w)

    Key fix for HOT start:
      - Explicit wall-to-metal heat transfer term Q_wm = k_wm*(T_w - T_m)
      - Separate wall losses to ambient (avoid double counting losses from metal)
      - Wall thermal capacitance C_w provides realistic thermal inertia / residual heat

    Notes:
      - This class focuses on temperature/energy dynamics for validation & RL.
      - For "fit to real data", strongly recommend using:
            use_fixed_scrap_schedule=True
        and feeding a fixed power profile (see optional power_profile in step()).
    """

    def __init__(
        self,
        initial_weight_kg=350,
        target_temp_c=900,
        scrap_addition_start=60 * 60,
        scrap_addition_end=75 * 60,
        start_mode="cold",  # "cold" or "hot"
        idle_time_min=0,  # idle time from previous batch
        wall_temp_c=None,  # preset wall temperature if known
        initial_metal_temp_c=None,  # preset initial metal temp if known
        seed=None,
        use_fixed_scrap_schedule=False,
        fixed_scrap_schedule=None,  # list of (time_sec, weight_kg)
        # ---- thermal parameters (tunable) ----
        # electrical power split
        eff_to_metal=0.90,  # fraction of electrical power to metal (induction coupling)
        eff_to_wall=0.45,  # fraction of electrical power to wall/lining
        # wall-metal coupling and inertia
        k_wall_metal=1800.0,  # W/K
        wall_heat_capacity_J_per_K=2.5e6,  # J/K (effective)
        # wall heat loss params
        wall_area_m2=3.5,
        wall_h_W_m2K=18.0,
        wall_emissivity=0.55,
        # metal loss params (keep small if wall loss is modeled)
        metal_area_m2=1.0,
        metal_h_W_m2K=5.0,
        metal_emissivity=0.10,
        # latent heat handling (simple band)
        melting_point_c=660.0,
        latent_band_c=50.0,
        latent_scale=0.25,  # scale dT in latent band
        post_melt_scale=0.80,
        # scrap cooling strength
        scrap_temp_drop_scale=0.50,
        scrap_recent_scale=0.80,
        scrap_recent_window_min=5,
        # simulation timing
        dt_sec=60,
        max_time_min=120,
        max_capacity_kg=500,
        ambient_temp_c=25,
        max_temp_c=1000,
    ):
        # RNG
        self.rng = np.random.default_rng(seed)

        # Physical constants
        self.specific_heat = 900.0  # J/kg·K (effective constant cp for simplicity)
        self.ambient_temp = float(ambient_temp_c)
        self.max_temp = float(max_temp_c)
        self.target_temp = float(target_temp_c)

        # Mass / capacity
        self.initial_mass = float(initial_weight_kg)
        self.current_mass = float(initial_weight_kg)
        self.max_capacity = float(max_capacity_kg)

        # Scrap addition params
        self.scrap_addition_start = float(scrap_addition_start)
        self.scrap_addition_end = float(scrap_addition_end)
        self.use_fixed_scrap_schedule = bool(use_fixed_scrap_schedule)
        self.fixed_scrap_schedule = fixed_scrap_schedule or []
        self.scrap_additions = []
        self.total_scrap_added = 0.0
        self.scrap_temp_drop_scale = float(scrap_temp_drop_scale)
        self.scrap_recent_scale = float(scrap_recent_scale)
        self.scrap_recent_window_sec = int(scrap_recent_window_min * 60)

        # Start condition
        self.start_mode = start_mode
        self.idle_time_min = float(idle_time_min)

        # Estimate wall temperature at t=0 if not provided
        if wall_temp_c is not None:
            self.wall_temp_c0 = float(wall_temp_c)
        else:
            if start_mode == "cold":
                self.wall_temp_c0 = self.ambient_temp
            else:
                # heuristic: wall cools exponentially with idle time
                decay = np.exp(-self.idle_time_min / 120.0)
                # 250C above ambient at idle=0; adjust if needed
                self.wall_temp_c0 = self.ambient_temp + 250.0 * decay

        # Estimate initial metal temperature at t=0 if not provided
        if initial_metal_temp_c is not None:
            self.metal_temp_c0 = float(initial_metal_temp_c)
        else:
            if start_mode == "cold":
                self.metal_temp_c0 = self.ambient_temp
            else:
                # heuristic: metal starts warmer than ambient in hot start
                # but not as hot as wall; tune factor if needed
                self.metal_temp_c0 = max(self.ambient_temp, self.wall_temp_c0 * 0.60)

        # Thermal parameters
        self.eff_to_metal = float(np.clip(eff_to_metal, 0.05, 0.95))
        self.eff_to_wall = float(np.clip(eff_to_wall, 0.00, 0.60))
        self.k_wall_metal = float(k_wall_metal)
        self.C_wall = float(wall_heat_capacity_J_per_K)

        self.wall_area = float(wall_area_m2)
        self.wall_h = float(wall_h_W_m2K)
        self.wall_eps = float(wall_emissivity)

        self.metal_area = float(metal_area_m2)
        self.metal_h = float(metal_h_W_m2K)
        self.metal_eps = float(metal_emissivity)

        self.sigma = 5.67e-8

        # Latent handling
        self.melting_point = float(melting_point_c)
        self.latent_band = float(latent_band_c)
        self.latent_scale = float(latent_scale)
        self.post_melt_scale = float(post_melt_scale)

        # Time settings
        self.dt = int(dt_sec)
        self.max_time = int(max_time_min * 60)

        # Action space (for RL usage)
        self.action_space = {
            0: "increase_power_strong",
            1: "increase_power_mild",
            2: "maintain",
            3: "decrease_power_mild",
            4: "decrease_power_strong",
        }

        # State
        self.state = {
            "temperature": float(self.metal_temp_c0),  # metal temperature
            "furnace_wall_temp": float(self.wall_temp_c0),  # wall temperature
            "weight": float(self.current_mass),
            "time": 0.0,
            "power": 0.0,  # kW
            "status": 0,
            "energy_consumption": 0.0,  # kWh
            "scrap_added": 0.0,
            "last_scrap_time": 0.0,
        }

        # Internal temp update from energy balance for wall
        self._dT_wall = 0.0

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
    # Scrap handling
    # ----------------------------
    def add_scrap(self):
        """Add scrap between scrap_addition_start and scrap_addition_end."""
        current_time = float(self.state["time"])

        # Deterministic schedule (recommended for validation)
        if self.use_fixed_scrap_schedule:
            for t_sec, w_kg in self.fixed_scrap_schedule:
                already = any(
                    abs(s["time"] - t_sec) < 1e-6 for s in self.scrap_additions
                )
                if (not already) and abs(current_time - t_sec) <= self.dt / 2:
                    return self._apply_scrap(float(w_kg), float(t_sec))
            return False

        # Random schedule (for RL training variety)
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

        # Mix/cooling effect (simple)
        # bigger drop when metal is much hotter than ambient; scaled by scrap fraction
        Tm = float(self.state["temperature"])
        temp_drop = (
            (scrap_weight / self.current_mass)
            * (Tm - self.ambient_temp)
            * self.scrap_temp_drop_scale
        )
        self.state["temperature"] = float(max(self.ambient_temp, Tm - temp_drop))

        return True

    # ----------------------------
    # Thermal model (2-node)
    # ----------------------------
    def _wall_losses(self, Tw: float):
        Ta = self.ambient_temp
        Aw = self.wall_area
        hw = self.wall_h
        eps = self.wall_eps
        sigma = self.sigma

        TwK = Tw + 273.15
        TaK = Ta + 273.15
        Q_conv = hw * Aw * (Tw - Ta)
        Q_rad = eps * sigma * Aw * (TwK**4 - TaK**4)
        return Q_conv + Q_rad  # W

    def _metal_losses(self, Tm: float):
        # Keep small to avoid double counting; still allows some effect if needed.
        Ta = self.ambient_temp
        Am = self.metal_area
        hm = self.metal_h
        eps = self.metal_eps
        sigma = self.sigma

        TmK = Tm + 273.15
        TaK = Ta + 273.15
        Q_conv = hm * Am * (Tm - Ta)
        Q_rad = eps * sigma * Am * (TmK**4 - TaK**4)
        return Q_conv + Q_rad  # W

    def calculate_temperature_change(self):
        dt = float(self.dt)
        Tm = float(self.state["temperature"])
        Tw = float(self.state["furnace_wall_temp"])
        Ta = self.ambient_temp

        P_watt = float(self.state["power"]) * 1000.0  # W

        # (1) Power split
        Q_to_m = P_watt * self.eff_to_metal
        Q_to_w = P_watt * self.eff_to_wall

        # (2) Wall-metal coupling (core hot-start mechanism)
        Q_wm = self.k_wall_metal * (Tw - Tm)  # W (+ adds to metal when wall hotter)

        # (3) Losses
        Q_loss_w = self._wall_losses(Tw)
        Q_loss_m = self._metal_losses(Tm)

        # (4) Metal energy balance
        m = float(self.current_mass)
        cp = float(self.specific_heat)

        net_m = Q_to_m + Q_wm - Q_loss_m
        dT_m = (net_m * dt) / (m * cp)

        # latent band scaling
        if (Tm > self.melting_point - self.latent_band) and (
            Tm < self.melting_point + self.latent_band
        ):
            dT_m *= self.latent_scale
        elif Tm >= self.melting_point + self.latent_band:
            dT_m *= self.post_melt_scale

        # recent scrap slows heating for a short time window
        if self.scrap_additions:
            recent = [
                s
                for s in self.scrap_additions
                if float(self.state["time"]) - s["time"] < self.scrap_recent_window_sec
            ]
            if recent:
                dT_m *= self.scrap_recent_scale

        # (5) Wall energy balance
        net_w = Q_to_w - Q_wm - Q_loss_w
        dT_w = (net_w * dt) / self.C_wall
        self._dT_wall = float(dT_w)

        return float(dT_m)

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, action, power_profile_kw=None):
        """
        If power_profile_kw is provided:
            - power is overridden by power_profile_kw(time_minute)->kW
            - action is ignored (useful for validation against real data)
        """
        # Add scrap if scheduled
        _ = self.add_scrap()

        # Power update
        if self.state["status"] != 1:
            self.state["power"] = 0.0
        else:
            if power_profile_kw is not None:
                # fixed power profile for validation
                t_min = float(self.state["time"]) / 60.0
                self.state["power"] = float(power_profile_kw(t_min))
            else:
                # RL action-based power changes
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

        # Optional: keep your phase constraints here if needed.
        # IMPORTANT: for validation, ensure constraints match real operation.
        # Example (same as your old logic):
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

        # Update temperatures
        if self.state["status"] == 1:
            dT_m = self.calculate_temperature_change()
            self.state["temperature"] = float(
                np.clip(
                    self.state["temperature"] + dT_m, self.ambient_temp, self.max_temp
                )
            )
            # wall update from energy balance
            self.state["furnace_wall_temp"] = float(
                max(self.ambient_temp, self.state["furnace_wall_temp"] + self._dT_wall)
            )

        # Time update
        self.state["time"] = float(self.state["time"]) + float(self.dt)

        # Energy consumption (kWh)
        self.state["energy_consumption"] += float(
            (self.state["power"] * self.dt) / 3600.0
        )

        # Weight update
        self.state["weight"] = float(self.current_mass)

        # Done conditions
        done = False
        if (
            self.state["time"] >= self.max_time
            or self.state["temperature"] >= self.max_temp
            or self.state["temperature"] >= self.target_temp
            or self.state["status"] == 0
        ):
            done = True

        reward = self.calculate_reward() if done else 0.0

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
    # Reward (kept similar)
    # ----------------------------
    def calculate_reward(self):
        # Temperature component
        if self.state["temperature"] >= self.target_temp:
            temp_component = 1.0
        else:
            temp_component = -1.0 + 2.0 * (
                (self.state["temperature"] - self.ambient_temp)
                / (self.target_temp - self.ambient_temp + 1e-9)
            )

        # Energy efficiency component
        if self.state["energy_consumption"] > 1e-6:
            efficiency = self.state["weight"] / self.state["energy_consumption"]
        else:
            efficiency = 0.0
        optimal_efficiency = 1.8
        norm_eff = min(efficiency / optimal_efficiency, 1.0)

        # Weights
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
