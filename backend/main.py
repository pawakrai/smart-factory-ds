from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import init_db, get_session, engine
from .routers import plans, batches, energy, settings as settings_router
from .routers import uploads

# Import models so SQLModel registers them before create_all
from .models import Plan, Batch, EnergyLog, Setting, PlantLoadProfile  # noqa: F401

app = FastAPI(
    title="FurnaceFlow API",
    version="1.0.0",
    description="Sharp Manufacturing Energy & Production Dashboard",
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plans.router, prefix="/api")
app.include_router(batches.router, prefix="/api")
app.include_router(energy.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    init_db()
    _seed_settings()


def _seed_settings():
    from sqlmodel import Session, select

    defaults = [
        # ── Section A: IF Furnace ──────────────────────────────────────────────
        ("if_visual_kw_max",                 "450",    "Max kW cap for visual IF power profile in charts (kW)"),
        ("if_power_option_low_kw",           "450",    "IF power option low (kW)"),
        ("if_power_option_mid_kw",           "475",    "IF power option mid (kW)"),
        ("if_power_option_high_kw",          "500",    "IF power option high (kW)"),
        ("if_batch_output_kg",               "500",    "Metal output per IF batch (kg)"),
        ("if_efficiency_factor_a",           "0.99",   "IF-A efficiency factor"),
        ("if_efficiency_factor_b",           "1.03",   "IF-B efficiency factor"),
        ("cold_start_gap_threshold_min",     "180",    "Gap (min) triggering cold start penalty"),
        ("cold_start_extra_duration_min",    "8",      "Extra melt time for cold start (min)"),
        ("cold_start_extra_energy_kwh",      "30",     "Extra energy for cold start (kWh)"),
        ("post_pour_downtime_min",           "10",     "Downtime after pour before furnace free (min)"),
        # ── Section B: M&H Furnace ────────────────────────────────────────────
        ("mh_a_capacity_kg",                 "400",    "M&H A maximum capacity (kg)"),
        ("mh_a_initial_level_kg",            "400",    "M&H A starting level at shift start (kg)"),
        ("mh_a_consumption_rate_kg_per_min", "2.20",   "M&H A consumption rate (kg/min)"),
        ("mh_a_min_operational_level_kg",    "200",    "M&H A minimum safe operational level (kg)"),
        ("mh_b_capacity_kg",                 "250",    "M&H B maximum capacity (kg)"),
        ("mh_b_initial_level_kg",            "230",    "M&H B starting level at shift start (kg)"),
        ("mh_b_consumption_rate_kg_per_min", "2.30",   "M&H B consumption rate (kg/min)"),
        ("mh_b_min_operational_level_kg",    "125",    "M&H B minimum safe operational level (kg)"),
        ("mh_empty_penalty_per_min",         "150",    "Penalty per minute when M&H furnace runs empty (Baht/min)"),
        ("mh_low_level_minute_penalty",      "40",     "Penalty per minute when M&H level is below minimum (Baht/min)"),
        ("mh_low_level_penalty_rate",        "200",    "Base penalty rate for low-level violation used in simulation"),
        ("mh_low_level_nonlinear_factor",    "3.0",    "Nonlinear exponent when M&H level drops critically low"),
        ("mh_max_empty_min_allow",           "120",    "Max cumulative minutes M&H may be empty before hard constraint violation"),
        ("mh_max_low_level_min_allow",       "240",    "Max cumulative minutes M&H may be below minimum before hard constraint violation"),
        # ── Section C: Energy & Tariff ────────────────────────────────────────
        ("tou_onpeak_baht_per_kwh",          "4.1839", "On-peak TOU energy price (Baht/kWh, excl. FT)"),
        ("tou_offpeak_baht_per_kwh",         "2.6037", "Off-peak TOU energy price (Baht/kWh, excl. FT)"),
        ("ft_baht_per_kwh",                  "0.0972", "Fuel tariff adder (Baht/kWh)"),
        ("demand_charge_baht_per_kw_month",  "132.93", "Monthly demand charge (Baht/kW/month)"),
        ("contract_demand_kw",               "1600",   "Plant contract demand limit (kW)"),
        ("peak_hours_start",                 "09:00",  "On-peak window start (HH:MM)"),
        ("peak_hours_end",                   "22:00",  "On-peak window end (HH:MM)"),
        ("solar_window_start",               "12:00",  "Solar window start (HH:MM)"),
        ("solar_window_end",                 "13:00",  "Solar window end (HH:MM)"),
        ("solar_price_factor",               "0.35",   "Effective price multiplier during solar window"),
        # ── Section D: GA Optimization ────────────────────────────────────────
        ("ga_pop_size",                      "80",     "GA population size"),
        ("ga_n_generations",                 "100",    "Maximum GA generations"),
        ("ga_early_stop_patience",           "20",     "Generations without improvement before early stop"),
        ("ga_random_seed",                   "42",     "RNG seed for reproducibility"),
        ("ga_obj_weight_empty_penalty",      "1.10",   "Objective weight for M&H empty penalty in GA scoring"),
        ("ga_obj_weight_low_level_min",      "0.80",   "Objective weight for M&H below-minimum penalty in GA scoring"),
        ("ga_obj_weight_low_level_shape",    "0.90",   "Objective weight for M&H low-level shape penalty in GA scoring"),
        # ── Section E: Shift Configuration ───────────────────────────────────
        ("shift_duration_hours",             "8",      "Default shift duration (hours)"),
        ("shift_start_hhmm",                 "08:00",  "Default shift start time of day (HH:MM)"),
        ("target_batches_default",           "8",      "Default number of batches per shift"),
    ]
    with Session(engine) as session:
        for key, value, desc in defaults:
            existing = session.exec(select(Setting).where(Setting.config_key == key)).first()
            if not existing:
                session.add(Setting(config_key=key, config_value=value, description=desc))

        # Force-correct known data-corruption cases (safe: only triggers on exact wrong value)
        force_corrections = [
            # tou_offpeak was accidentally seeded with on-peak value (4.1839) in some envs
            ("tou_offpeak_baht_per_kwh", "4.1839", "2.6037"),
        ]
        for key, wrong_val, correct_val in force_corrections:
            bad = session.exec(
                select(Setting).where(
                    Setting.config_key == key,
                    Setting.config_value == wrong_val,
                )
            ).first()
            if bad:
                bad.config_value = correct_val
                session.add(bad)

        session.commit()


@app.get("/health")
def health():
    return {"status": "ok", "service": "FurnaceFlow API"}
