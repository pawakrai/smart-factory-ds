from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from datetime import datetime
from pydantic import BaseModel
from ..database import get_session
from ..models import Setting

router = APIRouter(prefix="/settings", tags=["settings"])

# Canonical defaults matching app_v9.py hardcoded values — used for "Reset to default" UI
SETTING_DEFAULTS: dict[str, str] = {
    "if_visual_kw_max":                 "450",
    "if_power_option_low_kw":           "450",
    "if_power_option_mid_kw":           "475",
    "if_power_option_high_kw":          "500",
    "if_batch_output_kg":               "500",
    "if_efficiency_factor_a":           "0.99",
    "if_efficiency_factor_b":           "1.03",
    "cold_start_gap_threshold_min":     "180",
    "cold_start_extra_duration_min":    "8",
    "cold_start_extra_energy_kwh":      "30",
    "post_pour_downtime_min":           "10",
    "mh_a_capacity_kg":                 "400",
    "mh_a_initial_level_kg":            "400",
    "mh_a_consumption_rate_kg_per_min": "2.20",
    "mh_a_min_operational_level_kg":    "200",
    "mh_b_capacity_kg":                 "250",
    "mh_b_initial_level_kg":            "230",
    "mh_b_consumption_rate_kg_per_min": "2.30",
    "mh_b_min_operational_level_kg":    "125",
    "mh_empty_penalty_per_min":         "150",
    "mh_low_level_minute_penalty":      "40",
    "mh_low_level_penalty_rate":        "200",
    "mh_low_level_nonlinear_factor":    "3.0",
    "mh_max_empty_min_allow":           "120",
    "mh_max_low_level_min_allow":       "240",
    "tou_onpeak_baht_per_kwh":          "4.1839",
    "tou_offpeak_baht_per_kwh":         "2.6037",
    "ft_baht_per_kwh":                  "0.0972",
    "demand_charge_baht_per_kw_month":  "132.93",
    "contract_demand_kw":               "1600",
    "peak_hours_start":                 "09:00",
    "peak_hours_end":                   "22:00",
    "solar_window_start":               "12:00",
    "solar_window_end":                 "13:00",
    "solar_price_factor":               "0.35",
    "ga_pop_size":                      "80",
    "ga_n_generations":                 "100",
    "ga_early_stop_patience":           "20",
    "ga_random_seed":                   "42",
    "ga_obj_weight_empty_penalty":      "1.10",
    "ga_obj_weight_low_level_min":      "0.80",
    "ga_obj_weight_low_level_shape":    "0.90",
    "shift_duration_hours":             "8",
    "shift_start_hhmm":                 "08:00",
    "target_batches_default":           "8",
}


class SettingUpdate(BaseModel):
    config_value: str


@router.get("/defaults")
def get_defaults() -> dict[str, str]:
    """Return canonical default values sourced from app_v9.py hardcoded constants."""
    return SETTING_DEFAULTS


@router.get("", response_model=list[Setting])
def list_settings(session: Session = Depends(get_session)):
    return session.exec(select(Setting).order_by(Setting.config_key)).all()


@router.put("/{key}", response_model=Setting)
def update_setting(key: str, data: SettingUpdate, session: Session = Depends(get_session)):
    setting = session.exec(select(Setting).where(Setting.config_key == key)).first()
    if not setting:
        raise HTTPException(404, f"Setting '{key}' not found")
    setting.config_value = data.config_value
    setting.updated_at = datetime.utcnow()
    session.add(setting)
    session.commit()
    session.refresh(setting)
    return setting
