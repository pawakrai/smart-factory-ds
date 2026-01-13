from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service_core import run_hga_schedule

app = FastAPI(title="FurnaceFlow HGA Service", version="0.1.0")

# Allow local dev & typical frontends (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IFConfig(BaseModel):
    use_furnace_a: bool = Field(True, description="Enable/disable Furnace A")
    use_furnace_b: bool = Field(True, description="Enable/disable Furnace B")


class MHConfig(BaseModel):
    max_capacity: Dict[str, float] = Field(
        default_factory=lambda: {"A": 400.0, "B": 250.0},
        description="Max metal capacity in kg for each furnace in M&H",
    )
    initial_level: Dict[str, float] = Field(
        default_factory=lambda: {"A": 400.0, "B": 200.0},
        description="Initial metal level in M&H (kg) for each furnace",
    )
    consumption_rate: Dict[str, float] = Field(
        default_factory=lambda: {"A": 3.5, "B": 2.5},
        description="Consumption rate of metal in kg/min for each furnace",
    )


class SolarConfig(BaseModel):
    # list of [start_minute, end_minute], e.g. [[720, 780]] for 12:00–13:00
    windows: List[Tuple[int, int]] = Field(default_factory=lambda: [(12 * 60, 13 * 60)])
    # 0.0 = no discount, 1.0 = energy in window is free
    discount_factor: float = Field(0.5, ge=0.0, le=1.0)


class GAConfig(BaseModel):
    pop_size: int = Field(50, ge=4, description="Population size for NSGA-II")
    n_gen: int = Field(100, ge=10, description="Number of generations")
    seed: int = Field(42, description="Random seed for reproducibility")


class ScheduleRequest(BaseModel):
    num_batches: int = Field(10, ge=1, description="Number of melts in the horizon")
    if_cfg: IFConfig = Field(default_factory=IFConfig)
    mh: MHConfig = Field(default_factory=MHConfig)
    solar: SolarConfig = Field(default_factory=SolarConfig)
    ga: GAConfig = Field(default_factory=GAConfig)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/schedule")
def create_schedule(req: ScheduleRequest) -> Dict[str, Any]:
    """
    Run the HGA-based scheduler with the given configuration and
    return the best schedule and cost breakdown.
    """
    cfg: Dict[str, Any] = {
        "num_batches": req.num_batches,
        "if": {
            "use_furnace_a": req.if_cfg.use_furnace_a,
            "use_furnace_b": req.if_cfg.use_furnace_b,
        },
        "mh": {
            "max_capacity": req.mh.max_capacity,
            "initial_level": req.mh.initial_level,
            "consumption_rate": req.mh.consumption_rate,
        },
        "solar": {
            "windows": req.solar.windows,
            "discount_factor": req.solar.discount_factor,
        },
        "ga": {
            "pop_size": req.ga.pop_size,
            "n_gen": req.ga.n_gen,
            "seed": req.ga.seed,
        },
    }

    result = run_hga_schedule(cfg)
    return result


if __name__ == "__main__":
    # For local testing:
    import uvicorn

    uvicorn.run("backend.service_main:app", host="0.0.0.0", port=8000, reload=True)
