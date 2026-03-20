from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class EnergyLogCreate(BaseModel):
    batch_id: Optional[str] = None
    timestamp: datetime
    sim_kw: Optional[float] = None
    actual_kw: Optional[float] = None
