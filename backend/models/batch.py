from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid


class Batch(SQLModel, table=True):
    __tablename__ = "batches"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    plan_id: str = Field(foreign_key="plans.id")
    batch_number: int
    expected_start: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    ingot_kg: Optional[float] = None
    fe_kg: Optional[float] = None
    si_kg: Optional[float] = None
    scrap_kg: Optional[float] = None
    furnace: Optional[str] = None           # "A" or "B" — assigned by GA
    duration_min: Optional[int] = None     # melt duration in minutes (from GA)
    melt_finish_at: Optional[datetime] = None   # when melting phase ends
    pour_at: Optional[datetime] = None          # when batch is poured into M&H
    power_kw: Optional[float] = None           # IF power used: 450 | 475 | 500 kW
    is_cold_start: bool = Field(default=False) # cold start penalty applied
    energy_kwh: Optional[float] = None        # estimated energy for this batch (kWh)
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
