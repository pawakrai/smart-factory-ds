from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid


class EnergyLog(SQLModel, table=True):
    __tablename__ = "energy_logs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    batch_id: Optional[str] = Field(default=None, foreign_key="batches.id")
    timestamp: datetime
    sim_kw: Optional[float] = None
    actual_kw: Optional[float] = None
