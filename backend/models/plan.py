from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid


class Plan(SQLModel, table=True):
    __tablename__ = "plans"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    target_batches: int
    consumption_rate: float = Field(default=0.0)  # kept for DB compat, not used in new form
    shift_start: datetime
    status: str = Field(default="draft")
    opt_mode: str = Field(default="energy")         # "energy" | "service"
    schedule_metrics: Optional[str] = None          # JSON: ScheduleMetrics
    schedule_data: Optional[str] = None             # JSON: ScheduleData (time-series arrays)
    created_at: datetime = Field(default_factory=datetime.utcnow)
