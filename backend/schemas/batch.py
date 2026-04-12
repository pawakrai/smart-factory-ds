from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class BatchUpdate(BaseModel):
    actual_start: Optional[datetime] = None
    actual_finish: Optional[datetime] = None
    ingot_kg: Optional[float] = None
    fe_kg: Optional[float] = None
    si_kg: Optional[float] = None
    scrap_kg: Optional[float] = None
    status: Optional[str] = None
