from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid


class PlantLoadProfile(SQLModel, table=True):
    __tablename__ = "plant_load_profiles"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str
    # JSON array: [{minute: 0, load_kw: 450.0}, ...] — 1440 entries (minute 0..1439)
    entries_json: str = Field(default="[]")
    # JSON array of spike events: [{start_tod:"10:30", end_tod:"11:00", extra_kw:400.0}]
    spikes_json: str = Field(default="[]")
    is_active: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    uploaded_filename: Optional[str] = None
