from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field
import uuid


class Setting(SQLModel, table=True):
    __tablename__ = "settings"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    config_key: str = Field(unique=True, index=True)
    config_value: str
    description: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
