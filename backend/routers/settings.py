from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from datetime import datetime
from pydantic import BaseModel
from ..database import get_session
from ..models import Setting

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    config_value: str


@router.get("/", response_model=list[Setting])
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
