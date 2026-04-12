from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select
from typing import Optional
from datetime import datetime, timedelta
import random
from ..database import get_session
from ..models import EnergyLog
from ..schemas.energy_log import EnergyLogCreate

router = APIRouter(prefix="/energy-logs", tags=["energy"])


@router.get("", response_model=list[EnergyLog])
def list_energy_logs(
    limit: int = Query(default=200, le=1000),
    shift_start: Optional[str] = Query(default=None, description="ISO datetime — filter logs on or after this time"),
    session: Session = Depends(get_session),
):
    stmt = select(EnergyLog)
    if shift_start:
        try:
            dt = datetime.fromisoformat(shift_start.replace("Z", "+00:00"))
            stmt = stmt.where(EnergyLog.timestamp >= dt)
        except ValueError:
            pass
    stmt = stmt.order_by(EnergyLog.timestamp.asc()).limit(limit)
    return session.exec(stmt).all()


@router.post("", response_model=EnergyLog, status_code=201)
def create_energy_log(data: EnergyLogCreate, session: Session = Depends(get_session)):
    log = EnergyLog(**data.model_dump())
    session.add(log)
    session.commit()
    session.refresh(log)
    return log


@router.post("/seed-mock", status_code=201)
def seed_mock_energy_logs(session: Session = Depends(get_session)):
    """Seed 24 h of mock hourly energy data (batch_id=None rows) for dashboard development."""
    # Remove previous mock entries (no batch_id attached)
    existing = session.exec(select(EnergyLog).where(EnergyLog.batch_id == None)).all()  # noqa: E711
    for log in existing:
        session.delete(log)

    now = datetime.utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    logs = []
    for hour in range(24):
        ts = start + timedelta(hours=hour)
        is_peak = 9 <= hour < 22
        base_kw = 420.0 if is_peak else 190.0
        sim_kw = base_kw + random.uniform(-30, 50)
        noise = random.uniform(-0.08, 0.12)
        actual_kw = sim_kw * (1 + noise)
        logs.append(
            EnergyLog(
                timestamp=ts,
                sim_kw=round(sim_kw, 1),
                actual_kw=round(actual_kw, 1),
            )
        )

    session.add_all(logs)
    session.commit()
    return {"seeded": len(logs), "date": start.date().isoformat()}
