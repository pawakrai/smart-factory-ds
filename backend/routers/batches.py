from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select
from ..database import get_session
from ..models import Batch
from ..schemas.batch import BatchUpdate
from ..services.rl_service import get_power_profile

router = APIRouter(prefix="/batches", tags=["batches"])


@router.get("/", response_model=list[Batch])
def list_batches(
    plan_id: str | None = Query(None),
    session: Session = Depends(get_session),
):
    q = select(Batch).order_by(Batch.batch_number)
    if plan_id:
        q = q.where(Batch.plan_id == plan_id)
    return session.exec(q).all()


@router.get("/{batch_id}/power-profile")
def batch_power_profile(batch_id: str, session: Session = Depends(get_session)):
    batch = session.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    duration = batch.duration_min or 120
    return get_power_profile(batch_id, duration)


@router.get("/{batch_id}", response_model=Batch)
def get_batch(batch_id: str, session: Session = Depends(get_session)):
    batch = session.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return batch


@router.patch("/{batch_id}", response_model=Batch)
def update_batch(batch_id: str, data: BatchUpdate, session: Session = Depends(get_session)):
    batch = session.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(batch, k, v)
    session.add(batch)
    session.commit()
    session.refresh(batch)
    return batch
