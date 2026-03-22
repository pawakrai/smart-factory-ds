import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from ..database import get_session, engine
from ..models import Plan, Batch
from ..models.setting import Setting
from ..schemas.plan import PlanCreate, PlanUpdate, PlanRead, ScheduleData
from ..services.ga_service import generate_schedule

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/plans", tags=["plans"])


def _load_settings(session: Session) -> dict[str, str]:
    """Load all settings from DB as a plain dict {config_key: config_value}."""
    rows = session.exec(select(Setting)).all()
    return {r.config_key: r.config_value for r in rows}


def _run_ga_and_update(plan_id: str) -> None:
    """Background task: run GA and update plan/batches in DB."""
    with Session(engine) as session:
        plan = session.get(Plan, plan_id)
        if plan is None:
            return
        try:
            settings_dict = _load_settings(session)
            ga_result = generate_schedule(plan, settings_dict)

            for item in ga_result.schedule_items:
                batch = Batch(
                    plan_id=plan.id,
                    batch_number=item["batch_id"],
                    expected_start=item.get("expected_start"),
                    melt_finish_at=item.get("melt_finish_at"),
                    pour_at=item.get("pour_at"),
                    furnace=item.get("furnace"),
                    duration_min=item.get("duration_min"),
                    power_kw=item.get("power_kw"),
                    is_cold_start=bool(item.get("is_cold_start", False)),
                    energy_kwh=item.get("energy_kwh"),
                    status="pending",
                )
                session.add(batch)

            if ga_result.metrics:
                plan.schedule_metrics = ga_result.metrics.model_dump_json()
            if ga_result.schedule_data:
                plan.schedule_data = ga_result.schedule_data.model_dump_json()

            plan.status = "draft"
            logger.info("GA completed for plan %s", plan_id)
        except Exception as exc:
            logger.error("GA background task failed for plan %s: %s", plan_id, exc, exc_info=True)
            plan.status = "draft"

        session.add(plan)
        session.commit()


@router.get("/", response_model=list[PlanRead])
def list_plans(session: Session = Depends(get_session)):
    return session.exec(select(Plan).order_by(Plan.created_at.desc())).all()


@router.post("/", response_model=PlanRead, status_code=202)
def create_plan(data: PlanCreate, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    plan = Plan(
        target_batches=data.target_batches,
        shift_start=data.shift_start,
        opt_mode=data.opt_mode,
        consumption_rate=0.0,
        status="pending",
        if_a_enabled=data.if_a_enabled,
        if_b_enabled=data.if_b_enabled,
        mh_a_consumption_rate=data.mh_a_consumption_rate,
        mh_b_consumption_rate=data.mh_b_consumption_rate,
    )
    session.add(plan)
    session.commit()
    session.refresh(plan)

    background_tasks.add_task(_run_ga_and_update, plan.id)
    return plan


@router.get("/{plan_id}", response_model=PlanRead)
def get_plan(plan_id: str, session: Session = Depends(get_session)):
    plan = session.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    return plan


@router.patch("/{plan_id}", response_model=PlanRead)
def update_plan(plan_id: str, data: PlanUpdate, session: Session = Depends(get_session)):
    plan = session.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(plan, k, v)
    session.add(plan)
    session.commit()
    session.refresh(plan)
    return plan


@router.post("/{plan_id}/activate", response_model=PlanRead)
def activate_plan(plan_id: str, session: Session = Depends(get_session)):
    plan = session.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    if plan.status == "pending":
        raise HTTPException(400, "Cannot activate a plan while GA is still running")
    if plan.status == "completed":
        raise HTTPException(400, "Cannot re-activate a completed plan")
    for p in session.exec(select(Plan).where(Plan.status == "active")).all():
        if p.id != plan_id:
            p.status = "draft"
            session.add(p)
    plan.status = "active"
    session.add(plan)
    session.commit()
    session.refresh(plan)
    return plan


@router.delete("/{plan_id}", status_code=204)
def delete_plan(plan_id: str, session: Session = Depends(get_session)):
    plan = session.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    session.delete(plan)
    session.commit()


@router.get("/{plan_id}/batches", response_model=list[Batch])
def get_plan_batches(plan_id: str, session: Session = Depends(get_session)):
    return session.exec(
        select(Batch).where(Batch.plan_id == plan_id).order_by(Batch.batch_number)
    ).all()


@router.get("/{plan_id}/schedule-data", response_model=ScheduleData)
def get_schedule_data(plan_id: str, session: Session = Depends(get_session)):
    plan = session.get(Plan, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    if not plan.schedule_data:
        raise HTTPException(404, "Schedule data not available for this plan")
    return ScheduleData(**json.loads(plan.schedule_data))
