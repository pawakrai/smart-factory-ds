"""
uploads.py — Endpoints for TOU rate and plant load Excel upload/download/export.

Routes:
  GET  /api/uploads/tou-template           → download blank TOU template
  GET  /api/uploads/tou-rates/export       → export current TOU settings as Excel
  POST /api/uploads/tou-rates              → upload TOU Excel → update settings immediately

  GET  /api/uploads/plant-load-template    → download blank plant load template
  GET  /api/uploads/plant-load/export      → export active plant load profile as Excel
  POST /api/uploads/plant-load             → upload plant load Excel → replace active profile
  GET  /api/uploads/plant-load/active      → get active profile metadata + entries summary
"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from ..database import get_session
from ..models import Setting
from ..models.plant_load import PlantLoadProfile
from ..services import excel_service

router = APIRouter(prefix="/uploads", tags=["uploads"])

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# ══════════════════════════════════════════════════════════════════════════════
# TOU Rate
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/tou-template", summary="Download blank TOU rate Excel template")
def download_tou_template():
    buf = excel_service.generate_tou_template()
    return StreamingResponse(
        buf,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="tou_rate_template.xlsx"'},
    )


@router.get("/tou-rates/export", summary="Export current TOU settings as Excel")
def export_tou_rates(session: Session = Depends(get_session)):
    settings = session.exec(select(Setting)).all()
    settings_dict = {s.config_key: s.config_value for s in settings}
    buf = excel_service.export_tou_to_excel(settings_dict)
    return StreamingResponse(
        buf,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="tou_rates_current.xlsx"'},
    )


@router.post("/tou-rates", summary="Upload TOU rate Excel → replace settings immediately")
async def upload_tou_rates(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Only .xlsx files are accepted")

    raw = await file.read()
    try:
        updates = excel_service.parse_tou_excel(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    updated_keys: list[str] = []
    for key, value in updates.items():
        setting = session.exec(select(Setting).where(Setting.config_key == key)).first()
        if setting:
            setting.config_value = value
            setting.updated_at = datetime.utcnow()
            session.add(setting)
            updated_keys.append(key)

    session.commit()
    return {"updated_keys": updated_keys, "count": len(updated_keys)}


# ══════════════════════════════════════════════════════════════════════════════
# Plant Load
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/plant-load-template", summary="Download blank plant load Excel template")
def download_plant_load_template():
    buf = excel_service.generate_plant_load_template()
    return StreamingResponse(
        buf,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": 'attachment; filename="plant_load_template.xlsx"'},
    )


@router.get("/plant-load/export", summary="Export active plant load profile as Excel")
def export_plant_load(session: Session = Depends(get_session)):
    profile = session.exec(
        select(PlantLoadProfile).where(PlantLoadProfile.is_active == True)
    ).first()

    if profile is None:
        # No active profile — export default template values
        buf = excel_service.generate_plant_load_template()
        filename = "plant_load_default.xlsx"
    else:
        buf = excel_service.export_plant_load_to_excel(
            profile.entries_json, profile.spikes_json
        )
        filename = f"plant_load_{profile.name.replace(' ', '_')}.xlsx"

    return StreamingResponse(
        buf,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/plant-load/active", summary="Get active plant load profile metadata")
def get_active_plant_load(session: Session = Depends(get_session)):
    profile = session.exec(
        select(PlantLoadProfile).where(PlantLoadProfile.is_active == True)
    ).first()

    if profile is None:
        return {"active": False, "profile": None}

    entries = json.loads(profile.entries_json) if profile.entries_json else []
    spikes  = json.loads(profile.spikes_json)  if profile.spikes_json  else []

    # Return hourly summary (24 values) instead of 1440 raw entries for the UI
    hourly: list[dict] = []
    for hour in range(24):
        start = hour * 60
        end   = start + 60
        window = [e["load_kw"] for e in entries if start <= e["minute"] < end]
        hourly.append({
            "hour": hour,
            "time": f"{hour:02d}:00",
            "avg_load_kw": round(sum(window) / len(window), 1) if window else 0,
        })

    return {
        "active": True,
        "profile": {
            "id":                profile.id,
            "name":              profile.name,
            "created_at":        profile.created_at.isoformat(),
            "uploaded_filename": profile.uploaded_filename,
            "entry_count":       len(entries),
            "spike_count":       len(spikes),
            "spikes":            spikes,
            "hourly_summary":    hourly,
        },
    }


@router.post("/plant-load", summary="Upload plant load Excel → replace active profile")
async def upload_plant_load(
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(400, "Only .xlsx files are accepted")

    raw = await file.read()
    try:
        entries, spikes = excel_service.parse_plant_load_excel(raw)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # Deactivate any existing active profile
    existing = session.exec(
        select(PlantLoadProfile).where(PlantLoadProfile.is_active == True)
    ).all()
    for old in existing:
        old.is_active = False
        session.add(old)

    # Create new active profile
    profile_name = file.filename.removesuffix(".xlsx").replace("_", " ")
    profile = PlantLoadProfile(
        name=profile_name,
        entries_json=json.dumps(entries),
        spikes_json=json.dumps(spikes),
        is_active=True,
        uploaded_filename=file.filename,
    )
    session.add(profile)
    session.commit()
    session.refresh(profile)

    return {
        "id":            profile.id,
        "name":          profile.name,
        "entry_count":   len(entries),
        "spike_count":   len(spikes),
        "is_active":     profile.is_active,
    }
