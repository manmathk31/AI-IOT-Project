"""
Maintenance task routes — CRUD for maintenance assignments.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import MaintenanceTask

router = APIRouter()


class MaintenanceCreate(BaseModel):
    engineer: str
    machine: str
    notes: str


class MaintenanceStatusUpdate(BaseModel):
    status: str


@router.get("/api/maintenance")
def get_maintenance_tasks(db: Session = Depends(get_db)):
    """Return all maintenance tasks, newest first."""
    tasks = (
        db.query(MaintenanceTask)
        .order_by(MaintenanceTask.timestamp.desc())
        .all()
    )
    return {
        "tasks": [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                "engineer": t.engineer,
                "machine": t.machine,
                "notes": t.notes,
                "status": t.status,
            }
            for t in tasks
        ]
    }


@router.post("/api/maintenance")
def create_maintenance_task(body: MaintenanceCreate, db: Session = Depends(get_db)):
    """Create a new maintenance task."""
    task = MaintenanceTask(
        engineer=body.engineer,
        machine=body.machine,
        notes=body.notes,
        status="Pending",
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return {
        "id": task.id,
        "timestamp": task.timestamp.isoformat() if task.timestamp else None,
        "engineer": task.engineer,
        "machine": task.machine,
        "notes": task.notes,
        "status": task.status,
    }


@router.patch("/api/maintenance/{task_id}")
def update_maintenance_status(
    task_id: int,
    body: MaintenanceStatusUpdate,
    db: Session = Depends(get_db),
):
    """Update the status of a maintenance task."""
    task = db.query(MaintenanceTask).filter(MaintenanceTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Maintenance task not found")
    task.status = body.status
    db.commit()
    db.refresh(task)
    return {
        "id": task.id,
        "status": task.status,
    }
