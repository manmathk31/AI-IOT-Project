"""
Alert routes — list and resolve alerts.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Alert

router = APIRouter()


@router.get("/api/alerts")
def get_alerts(db: Session = Depends(get_db)):
    """Return all alerts, newest first."""
    alerts = (
        db.query(Alert)
        .order_by(Alert.timestamp.desc())
        .all()
    )
    return {
        "alerts": [
            {
                "id": a.id,
                "type": a.type,
                "severity": a.severity,
                "message": a.message,
                "status": a.status,
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
            }
            for a in alerts
        ]
    }


@router.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as resolved."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = "resolved"
    db.commit()
    return {"success": True}
