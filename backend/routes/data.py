"""
Data routes — live sensor data and historical readings.
Now includes humidity in all responses.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models import SensorReading

router = APIRouter()


@router.get("/api/live-data")
def get_live_data(db: Session = Depends(get_db)):
    """Return the last 50 sensor readings, newest first."""
    readings = (
        db.query(SensorReading)
        .order_by(SensorReading.timestamp.desc())
        .limit(50)
        .all()
    )
    return {
        "readings": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "temp": r.temp,
                "humidity": r.humidity if r.humidity is not None else 0.0,
                "vibration": r.vibration,
                "current": r.current,
                "flame": r.flame,
            }
            for r in readings
        ]
    }


@router.get("/api/history")
def get_history(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return last N readings in ascending time order for charting."""
    readings = (
        db.query(SensorReading)
        .order_by(SensorReading.timestamp.desc())
        .limit(limit)
        .all()
    )
    # Reverse to ascending order for charts
    readings.reverse()
    return {
        "readings": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "temp": r.temp,
                "humidity": r.humidity if r.humidity is not None else 0.0,
                "vibration": r.vibration,
                "current": r.current,
                "flame": r.flame,
            }
            for r in readings
        ]
    }
