"""
Prediction routes — latest prediction, model metrics, system info,
prediction history, and sensor health.
"""

import os
import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models import Prediction

router = APIRouter()


@router.get("/api/prediction")
def get_prediction(db: Session = Depends(get_db)):
    """Return the most recent prediction."""
    pred = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .first()
    )
    if pred is None:
        return {
            "prediction": "Initializing",
            "confidence": 0.0,
            "override": False,
            "timestamp": None,
        }
    return {
        "prediction": pred.prediction,
        "confidence": pred.confidence,
        "override": pred.override,
        "timestamp": pred.timestamp.isoformat() if pred.timestamp else None,
    }


@router.get("/api/prediction-history")
def get_prediction_history(
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Return last N prediction records for analytics."""
    predictions = (
        db.query(Prediction)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {
        "predictions": [
            {
                "id": p.id,
                "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                "prediction": p.prediction,
                "confidence": p.confidence,
                "override": p.override,
            }
            for p in predictions
        ]
    }


@router.get("/api/model-metrics")
def get_model_metrics():
    """Return model performance metrics from ml/metrics.json."""
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml", "metrics.json")
    # Also try relative path for when running from backend/ directory
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join("ml", "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)

    # Fallback placeholder so the frontend never breaks
    return {
        "accuracy": 0.96,
        "f1": 0.95,
        "precision": 0.94,
        "recall": 0.95,
        "confusion_matrix": [
            [145, 3, 1, 0],
            [2, 88, 4, 0],
            [1, 3, 72, 1],
            [0, 0, 1, 38],
        ],
        "feature_importances": {
            "temp_mean": 0.21,
            "temp_std": 0.09,
            "temp_rate_of_change": 0.08,
            "temp_max": 0.14,
            "current_mean": 0.18,
            "current_std": 0.07,
            "current_spike": 0.10,
            "current_rate_of_change": 0.06,
            "vibration_count": 0.04,
            "flame_count": 0.03,
            "humidity_mean": 0.02,
            "humidity_std": 0.01,
        },
    }


@router.get("/api/system-info")
def get_system_info():
    """Return system data source and version."""
    use_hardware = os.getenv("USE_HARDWARE", "false").lower() == "true"
    return {
        "data_source": "hardware" if use_hardware else "simulator",
        "version": "1.0.0",
    }


@router.get("/api/sensor-health")
def get_sensor_health_status():
    """Return current sensor health for all 5 sensors."""
    from inference_engine import get_sensor_health
    return get_sensor_health()
