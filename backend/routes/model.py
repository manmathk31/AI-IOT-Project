"""
Model management routes -- data collection, training, model lifecycle,
and sensor configuration.

This is the backend for the AI Control Panel in the frontend.
"""

import os
from fastapi import APIRouter
from model_trainer import train_model, delete_model, get_model_status
import data_collector_service

router = APIRouter()


# ════════════════════════════════════════════════════════════════
# MODEL STATUS
# ════════════════════════════════════════════════════════════════

@router.get("/api/model/status")
def model_status():
    """Get comprehensive model status: loaded, collecting, training info."""
    status = get_model_status()
    collection = data_collector_service.get_stats()
    training_data = data_collector_service.get_training_data_info()

    # Get sensor config from inference engine
    from inference_engine import get_sensor_config
    sensor_config = get_sensor_config()

    return {
        "model": status,
        "collection": collection,
        "training_data": training_data,
        "sensor_config": sensor_config,
    }


# ════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ════════════════════════════════════════════════════════════════

@router.post("/api/model/collect-start")
def collect_start():
    """Start collecting normal operating data from live sensor feed."""
    return data_collector_service.start_collection()


@router.post("/api/model/collect-stop")
def collect_stop():
    """Stop data collection and finalize CSV."""
    return data_collector_service.stop_collection()


@router.get("/api/model/collection-stats")
def collection_stats():
    """Get live collection stats (count, duration, progress)."""
    return data_collector_service.get_stats()


# ════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ════════════════════════════════════════════════════════════════

@router.post("/api/model/train")
def train():
    """Train One-Class SVM on collected data. Synchronous (takes 1-3 seconds)."""
    # Don't train while collecting
    if data_collector_service.is_collecting():
        return {"error": "Cannot train while data collection is active. Stop collection first."}

    result = train_model()

    # If training succeeded, hot-reload the model into inference engine
    if result.get("success"):
        from inference_engine import reload_model
        reload_result = reload_model()
        result["reload"] = reload_result

    return result


# ════════════════════════════════════════════════════════════════
# MODEL LIFECYCLE
# ════════════════════════════════════════════════════════════════

@router.post("/api/model/delete")
def delete():
    """Delete the current model. System falls back to simulation mode."""
    result = delete_model()

    # Unload from inference engine
    from inference_engine import unload_model
    unload_model()

    return result


@router.post("/api/model/reload")
def reload():
    """Hot-reload model from disk without restarting server."""
    from inference_engine import reload_model
    return reload_model()


# ════════════════════════════════════════════════════════════════
# TRAINING DATA MANAGEMENT
# ════════════════════════════════════════════════════════════════

@router.get("/api/model/training-data")
def training_data_info():
    """List all collected training data CSV files."""
    return data_collector_service.get_training_data_info()


@router.delete("/api/model/training-data")
def delete_training_data():
    """Delete all collected training data CSV files."""
    if data_collector_service.is_collecting():
        return {"error": "Cannot delete data while collection is active."}
    return data_collector_service.delete_training_data()


# ════════════════════════════════════════════════════════════════
# SENSOR CONFIGURATION
# ════════════════════════════════════════════════════════════════

@router.get("/api/sensor-config")
def get_sensor_config():
    """Get current sensor enable/disable configuration."""
    from inference_engine import get_sensor_config
    return get_sensor_config()


@router.post("/api/sensor-config")
def update_sensor_config(config: dict):
    """
    Update sensor enable/disable configuration.
    Body: {"flame_enabled": true/false, "vibration_enabled": true/false}
    """
    from inference_engine import update_sensor_config
    return update_sensor_config(config)
