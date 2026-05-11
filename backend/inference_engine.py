"""
ML inference engine for the Wind Turbine Monitoring System.
Consumes sensor data from a queue, runs sliding-window feature extraction,
and detects anomalies using a One-Class SVM model.

Architecture — Hybrid detection:
  Layer 1: Rule-based overrides (flame, vibration) — highest priority
  Layer 2: One-Class SVM anomaly detection — for everything else

Features:
- One-Class SVM anomaly detection with severity scoring
- Hardware rule-based overrides (flame, vibration)
- Alert flood prevention (deduplication)
- Sensor fault tolerance (None handling, fallback values, health tracking)
- Sensor failure alerts after 10+ consecutive missing readings
"""

import os
import random
import collections
import traceback
import threading
from datetime import datetime

import numpy as np

from models import SensorReading, Prediction, Alert
from feature_engineering import extract_features, get_model_input
import data_collector_service

# ── Sensor Configuration ──
_sensor_config_lock = threading.Lock()
_sensor_config = {
    "flame_enabled": True,
    "vibration_enabled": True,
}

def get_sensor_config():
    with _sensor_config_lock:
        return dict(_sensor_config)

def update_sensor_config(config):
    with _sensor_config_lock:
        if "flame_enabled" in config:
            _sensor_config["flame_enabled"] = bool(config["flame_enabled"])
        if "vibration_enabled" in config:
            _sensor_config["vibration_enabled"] = bool(config["vibration_enabled"])
        return dict(_sensor_config)

# ── Model state ──
_model_lock = threading.Lock()
_model = None
_scaler = None

def reload_model():
    global _model, _scaler
    model_path = os.path.join(os.path.dirname(__file__), "ml", "model_v1.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "ml", "scaler.pkl")
    try:
        import joblib
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            new_model = joblib.load(model_path)
            new_scaler = joblib.load(scaler_path)
            with _model_lock:
                _model = new_model
                _scaler = new_scaler
            print("[Inference] Model reloaded successfully.")
            return {"success": True, "message": "Model reloaded successfully."}
        else:
            with _model_lock:
                _model = None
                _scaler = None
            print("[Inference] Model files not found. Using simulation mode.")
            return {"success": False, "error": "Model files not found. System running in simulation mode."}
    except Exception as e:
        print(f"[Inference] Reload error: {e}")
        return {"success": False, "error": f"Failed to reload model: {e}"}

def unload_model():
    global _model, _scaler
    with _model_lock:
        _model = None
        _scaler = None
    print("[Inference] Model unloaded.")

# ── Global sensor health state (read by /api/sensor-health route) ──
_sensor_health_lock = threading.Lock()
_sensor_health = {
    "temp": "ok",
    "humidity": "ok",
    "current": "ok",
    "vibration": "ok",
    "flame": "ok",
}


def get_sensor_health():
    """Return a snapshot of current sensor health status (thread-safe)."""
    with _sensor_health_lock:
        return dict(_sensor_health)


def run_inference(data_queue, session_factory):
    """
    Run the inference engine forever, consuming readings from data_queue
    and writing predictions + alerts to the database.

    Uses a hybrid approach:
    1. Rule-based overrides for flame (CRITICAL_FLAME) and vibration (HIGH_VIBRATION)
    2. One-Class SVM anomaly detection for everything else

    Args:
        data_queue: queue.Queue with sensor reading dicts
        session_factory: SQLAlchemy sessionmaker (SessionLocal)
    """
    global _sensor_health

    # Load model initially
    reload_model()

    label_map = {
        "Normal": "Normal",
        "Warning": "Warning",
        "Fault": "Fault",
        "CRITICAL_FLAME": "CRITICAL_FLAME",
        "HIGH_VIBRATION": "HIGH_VIBRATION",
    }
    window = collections.deque(maxlen=20)

    # ── Sensor fault tolerance state ──
    last_valid = {
        "temp": 25.0,
        "humidity": 50.0,
        "current": 0.0,
        "vibration": 0,
        "flame": 0,
    }
    consecutive_missing = {
        "temp": 0,
        "humidity": 0,
        "current": 0,
        "vibration": 0,
        "flame": 0,
    }
    SENSOR_FIELDS = ["temp", "humidity", "current", "vibration", "flame"]
    SENSOR_DEFAULTS = {"temp": 25.0, "humidity": 50.0, "current": 0.0, "vibration": 0, "flame": 0}
    FAILURE_THRESHOLD = 10

    print("[Inference] Inference engine started (One-Class SVM mode). Waiting for readings...")

    while True:
        try:
            # 1. Get next reading (blocking)
            reading = data_queue.get()

            # ── Sensor health check and fallback ──
            for field in SENSOR_FIELDS:
                raw_val = reading.get(field)
                if raw_val is not None:
                    # Sensor is sending data — update last known valid
                    last_valid[field] = raw_val
                    consecutive_missing[field] = 0
                else:
                    # Sensor value missing
                    consecutive_missing[field] += 1
                    # GOLDEN RULE: missing flame sensor = 0 (never assume flame from silence)
                    if field == "flame":
                        reading[field] = 0
                    elif field == "vibration":
                        reading[field] = 0
                    else:
                        # Use last known valid value
                        reading[field] = last_valid[field]

            # Update global sensor health status
            with _sensor_health_lock:
                for field in SENSOR_FIELDS:
                    if consecutive_missing[field] == 0:
                        _sensor_health[field] = "ok"
                    elif consecutive_missing[field] < FAILURE_THRESHOLD:
                        _sensor_health[field] = "fallback"
                    else:
                        _sensor_health[field] = "failed"

            # Log sensor failures
            failed_sensors = [f for f in SENSOR_FIELDS if consecutive_missing[f] >= FAILURE_THRESHOLD]
            if failed_sensors:
                print(f"[Inference] WARNING: Sensors failed ({FAILURE_THRESHOLD}+ missing): {failed_sensors}")

            # ── Data Collection Hook ──
            data_collector_service.add_reading(reading)

            # 2. Append to sliding window
            window.append(reading)

            # 3. Wait until window is full
            if len(window) < 20:
                continue

            # 4. Open a new DB session
            session = session_factory()

            try:
                # 5. Save raw sensor reading to DB
                sensor_record = SensorReading(
                    timestamp=datetime.utcnow(),
                    temp=reading["temp"],
                    humidity=reading.get("humidity", 0.0),
                    vibration=reading["vibration"],
                    current=reading["current"],
                    flame=reading["flame"],
                )
                session.add(sensor_record)

                # ── Create sensor failure alerts (deduplicated) ──
                for field in SENSOR_FIELDS:
                    if consecutive_missing[field] == FAILURE_THRESHOLD:
                        # Only fire alert exactly at threshold (not every reading after)
                        existing = session.query(Alert).filter(
                            Alert.type == "Sensor Failure",
                            Alert.status == "active",
                            Alert.message.contains(f"'{field}'"),
                        ).first()
                        if not existing:
                            sensor_alert = Alert(
                                type="Sensor Failure",
                                severity="warning",
                                message=f"Sensor '{field}' has not sent data for {FAILURE_THRESHOLD}+ readings. Check wiring.",
                            )
                            session.add(sensor_alert)
                            print(f"[Inference] WARNING: Sensor failure alert created for '{field}'")

                # ════════════════════════════════════════════════════
                # STEP 1 — Flame check (rule-based override)
                # ════════════════════════════════════════════════════
                # GOLDEN RULE: only trigger flame when sensor EXPLICITLY sends flame=1
                flame_enabled = get_sensor_config().get("flame_enabled", True)
                if flame_enabled and reading["flame"] == 1 and consecutive_missing.get("flame", 0) == 0:
                    prediction_record = Prediction(
                        prediction="CRITICAL_FLAME",
                        confidence=1.0,
                        override=True,
                        anomaly_score=None,
                    )
                    # Flame alerts always create (no dedup — fire alerts are urgent)
                    alert_record = Alert(
                        type="Flame Detected",
                        severity="critical",
                        message=(
                            "Flame sensor triggered on turbine. "
                            "Emergency shutdown recommended. "
                            "Contact safety team immediately."
                        ),
                    )
                    session.add(prediction_record)
                    session.add(alert_record)
                    session.commit()
                    print("[Inference] CRITICAL_FLAME -- rule-based override prediction saved.")
                    continue

                # ════════════════════════════════════════════════════
                # STEP 2 — Vibration check (rule-based override)
                # ════════════════════════════════════════════════════
                vibration_enabled = get_sensor_config().get("vibration_enabled", True)
                vibration_count = sum(r.get("vibration", 0) or 0 for r in window)
                if vibration_enabled and vibration_count >= 15:
                    prediction_record = Prediction(
                        prediction="HIGH_VIBRATION",
                        confidence=1.0,
                        override=True,
                        anomaly_score=None,
                    )
                    # Deduplicated vibration alert
                    existing_vib = session.query(Alert).filter(
                        Alert.type == "High Vibration",
                        Alert.status == "active",
                    ).first()
                    if not existing_vib:
                        alert_record = Alert(
                            type="High Vibration",
                            severity="warning",
                            message=(
                                f"Sustained high vibration detected: {vibration_count}/20 readings "
                                f"triggered vibration sensor. Check bearings and rotor balance. "
                                f"Temp: {reading['temp']:.1f}C, Current: {reading['current']:.1f}mA."
                            ),
                        )
                        session.add(alert_record)
                    session.add(prediction_record)
                    session.commit()
                    print(f"[Inference] HIGH_VIBRATION -- rule-based override ({vibration_count}/20 readings).")
                    continue

                # ════════════════════════════════════════════════════
                # STEP 3 — One-Class SVM inference
                # ════════════════════════════════════════════════════
                with _model_lock:
                    local_model = _model
                    local_scaler = _scaler

                if local_model is not None and local_scaler is not None:
                    feat_dict, full_array = extract_features(list(window))
                    model_array = get_model_input(full_array)
                    scaled = local_scaler.transform([model_array])
                    score = local_model.decision_function(scaled)[0]
                    raw_pred = local_model.predict(scaled)[0]  # 1=normal, -1=anomaly

                    # Map to label and confidence
                    if raw_pred == 1:
                        label = "Normal"
                        confidence = min(1.0, (score + 1.0) / 1.0)  # normalize score to 0-1
                        confidence = max(0.5, confidence)
                    else:  # anomaly (raw_pred == -1)
                        if score > -0.5:
                            label = "Warning"
                            confidence = min(0.99, abs(score) / 0.5)
                            confidence = max(0.5, confidence)
                        else:
                            label = "Fault"
                            confidence = min(0.99, abs(score) / 1.0)
                            confidence = max(0.5, confidence)

                    prediction_record = Prediction(
                        prediction=label,
                        confidence=float(confidence),
                        override=False,
                        anomaly_score=float(score),
                    )
                    session.add(prediction_record)

                    # Generate alerts for non-Normal states (DEDUPLICATED)
                    if label == "Warning":
                        existing = session.query(Alert).filter(
                            Alert.type == "Warning",
                            Alert.status == "active",
                        ).first()
                        if not existing:
                            alert_record = Alert(
                                type="Warning",
                                severity="warning",
                                message=(
                                    f"Anomaly detected (mild). Score: {score:.3f}. "
                                    f"Temp: {reading['temp']:.1f}C, "
                                    f"Current: {reading['current']:.1f}mA. "
                                    f"Monitor closely."
                                ),
                            )
                            session.add(alert_record)

                    elif label == "Fault":
                        existing = session.query(Alert).filter(
                            Alert.type == "Fault Detected",
                            Alert.status == "active",
                        ).first()
                        if not existing:
                            alert_record = Alert(
                                type="Fault Detected",
                                severity="fault",
                                message=(
                                    f"Strong anomaly detected. Score: {score:.3f}. "
                                    f"Temp: {reading['temp']:.1f}C, "
                                    f"Current: {reading['current']:.1f}mA. "
                                    f"Schedule maintenance immediately."
                                ),
                            )
                            session.add(alert_record)

                    session.commit()
                    print(
                        f"[Inference] Prediction: {label} "
                        f"(confidence: {confidence:.2%}, anomaly_score: {score:.3f})"
                    )

                # ════════════════════════════════════════════════════
                # STEP 4 — Simulation mode (no model loaded)
                # ════════════════════════════════════════════════════
                else:
                    rand = random.random()
                    if rand < 0.60:
                        sim_label = "Normal"
                    elif rand < 0.85:
                        sim_label = "Warning"
                    else:
                        sim_label = "Fault"

                    prediction_record = Prediction(
                        prediction=sim_label,
                        confidence=0.0,
                        override=False,
                        anomaly_score=None,
                    )
                    session.add(prediction_record)
                    session.commit()
                    print(f"[Inference] Simulation mode — {sim_label} (no model loaded)")

            except Exception as e:
                session.rollback()
                print(f"[Inference] DB error: {e}")
                traceback.print_exc()
            finally:
                session.close()

        except Exception as e:
            print(f"[Inference] Loop error: {e}")
            traceback.print_exc()
