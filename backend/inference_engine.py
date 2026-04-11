"""
ML inference engine for the Wind Turbine Monitoring System.
Consumes sensor data from a queue, runs sliding-window feature extraction,
and classifies turbine state using a trained ML model.
"""

import os
import collections
import traceback
from datetime import datetime

import numpy as np

from models import SensorReading, Prediction, Alert
from feature_engineering import extract_features


def run_inference(data_queue, session_factory):
    """
    Run the inference engine forever, consuming readings from data_queue
    and writing predictions + alerts to the database.

    Args:
        data_queue: queue.Queue with sensor reading dicts
        session_factory: SQLAlchemy sessionmaker (SessionLocal)
    """
    # ── Load ML model and scaler ──
    model = None
    scaler = None

    model_path = os.path.join(os.path.dirname(__file__), "ml", "model_v1.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "ml", "scaler.pkl")

    try:
        import joblib
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("[Inference] Model and scaler loaded successfully.")
        else:
            print(
                "[Inference] WARNING: Model files not found — running in simulation mode. "
                "Predictions will be placeholder only."
            )
    except Exception as e:
        print(f"[Inference] ERROR loading model: {e}")
        model = None
        scaler = None

    label_map = {0: "Normal", 1: "Warning", 2: "Fault", 3: "CRITICAL_FLAME"}
    window = collections.deque(maxlen=20)

    print("[Inference] Inference engine started. Waiting for readings...")

    while True:
        try:
            # 1. Get next reading (blocking)
            reading = data_queue.get()

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
                    vibration=reading["vibration"],
                    current=reading["current"],
                    flame=reading["flame"],
                )
                session.add(sensor_record)

                # 6. Check for flame override
                if reading["flame"] == 1:
                    prediction_record = Prediction(
                        prediction="CRITICAL_FLAME",
                        confidence=1.0,
                        override=True,
                    )
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
                    print("[Inference] 🔥 CRITICAL_FLAME — override prediction saved.")
                    continue

                # 7. Model-based prediction (no flame)
                if model is not None and scaler is not None:
                    feat_dict, feat_array = extract_features(list(window))
                    scaled_array = scaler.transform([feat_array])
                    probabilities = model.predict_proba(scaled_array)[0]
                    predicted_index = int(np.argmax(probabilities))
                    confidence = float(np.max(probabilities))
                    label = label_map.get(predicted_index, "Normal")

                    prediction_record = Prediction(
                        prediction=label,
                        confidence=confidence,
                        override=False,
                    )
                    session.add(prediction_record)

                    # Generate alerts for non-Normal states
                    if label == "Warning":
                        alert_record = Alert(
                            type="Warning",
                            severity="warning",
                            message=(
                                f"System showing warning signs. "
                                f"Temp: {reading['temp']:.1f}C, "
                                f"Current: {reading['current']:.1f}mA. "
                                f"Monitor closely."
                            ),
                        )
                        session.add(alert_record)

                    elif label == "Fault":
                        alert_record = Alert(
                            type="Fault Detected",
                            severity="fault",
                            message=(
                                f"Fault condition detected. "
                                f"Temp: {reading['temp']:.1f}C, "
                                f"Current: {reading['current']:.1f}mA, "
                                f"Vibration active. "
                                f"Schedule maintenance immediately."
                            ),
                        )
                        session.add(alert_record)

                    session.commit()
                    print(
                        f"[Inference] Prediction: {label} "
                        f"(confidence: {confidence:.2%})"
                    )

                # 8. No model loaded — placeholder prediction
                else:
                    prediction_record = Prediction(
                        prediction="Simulating — No Model",
                        confidence=0.0,
                        override=False,
                    )
                    session.add(prediction_record)
                    session.commit()
                    print("[Inference] No model — placeholder prediction saved.")

            except Exception as e:
                session.rollback()
                print(f"[Inference] DB error: {e}")
                traceback.print_exc()
            finally:
                session.close()

        except Exception as e:
            print(f"[Inference] Loop error: {e}")
            traceback.print_exc()
