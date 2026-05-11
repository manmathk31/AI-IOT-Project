"""
Model Trainer — In-app One-Class SVM training pipeline.

Replaces the Google Colab workflow. Loads collected CSV data,
extracts sliding window features, trains a One-Class SVM,
computes permutation importances, and saves model + scaler + metrics.

All training happens synchronously (One-Class SVM on 600-2000 samples
trains in under 2 seconds — no need for async).
"""

import os
import json
import glob
import time
from datetime import datetime

import numpy as np
import pandas as pd

WINDOW_SIZE = 20
STEP = 5  # Step of 5 to reduce overlap and speed up training

ML_DIR = os.path.join(os.path.dirname(__file__), "ml")
TRAINING_DATA_DIR = os.path.join(ML_DIR, "training_data")
MODEL_PATH = os.path.join(ML_DIR, "model_v1.pkl")
SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(ML_DIR, "metrics.json")

# 12 model features (same order as feature_engineering.py, excluding vibration_count)
FEATURE_NAMES = [
    "temp_mean", "temp_std", "temp_rate_of_change", "temp_max",
    "current_mean", "current_std", "current_spike", "current_rate_of_change",
    "humidity_mean", "humidity_std",
    "temp_rms", "current_rms",
]


def _extract_features_from_window(window_df):
    """
    Extract 12 model features from a DataFrame window.
    Same feature order as backend feature_engineering.py (excluding vibration_count).
    """
    temps = window_df["temp"].values.astype(float)
    currents = window_df["current"].values.astype(float)

    if "humidity" in window_df.columns:
        humidities = window_df["humidity"].values.astype(float)
    else:
        humidities = np.zeros(len(temps))

    temp_mean = np.mean(temps)
    temp_std = np.std(temps)
    temp_rate_of_change = temps[-1] - temps[0]
    temp_max = np.max(temps)

    current_mean = np.mean(currents)
    current_std = np.std(currents)
    current_spike = np.max(currents) - current_mean
    current_rate_of_change = currents[-1] - currents[0]

    humidity_mean = np.mean(humidities)
    humidity_std = np.std(humidities)

    temp_rms = np.sqrt(np.mean(temps ** 2))
    current_rms = np.sqrt(np.mean(currents ** 2))

    return [
        temp_mean, temp_std, temp_rate_of_change, temp_max,
        current_mean, current_std, current_spike, current_rate_of_change,
        humidity_mean, humidity_std,
        temp_rms, current_rms,
    ]


def get_model_status():
    """
    Check model files and return status info.

    Returns:
        dict with model state information
    """
    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)
    metrics_exists = os.path.exists(METRICS_PATH)

    result = {
        "model_loaded": model_exists and scaler_exists,
        "model_file_exists": model_exists,
        "scaler_file_exists": scaler_exists,
        "metrics_file_exists": metrics_exists,
    }

    if model_exists:
        stat = os.stat(MODEL_PATH)
        result["model_size_bytes"] = stat.st_size
        result["model_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    if metrics_exists:
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            result["training_samples"] = metrics.get("training_samples", 0)
            result["normal_detection_rate"] = metrics.get("normal_detection_rate", 0)
            result["trained_at"] = metrics.get("trained_at", None)
        except Exception:
            pass

    return result


def train_model():
    """
    Train a One-Class SVM on all collected CSV data.

    Returns:
        dict with training results and metrics, or error
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    import joblib

    start_time = time.time()

    # 1. Find and load all training CSVs
    if not os.path.exists(TRAINING_DATA_DIR):
        return {"error": "No training data directory found. Collect data first."}

    csv_files = sorted(glob.glob(os.path.join(TRAINING_DATA_DIR, "*.csv")))
    if not csv_files:
        return {"error": "No training data CSV files found. Collect data first."}

    print(f"[Trainer] Loading {len(csv_files)} CSV files...")
    dfs = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
            print(f"[Trainer]   {os.path.basename(fp)}: {len(df)} rows")
        except Exception as e:
            print(f"[Trainer]   Error loading {fp}: {e}")

    if not dfs:
        return {"error": "Could not load any training data files."}

    df = pd.concat(dfs, ignore_index=True)
    print(f"[Trainer] Total raw rows: {len(df)}")

    # 2. Clean data
    before = len(df)
    df = df.dropna(subset=["temp", "current"])
    # Fill missing humidity with 50.0
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].fillna(50.0)
    else:
        df["humidity"] = 50.0

    after = len(df)
    if before != after:
        print(f"[Trainer] Dropped {before - after} rows with NaN in temp/current.")

    if len(df) < WINDOW_SIZE:
        return {"error": f"Only {len(df)} valid rows. Need at least {WINDOW_SIZE}."}

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # 3. Extract features with sliding windows
    feature_rows = []
    for i in range(0, len(df) - WINDOW_SIZE + 1, STEP):
        window = df.iloc[i:i + WINDOW_SIZE]
        features = _extract_features_from_window(window)
        feature_rows.append(features)

    X = np.array(feature_rows)
    print(f"[Trainer] Extracted {len(X)} feature windows.")

    if len(X) < 10:
        return {"error": f"Only {len(X)} feature windows. Need at least 10. Collect more data."}

    # 4. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Train One-Class SVM
    print("[Trainer] Training One-Class SVM (kernel=rbf, nu=0.05)...")
    model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
    model.fit(X_scaled)

    # 6. Evaluate on training data
    pred_train = model.predict(X_scaled)
    scores_train = model.decision_function(X_scaled)

    normal_rate = float((pred_train == 1).mean())
    anomaly_rate = float((pred_train == -1).mean())

    print(f"[Trainer] Normal detection rate: {normal_rate:.2%}")
    print(f"[Trainer] Score range: [{scores_train.min():.3f}, {scores_train.max():.3f}]")

    # 7. Compute permutation importances
    original_mean_score = scores_train.mean()
    importances = []

    for i in range(X_scaled.shape[1]):
        X_permuted = X_scaled.copy()
        np.random.shuffle(X_permuted[:, i])
        score_permuted = model.decision_function(X_permuted).mean()
        importance = abs(original_mean_score - score_permuted)
        importances.append(importance)

    # Normalize to sum to 1
    total_imp = sum(importances)
    if total_imp > 0:
        importances = [imp / total_imp for imp in importances]

    # 8. Save model, scaler, metrics
    os.makedirs(ML_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Trainer] Saved model_v1.pkl and scaler.pkl")

    train_duration = time.time() - start_time

    metrics = {
        "model_type": "One-Class SVM",
        "kernel": "rbf",
        "nu": 0.05,
        "training_samples": int(len(X)),
        "raw_data_rows": int(len(df)),
        "csv_files_used": len(csv_files),
        "normal_detection_rate": round(normal_rate, 4),
        "anomaly_detection_rate": round(anomaly_rate, 4),
        "false_positive_rate": round(anomaly_rate, 4),
        "score_mean": round(float(scores_train.mean()), 4),
        "score_std": round(float(scores_train.std()), 4),
        "score_min": round(float(scores_train.min()), 4),
        "score_max": round(float(scores_train.max()), 4),
        "threshold_warning": -0.2,
        "threshold_fault": -0.5,
        "train_duration_seconds": round(train_duration, 2),
        "trained_at": datetime.now().isoformat(),
        "feature_importances": {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, importances)
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[Trainer] Saved metrics.json. Training completed in {train_duration:.2f}s.")

    return {
        "success": True,
        "metrics": metrics,
        "message": f"Model trained on {len(X)} samples in {train_duration:.1f}s. Normal detection rate: {normal_rate:.1%}",
    }


def delete_model():
    """
    Delete model files (model_v1.pkl, scaler.pkl).
    Keeps metrics.json for reference.

    Returns:
        dict with deletion result
    """
    deleted = []
    for fp in [MODEL_PATH, SCALER_PATH]:
        if os.path.exists(fp):
            os.remove(fp)
            deleted.append(os.path.basename(fp))

    print(f"[Trainer] Deleted model files: {deleted}")
    return {
        "deleted": deleted,
        "message": "Model deleted. System will run in simulation mode.",
    }
