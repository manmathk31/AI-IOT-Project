"""
Feature engineering for the Wind Turbine ML pipeline.
Extracts 13 features from a sliding window of 20 sensor readings.
Handles None values gracefully so a missing sensor never crashes inference.

One-Class SVM approach:
- full_array: 13 features (for logging and completeness)
- model_array: 12 features (vibration_count excluded — handled by rule-based override)
- flame_count is NOT a feature (flame handled by rule-based override before model)

Feature order contract (NEVER change this order):
  0: temp_mean
  1: temp_std
  2: temp_rate_of_change
  3: temp_max
  4: current_mean
  5: current_std
  6: current_spike
  7: current_rate_of_change
  8: vibration_count       ← excluded from model input (rule-based only)
  9: humidity_mean
 10: humidity_std
 11: temp_rms
 12: current_rms
"""

import numpy as np


def _safe_values(raw_list, default=0.0):
    """Replace None values with the mean of non-None values, or default if all None."""
    valid = [v for v in raw_list if v is not None]
    if not valid:
        return [default] * len(raw_list)
    fill = np.mean(valid)
    return [v if v is not None else fill for v in raw_list]


def extract_features(window):
    """
    Extract features from a window of 20 sensor readings.

    Args:
        window: list of 20 dicts, each with keys: temp, vibration, current, flame, humidity

    Returns:
        tuple: (feature_dict, full_array) with 13 features in fixed order
               full_array includes vibration_count at index 8.
               Use get_model_input(full_array) to get the 12-feature model input.
    """
    # Extract raw values, tolerant of missing keys and None values
    raw_temps = [r.get("temp") for r in window]
    raw_currents = [r.get("current") for r in window]
    raw_humidities = [r.get("humidity") for r in window]
    vibrations = [r.get("vibration", 0) or 0 for r in window]

    # Safe-fill None values
    temps = _safe_values(raw_temps, default=25.0)
    currents = _safe_values(raw_currents, default=0.0)
    humidities = _safe_values(raw_humidities, default=50.0)

    temp_mean = np.mean(temps)
    temp_std = np.std(temps)
    temp_rate_of_change = temps[-1] - temps[0]
    temp_max = np.max(temps)

    current_mean = np.mean(currents)
    current_std = np.std(currents)
    current_spike = np.max(currents) - current_mean
    current_rate_of_change = currents[-1] - currents[0]

    vibration_count = sum(vibrations)

    humidity_mean = np.mean(humidities)
    humidity_std = np.std(humidities)

    # RMS calculations
    temp_rms = np.sqrt(np.mean(np.array(temps) ** 2))
    current_rms = np.sqrt(np.mean(np.array(currents) ** 2))

    feature_dict = {
        "temp_mean": float(temp_mean),
        "temp_std": float(temp_std),
        "temp_rate_of_change": float(temp_rate_of_change),
        "temp_max": float(temp_max),
        "current_mean": float(current_mean),
        "current_std": float(current_std),
        "current_spike": float(current_spike),
        "current_rate_of_change": float(current_rate_of_change),
        "vibration_count": int(vibration_count),
        "humidity_mean": float(humidity_mean),
        "humidity_std": float(humidity_std),
        "temp_rms": float(temp_rms),
        "current_rms": float(current_rms),
    }

    # FIXED ORDER — must match Colab training notebook exactly
    # 13 features total (vibration_count at index 8 is for logging only)
    feature_array = np.array([
        temp_mean,              # index 0
        temp_std,               # index 1
        temp_rate_of_change,    # index 2
        temp_max,               # index 3
        current_mean,           # index 4
        current_std,            # index 5
        current_spike,          # index 6
        current_rate_of_change, # index 7
        vibration_count,        # index 8  ← NOT passed to model
        humidity_mean,          # index 9
        humidity_std,           # index 10
        temp_rms,               # index 11
        current_rms,            # index 12
    ])

    return feature_dict, feature_array


def get_model_input(full_array):
    """
    Remove vibration_count (index 8) from the full feature array
    to produce the 12-feature input expected by the One-Class SVM model.

    Args:
        full_array: numpy array of 13 features from extract_features()

    Returns:
        numpy array of 12 features (indices 0-7 + 9-12)
    """
    return np.delete(full_array, 8)
