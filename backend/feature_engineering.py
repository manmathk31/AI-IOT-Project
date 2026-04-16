"""
Feature engineering for the Wind Turbine ML pipeline.
Extracts 12 features from a sliding window of 20 sensor readings.
Handles None values gracefully so a missing sensor never crashes inference.
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
        tuple: (feature_dict, numpy_array) with 12 features in fixed order
    """
    # Extract raw values, tolerant of missing keys and None values
    raw_temps = [r.get("temp") for r in window]
    raw_currents = [r.get("current") for r in window]
    raw_humidities = [r.get("humidity") for r in window]
    vibrations = [r.get("vibration", 0) or 0 for r in window]
    flames = [r.get("flame", 0) or 0 for r in window]

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
    flame_count = sum(flames)

    humidity_mean = np.mean(humidities)
    humidity_std = np.std(humidities)

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
        "flame_count": int(flame_count),
        "humidity_mean": float(humidity_mean),
        "humidity_std": float(humidity_std),
    }

    # FIXED ORDER — must match Colab training notebook exactly
    feature_array = np.array([
        temp_mean,              # index 0
        temp_std,               # index 1
        temp_rate_of_change,    # index 2
        temp_max,               # index 3
        current_mean,           # index 4
        current_std,            # index 5
        current_spike,          # index 6
        current_rate_of_change, # index 7
        vibration_count,        # index 8
        flame_count,            # index 9
        humidity_mean,          # index 10
        humidity_std,           # index 11
    ])

    return feature_dict, feature_array
