"""
Feature engineering for the Wind Turbine ML pipeline.
Extracts 10 features from a sliding window of 20 sensor readings.
"""

import numpy as np


def extract_features(window):
    """
    Extract features from a window of 20 sensor readings.

    Args:
        window: list of 20 dicts, each with keys: temp, vibration, current, flame

    Returns:
        tuple: (feature_dict, numpy_array) with 10 features in fixed order
    """
    temps = [r["temp"] for r in window]
    currents = [r["current"] for r in window]
    vibrations = [r["vibration"] for r in window]
    flames = [r["flame"] for r in window]

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
    }

    feature_array = np.array([
        temp_mean,
        temp_std,
        temp_rate_of_change,
        temp_max,
        current_mean,
        current_std,
        current_spike,
        current_rate_of_change,
        vibration_count,
        flame_count,
    ])

    return feature_dict, feature_array
