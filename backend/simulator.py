"""
Sensor data simulator for the Wind Turbine Monitoring System.
Cycles through Normal → Warning → Fault → HIGH_VIBRATION → Critical Flame states.
Now includes humidity from DHT11 sensor simulation.

Updated for One-Class SVM anomaly detection approach.
Console output shows simulated anomaly scores alongside state names.
"""

import time
import random
from datetime import datetime


STATE_NAMES = {
    0: "Normal",
    1: "Warning",
    2: "Fault",
    3: "HIGH_VIBRATION",
    4: "Critical Flame",
}
STATE_DURATIONS = {0: 30, 1: 15, 2: 10, 3: 8, 4: 5}

# Simulated anomaly scores for each state (mimics One-Class SVM decision_function output)
STATE_SCORE_RANGES = {
    0: (0.2, 0.8),       # Normal: positive scores
    1: (-0.4, -0.1),     # Warning: mild negative scores
    2: (-1.0, -0.5),     # Fault: strong negative scores
    3: (None, None),      # HIGH_VIBRATION: rule-based, no score
    4: (None, None),      # Critical Flame: rule-based, no score
}


def run_simulator(data_queue):
    """
    Run the sensor simulator forever, pushing readings into data_queue every second.

    Args:
        data_queue: queue.Queue instance to push sensor dicts into
    """
    state = 0
    counter = 0

    print("[Simulator] Starting sensor data simulator (One-Class SVM mode)...")
    print("[Simulator] State cycle: Normal -> Warning -> Fault -> HIGH_VIBRATION -> Critical Flame")

    while True:
        # Generate reading based on current state
        if state == 0:  # Normal
            temp = max(30, min(55, random.gauss(42, 3)))
            humidity = max(40, min(65, random.gauss(52, 5)))
            current = max(240, min(330, random.gauss(285, 15)))
            vibration = 1 if random.random() < 0.05 else 0
            flame = 0

        elif state == 1:  # Warning
            temp = max(52, min(72, random.gauss(62, 4)))
            humidity = max(35, min(58, random.gauss(45, 5)))
            current = max(340, min(430, random.gauss(385, 20)))
            vibration = 1 if random.random() < 0.35 else 0
            flame = 0

        elif state == 2:  # Fault
            temp = max(68, min(84, random.gauss(76, 3)))
            humidity = max(28, min(50, random.gauss(38, 4)))
            current = max(445, min(500, random.gauss(472, 12)))
            vibration = 1 if random.random() < 0.75 else 0
            flame = 0

        elif state == 3:  # HIGH_VIBRATION
            temp = max(65, min(72, random.gauss(68, 2)))
            humidity = max(38, min(55, random.gauss(48, 4)))
            current = max(300, min(400, random.gauss(350, 15)))
            vibration = 1  # All readings have vibration active
            flame = 0

        elif state == 4:  # Critical Flame
            temp = max(74, min(88, random.gauss(82, 2)))
            humidity = max(18, min(38, random.gauss(28, 3)))
            current = max(470, min(500, random.gauss(490, 8)))
            vibration = 1
            flame = 1

        reading = {
            "timestamp": datetime.utcnow().isoformat(),
            "temp": round(temp, 2),
            "humidity": round(humidity, 2),
            "vibration": vibration,
            "current": round(current, 2),
            "flame": flame,
        }

        data_queue.put(reading)

        # Generate simulated anomaly score for console display
        score_range = STATE_SCORE_RANGES[state]
        if score_range[0] is not None:
            sim_score = round(random.uniform(score_range[0], score_range[1]), 3)
            score_display = f"score: {sim_score:+.3f}"
        else:
            score_display = "score: OVERRIDE"

        print(
            f"[Simulator] State: {STATE_NAMES[state]:15s} | "
            f"Temp: {reading['temp']:6.2f}C | "
            f"Humidity: {reading['humidity']:5.2f}% | "
            f"Current: {reading['current']:7.2f}mA | "
            f"Vib: {reading['vibration']} | "
            f"Flame: {reading['flame']} | "
            f"{score_display}"
        )

        counter += 1

        # Advance state when duration expires
        if counter >= STATE_DURATIONS[state]:
            counter = 0
            state = (state + 1) % 5  # 5 states now (0-4)
            print(f"[Simulator] >>> Transitioning to state: {STATE_NAMES[state]}")

        time.sleep(1)
