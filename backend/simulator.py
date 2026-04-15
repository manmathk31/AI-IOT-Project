"""
Sensor data simulator for the Wind Turbine Monitoring System.
Cycles through Normal → Warning → Fault → Critical Flame states.
Now includes humidity from DHT11 sensor simulation.
"""

import time
import random
from datetime import datetime


STATE_NAMES = {0: "Normal", 1: "Warning", 2: "Fault", 3: "Critical Flame"}
STATE_DURATIONS = {0: 30, 1: 15, 2: 10, 3: 5}


def run_simulator(data_queue):
    """
    Run the sensor simulator forever, pushing readings into data_queue every second.

    Args:
        data_queue: queue.Queue instance to push sensor dicts into
    """
    state = 0
    counter = 0

    print("[Simulator] Starting sensor data simulator...")

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

        elif state == 3:  # Critical Flame
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

        print(
            f"[Simulator] State: {STATE_NAMES[state]:15s} | "
            f"Temp: {reading['temp']:6.2f}°C | "
            f"Humidity: {reading['humidity']:5.2f}% | "
            f"Current: {reading['current']:7.2f}mA | "
            f"Vib: {reading['vibration']} | "
            f"Flame: {reading['flame']}"
        )

        counter += 1

        # Advance state when duration expires
        if counter >= STATE_DURATIONS[state]:
            counter = 0
            state = (state + 1) % 4
            print(f"[Simulator] >>> Transitioning to state: {STATE_NAMES[state]}")

        time.sleep(1)
