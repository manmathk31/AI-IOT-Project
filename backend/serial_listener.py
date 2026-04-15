"""
Arduino serial listener for the Wind Turbine Monitoring System.
Replaces the simulator when real hardware is connected.
Reads JSON sensor data from Arduino via serial port and pushes into data queue.

Features:
- Auto-detect Arduino port
- Graceful handling of missing/malformed sensor fields
- Reconnection on serial errors (up to 5 retries)
- Falls back to simulator if no serial port found
"""

import json
import time
import traceback
from datetime import datetime


def find_arduino_port():
    """Scan available serial ports and return the first Arduino-like port."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("[Serial] No serial ports found.")
            return None

        for port in ports:
            print(f"[Serial] Found port: {port.device} — {port.description}")

        # Auto-select first port that looks like Arduino
        KEYWORDS = ["arduino", "usb", "ch340", "cp210", "ftdi", "serial"]
        for port in ports:
            desc_lower = (port.description or "").lower()
            mfr_lower = (port.manufacturer or "").lower()
            for kw in KEYWORDS:
                if kw in desc_lower or kw in mfr_lower:
                    print(f"[Serial] Auto-selected port: {port.device}")
                    return port.device

        # Fallback: return first available port
        print(f"[Serial] No Arduino keyword match. Using first port: {ports[0].device}")
        return ports[0].device

    except ImportError:
        print("[Serial] ERROR: pyserial not installed. Run: pip install pyserial")
        return None
    except Exception as e:
        print(f"[Serial] Error scanning ports: {e}")
        return None


def run_serial_listener(data_queue, port=None, baud_rate=9600):
    """
    Read sensor data from Arduino serial port and push into data_queue.
    Falls back to simulator if serial connection fails.

    Args:
        data_queue: queue.Queue to push sensor reading dicts into
        port: serial port (e.g. 'COM3'). Auto-detected if None.
        baud_rate: baud rate for serial connection (default 9600)
    """
    try:
        import serial
    except ImportError:
        print("[Serial] ERROR: pyserial not installed. Falling back to simulator.")
        _fallback_to_simulator(data_queue)
        return

    # Auto-detect port if not specified
    if port is None:
        port = find_arduino_port()

    if port is None:
        print("[Serial] No serial port available. Falling back to simulator.")
        _fallback_to_simulator(data_queue)
        return

    MAX_RETRIES = 5
    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            print(f"[Serial] Connecting to {port} at {baud_rate} baud...")
            ser = serial.Serial(port, baud_rate, timeout=2)
            time.sleep(2)  # Wait for Arduino to reset after serial connect
            print(f"[Serial] Connected to {port}. Reading data...")

            retry_count = 0  # Reset retry count on successful connection

            while True:
                try:
                    raw_line = ser.readline()
                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # Try to parse JSON
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        # Not valid JSON — skip (could be debug output from Arduino)
                        print(f"[Serial] Skipping non-JSON: {line[:80]}")
                        continue

                    if not isinstance(data, dict):
                        continue

                    # Build safe reading with fallbacks for missing fields
                    safe_reading = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "temp": data.get("temp", None),
                        "humidity": data.get("humidity", None),
                        "vibration": data.get("vibration", 0),
                        "current": data.get("current", None),
                        "flame": data.get("flame", 0),  # Missing flame = 0 (NEVER assume flame)
                    }

                    # Validate we have at minimum temp or current (at least one useful sensor)
                    if safe_reading["temp"] is None and safe_reading["current"] is None:
                        print("[Serial] Warning: Both temp and current are None. Skipping reading.")
                        continue

                    data_queue.put(safe_reading)

                    print(
                        f"[Serial] "
                        f"Temp: {safe_reading['temp']} | "
                        f"Humidity: {safe_reading['humidity']} | "
                        f"Current: {safe_reading['current']} | "
                        f"Vib: {safe_reading['vibration']} | "
                        f"Flame: {safe_reading['flame']}"
                    )

                except serial.SerialException as e:
                    print(f"[Serial] Connection lost: {e}")
                    break  # Break inner loop to trigger reconnection
                except Exception as e:
                    print(f"[Serial] Read error: {e}")
                    continue

            # Cleanup on disconnect
            try:
                ser.close()
            except Exception:
                pass

        except serial.SerialException as e:
            print(f"[Serial] Connection error: {e}")
        except Exception as e:
            print(f"[Serial] Unexpected error: {e}")
            traceback.print_exc()

        retry_count += 1
        if retry_count < MAX_RETRIES:
            wait_time = 3 * retry_count
            print(f"[Serial] Retry {retry_count}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)

            # Try to re-detect port (Arduino may have been reconnected on different port)
            new_port = find_arduino_port()
            if new_port:
                port = new_port
        else:
            print(f"[Serial] {MAX_RETRIES} retries exhausted. Falling back to simulator.")
            _fallback_to_simulator(data_queue)


def _fallback_to_simulator(data_queue):
    """Import and run the simulator as a fallback."""
    print("[Serial] >>> FALLBACK: Starting simulator instead of serial listener.")
    from simulator import run_simulator
    run_simulator(data_queue)
