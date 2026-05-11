"""
Data Collector — Standalone script for collecting NORMAL operating data from Arduino.
Used by the hardware team to build training datasets for the One-Class SVM model.

One-Class SVM only needs normal operating data for training.
No --label argument needed — all collected data is assumed normal.

Usage:
    python data_collector.py --port COM3 --session normal_load
    python data_collector.py --port COM3 --session idle --duration 600
    python data_collector.py --port COM3 --session high_speed --output data/high_speed.csv
    python data_collector.py --list-ports
    python data_collector.py --help
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


CSV_HEADER = ["timestamp", "temp", "humidity", "vibration", "current", "flame"]


def list_ports():
    """Print all available serial ports."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in ports:
        print(f"  {p.device:12s} — {p.description}")


def collect_data(port, baud, session, output_file, duration):
    """
    Collect normal operating data from Arduino and save to CSV.
    No label column — One-Class SVM trains on normal data only.

    Args:
        port: Serial port (e.g. COM3)
        baud: Baud rate
        session: Session description (e.g. "idle", "normal_load", "high_speed")
        output_file: Path to output CSV file
        duration: Seconds to collect (0 = until Ctrl+C)
    """
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Check if file exists (to decide whether to write header)
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    print(f"\n{'=' * 70}")
    print(f"  Wind Turbine Data Collector — One-Class SVM Training Data")
    print(f"{'=' * 70}")
    print()
    print(f"  ⚠️  Collecting NORMAL operating data for One-Class SVM training.")
    print(f"  ⚠️  Run the turbine in its normal operating condition now.")
    print(f"  ⚠️  Do NOT simulate faults during this collection.")
    print()
    print(f"  Port:     {port}")
    print(f"  Baud:     {baud}")
    print(f"  Session:  {session}")
    print(f"  Output:   {output_file}")
    print(f"  Duration: {'Until Ctrl+C' if duration == 0 else f'{duration} seconds'}")
    print(f"  Mode:     {'Appending to existing file' if file_exists else 'Creating new file'}")
    print(f"  Target:   At least 600 readings (10 minutes at 1Hz)")
    print(f"{'=' * 70}")
    print()

    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Wait for Arduino reset
        print(f"Connected to {port}. Collecting NORMAL data...")
        print()
    except serial.SerialException as e:
        print(f"ERROR: Could not open {port}: {e}")
        print("\nAvailable ports:")
        list_ports()
        sys.exit(1)

    count = 0
    start_time = time.time()

    try:
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(CSV_HEADER)
                f.flush()

            while True:
                # Check duration
                if duration > 0:
                    elapsed = time.time() - start_time
                    if elapsed >= duration:
                        print(f"\nDuration reached ({duration}s). Stopping.")
                        break

                # Read line from serial
                try:
                    raw = ser.readline()
                    if not raw:
                        continue

                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # Parse JSON
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip non-JSON output

                    if not isinstance(data, dict):
                        continue

                    # Extract fields with defaults
                    timestamp = datetime.utcnow().isoformat()
                    temp = data.get("temp", 0.0)
                    humidity = data.get("humidity", 0.0)
                    vibration = data.get("vibration", 0)
                    current = data.get("current", 0.0)
                    flame = data.get("flame", 0)

                    # Write CSV row (no label column)
                    row = [timestamp, temp, humidity, vibration, current, flame]
                    writer.writerow(row)
                    count += 1

                    # Flush every 10 rows for safety
                    if count % 10 == 0:
                        f.flush()

                    # Progress logging
                    if duration > 0:
                        remaining = duration - int(time.time() - start_time)
                        progress = f"{count}/{duration}"
                        time_info = f"({remaining}s remaining)"
                    else:
                        progress = f"{count}"
                        time_info = ""

                    if count % 10 == 0 or count == 1:
                        print(
                            f"  [{session}] Collected {progress} readings {time_info} — "
                            f"Temp: {temp:.1f}°C  "
                            f"Humidity: {humidity:.1f}%  "
                            f"Current: {current:.1f}mA  "
                            f"Vib: {vibration}  "
                            f"Flame: {flame}"
                        )

                except serial.SerialException as e:
                    print(f"\nSerial error: {e}")
                    break

    except KeyboardInterrupt:
        print(f"\n\nStopped by user (Ctrl+C).")

    finally:
        try:
            ser.close()
        except Exception:
            pass

    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Collection complete!")
    print(f"  Total readings:  {count}")
    print(f"  Duration:        {elapsed_total:.1f} seconds")
    print(f"  Session:         {session}")
    print(f"  Saved to:        {output_file}")
    if count < 600:
        print(f"  ⚠ WARNING: Only {count} readings. Aim for 600+ for good training data.")
    else:
        print(f"  ✓ Sufficient data collected for One-Class SVM training.")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect NORMAL operating data from Arduino for One-Class SVM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
One-Class SVM Training Data Collection
=======================================
This tool collects ONLY normal operating data. The One-Class SVM model
learns what "normal" looks like and flags anything outside that as anomalous.

Examples:
  python data_collector.py --port COM3 --session idle
  python data_collector.py --port COM3 --session normal_load --duration 600
  python data_collector.py --port COM3 --session high_speed --output data/high_speed.csv
  python data_collector.py --list-ports

After collection, upload the CSV to Google Colab and run colab_training.py.
        """,
    )

    parser.add_argument("--port", type=str, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument(
        "--session",
        type=str,
        default="normal",
        help="Session description for logging (e.g. idle, normal_load, high_speed). Default: normal",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: data/normal_<session>.csv)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Duration in seconds (default: 600 = 10 minutes, 0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available serial ports and exit",
    )

    args = parser.parse_args()

    if args.list_ports:
        list_ports()
        sys.exit(0)

    if not args.port:
        parser.error("--port is required (use --help for usage)")

    # Auto-generate output filename based on session
    output_file = args.output or f"data/normal_{args.session}.csv"

    collect_data(args.port, args.baud, args.session, output_file, args.duration)


if __name__ == "__main__":
    main()
