"""
Data Collector Service — Thread-safe in-memory data collection for training.

When collection is active, the inference engine feeds each sensor reading
into this service. Readings are buffered and written to a CSV file in
backend/ml/training_data/ for use by the model trainer.

This service is designed to be called from the inference engine thread
and controlled from the API routes — all state is protected by locks.


"""

import os
import csv
import threading
import time
from datetime import datetime


# ── Module-level state (thread-safe) ──
_lock = threading.Lock()
_collecting = False
_csv_file = None
_csv_writer = None
_csv_path = None
_start_time = None
_count = 0

# Directory for training data CSVs
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "ml", "training_data")
CSV_HEADER = ["timestamp", "temp", "humidity", "vibration", "current", "flame"]


def _ensure_dir():
    """Create training_data directory if it doesn't exist."""
    if not os.path.exists(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)


def start_collection():
    """
    Begin collecting sensor readings into a CSV file.

    Returns:
        dict with status info, or error message
    """
    global _collecting, _csv_file, _csv_writer, _csv_path, _start_time, _count

    with _lock:
        if _collecting:
            return {"error": "Already collecting data", "status": "collecting"}

        _ensure_dir()

        # Generate timestamped filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _csv_path = os.path.join(TRAINING_DATA_DIR, f"normal_{ts}.csv")

        try:
            _csv_file = open(_csv_path, "w", newline="", encoding="utf-8")
            _csv_writer = csv.writer(_csv_file)
            _csv_writer.writerow(CSV_HEADER)
            _csv_file.flush()
        except Exception as e:
            _csv_file = None
            _csv_writer = None
            _csv_path = None
            return {"error": f"Failed to create CSV: {e}"}

        _collecting = True
        _start_time = time.time()
        _count = 0

        print(f"[Collector] Started collecting normal data to {_csv_path}")
        return {
            "status": "collecting",
            "file": os.path.basename(_csv_path),
            "message": "Collection started. Run turbine in normal conditions.",
        }


def stop_collection():
    """
    Stop collecting and close the CSV file.

    Returns:
        dict with collection summary
    """
    global _collecting, _csv_file, _csv_writer, _csv_path, _start_time, _count

    with _lock:
        if not _collecting:
            return {"error": "Not currently collecting", "status": "idle"}

        _collecting = False
        duration = time.time() - _start_time if _start_time else 0
        final_count = _count
        final_path = _csv_path

        # Close file
        try:
            if _csv_file:
                _csv_file.flush()
                _csv_file.close()
        except Exception:
            pass

        _csv_file = None
        _csv_writer = None

        # Delete file if too few readings (less than 20 — can't even fill one window)
        if final_count < 20 and final_path and os.path.exists(final_path):
            os.remove(final_path)
            print(f"[Collector] Deleted {final_path} — only {final_count} readings (need >= 20)")
            return {
                "status": "idle",
                "readings": final_count,
                "duration_seconds": round(duration, 1),
                "error": f"Only {final_count} readings collected. Need at least 20. Data discarded.",
            }

        print(f"[Collector] Stopped. {final_count} readings in {duration:.1f}s saved to {final_path}")
        return {
            "status": "idle",
            "readings": final_count,
            "duration_seconds": round(duration, 1),
            "file": os.path.basename(final_path) if final_path else None,
            "message": f"Collected {final_count} readings in {duration:.0f}s.",
        }


def add_reading(reading):
    """
    Add a sensor reading to the collection buffer.
    Called from the inference engine thread on every tick.
    Only writes if collection is active.

    Args:
        reading: dict with keys: temp, humidity, vibration, current, flame
    """
    global _count

    with _lock:
        if not _collecting or _csv_writer is None:
            return

        try:
            row = [
                datetime.utcnow().isoformat(),
                reading.get("temp", 0.0),
                reading.get("humidity", 0.0),
                reading.get("vibration", 0),
                reading.get("current", 0.0),
                reading.get("flame", 0),
            ]
            _csv_writer.writerow(row)
            _count += 1

            # Flush every 10 rows
            if _count % 10 == 0 and _csv_file:
                _csv_file.flush()

        except Exception as e:
            print(f"[Collector] Write error: {e}")


def get_stats():
    """
    Get current collection statistics.

    Returns:
        dict with status, count, duration, minimum met flag
    """
    with _lock:
        if _collecting:
            duration = time.time() - _start_time if _start_time else 0
            return {
                "status": "collecting",
                "readings": _count,
                "duration_seconds": round(duration, 1),
                "minimum_met": _count >= 600,
                "file": os.path.basename(_csv_path) if _csv_path else None,
            }
        else:
            return {
                "status": "idle",
                "readings": 0,
                "duration_seconds": 0,
                "minimum_met": False,
                "file": None,
            }


def is_collecting():
    """Thread-safe check if collection is active."""
    with _lock:
        return _collecting


def get_training_data_info():
    """
    List all training data CSV files with their stats.

    Returns:
        dict with files list and totals
    """
    _ensure_dir()
    files = []
    total_rows = 0

    for fname in sorted(os.listdir(TRAINING_DATA_DIR)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(TRAINING_DATA_DIR, fname)
        size = os.path.getsize(fpath)

        # Count rows (subtract 1 for header)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1
        except Exception:
            rows = 0

        total_rows += max(0, rows)
        files.append({
            "name": fname,
            "size_bytes": size,
            "rows": max(0, rows),
            "created": datetime.fromtimestamp(os.path.getctime(fpath)).isoformat(),
        })

    return {
        "files": files,
        "total_files": len(files),
        "total_rows": total_rows,
    }


def delete_training_data():
    """Delete all training data CSV files."""
    _ensure_dir()
    deleted = 0
    for fname in os.listdir(TRAINING_DATA_DIR):
        if fname.endswith(".csv"):
            try:
                os.remove(os.path.join(TRAINING_DATA_DIR, fname))
                deleted += 1
            except Exception:
                pass

    print(f"[Collector] Deleted {deleted} training data files.")
    return {"deleted": deleted}
