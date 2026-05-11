"""
Main FastAPI application for the Wind Turbine Monitoring System.
Serves both the REST API and the frontend static files.
Supports switching between simulator and real hardware via USE_HARDWARE env var.
"""

import os
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import engine, SessionLocal, Base
from routes.data import router as data_router
from routes.alerts import router as alerts_router
from routes.predictions import router as predictions_router
from routes.maintenance import router as maintenance_router
from routes.model import router as model_router

# ── Environment ──
USE_HARDWARE = os.getenv("USE_HARDWARE", "false").lower() == "true"

# ── Create App ──
app = FastAPI(
    title="Wind Turbine Monitoring API",
    version="1.0.0",
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API Routers ──
app.include_router(data_router)
app.include_router(predictions_router)
app.include_router(alerts_router)
app.include_router(maintenance_router)
app.include_router(model_router)

# ── Serve Static Files (CSS, JS, images) ──
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Startup Event ──
@app.on_event("startup")
def startup_event():
    """Create tables and start background threads on startup."""
    # Create all database tables
    Base.metadata.create_all(bind=engine)

    # Create shared queue for sensor data
    sensor_queue = queue.Queue()

    # ── Data source: Hardware or Simulator ──
    if USE_HARDWARE:
        from serial_listener import run_serial_listener
        t1 = threading.Thread(
            target=run_serial_listener,
            args=(sensor_queue,),
            daemon=True,
        )
        print("[App] Starting with REAL HARDWARE mode — Serial listener active")
    else:
        from simulator import run_simulator
        t1 = threading.Thread(
            target=run_simulator,
            args=(sensor_queue,),
            daemon=True,
        )
        print("[App] Starting with SIMULATOR mode — No hardware required")

    t1.start()

    # Start inference engine thread
    from inference_engine import run_inference
    t2 = threading.Thread(
        target=run_inference,
        args=(sensor_queue, SessionLocal),
        daemon=True,
    )
    t2.start()

    print("System started. Data source and inference engine running.")


# ── Serve Frontend ──
@app.get("/")
async def serve_frontend():
    """Serve the main dashboard HTML."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Catch-all: always serve index.html for non-API, non-static paths ──
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Fallback route — serves index.html for any unknown path."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
