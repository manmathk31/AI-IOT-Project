"""
Main FastAPI application for the Wind Turbine Monitoring System.
"""

import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import engine, SessionLocal, Base
from simulator import run_simulator
from inference_engine import run_inference
from routes.data import router as data_router
from routes.predictions import router as predictions_router
from routes.alerts import router as alerts_router
from routes.maintenance import router as maintenance_router

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

# ── Include Routers ──
app.include_router(data_router)
app.include_router(predictions_router)
app.include_router(alerts_router)
app.include_router(maintenance_router)


# ── Startup Event ──
@app.on_event("startup")
def startup_event():
    """Create tables and start background threads on startup."""
    # Create all database tables
    Base.metadata.create_all(bind=engine)

    # Create shared queue for sensor data
    sensor_queue = queue.Queue()

    # Start simulator thread
    t1 = threading.Thread(
        target=run_simulator,
        args=(sensor_queue,),
        daemon=True,
    )
    t1.start()

    # Start inference engine thread
    t2 = threading.Thread(
        target=run_inference,
        args=(sensor_queue, SessionLocal),
        daemon=True,
    )
    t2.start()

    print("System started. Simulator and inference engine running.")


# ── Root Endpoint ──
@app.get("/")
def root():
    return {
        "status": "Wind Turbine Monitoring API is running",
        "docs": "/docs",
    }
