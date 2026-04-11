"""
SQLAlchemy ORM models for the Wind Turbine Monitoring System.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    temp = Column(Float)          # Temperature in Celsius
    vibration = Column(Integer)   # 0 or 1
    current = Column(Float)       # Current in mA
    flame = Column(Integer)       # 0 or 1


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prediction = Column(String)   # Normal / Warning / Fault / CRITICAL_FLAME
    confidence = Column(Float)    # 0.0 to 1.0
    override = Column(Boolean, default=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String)         # e.g. "Fault Detected", "Flame Detected", "Warning"
    severity = Column(String)     # "info" | "warning" | "fault" | "critical"
    message = Column(String)
    status = Column(String, default="active")  # "active" | "resolved"


class MaintenanceTask(Base):
    __tablename__ = "maintenance_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    engineer = Column(String)
    machine = Column(String)
    notes = Column(String)
    status = Column(String, default="Pending")  # "Pending" | "In Progress" | "Completed"
