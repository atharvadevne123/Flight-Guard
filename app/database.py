"""SQLAlchemy database models and session management."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./flight_guard.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Flight(Base):
    """Raw flight record for training data."""

    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, index=True)
    carrier = Column(String(10), nullable=False, index=True)
    origin = Column(String(5), nullable=False, index=True)
    destination = Column(String(5), nullable=False, index=True)
    scheduled_departure = Column(DateTime, nullable=False)
    actual_departure = Column(DateTime, nullable=True)
    delay_minutes = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    aircraft_type = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionLog(Base):
    """Log of all predictions made by the API."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    correlation_id = Column(String(36), nullable=True, index=True)
    carrier = Column(String(10), nullable=False)
    origin = Column(String(5), nullable=False)
    destination = Column(String(5), nullable=False)
    scheduled_hour = Column(Integer, nullable=False)
    day_of_week = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    delay_probability = Column(Float, nullable=False)
    predicted_class_name = Column(String(20), nullable=False)
    model_version = Column(String(20), nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    """KS-test drift detection reports."""

    __tablename__ = "drift_reports"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(50), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Integer, nullable=False)
    reference_size = Column(Integer, nullable=True)
    current_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class RetrainLog(Base):
    """History of model retraining runs."""

    __tablename__ = "retrain_logs"

    id = Column(Integer, primary_key=True, index=True)
    trigger = Column(String(50), nullable=False)
    auc_before = Column(Float, nullable=True)
    auc_after = Column(Float, nullable=True)
    n_samples = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db() -> Session:
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables."""
    Base.metadata.create_all(bind=engine)
