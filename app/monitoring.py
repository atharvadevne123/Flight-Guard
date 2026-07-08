"""Model monitoring: drift detection, prediction logging, health checks."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

from app.database import DriftReport, PredictionLog

logger = logging.getLogger(__name__)

# Rolling window for online drift detection (last N predictions)
_prediction_window: deque[float] = deque(maxlen=500)
_reference_scores: list[float] = []


def compute_drift(reference: list[float], current: list[float]) -> dict[str, Any]:
    """Compute KS-test drift between reference and current distributions."""
    if len(reference) < 30 or len(current) < 30:
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "reason": "insufficient_data",
        }
    stat, p = ks_2samp(reference, current)
    return {
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < 0.05),
        "reference_size": len(reference),
        "current_size": len(current),
    }


def compute_psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current."""
    eps = 1e-8
    ref_arr = np.array(reference)
    cur_arr = np.array(current)
    bins_edges = np.histogram_bin_edges(ref_arr, bins=bins, range=(0.0, 1.0))
    ref_counts, _ = np.histogram(ref_arr, bins=bins_edges)
    cur_counts, _ = np.histogram(cur_arr, bins=bins_edges)
    ref_pct = (ref_counts + eps) / (len(reference) + eps * bins)
    cur_pct = (cur_counts + eps) / (len(current) + eps * bins)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 4)


def update_prediction_window(delay_prob: float) -> None:
    """Add a new prediction probability to the rolling window."""
    _prediction_window.append(delay_prob)


def set_reference_scores(scores: list[float]) -> None:
    """Set the reference distribution for drift monitoring."""
    global _reference_scores
    _reference_scores = list(scores)
    logger.info("Reference distribution set with %d scores", len(_reference_scores))


def get_online_drift() -> dict[str, Any]:
    """Check drift of recent predictions against reference."""
    current = list(_prediction_window)
    return compute_drift(_reference_scores, current)


def log_prediction(
    db: Session,
    carrier: str,
    origin: str,
    destination: str,
    scheduled_hour: int,
    day_of_week: int,
    month: int,
    delay_probability: float,
    predicted_class: str,
    model_version: str = "1.0.0",
    latency_ms: float | None = None,
    correlation_id: str | None = None,
) -> PredictionLog:
    """Persist a prediction to the database."""
    record = PredictionLog(
        correlation_id=correlation_id,
        carrier=carrier,
        origin=origin,
        destination=destination,
        scheduled_hour=scheduled_hour,
        day_of_week=day_of_week,
        month=month,
        delay_probability=delay_probability,
        predicted_class_name=predicted_class,
        model_version=model_version,
        latency_ms=latency_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    update_prediction_window(delay_probability)
    return record


def log_drift_report(db: Session, feature_name: str, drift_result: dict[str, Any]) -> DriftReport:
    """Persist a drift detection result to the database."""
    report = DriftReport(
        feature_name=feature_name,
        ks_statistic=drift_result.get("ks_statistic", 0.0),
        p_value=drift_result.get("p_value", 1.0),
        drift_detected=int(drift_result.get("drift_detected", False)),
        reference_size=drift_result.get("reference_size"),
        current_size=drift_result.get("current_size"),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_prediction_stats(db: Session, hours: int = 24) -> dict[str, Any]:
    """Compute summary statistics of recent predictions."""
    since = datetime.utcnow() - timedelta(hours=hours)
    records = db.query(PredictionLog).filter(PredictionLog.created_at >= since).all()
    if not records:
        return {
            "count": 0,
            "avg_delay_probability": None,
            "delayed_count": 0,
            "on_time_count": 0,
            "window_hours": hours,
        }
    probs = [r.delay_probability for r in records]
    delayed = sum(1 for r in records if r.predicted_class_name == "delayed")
    return {
        "count": len(records),
        "avg_delay_probability": round(float(np.mean(probs)), 4),
        "p95_delay_probability": round(float(np.percentile(probs, 95)), 4),
        "delayed_count": delayed,
        "on_time_count": len(records) - delayed,
        "delay_rate": round(delayed / len(records), 4),
        "window_hours": hours,
    }
