"""Airflow DAG for automated weekly model retraining."""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "reflective-lantern",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

AUC_GATE = float(os.getenv("RETRAIN_AUC_GATE", "0.70"))
MIN_SAMPLES = int(os.getenv("RETRAIN_MIN_SAMPLES", "200"))


def _check_drift() -> dict:
    """Task: Check if drift is detected and retrain is warranted."""
    from app.monitoring import get_online_drift

    drift = get_online_drift()
    logger.info("Drift check result: %s", drift)
    return drift


def _load_training_data() -> tuple:
    """Task: Load fresh training data from the database or generate synthetic."""
    from app.database import Flight, SessionLocal
    from app.features import generate_synthetic_data

    db = SessionLocal()
    try:
        count = db.query(Flight).count()
        logger.info("Found %d flight records in DB", count)
    finally:
        db.close()

    # Use synthetic data if not enough real data
    X, y = generate_synthetic_data(n_samples=max(MIN_SAMPLES, 2000))
    return X, y


def _retrain(**context: object) -> dict:
    """Task: Retrain the model and gate on AUC."""
    from app.database import RetrainLog, SessionLocal
    from app.features import generate_synthetic_data
    from app.model import train_model

    X, y = generate_synthetic_data(n_samples=2000)
    pipeline, metrics = train_model(X, y)

    auc = metrics.get("auc_mean", 0.0)
    status = "success" if auc >= AUC_GATE else "failed_auc_gate"
    logger.info("Retrain complete: AUC=%.4f, gate=%.2f, status=%s", auc, AUC_GATE, status)

    db = SessionLocal()
    try:
        record = RetrainLog(
            trigger="scheduled_dag",
            auc_after=auc,
            n_samples=metrics.get("n_samples"),
            status=status,
            notes=f"cv_auc_mean={auc:.4f}",
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    if status == "failed_auc_gate":
        raise ValueError(f"Model AUC {auc:.4f} below gate {AUC_GATE}")

    return metrics


def _notify_completion(**context: object) -> None:
    """Task: Log retraining completion notification."""
    logger.info("Weekly retrain DAG completed successfully at %s", datetime.utcnow().isoformat())


if _AIRFLOW_AVAILABLE and DAG is not None:
    with DAG(
        dag_id="flight_guard_weekly_retrain",
        default_args=DEFAULT_ARGS,
        description="Weekly retrain of Flight-Guard delay prediction model",
        schedule_interval="0 3 * * 1",  # Every Monday at 03:00 UTC
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["flight-guard", "ml", "retrain"],
    ) as dag:
        check_drift_task = PythonOperator(
            task_id="check_drift",
            python_callable=_check_drift,
        )
        load_data_task = PythonOperator(
            task_id="load_training_data",
            python_callable=_load_training_data,
        )
        retrain_task = PythonOperator(
            task_id="retrain_model",
            python_callable=_retrain,
        )
        notify_task = PythonOperator(
            task_id="notify_completion",
            python_callable=_notify_completion,
        )
        check_drift_task >> load_data_task >> retrain_task >> notify_task
