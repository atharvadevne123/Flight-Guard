"""Experiment tracking with MLflow when available, JSON-file fallback otherwise."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore[import-untyped]

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

FALLBACK_LOG = Path(os.getenv("EXPERIMENT_LOG_PATH", "experiments.jsonl"))
EXPERIMENT_NAME = "flight-guard"


def log_training_run(
    metrics: dict[str, Any],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Record a training run to MLflow or the JSONL fallback.

    Args:
        metrics: Numeric metrics (auc_mean, accuracy_mean, ...).
        params: Hyperparameters used for the run.
        tags: Free-form string tags.

    Returns:
        The run identifier (MLflow run id or fallback line id).
    """
    params = params or {}
    tags = tags or {}

    if _MLFLOW_AVAILABLE:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run() as run:
            numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(numeric)
            mlflow.log_params(params)
            mlflow.set_tags(tags)
            logger.info("Logged run to MLflow: %s", run.info.run_id)
            return str(run.info.run_id)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment": EXPERIMENT_NAME,
        "metrics": metrics,
        "params": params,
        "tags": tags,
    }
    with FALLBACK_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    run_id = f"local-{int(datetime.utcnow().timestamp())}"
    logger.info("Logged run to fallback %s: %s", FALLBACK_LOG, run_id)
    return run_id


def load_run_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent fallback-logged runs (newest last)."""
    if not FALLBACK_LOG.exists():
        return []
    lines = FALLBACK_LOG.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]
