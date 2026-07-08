"""Centralised application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    Attributes:
        database_url: SQLAlchemy connection string (SQLite dev, PostgreSQL prod).
        model_path: Filesystem path where the trained pipeline is persisted.
        metrics_path: Filesystem path where CV metrics JSON is persisted.
        retrain_auc_gate: Minimum CV AUC required to promote a retrained model.
        retrain_min_samples: Minimum training rows required for a retrain run.
        rate_limit_requests: Max requests per IP per window.
        rate_limit_window_seconds: Sliding rate-limit window length.
        log_level: Root logging level name.
    """

    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./flight_guard.db")
    )
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "model.joblib"))
    metrics_path: str = field(default_factory=lambda: os.getenv("METRICS_PATH", "metrics.json"))
    retrain_auc_gate: float = field(default_factory=lambda: _env_float("RETRAIN_AUC_GATE", 0.70))
    retrain_min_samples: int = field(default_factory=lambda: _env_int("RETRAIN_MIN_SAMPLES", 200))
    rate_limit_requests: int = field(default_factory=lambda: _env_int("RATE_LIMIT_REQUESTS", 200))
    rate_limit_window_seconds: int = field(
        default_factory=lambda: _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


def get_settings() -> Settings:
    """Return a Settings snapshot from the current environment."""
    return Settings()
