"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class FlightPredictRequest(BaseModel):
    """Input schema for flight delay prediction."""

    carrier: str = Field(
        ..., min_length=2, max_length=10, description="IATA carrier code (e.g. AA, UA)"
    )
    origin: str = Field(..., min_length=3, max_length=5, description="IATA origin airport code")
    destination: str = Field(
        ..., min_length=3, max_length=5, description="IATA destination airport code"
    )
    scheduled_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month of departure (1-12)")
    distance_km: float = Field(1000.0, gt=0, le=20000.0, description="Route distance in kilometres")

    @field_validator("carrier")
    @classmethod
    def uppercase_carrier(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("origin", "destination")
    @classmethod
    def uppercase_airport(cls, v: str) -> str:
        return v.upper().strip()


class FlightPredictResponse(BaseModel):
    """Prediction response schema."""

    delay_probability: float
    on_time_probability: float
    predicted_class: Literal["on_time", "delayed"]
    risk_tier: Literal["low", "medium", "high"]
    model_version: str
    correlation_id: str | None = None


class BatchPredictRequest(BaseModel):
    """Batch prediction request schema."""

    flights: list[FlightPredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    """Batch prediction response schema."""

    predictions: list[FlightPredictResponse]
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: str
    db_connected: bool


class MetricsResponse(BaseModel):
    """Model metrics response."""

    auc_mean: float | None = None
    auc_std: float | None = None
    accuracy_mean: float | None = None
    n_samples: int | None = None
    model_version: str
    status: str | None = None


class DriftResponse(BaseModel):
    """Drift detection response."""

    ks_statistic: float
    p_value: float
    drift_detected: bool
    reference_size: int | None = None
    current_size: int | None = None
    psi: float | None = None


class PredictionStatsResponse(BaseModel):
    """Summary statistics of recent predictions."""

    count: int
    avg_delay_probability: float | None
    p95_delay_probability: float | None = None
    delayed_count: int
    on_time_count: int
    delay_rate: float | None = None
    window_hours: int
