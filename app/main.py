"""FastAPI application with versioned API endpoints."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.features import prepare_dataframe
from app.middleware import CorrelationIDMiddleware, RateLimitMiddleware
from app.model import load_metrics, load_model, predict
from app.monitoring import (
    get_online_drift,
    get_prediction_stats,
    log_drift_report,
    set_reference_scores,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftResponse,
    FlightPredictRequest,
    FlightPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictionStatsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _model
    init_db()
    _model = load_model()
    # Seed reference distribution from synthetic training data

    from app.features import generate_synthetic_data

    X_ref, _ = generate_synthetic_data(500)
    probs = _model.predict_proba(X_ref)[:, 1].tolist()
    set_reference_scores(probs)
    logger.info("Flight-Guard API started. Model loaded.")
    yield
    logger.info("Flight-Guard API shutting down.")


app = FastAPI(
    title="Flight-Guard",
    description=(
        "Real-time flight delay prediction API using XGBoost-LightGBM ensemble "
        "with carrier risk scoring, route analysis, seasonal patterns, "
        "KS-drift monitoring, and automated retraining pipelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RateLimitMiddleware)

api = FastAPI(title="Flight-Guard API v1", version="1.0.0")


@api.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the API and its dependencies.",
)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Check API and model health."""
    db_ok = True
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="healthy" if _model is not None and db_ok else "degraded",
        model_loaded=_model is not None,
        model_version=load_metrics().get("model_version", "unknown"),
        db_connected=db_ok,
    )


@api.post(
    "/predict",
    response_model=FlightPredictResponse,
    summary="Predict flight delay",
    description=(
        "Predict whether a flight will be delayed (>15 min) and return "
        "probability, risk tier, and model version."
    ),
)
def predict_delay(
    request: FlightPredictRequest,
    http_request: Request,
    db: Session = Depends(get_db),
) -> FlightPredictResponse:
    """Predict delay probability for a single flight."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    df = prepare_dataframe(request.model_dump())
    result = predict(_model, df)
    latency_ms = (time.perf_counter() - start) * 1000

    corr_id = getattr(getattr(http_request, "state", None), "correlation_id", None)

    # Log to DB (non-blocking, best-effort)
    try:
        from app.database import PredictionLog

        record = PredictionLog(
            correlation_id=corr_id,
            carrier=request.carrier,
            origin=request.origin,
            destination=request.destination,
            scheduled_hour=request.scheduled_hour,
            day_of_week=request.day_of_week,
            month=request.month,
            delay_probability=result["delay_probability"],
            predicted_class_name=result["predicted_class"],
            model_version=result["model_version"],
            latency_ms=latency_ms,
        )
        db.add(record)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to log prediction: %s", exc)

    from app.monitoring import update_prediction_window

    update_prediction_window(result["delay_probability"])

    return FlightPredictResponse(**result, correlation_id=corr_id)


@api.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Batch predict flight delays",
    description="Predict delays for up to 100 flights in a single request.",
)
def predict_batch(
    request: BatchPredictRequest,
    http_request: Request,
) -> BatchPredictResponse:
    """Batch predict delay probabilities."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    corr_id = getattr(getattr(http_request, "state", None), "correlation_id", None)
    predictions = []
    for flight in request.flights:
        df = prepare_dataframe(flight.model_dump())
        result = predict(_model, df)
        predictions.append(FlightPredictResponse(**result, correlation_id=corr_id))

    return BatchPredictResponse(predictions=predictions, batch_size=len(predictions))


@api.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Model training metrics",
    description="Returns CV AUC, accuracy, and training metadata.",
)
def model_metrics() -> MetricsResponse:
    """Return model training metrics (cached for 15 seconds)."""
    from app.cache import metrics_cache

    cached = metrics_cache.get("model_metrics")
    if cached is not None:
        return cached
    m = load_metrics()
    response = MetricsResponse(
        auc_mean=m.get("auc_mean"),
        auc_std=m.get("auc_std"),
        accuracy_mean=m.get("accuracy_mean"),
        n_samples=m.get("n_samples"),
        model_version=m.get("model_version", "unknown"),
        status=m.get("status"),
    )
    metrics_cache.set("model_metrics", response)
    return response


@api.get(
    "/drift",
    response_model=DriftResponse,
    summary="Online drift detection",
    description="KS-test comparing recent prediction scores against reference distribution.",
)
def drift_status(db: Session = Depends(get_db)) -> DriftResponse:
    """Return current drift metrics."""
    result = get_online_drift()
    try:
        log_drift_report(db, "delay_probability", result)
    except Exception as exc:
        logger.warning("Failed to log drift: %s", exc)
    return DriftResponse(**{k: v for k, v in result.items() if k in DriftResponse.model_fields})


@api.get(
    "/stats",
    response_model=PredictionStatsResponse,
    summary="Prediction statistics",
    description="Aggregated statistics of predictions in the last N hours.",
)
def prediction_stats(
    hours: int = 24,
    db: Session = Depends(get_db),
) -> PredictionStatsResponse:
    """Return prediction volume and delay rate statistics."""
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=422, detail="hours must be between 1 and 720")
    stats = get_prediction_stats(db, hours=hours)
    return PredictionStatsResponse(**stats)


@api.get(
    "/forecast",
    summary="Delay trend forecast",
    description=(
        "Fits a linear trend to recent prediction scores and projects the "
        "delay-probability trajectory over the requested horizon."
    ),
)
def delay_forecast(horizon: int = 7) -> dict:
    """Forecast the delay-probability trend from the rolling prediction window."""
    if horizon < 1 or horizon > 90:
        raise HTTPException(status_code=422, detail="horizon must be between 1 and 90")
    from app.forecasting import linear_trend_forecast
    from app.monitoring import _prediction_window

    values = list(_prediction_window)
    result = linear_trend_forecast(values, horizon=horizon)
    result["history_size"] = len(values)
    return result


# Mount versioned router
app.mount("/api/v1", api)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "Flight-Guard",
        "version": "1.0.0",
        "docs": "/docs",
        "api": "/api/v1",
    }
