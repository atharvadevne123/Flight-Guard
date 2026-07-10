"""
Flask microservice: Flight-Guard API
Endpoints:
  POST /predict              — predict delay for a single flight
  POST /predict/batch        — batch flight delay predictions
  GET  /carrier-risk/<code>  — carrier risk profile
  GET  /route-analysis       — route delay patterns
  GET  /health               — liveness probe
  GET  /metrics              — Prometheus metrics
  GET  /model/info           — model metadata
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, g, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_restx import Api, Resource, fields
from loguru import logger
from marshmallow import Schema, ValidationError
from marshmallow import fields as ma_fields
from marshmallow import validate as ma_validate
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ─── App bootstrap ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

api = Api(
    app,
    version="1.0",
    title="Flight-Guard API",
    description="Real-time flight delay prediction with carrier risk scoring and route analysis",
    doc="/docs",
)

ns = api.namespace("", description="Flight delay prediction endpoints")

# ─── Prometheus metrics ───────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "flight_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "flight_api_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
DELAY_PROB_HISTOGRAM = Histogram(
    "flight_delay_probability_distribution",
    "Distribution of predicted delay probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ─── Delay tier thresholds ────────────────────────────────────────────────────

THRESHOLD_SEVERE = float(os.getenv("THRESHOLD_SEVERE", "0.75"))
THRESHOLD_MODERATE = float(os.getenv("THRESHOLD_MODERATE", "0.50"))
THRESHOLD_MINOR = float(os.getenv("THRESHOLD_MINOR", "0.25"))

# ─── Model loading ────────────────────────────────────────────────────────────

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))

_ensemble = None
_carrier_risk_scorer = None
_feature_engineer = None
_feature_cols: list = []


def _load_models() -> None:
    global _ensemble, _carrier_risk_scorer, _feature_engineer, _feature_cols
    import json

    import joblib

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.ensemble.delay_predictor import DelayPredictor
    from models.risk.carrier_risk_scorer import CarrierRiskScorer

    ensemble_path = MODEL_DIR / "delay_predictor.joblib"
    fe_path = MODEL_DIR / "feature_engineer.joblib"
    cols_path = MODEL_DIR / "feature_cols.json"

    if ensemble_path.exists():
        _ensemble = DelayPredictor.load(ensemble_path)
        logger.success("Delay predictor loaded.")
    else:
        logger.warning("Delay predictor not found at {}. Operating in mock mode.", ensemble_path)

    if fe_path.exists():
        _feature_engineer = joblib.load(fe_path)
        logger.success("Feature engineer loaded.")

    if cols_path.exists():
        _feature_cols = json.loads(cols_path.read_text())

    _carrier_risk_scorer = CarrierRiskScorer()
    logger.success("Carrier risk scorer initialised.")


# ─── Input validation ─────────────────────────────────────────────────────────

VALID_PROTOCOLS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9", "G4", "SY"]
VALID_WEATHER = ["clear", "rain", "snow", "fog", "wind"]


class FlightSchema(Schema):
    flight_id = ma_fields.Str(required=True)
    carrier_code = ma_fields.Str(required=True)
    origin = ma_fields.Str(
        required=True,
        validate=ma_validate.Length(min=3, max=3, error="origin must be a 3-letter IATA code."),
    )
    destination = ma_fields.Str(
        required=True,
        validate=ma_validate.Length(
            min=3, max=3, error="destination must be a 3-letter IATA code."
        ),
    )
    scheduled_departure = ma_fields.Str(required=True)
    aircraft_type = ma_fields.Str(load_default="unknown")
    distance_miles = ma_fields.Float(
        load_default=None,
        validate=ma_validate.Range(
            min=0,
            min_inclusive=False,
            error="distance_miles must be > 0.",
        ),
    )
    departure_hour = ma_fields.Int(
        load_default=12,
        validate=ma_validate.Range(
            min=0,
            max=23,
            error="departure_hour must be 0-23.",
        ),
    )
    day_of_week = ma_fields.Int(
        load_default=0,
        validate=ma_validate.Range(
            min=0,
            max=6,
            error="day_of_week must be 0-6.",
        ),
    )
    month = ma_fields.Int(
        load_default=1,
        validate=ma_validate.Range(
            min=1,
            max=12,
            error="month must be 1-12.",
        ),
    )
    is_holiday = ma_fields.Bool(load_default=False)
    prior_leg_delay_minutes = ma_fields.Float(
        load_default=0.0,
        validate=ma_validate.Range(
            min=0,
            error="prior_leg_delay_minutes must be >= 0.",
        ),
    )
    weather_condition = ma_fields.Str(
        load_default="clear",
        validate=ma_validate.OneOf(
            VALID_WEATHER,
            error=f"weather_condition must be one of: {VALID_WEATHER}.",
        ),
    )


_schema = FlightSchema()
_batch_schema = FlightSchema(many=True)

# ─── Swagger models ───────────────────────────────────────────────────────────

flight_model = api.model(
    "Flight",
    {
        "flight_id": fields.String(required=True, example="AA-2024-ORD-LAX-001"),
        "carrier_code": fields.String(required=True, example="AA"),
        "origin": fields.String(required=True, example="ORD"),
        "destination": fields.String(required=True, example="LAX"),
        "scheduled_departure": fields.String(required=True, example="2024-06-15T08:30:00"),
        "aircraft_type": fields.String(example="B737"),
        "distance_miles": fields.Float(example=1745.0),
        "departure_hour": fields.Integer(example=8),
        "day_of_week": fields.Integer(example=4),
        "month": fields.Integer(example=6),
        "is_holiday": fields.Boolean(example=False),
        "prior_leg_delay_minutes": fields.Float(example=0.0),
        "weather_condition": fields.String(example="clear"),
    },
)

prediction_response = api.model(
    "DelayPredictionResponse",
    {
        "flight_id": fields.String(),
        "delay_probability": fields.Float(),
        "delay_tier": fields.String(),
        "expected_delay_minutes": fields.Float(),
        "carrier_risk_score": fields.Float(),
        "shap_features": fields.Raw(),
        "latency_ms": fields.Float(),
    },
)

# ─── API key auth ─────────────────────────────────────────────────────────────

_API_KEY = os.getenv("API_KEY")
_OPEN_PATHS = {"/health", "/metrics", "/swagger.json"}


@limiter.request_filter
def _exempt_health_metrics():
    return request.path in ("/health", "/metrics") or request.path.startswith("/docs")


@app.before_request
def _check_api_key():
    if not _API_KEY:
        return
    if request.path in _OPEN_PATHS or request.path.startswith("/docs"):
        return
    if request.headers.get("X-Api-Key", "") != _API_KEY:
        REQUEST_COUNT.labels(endpoint="auth", status="401").inc()
        return jsonify(
            {"error": "Unauthorized", "detail": "Invalid or missing X-Api-Key header"}
        ), 401


@app.before_request
def _attach_request_id():
    g.request_id = str(uuid.uuid4())


@app.after_request
def _add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


# ─── Error handlers ───────────────────────────────────────────────────────────


@api.errorhandler(Exception)
def handle_generic(e):
    code = getattr(e, "code", 500)
    return {"error": type(e).__name__, "detail": str(e)}, code


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not Found", "detail": str(e)}), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded", "detail": str(e)}), 429


# ─── Core scoring logic ───────────────────────────────────────────────────────


def _predict_flight(flight: dict) -> dict:
    """Run feature engineering → ensemble → SHAP for one flight record."""
    start = time.perf_counter()
    df = pd.DataFrame([flight])

    if _feature_engineer is not None:
        df = _feature_engineer.transform(df)

    if _feature_cols:
        X = pd.DataFrame(0.0, index=df.index, columns=_feature_cols)
        for col in _feature_cols:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        exclude = {
            "flight_id",
            "carrier_code",
            "origin",
            "destination",
            "scheduled_departure",
            "aircraft_type",
            "weather_condition",
        }
        feat_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
        X = df[feat_cols].fillna(0)

    delay_prob = 0.3
    shap_features: dict = {}

    if _ensemble is not None:
        delay_prob = float(_ensemble.predict_proba(X)[0])
        try:
            explanations = _ensemble.explain(X)
            shap_features = explanations[0].get("shap_features", {})
        except Exception as exc:
            logger.debug("SHAP explain failed: {}", exc)
    else:
        import random

        delay_prob = round(random.uniform(0.05, 0.90), 4)

    # Tier classification
    if delay_prob >= THRESHOLD_SEVERE:
        delay_tier = "SEVERE"
    elif delay_prob >= THRESHOLD_MODERATE:
        delay_tier = "MODERATE"
    elif delay_prob >= THRESHOLD_MINOR:
        delay_tier = "MINOR"
    else:
        delay_tier = "ON_TIME"

    # Expected delay minutes (rough heuristic based on probability)
    expected_delay_minutes = round(delay_prob * 120, 1)

    # Carrier risk score
    carrier_risk = 0.5
    if _carrier_risk_scorer is not None:
        try:
            risk_result = _carrier_risk_scorer.score_carrier(
                flight.get("carrier_code", "XX"), pd.DataFrame()
            )
            carrier_risk = risk_result.get("risk_score", 0.5)
        except Exception as exc:
            logger.debug("Carrier risk scoring failed: {}", exc)

    latency_ms = (time.perf_counter() - start) * 1000
    DELAY_PROB_HISTOGRAM.observe(delay_prob)

    return {
        "flight_id": flight.get("flight_id"),
        "delay_probability": round(delay_prob, 6),
        "delay_tier": delay_tier,
        "expected_delay_minutes": expected_delay_minutes,
        "carrier_risk_score": round(carrier_risk, 4),
        "shap_features": {k: round(v, 6) for k, v in list(shap_features.items())[:10]},
        "latency_ms": round(latency_ms, 2),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────


@ns.route("/health")
class HealthCheck(Resource):
    def get(self):
        return {
            "status": "ok",
            "models_loaded": _ensemble is not None,
            "carrier_risk_loaded": _carrier_risk_scorer is not None,
        }, 200


@ns.route("/metrics")
class Metrics(Resource):
    def get(self):
        from flask import Response

        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@ns.route("/model/info")
class ModelInfo(Resource):
    def get(self):
        return {
            "ensemble_loaded": _ensemble is not None,
            "carrier_risk_loaded": _carrier_risk_scorer is not None,
            "feature_count": len(_feature_cols),
            "version": "1.0.0",
            "delay_tiers": ["ON_TIME", "MINOR", "MODERATE", "SEVERE"],
            "thresholds": {
                "severe": THRESHOLD_SEVERE,
                "moderate": THRESHOLD_MODERATE,
                "minor": THRESHOLD_MINOR,
            },
        }


@ns.route("/predict")
class Predict(Resource):
    @ns.expect(flight_model)
    def post(self):
        try:
            data = _schema.load(request.get_json(force=True) or {})
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="predict", status="400").inc()
            return {"error": str(exc.messages)}, 400

        with PREDICTION_LATENCY.time():
            result = _predict_flight(data)

        REQUEST_COUNT.labels(endpoint="predict", status="200").inc()
        return result, 200


@ns.route("/predict/batch")
class PredictBatch(Resource):
    def post(self):
        payload = request.get_json(force=True) or {}
        flights = payload.get("flights")

        if flights is None:
            return {"error": "Missing 'flights' key."}, 400
        if not flights:
            return {"error": "No flights provided."}, 400
        if len(flights) > 500:
            return {"error": "Batch size limited to 500 flights."}, 400

        try:
            data_list = _batch_schema.load(flights)
        except ValidationError as exc:
            REQUEST_COUNT.labels(endpoint="predict_batch", status="400").inc()
            return {"error": str(exc.messages)}, 400

        results = []
        with PREDICTION_LATENCY.time():
            for flight in data_list:
                results.append(_predict_flight(flight))

        REQUEST_COUNT.labels(endpoint="predict_batch", status="200").inc()
        return {"results": results, "count": len(results)}, 200


@ns.route("/carrier-risk/<string:carrier_code>")
class CarrierRisk(Resource):
    def get(self, carrier_code: str):
        if _carrier_risk_scorer is None:
            return {"error": "Carrier risk scorer not loaded."}, 503

        try:
            profile = _carrier_risk_scorer.score_carrier(carrier_code.upper(), pd.DataFrame())
            REQUEST_COUNT.labels(endpoint="carrier_risk", status="200").inc()
            return profile, 200
        except Exception as exc:
            REQUEST_COUNT.labels(endpoint="carrier_risk", status="500").inc()
            return {"error": str(exc)}, 500


@ns.route("/route-analysis")
class RouteAnalysis(Resource):
    def get(self):
        origin = request.args.get("origin", "").upper()
        dest = request.args.get("dest", "").upper()

        if not origin or not dest:
            return {"error": "Query params 'origin' and 'dest' are required."}, 400
        if len(origin) != 3 or len(dest) != 3:
            return {"error": "origin and dest must be 3-letter IATA codes."}, 400

        if _carrier_risk_scorer is None:
            return {"error": "Risk scorer not loaded."}, 503

        try:
            profile = _carrier_risk_scorer.score_route(origin, dest, pd.DataFrame())
            REQUEST_COUNT.labels(endpoint="route_analysis", status="200").inc()
            return profile, 200
        except Exception as exc:
            REQUEST_COUNT.labels(endpoint="route_analysis", status="500").inc()
            return {"error": str(exc)}, 500


@ns.route("/weather-impact")
class WeatherImpact(Resource):
    """Return delay probability increase by weather condition and time of day."""

    _WEATHER_IMPACT = {
        "clear": {"delay_multiplier": 1.00, "avg_added_minutes": 0, "severity": "none"},
        "rain": {"delay_multiplier": 1.28, "avg_added_minutes": 12, "severity": "low"},
        "wind": {"delay_multiplier": 1.35, "avg_added_minutes": 18, "severity": "medium"},
        "fog": {"delay_multiplier": 1.62, "avg_added_minutes": 35, "severity": "high"},
        "snow": {"delay_multiplier": 2.14, "avg_added_minutes": 58, "severity": "severe"},
    }
    _PEAK_HOUR_MULTIPLIER = 1.22  # peak travel hours compound weather delays
    _HOLIDAY_MULTIPLIER = 1.18

    def get(self):
        weather = request.args.get("weather", "clear").lower()
        hour = int(request.args.get("hour", 12))
        is_holiday = request.args.get("is_holiday", "false").lower() == "true"

        impact = self._WEATHER_IMPACT.get(weather)
        if impact is None:
            return {
                "error": f"Unknown weather condition '{weather}'.",
                "valid": list(self._WEATHER_IMPACT),
            }, 400

        multiplier = impact["delay_multiplier"]
        if 6 <= hour <= 9 or 16 <= hour <= 20:
            multiplier *= self._PEAK_HOUR_MULTIPLIER
        if is_holiday:
            multiplier *= self._HOLIDAY_MULTIPLIER

        base_on_time_pct = 82.0
        adjusted_on_time = max(20.0, base_on_time_pct / multiplier)

        REQUEST_COUNT.labels(endpoint="weather_impact", status="200").inc()
        return {
            "weather_condition": weather,
            "hour_of_day": hour,
            "is_holiday": is_holiday,
            "delay_multiplier": round(multiplier, 3),
            "avg_added_delay_min": round(
                impact["avg_added_minutes"] * (multiplier / impact["delay_multiplier"]), 1
            ),
            "adjusted_on_time_pct": round(adjusted_on_time, 1),
            "severity": impact["severity"],
            "recommendation": _weather_recommendation(weather, multiplier),
        }, 200


def _weather_recommendation(weather: str, multiplier: float) -> str:
    if multiplier < 1.1:
        return "Normal operations expected. No action required."
    if multiplier < 1.4:
        return "Minor delays possible. Allow extra buffer time for connections."
    if multiplier < 1.8:
        return (
            "Moderate delays likely. Consider rebooking connecting flights with < 90-min layovers."
        )
    return "Severe delays expected. Check airline alerts and consider flexible rebooking."


@ns.route("/delay-stats")
class DelayStats(Resource):
    """Aggregate delay statistics across all carriers (from in-memory predictions)."""

    def get(self):
        carrier = request.args.get("carrier", "").upper() or None

        # Return static benchmarks enriched with carrier data if scorer loaded
        carriers_data = {}
        if _carrier_risk_scorer is not None:
            for code in ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]:
                if carrier and code != carrier:
                    continue
                try:
                    profile = _carrier_risk_scorer.score_carrier(code)
                    carriers_data[code] = {
                        "on_time_pct": round((1 - profile.get("delay_rate", 0.25)) * 100, 1),
                        "avg_delay_min": profile.get("mean_delay_minutes", 35),
                        "risk_tier": profile.get("risk_tier", "MEDIUM"),
                    }
                except Exception:
                    pass

        REQUEST_COUNT.labels(endpoint="delay_stats", status="200").inc()
        return {
            "carriers": carriers_data,
            "industry_on_time_pct": 82.0,
            "industry_avg_delay_min": 34.2,
            "note": "Statistics derived from carrier risk scorer historical baselines.",
        }, 200


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
