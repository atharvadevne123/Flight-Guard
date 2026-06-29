# Flight-Guard

> Real-time flight delay prediction API — XGBoost-LightGBM ensemble with carrier risk scoring, route analysis, weather encoding, and Palantir Foundry integration.

## Overview

Flight-Guard predicts the probability and expected magnitude of a flight delay in real time. It combines a soft-voting ensemble of XGBoost, LightGBM, and Random Forest with a carrier-level risk scorer and route analysis, returning a structured prediction with SHAP explanations. A daily Airflow pipeline syncs data with Palantir Foundry, checks for drift, and retrains the model automatically.

## Features

- **Delay Prediction** — probability score with tier labels: `ON_TIME / MINOR / MODERATE / SEVERE`
- **Expected Delay Minutes** — calibrated delay magnitude estimate
- **Carrier Risk Profile** — historical delay rate and mean delay per carrier (AA, DL, UA, WN, B6, AS, NK, F9, G4, SY)
- **Route Analysis** — hub-to-hub patterns, distance buckets, directional encoding
- **Weather Severity Encoding** — `clear / rain / wind / fog / snow` mapped to severity scores
- **Prior Leg Propagation** — upstream delay passed as a feature for chained flights
- **SHAP Explanations** — per-prediction feature importances
- **KS-Drift Monitoring** — Evidently-based covariate drift detection
- **Palantir Foundry Integration** — flight dataset sync and model registry
- **Automated Retraining** — Airflow DAG with Foundry data I/O
- **Prometheus Metrics** — prediction count, latency, delay probability distribution

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | Flask 3, Flask-RESTX, Gunicorn |
| ML Models | XGBoost 2, LightGBM 4, scikit-learn (Random Forest, isotonic calibration) |
| Explainability | SHAP |
| Data | pandas, NumPy |
| Validation | marshmallow |
| Rate Limiting | Flask-Limiter |
| Drift Monitoring | Evidently |
| Experiment Tracking | MLflow |
| Orchestration | Apache Airflow 2 |
| Data Platform | Palantir Foundry REST API (Parquet, transaction writes) |
| Observability | Prometheus, prometheus-client |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Containerisation | Docker, docker-compose |
| Testing | pytest, pytest-mock |
| Runtime | Python 3.11 |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Predict delay for a single flight |
| `POST` | `/predict/batch` | Score up to 100 flights |
| `GET` | `/carrier-risk/<code>` | Carrier risk profile (e.g. `/carrier-risk/AA`) |
| `GET` | `/route-analysis` | Route delay statistics (`?origin=ORD&dest=LAX`) |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/model/info` | Model version and feature metadata |
| `GET` | `/docs` | Swagger UI |

### POST `/predict` — Request

```json
{
  "flight_id": "AA-2345",
  "carrier_code": "AA",
  "origin": "ORD",
  "destination": "LAX",
  "scheduled_departure": "2024-06-15T08:30:00",
  "aircraft_type": "B737",
  "distance_miles": 1745.0,
  "departure_hour": 8,
  "day_of_week": 5,
  "month": 6,
  "is_holiday": false,
  "prior_leg_delay_minutes": 0,
  "weather_condition": "clear"
}
```

### POST `/predict` — Response

```json
{
  "flight_id": "AA-2345",
  "delay_probability": 0.312,
  "delay_tier": "MINOR",
  "expected_delay_minutes": 22,
  "carrier_risk_score": 0.28,
  "shap_features": {
    "weather_severity": 0.08,
    "departure_hour": 0.06,
    "carrier_delay_rate": 0.04
  },
  "request_id": "req-abc123",
  "latency_ms": 14.2
}
```

## Project Structure

```
Flight-Guard/
├── api/
│   ├── app.py               # Flask-RESTX application
│   └── wsgi.py
├── foundry/
│   └── foundry_client.py    # Palantir Foundry REST client
├── models/
│   ├── ensemble/
│   │   └── delay_predictor.py       # XGBoost + LightGBM + RF ensemble
│   └── risk/
│       └── carrier_risk_scorer.py   # Per-carrier risk profiles
├── pipeline/
│   ├── feature_engineering.py       # Temporal, route, weather, carrier features
│   └── airflow/
│       └── retrain_dag.py           # Daily retraining DAG
├── monitoring/
│   └── drift_monitor.py
├── scripts/
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_feature_engineering.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Palantir Foundry Integration

Flight-Guard syncs with Foundry for:

- **Flights Dataset** — historical flight records with delay labels
- **Model Registry** — ensemble artifacts with evaluation metrics
- **Carrier Risk Scores** — computed risk profiles pushed back to Foundry

Configure via `.env`:

```env
FOUNDRY_HOST=https://your-instance.palantirfoundry.com
FOUNDRY_TOKEN=your-bearer-token
FLIGHTS_DATASET_RID=ri.foundry.main.dataset.xxxxxxxx
PREDICTIONS_DATASET_RID=ri.foundry.main.dataset.yyyyyyyy
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
gunicorn -b 0.0.0.0:8001 api.wsgi:app
# Or: docker-compose -f docker/docker-compose.yml up
```

## Running Tests

```bash
pytest tests/ -v
```

33 tests — API endpoints, carrier risk, route analysis, and feature engineering.

## Airflow DAG

The `flight_guard_retrain` DAG runs daily at 03:00 UTC:

```
fetch_flight_data
    → compute_carrier_risk_scores
    → check_drift
    → retrain_model
    → evaluate_model
    → push_model_to_foundry
```
