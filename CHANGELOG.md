# Changelog

All notable changes to Flight-Guard are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-08

### Added
- XGBoost + LightGBM + RandomForest soft-voting ensemble for flight delay classification.
- Six-stage sklearn feature pipeline: carrier risk encoding, route congestion features,
  cyclical temporal features, lag/rolling delay pressure scores, categorical drop, scaling.
- FastAPI application with versioned `/api/v1` endpoints: `/predict`, `/predict/batch`,
  `/health`, `/metrics`, `/drift`, `/stats`.
- KS-test and PSI drift detection with online rolling prediction window.
- SQLAlchemy models: `Flight`, `PredictionLog`, `DriftReport`, `RetrainLog`
  (SQLite dev, PostgreSQL prod).
- Airflow weekly retraining DAG with AUC quality gate (default ≥ 0.70).
- Rate limiting middleware (200 req/min per IP) and correlation ID middleware.
- Pydantic request/response validation with field validators.
- 5-fold stratified cross-validation with AUC-ROC and accuracy reporting.
- Docker + docker-compose with PostgreSQL and health checks.
- GitHub Actions CI: ruff lint, format check, pytest.
- Full pytest suite covering API, model, features, and monitoring.
- Architecture diagram generator script.
- Time-series delay-trend forecasting (`/api/v1/forecast`) with SMA, linear trend,
  and naive seasonal decomposition.
- FAISS route similarity index with NumPy brute-force fallback.
- MLflow experiment tracking with JSONL file fallback, wired into every training run.
- Thread-safe TTL cache on the metrics endpoint.
- Alembic migrations, CLI training script, dev data seeder, and example API client.
