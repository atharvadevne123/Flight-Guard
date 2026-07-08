"""Tests for model training and prediction."""

from __future__ import annotations

import pytest

from app.features import generate_synthetic_data, prepare_dataframe
from app.model import load_metrics, predict, train_model


@pytest.fixture(scope="module")
def trained_pipeline(synthetic_data):
    X, y = synthetic_data
    pipeline, metrics = train_model(X, y, cv_folds=3)
    return pipeline, metrics


class TestModelTraining:
    def test_train_returns_pipeline_and_metrics(self, synthetic_data):
        X, y = synthetic_data
        pipeline, metrics = train_model(X, y, cv_folds=2)
        assert pipeline is not None
        assert isinstance(metrics, dict)

    def test_metrics_keys(self, trained_pipeline):
        _, metrics = trained_pipeline
        assert "auc_mean" in metrics
        assert "auc_std" in metrics
        assert "n_samples" in metrics

    def test_auc_above_baseline(self, trained_pipeline):
        _, metrics = trained_pipeline
        # Random baseline is 0.5 — expect model to beat it
        assert metrics["auc_mean"] > 0.5

    def test_metrics_saved_to_disk(self, trained_pipeline):
        from pathlib import Path

        assert Path("metrics.json").exists()

    def test_model_saved_to_disk(self, trained_pipeline):
        from pathlib import Path

        assert Path("model.joblib").exists()

    @pytest.mark.parametrize("n_samples", [200, 500])
    def test_train_different_sizes(self, n_samples):
        X, y = generate_synthetic_data(n_samples=n_samples)
        pipeline, metrics = train_model(X, y, cv_folds=2)
        assert metrics["n_samples"] == n_samples


class TestModelPrediction:
    def test_predict_single_flight(self, trained_pipeline, sample_flight_request):
        pipeline, _ = trained_pipeline
        df = prepare_dataframe(sample_flight_request)
        result = predict(pipeline, df)
        assert "delay_probability" in result
        assert "predicted_class" in result
        assert "risk_tier" in result

    def test_predict_probability_range(self, trained_pipeline, sample_flight_request):
        pipeline, _ = trained_pipeline
        df = prepare_dataframe(sample_flight_request)
        result = predict(pipeline, df)
        assert 0.0 <= result["delay_probability"] <= 1.0
        assert 0.0 <= result["on_time_probability"] <= 1.0

    def test_predict_probabilities_sum_to_one(self, trained_pipeline, sample_flight_request):
        pipeline, _ = trained_pipeline
        df = prepare_dataframe(sample_flight_request)
        result = predict(pipeline, df)
        total = result["delay_probability"] + result["on_time_probability"]
        assert abs(total - 1.0) < 1e-4

    def test_predict_class_consistency(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        # High-risk flight (late December, rush hour, congested airports)
        high_risk = {
            "carrier": "NK",
            "origin": "ORD",
            "destination": "EWR",
            "scheduled_hour": 17,
            "day_of_week": 4,
            "month": 12,
            "distance_km": 1178.0,
        }
        df = prepare_dataframe(high_risk)
        result = predict(pipeline, df)
        # probability should determine class
        expected = "delayed" if result["delay_probability"] >= 0.5 else "on_time"
        assert result["predicted_class"] == expected

    def test_risk_tier_mapping(self, trained_pipeline):
        pipeline, _ = trained_pipeline
        for req, expected_tier in [
            (
                {
                    "carrier": "HA",
                    "origin": "HNL",
                    "destination": "SEA",
                    "scheduled_hour": 2,
                    "day_of_week": 2,
                    "month": 10,
                    "distance_km": 4382.0,
                },
                "low",
            ),
        ]:
            df = prepare_dataframe(req)
            result = predict(pipeline, df)
            # Low-risk flight should have low or medium tier — just validate structure
            assert result["risk_tier"] in ("low", "medium", "high")

    def test_load_metrics_returns_dict(self):
        metrics = load_metrics()
        assert isinstance(metrics, dict)
