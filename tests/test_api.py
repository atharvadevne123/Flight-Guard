"""Tests for FastAPI endpoints."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_structure(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "db_connected" in data

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Flight-Guard" in resp.json().get("service", "")


class TestPredictEndpoint:
    def test_predict_valid_request(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        assert resp.status_code == 200

    def test_predict_response_structure(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        data = resp.json()
        assert "delay_probability" in data
        assert "predicted_class" in data
        assert "risk_tier" in data
        assert "model_version" in data

    def test_predict_probability_in_range(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        data = resp.json()
        assert 0.0 <= data["delay_probability"] <= 1.0
        assert 0.0 <= data["on_time_probability"] <= 1.0

    def test_predict_class_valid(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        assert resp.json()["predicted_class"] in ("on_time", "delayed")

    def test_predict_risk_tier_valid(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        assert resp.json()["risk_tier"] in ("low", "medium", "high")

    def test_predict_peak_flight(self, client, peak_flight_request):
        resp = client.post("/api/v1/predict", json=peak_flight_request)
        assert resp.status_code == 200

    def test_predict_unknown_carrier(self, client, unknown_carrier_request):
        resp = client.post("/api/v1/predict", json=unknown_carrier_request)
        assert resp.status_code == 200

    @pytest.mark.parametrize("carrier", ["AA", "UA", "DL", "WN", "NK"])
    def test_predict_multiple_carriers(self, client, carrier):
        req = {
            "carrier": carrier,
            "origin": "JFK",
            "destination": "LAX",
            "scheduled_hour": 10,
            "day_of_week": 1,
            "month": 6,
            "distance_km": 3983.0,
        }
        resp = client.post("/api/v1/predict", json=req)
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["delay_probability"] <= 1.0

    def test_predict_invalid_hour(self, client, sample_flight_request):
        req = {**sample_flight_request, "scheduled_hour": 25}
        resp = client.post("/api/v1/predict", json=req)
        assert resp.status_code == 422

    def test_predict_invalid_month(self, client, sample_flight_request):
        req = {**sample_flight_request, "month": 13}
        resp = client.post("/api/v1/predict", json=req)
        assert resp.status_code == 422

    def test_predict_negative_distance(self, client, sample_flight_request):
        req = {**sample_flight_request, "distance_km": -100.0}
        resp = client.post("/api/v1/predict", json=req)
        assert resp.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_predict_single(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict/batch", json={"flights": [sample_flight_request]})
        assert resp.status_code == 200
        assert resp.json()["batch_size"] == 1

    def test_batch_predict_multiple(self, client, sample_flight_request, peak_flight_request):
        resp = client.post(
            "/api/v1/predict/batch",
            json={"flights": [sample_flight_request, peak_flight_request]},
        )
        assert resp.status_code == 200
        assert resp.json()["batch_size"] == 2

    def test_batch_predict_empty_rejected(self, client):
        resp = client.post("/api/v1/predict/batch", json={"flights": []})
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200

    def test_metrics_has_version(self, client):
        resp = client.get("/api/v1/metrics")
        assert "model_version" in resp.json()


class TestDriftEndpoint:
    def test_drift_returns_200(self, client):
        resp = client.get("/api/v1/drift")
        assert resp.status_code == 200

    def test_drift_structure(self, client):
        resp = client.get("/api/v1/drift")
        data = resp.json()
        assert "ks_statistic" in data
        assert "p_value" in data
        assert "drift_detected" in data


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200

    def test_stats_default_window(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.json()["window_hours"] == 24

    def test_stats_custom_window(self, client):
        resp = client.get("/api/v1/stats?hours=48")
        assert resp.json()["window_hours"] == 48

    def test_stats_invalid_hours(self, client):
        resp = client.get("/api/v1/stats?hours=0")
        assert resp.status_code == 422

    def test_correlation_id_header(self, client, sample_flight_request):
        resp = client.post(
            "/api/v1/predict",
            json=sample_flight_request,
            headers={"X-Correlation-ID": "test-corr-123"},
        )
        assert resp.headers.get("X-Correlation-ID") == "test-corr-123"
