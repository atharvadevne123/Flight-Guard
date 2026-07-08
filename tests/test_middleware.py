"""Tests for rate limiting and correlation ID middleware."""

from __future__ import annotations

import uuid

from app.middleware import (
    RATE_LIMIT_REQUESTS,
    _rate_store,
)


class TestCorrelationID:
    def test_generated_when_absent(self, client, sample_flight_request):
        resp = client.post("/api/v1/predict", json=sample_flight_request)
        corr = resp.headers.get("X-Correlation-ID")
        assert corr is not None
        # Should be a valid UUID when auto-generated
        uuid.UUID(corr)

    def test_propagated_when_present(self, client, sample_flight_request):
        resp = client.post(
            "/api/v1/predict",
            json=sample_flight_request,
            headers={"X-Correlation-ID": "my-trace-id"},
        )
        assert resp.headers.get("X-Correlation-ID") == "my-trace-id"

    def test_response_time_header_present(self, client):
        resp = client.get("/api/v1/health")
        assert "X-Response-Time-Ms" in resp.headers
        assert float(resp.headers["X-Response-Time-Ms"]) >= 0


class TestRateLimit:
    def test_health_exempt_from_rate_limit(self, client):
        _rate_store.clear()
        for _ in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_rate_limit_blocks_excess(self, client):
        _rate_store.clear()
        # Pre-fill the store to simulate an exhausted window
        import time

        now = time.time()
        _rate_store["testclient"] = [now] * RATE_LIMIT_REQUESTS
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "60"
        _rate_store.clear()

    def test_requests_allowed_under_limit(self, client):
        _rate_store.clear()
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 200
