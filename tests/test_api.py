from __future__ import annotations

import json


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "models_loaded" in data

    def test_health_models_not_loaded(self, client):
        r = client.get("/health")
        assert r.get_json()["models_loaded"] is False

    def test_health_carrier_risk_key(self, client):
        data = client.get("/health").get_json()
        assert "carrier_risk_loaded" in data


class TestModelInfo:
    def test_model_info_structure(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        data = r.get_json()
        for key in ("ensemble_loaded", "carrier_risk_loaded", "feature_count", "version"):
            assert key in data

    def test_model_info_version(self, client):
        assert client.get("/model/info").get_json()["version"] == "1.0.0"

    def test_model_info_delay_tiers(self, client):
        data = client.get("/model/info").get_json()
        assert set(data["delay_tiers"]) == {"ON_TIME", "MINOR", "MODERATE", "SEVERE"}


class TestPredict:
    def test_predict_valid(self, client, patch_models, valid_flight):
        r = client.post("/predict", data=json.dumps(valid_flight),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "delay_probability" in data
        assert "delay_tier" in data
        assert "expected_delay_minutes" in data
        assert "carrier_risk_score" in data
        assert 0.0 <= data["delay_probability"] <= 1.0
        assert data["delay_tier"] in ("ON_TIME", "MINOR", "MODERATE", "SEVERE")

    def test_predict_missing_flight_id(self, client, patch_models, valid_flight):
        payload = {k: v for k, v in valid_flight.items() if k != "flight_id"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_carrier(self, client, patch_models, valid_flight):
        payload = {k: v for k, v in valid_flight.items() if k != "carrier_code"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_origin(self, client, patch_models, valid_flight):
        payload = {k: v for k, v in valid_flight.items() if k != "origin"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_destination(self, client, patch_models, valid_flight):
        payload = {k: v for k, v in valid_flight.items() if k != "destination"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_missing_scheduled_departure(self, client, patch_models, valid_flight):
        payload = {k: v for k, v in valid_flight.items() if k != "scheduled_departure"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_invalid_departure_hour(self, client, patch_models, valid_flight):
        payload = {**valid_flight, "departure_hour": 25}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_invalid_weather(self, client, patch_models, valid_flight):
        payload = {**valid_flight, "weather_condition": "tornado"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_invalid_iata_too_short(self, client, patch_models, valid_flight):
        payload = {**valid_flight, "origin": "OR"}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_prior_delay_negative(self, client, patch_models, valid_flight):
        payload = {**valid_flight, "prior_leg_delay_minutes": -5.0}
        r = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_predict_response_has_request_id(self, client, patch_models, valid_flight):
        r = client.post("/predict", data=json.dumps(valid_flight), content_type="application/json")
        assert "X-Request-ID" in r.headers


class TestCarrierRisk:
    def test_carrier_risk_returns_profile(self, client, patch_models):
        r = client.get("/carrier-risk/AA")
        assert r.status_code == 200
        data = r.get_json()
        assert "risk_score" in data

    def test_carrier_risk_lowercase_normalized(self, client, patch_models):
        r = client.get("/carrier-risk/aa")
        assert r.status_code == 200


class TestRouteAnalysis:
    def test_route_analysis_valid(self, client, patch_models):
        r = client.get("/route-analysis?origin=ORD&dest=LAX")
        assert r.status_code == 200
        data = r.get_json()
        assert "risk_score" in data

    def test_route_analysis_missing_params(self, client, patch_models):
        r = client.get("/route-analysis?origin=ORD")
        assert r.status_code == 400

    def test_route_analysis_invalid_code(self, client, patch_models):
        r = client.get("/route-analysis?origin=OR&dest=LAX")
        assert r.status_code == 400


class TestBatch:
    def test_batch_predict_valid(self, client, patch_models, valid_flight):
        payload = {"flights": [valid_flight, {**valid_flight, "flight_id": "AA-002"}]}
        r = client.post("/predict/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 2

    def test_batch_too_large(self, client, patch_models, valid_flight):
        payload = {"flights": [valid_flight] * 501}
        r = client.post("/predict/batch", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 400

    def test_batch_missing_key(self, client, patch_models):
        r = client.post("/predict/batch", data=json.dumps({}), content_type="application/json")
        assert r.status_code == 400
