from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def patch_models(monkeypatch):
    import api.app as app_module

    mock_ensemble = MagicMock()
    mock_ensemble.predict_proba.return_value = [0.65]
    mock_ensemble.explain.return_value = [
        {"shap_features": {"departure_hour": 0.15, "carrier_risk": 0.12}}
    ]

    mock_risk = MagicMock()
    mock_risk.score_carrier.return_value = {
        "carrier_code": "AA",
        "risk_score": 0.45,
        "baseline_delay_rate": 0.218,
        "weather_sensitivity": 1.1,
    }
    mock_risk.score_route.return_value = {
        "origin": "ORD",
        "destination": "LAX",
        "avg_delay_minutes": 22.5,
        "risk_score": 0.38,
    }

    mock_fe = MagicMock()
    mock_fe.transform.side_effect = lambda df: df

    monkeypatch.setattr(app_module, "_ensemble", mock_ensemble)
    monkeypatch.setattr(app_module, "_carrier_risk_scorer", mock_risk)
    monkeypatch.setattr(app_module, "_feature_engineer", mock_fe)
    monkeypatch.setattr(app_module, "_feature_cols", [])
    return mock_ensemble


@pytest.fixture
def valid_flight():
    return {
        "flight_id": "AA-2024-ORD-LAX-001",
        "carrier_code": "AA",
        "origin": "ORD",
        "destination": "LAX",
        "scheduled_departure": "2024-06-15T08:30:00",
        "aircraft_type": "B737",
        "distance_miles": 1745.0,
        "departure_hour": 8,
        "day_of_week": 4,
        "month": 6,
        "is_holiday": False,
        "prior_leg_delay_minutes": 0.0,
        "weather_condition": "clear",
    }
