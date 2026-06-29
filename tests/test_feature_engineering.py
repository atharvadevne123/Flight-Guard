from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.feature_engineering import FlightFeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "flight_id": "AA-001", "carrier_code": "AA", "origin": "ORD",
            "destination": "LAX", "scheduled_departure": "2024-06-15T08:30:00",
            "aircraft_type": "B737", "distance_miles": 1745.0, "departure_hour": 8,
            "day_of_week": 4, "month": 6, "is_holiday": False,
            "prior_leg_delay_minutes": 0.0, "weather_condition": "clear",
        },
        {
            "flight_id": "DL-002", "carrier_code": "DL", "origin": "ATL",
            "destination": "JFK", "scheduled_departure": "2024-01-10T17:45:00",
            "aircraft_type": "A320", "distance_miles": 880.0, "departure_hour": 17,
            "day_of_week": 2, "month": 1, "is_holiday": False,
            "prior_leg_delay_minutes": 45.0, "weather_condition": "snow",
        },
    ])


class TestFlightFeatureEngineer:
    def test_fit_transform_returns_dataframe(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_schedule_slot_feature(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "schedule_slot" in result.columns

    def test_is_weekend_feature(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "is_weekend" in result.columns

    def test_is_hub_origin_feature(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "is_hub_origin" in result.columns
        assert result["is_hub_origin"].iloc[0] == 1  # ORD is a hub

    def test_weather_severity_encoding(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "weather_severity" in result.columns
        clear_idx = result[result.index == 0]["weather_severity"].values[0]
        snow_idx  = result[result.index == 1]["weather_severity"].values[0]
        assert snow_idx > clear_idx

    def test_prior_leg_propagation(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        assert "prior_leg_delay_minutes" in result.columns

    def test_no_nans_in_output(self, sample_df):
        fe = FlightFeatureEngineer()
        result = fe.fit_transform(sample_df)
        numeric_cols = result.select_dtypes(include="number").columns
        assert not result[numeric_cols].isnull().any().any()

    def test_transform_without_fit_raises(self, sample_df):
        fe = FlightFeatureEngineer()
        with pytest.raises(RuntimeError):
            fe.transform(sample_df)
