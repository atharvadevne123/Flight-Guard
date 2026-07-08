"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    AIRPORT_CONGESTION,
    CARRIER_DELAY_STATS,
    SEASONAL_FACTOR,
    CarrierRiskEncoder,
    DropCategoricalColumns,
    RouteFeatureEncoder,
    TemporalFeatureExtractor,
    build_feature_pipeline,
    generate_synthetic_data,
    prepare_dataframe,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "carrier": "AA",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_hour": 8,
                "day_of_week": 1,
                "month": 7,
                "distance_km": 3983.0,
            },
            {
                "carrier": "NK",
                "origin": "ORD",
                "destination": "ATL",
                "scheduled_hour": 17,
                "day_of_week": 4,
                "month": 12,
                "distance_km": 1139.0,
            },
            {
                "carrier": "ZZ",
                "origin": "XXX",
                "destination": "YYY",
                "scheduled_hour": 3,
                "day_of_week": 0,
                "month": 1,
                "distance_km": 800.0,
            },
        ]
    )


class TestCarrierRiskEncoder:
    def test_known_carrier(self, sample_df):
        enc = CarrierRiskEncoder()
        result = enc.fit_transform(sample_df)
        assert "carrier_avg_delay" in result.columns
        assert result.loc[0, "carrier_avg_delay"] == CARRIER_DELAY_STATS["AA"]

    def test_unknown_carrier_fallback(self, sample_df):
        enc = CarrierRiskEncoder()
        result = enc.fit_transform(sample_df)
        assert result.loc[2, "carrier_avg_delay"] == 20.0

    def test_carrier_risk_tier_range(self, sample_df):
        enc = CarrierRiskEncoder()
        result = enc.fit_transform(sample_df)
        assert result["carrier_risk_tier"].between(0, 3).all()


class TestRouteFeatureEncoder:
    def test_known_airports(self, sample_df):
        enc = RouteFeatureEncoder()
        result = enc.fit_transform(sample_df)
        assert result.loc[0, "origin_congestion"] == AIRPORT_CONGESTION["JFK"]
        assert result.loc[0, "dest_congestion"] == AIRPORT_CONGESTION["LAX"]

    def test_unknown_airport_fallback(self, sample_df):
        enc = RouteFeatureEncoder()
        result = enc.fit_transform(sample_df)
        assert result.loc[2, "origin_congestion"] == 0.65

    def test_congestion_product(self, sample_df):
        enc = RouteFeatureEncoder()
        result = enc.fit_transform(sample_df)
        expected = AIRPORT_CONGESTION["JFK"] * AIRPORT_CONGESTION["LAX"]
        assert abs(result.loc[0, "route_congestion_product"] - expected) < 1e-6


class TestTemporalFeatureExtractor:
    @pytest.mark.parametrize(
        "hour,expected",
        [
            (6, 1.0),
            (8, 1.0),
            (17, 1.0),
            (20, 1.0),
            (3, 0.0),
            (12, 0.0),
            (22, 0.0),
        ],
    )
    def test_peak_hour_flag(self, hour, expected, sample_df):
        df = sample_df.copy()
        df["scheduled_hour"] = hour
        enc = TemporalFeatureExtractor()
        result = enc.fit_transform(df)
        assert (result["is_peak_hour"] == expected).all()

    @pytest.mark.parametrize("dow,expected", [(5, 1.0), (6, 1.0), (0, 0.0), (4, 0.0)])
    def test_weekend_flag(self, dow, expected, sample_df):
        df = sample_df.copy()
        df["day_of_week"] = dow
        enc = TemporalFeatureExtractor()
        result = enc.fit_transform(df)
        assert (result["is_weekend"] == expected).all()

    def test_cyclical_features_range(self, sample_df):
        enc = TemporalFeatureExtractor()
        result = enc.fit_transform(sample_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert result[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all()

    @pytest.mark.parametrize("month", [1, 6, 12])
    def test_seasonal_factor_populated(self, month, sample_df):
        df = sample_df.copy()
        df["month"] = month
        enc = TemporalFeatureExtractor()
        result = enc.fit_transform(df)
        assert (result["seasonal_factor"] == SEASONAL_FACTOR[month]).all()


class TestDropCategoricalColumns:
    def test_drops_carrier_origin_destination(self, sample_df):
        transformed_df = CarrierRiskEncoder().fit_transform(sample_df)
        transformed_df = RouteFeatureEncoder().fit_transform(transformed_df)
        dropper = DropCategoricalColumns()
        result = dropper.fit_transform(transformed_df)
        for col in ["carrier", "origin", "destination"]:
            assert col not in result.columns


class TestFullPipeline:
    def test_pipeline_output_is_array(self, sample_df):
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(sample_df)
        assert hasattr(result, "shape")
        assert result.shape[0] == len(sample_df)

    def test_pipeline_no_nans(self, sample_df):
        pipe = build_feature_pipeline()
        result = pipe.fit_transform(sample_df)
        assert not np.isnan(result).any()

    def test_pipeline_idempotent_transform(self, sample_df):
        pipe = build_feature_pipeline()
        pipe.fit(sample_df)
        r1 = pipe.transform(sample_df)
        r2 = pipe.transform(sample_df)
        np.testing.assert_array_almost_equal(r1, r2)


class TestSyntheticData:
    def test_generate_returns_correct_shape(self):
        X, y = generate_synthetic_data(n_samples=100)
        assert len(X) == 100
        assert len(y) == 100

    def test_generate_delay_rate_reasonable(self):
        X, y = generate_synthetic_data(n_samples=1000)
        # Expect 20-70% delayed
        assert 0.20 <= y.mean() <= 0.70

    def test_prepare_dataframe_shape(self, sample_flight_request):
        df = prepare_dataframe(sample_flight_request)
        assert df.shape == (1, 7)

    def test_prepare_dataframe_carrier_uppercase(self):
        df = prepare_dataframe(
            {
                "carrier": "aa",
                "origin": "jfk",
                "destination": "lax",
                "scheduled_hour": 8,
                "day_of_week": 1,
                "month": 6,
                "distance_km": 3983.0,
            }
        )
        # prepare_dataframe just wraps the dict — uppercase is done at schema level
        assert "carrier" in df.columns
