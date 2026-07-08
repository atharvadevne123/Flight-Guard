"""Feature engineering pipeline for flight delay prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Carrier delay statistics derived from historical data (minutes)
CARRIER_DELAY_STATS: dict[str, float] = {
    "AA": 18.5,
    "UA": 22.1,
    "DL": 12.3,
    "WN": 16.8,
    "B6": 25.4,
    "AS": 10.2,
    "NK": 28.9,
    "F9": 24.6,
    "G4": 31.2,
    "SY": 19.7,
    "HA": 8.1,
    "VX": 14.3,
    "OO": 20.5,
    "MQ": 23.8,
    "YV": 21.0,
}

# Airport congestion index (higher = more congested)
AIRPORT_CONGESTION: dict[str, float] = {
    "ATL": 0.92,
    "ORD": 0.89,
    "LAX": 0.87,
    "DFW": 0.85,
    "DEN": 0.78,
    "JFK": 0.88,
    "SFO": 0.83,
    "SEA": 0.72,
    "LAS": 0.71,
    "MCO": 0.74,
    "EWR": 0.91,
    "CLT": 0.80,
    "PHX": 0.70,
    "MIA": 0.82,
    "IAH": 0.76,
    "BOS": 0.79,
    "MSP": 0.69,
    "DTW": 0.73,
    "FLL": 0.75,
    "BWI": 0.65,
}

# Seasonal delay multipliers (month 1-12)
SEASONAL_FACTOR: dict[int, float] = {
    1: 1.35,
    2: 1.28,
    3: 1.15,
    4: 1.05,
    5: 1.08,
    6: 1.22,
    7: 1.25,
    8: 1.20,
    9: 1.02,
    10: 0.98,
    11: 1.18,
    12: 1.42,
}

# Peak hours 6-9 AM and 4-8 PM get higher congestion
PEAK_HOURS = frozenset(range(6, 10)) | frozenset(range(16, 21))


class CarrierRiskEncoder(BaseEstimator, TransformerMixin):
    """Encode carrier as historical average delay score."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> CarrierRiskEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["carrier_avg_delay"] = X["carrier"].map(CARRIER_DELAY_STATS).fillna(20.0)
        X["carrier_risk_tier"] = pd.cut(
            X["carrier_avg_delay"],
            bins=[0, 12, 18, 24, 100],
            labels=[0, 1, 2, 3],
        ).astype(float)
        return X


class RouteFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode origin/destination airport congestion and route distance tier."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> RouteFeatureEncoder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["origin_congestion"] = X["origin"].map(AIRPORT_CONGESTION).fillna(0.65)
        X["dest_congestion"] = X["destination"].map(AIRPORT_CONGESTION).fillna(0.65)
        X["route_congestion_product"] = X["origin_congestion"] * X["dest_congestion"]
        distance = X.get("distance_km", pd.Series(dtype=float))
        if distance.isna().all() or len(distance) == 0:
            X["distance_tier"] = 1.0
        else:
            X["distance_tier"] = pd.cut(
                X["distance_km"].fillna(1000.0),
                bins=[0, 500, 1500, 3000, 15000],
                labels=[0, 1, 2, 3],
            ).astype(float)
        return X


class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time-of-day, day-of-week, and seasonal features."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> TemporalFeatureExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["is_peak_hour"] = X["scheduled_hour"].apply(lambda h: 1.0 if h in PEAK_HOURS else 0.0)
        X["is_weekend"] = X["day_of_week"].apply(lambda d: 1.0 if d >= 5 else 0.0)
        X["seasonal_factor"] = X["month"].map(SEASONAL_FACTOR).fillna(1.1)
        X["hour_sin"] = np.sin(2 * np.pi * X["scheduled_hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["scheduled_hour"] / 24)
        X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        return X


class LagRollingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build lag and rolling average delay features per carrier-route."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> LagRollingFeatureBuilder:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Composite delay pressure score (proxy for lag/rolling)
        X["route_delay_pressure"] = (
            X["carrier_avg_delay"] * 0.4
            + X["origin_congestion"] * 30 * 0.3
            + X["dest_congestion"] * 30 * 0.2
            + X["seasonal_factor"] * 10 * 0.1
        )
        X["delay_risk_score"] = (
            X["carrier_risk_tier"] * 10
            + X["route_congestion_product"] * 20
            + X["is_peak_hour"] * 8
            + X["is_weekend"] * 3
        )
        return X


class DropCategoricalColumns(BaseEstimator, TransformerMixin):
    """Drop raw string columns after encoding."""

    COLS_TO_DROP = ["carrier", "origin", "destination"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> DropCategoricalColumns:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in self.COLS_TO_DROP if c in X.columns]
        return X.drop(columns=drop)


FEATURE_COLUMNS = [
    "carrier",
    "origin",
    "destination",
    "scheduled_hour",
    "day_of_week",
    "month",
    "distance_km",
]

MODEL_FEATURE_COLUMNS = [
    "carrier_avg_delay",
    "carrier_risk_tier",
    "origin_congestion",
    "dest_congestion",
    "route_congestion_product",
    "distance_tier",
    "is_peak_hour",
    "is_weekend",
    "seasonal_factor",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "route_delay_pressure",
    "delay_risk_score",
]


def build_feature_pipeline() -> Pipeline:
    """Build the full sklearn feature engineering pipeline."""
    return Pipeline(
        [
            ("carrier_risk", CarrierRiskEncoder()),
            ("route_features", RouteFeatureEncoder()),
            ("temporal", TemporalFeatureExtractor()),
            ("lag_rolling", LagRollingFeatureBuilder()),
            ("drop_categorical", DropCategoricalColumns()),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_dataframe(data: dict) -> pd.DataFrame:
    """Convert a prediction request dict to a DataFrame row."""
    row = {
        "carrier": data.get("carrier", "XX"),
        "origin": data.get("origin", "XXX"),
        "destination": data.get("destination", "XXX"),
        "scheduled_hour": int(data.get("scheduled_hour", 12)),
        "day_of_week": int(data.get("day_of_week", 1)),
        "month": int(data.get("month", 6)),
        "distance_km": float(data.get("distance_km", 1000.0)),
    }
    return pd.DataFrame([row])


def generate_synthetic_data(n_samples: int = 2000) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic flight data for training."""
    rng = np.random.default_rng(42)
    carriers = list(CARRIER_DELAY_STATS.keys())
    airports = list(AIRPORT_CONGESTION.keys())

    df = pd.DataFrame(
        {
            "carrier": rng.choice(carriers, n_samples),
            "origin": rng.choice(airports, n_samples),
            "destination": rng.choice(airports, n_samples),
            "scheduled_hour": rng.integers(5, 23, n_samples),
            "day_of_week": rng.integers(0, 7, n_samples),
            "month": rng.integers(1, 13, n_samples),
            "distance_km": rng.uniform(300, 5000, n_samples),
        }
    )

    # Target: 1 = delayed (>15 min), 0 = on time
    base_delay_prob = (
        df["carrier"].map({k: v / 100 for k, v in CARRIER_DELAY_STATS.items()}).fillna(0.2)
    )
    congestion_factor = df["origin"].map(AIRPORT_CONGESTION).fillna(0.65)
    season_factor = (
        df["month"].map({k: (v - 1.0) * 0.3 for k, v in SEASONAL_FACTOR.items()}).fillna(0.05)
    )
    peak_factor = df["scheduled_hour"].apply(lambda h: 0.12 if h in PEAK_HOURS else 0.0)

    delay_prob = (base_delay_prob + congestion_factor * 0.15 + season_factor + peak_factor).clip(
        0.05, 0.95
    )
    y = pd.Series(rng.binomial(1, delay_prob, n_samples), name="delayed")

    logger.info(
        "Generated %d synthetic training samples (%.1f%% delayed)", n_samples, y.mean() * 100
    )
    return df, y
