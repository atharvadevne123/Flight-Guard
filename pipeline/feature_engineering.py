"""
Flight feature engineering for delay prediction.

FlightFeatureEngineer builds a rich, model-ready feature matrix from raw
flight records covering:
  - Temporal patterns (hour buckets, day-of-week, month, holidays, quarter)
  - Route characteristics (distance buckets, hub-to-hub, direction)
  - Carrier statistics (historical delay rate from training data)
  - Aircraft type (narrow-body vs wide-body encoding)
  - Schedule slot (red-eye, early morning, morning, afternoon, evening)
  - Flight chaining (prior leg delay propagation)
  - Weather severity encoding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hub airports — same set used in carrier_risk_scorer for consistency
HUB_AIRPORTS = {
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "CLT", "PHX", "MIA", "EWR", "BOS", "MSP", "DTW", "FLL", "LGA", "BWI",
}

# US federal holidays (month, day) — simplified fixed-date set
US_FEDERAL_HOLIDAYS = {
    (1, 1), (7, 4), (11, 11), (12, 25), (12, 26), (1, 2),
}

# Approximate airport longitudes for direction encoding
_AIRPORT_LON: dict[str, float] = {
    "ATL": -84.4, "BOS": -71.0, "BWI": -76.7, "CLT": -80.9, "DCA": -77.0,
    "DEN": -104.7, "DFW": -97.0, "DTW": -83.4, "EWR": -74.2, "FLL": -80.2,
    "IAD": -77.5, "IAH": -95.3, "JFK": -73.8, "LAS": -115.2, "LAX": -118.4,
    "LGA": -73.9, "MCO": -81.3, "MDW": -87.7, "MIA": -80.3, "MSP": -93.2,
    "ORD": -87.9, "PDX": -122.6, "PHL": -75.2, "PHX": -112.0, "SEA": -122.3,
    "SFO": -122.4, "SLC": -112.0, "STL": -90.4,
}

_AIRPORT_LAT: dict[str, float] = {
    "ATL": 33.6, "BOS": 42.4, "BWI": 39.2, "CLT": 35.2, "DCA": 38.9,
    "DEN": 39.9, "DFW": 32.9, "DTW": 42.2, "EWR": 40.7, "FLL": 26.1,
    "IAD": 38.9, "IAH": 29.7, "JFK": 40.6, "LAS": 36.1, "LAX": 33.9,
    "LGA": 40.8, "MCO": 28.4, "MDW": 41.8, "MIA": 25.8, "MSP": 44.9,
    "ORD": 42.0, "PDX": 45.6, "PHL": 39.9, "PHX": 33.4, "SEA": 47.4,
    "SFO": 37.6, "SLC": 40.8, "STL": 38.7,
}

# Narrow-body aircraft types (single-aisle)
NARROW_BODY = {
    "B737", "B738", "B739", "B73W", "B73H", "A319", "A320", "A321",
    "MD80", "MD88", "MD90", "E175", "E170", "CRJ9", "CRJ7", "ERJ",
}

# Wide-body types
WIDE_BODY = {
    "B777", "B787", "B788", "B789", "B78X", "B767", "B763", "B764",
    "A330", "A332", "A333", "A350", "A359", "A380", "A388",
}

WEATHER_SEVERITY_MAP = {
    "clear": 0,
    "rain":  1,
    "wind":  2,
    "fog":   3,
    "snow":  4,
}


# ---------------------------------------------------------------------------
# Feature engineer
# ---------------------------------------------------------------------------

class FlightFeatureEngineer:
    """
    Transforms raw flight records into a model-ready feature matrix.

    fit() learns carrier statistics from training data.
    transform() applies all feature groups to any DataFrame.
    fit_transform() is a convenience wrapper.
    """

    def __init__(self):
        self._fitted = False
        self._carrier_delay_rates: dict[str, float] = {}
        self._carrier_mean_delays: dict[str, float] = {}
        self._global_delay_rate: float = 0.22
        self._global_mean_delay: float = 32.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FlightFeatureEngineer":
        """Learn carrier-level statistics from training data.

        Requires columns: carrier_code, dep_delay_minutes (or is_delayed).
        """
        if "carrier_code" in df.columns:
            grp = df.groupby("carrier_code")
            if "dep_delay_minutes" in df.columns:
                # Delay rate (≥15 min) and mean delay per carrier
                self._carrier_delay_rates = (
                    grp["dep_delay_minutes"]
                    .apply(lambda s: (s >= 15).mean())
                    .to_dict()
                )
                self._carrier_mean_delays = (
                    grp["dep_delay_minutes"]
                    .apply(lambda s: s[s >= 15].mean() if (s >= 15).any() else 0.0)
                    .to_dict()
                )
                self._global_delay_rate = float((df["dep_delay_minutes"] >= 15).mean())
                self._global_mean_delay = float(
                    df.loc[df["dep_delay_minutes"] >= 15, "dep_delay_minutes"].mean()
                    if (df["dep_delay_minutes"] >= 15).any() else 32.0
                )
            elif "is_delayed" in df.columns:
                self._carrier_delay_rates = grp["is_delayed"].mean().to_dict()
                self._global_delay_rate = float(df["is_delayed"].mean())
        self._fitted = True
        logger.info(
            "FlightFeatureEngineer fitted: {:,} rows, {} carriers, global delay rate {:.2%}",
            len(df), len(self._carrier_delay_rates), self._global_delay_rate,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        df = df.copy()
        df = self._temporal_features(df)
        df = self._schedule_slot(df)
        df = self._route_features(df)
        df = self._carrier_features(df)
        df = self._aircraft_features(df)
        df = self._chaining_features(df)
        df = self._weather_features(df)
        logger.debug("Transformed {:,} flights → {} features.", len(df), df.shape[1])
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from scheduled_departure or scalar columns."""
        if "scheduled_departure" in df.columns:
            ts = pd.to_datetime(df["scheduled_departure"], errors="coerce", utc=False)
            df["departure_hour"] = ts.dt.hour
            df["day_of_week"]    = ts.dt.dayofweek
            df["month"]          = ts.dt.month
            df["day_of_month"]   = ts.dt.day

        # Hour bucket (0-23 → 0-5 sextile)
        hour = df.get("departure_hour", pd.Series(np.zeros(len(df), dtype=int), index=df.index))
        df["hour_bucket"] = pd.cut(
            hour,
            bins=[-1, 5, 11, 13, 17, 20, 23],
            labels=[0, 1, 2, 3, 4, 5],
        ).astype(float)

        dow = df.get("day_of_week", pd.Series(np.zeros(len(df), dtype=int), index=df.index))
        df["is_weekend"] = (dow >= 5).astype(int)

        month = df.get("month", pd.Series(np.ones(len(df), dtype=int), index=df.index))
        df["quarter"] = np.ceil(month / 3).astype(int)

        # Holiday flag — honour explicit column first, then derive
        if "is_holiday" not in df.columns or df["is_holiday"].isnull().all():
            if "scheduled_departure" in df.columns:
                ts2 = pd.to_datetime(df["scheduled_departure"], errors="coerce", utc=False)
                df["is_holiday"] = [
                    int((t.month, t.day) in US_FEDERAL_HOLIDAYS)
                    if not pd.isnull(t) else 0
                    for t in ts2
                ]
            else:
                df["is_holiday"] = 0
        else:
            df["is_holiday"] = df["is_holiday"].fillna(False).astype(int)

        # Peak travel season (Jun-Aug, Dec)
        df["is_peak_season"] = month.isin([6, 7, 8, 12]).astype(int)

        return df

    def _schedule_slot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode departure slot category as integer."""
        hour = df.get("departure_hour", pd.Series(np.zeros(len(df), dtype=int), index=df.index))
        conditions = [
            hour.between(0,  4),   # red-eye
            hour.between(5,  8),   # early morning
            hour.between(9, 11),   # morning
            hour.between(12, 16),  # afternoon
            hour.between(17, 19),  # evening
            hour.between(20, 23),  # night
        ]
        choices = [0, 1, 2, 3, 4, 5]
        df["schedule_slot"] = np.select(conditions, choices, default=3)
        return df

    def _route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode route structural attributes."""
        origin = df.get("origin", pd.Series([""] * len(df), index=df.index))
        dest   = df.get("destination", pd.Series([""] * len(df), index=df.index))

        df["is_hub_origin"]  = origin.str.upper().isin(HUB_AIRPORTS).astype(int)
        df["is_hub_dest"]    = dest.str.upper().isin(HUB_AIRPORTS).astype(int)
        df["is_hub_to_hub"]  = (df["is_hub_origin"] & df["is_hub_dest"]).astype(int)

        # Distance buckets: short (<500), medium (500-1500), long (>1500)
        if "distance_miles" in df.columns:
            dist = df["distance_miles"].fillna(800)
            df["distance_bucket"] = pd.cut(
                dist,
                bins=[-1, 500, 1500, np.inf],
                labels=[0, 1, 2],
            ).astype(float)
            df["distance_log"] = np.log1p(dist)
        else:
            df["distance_bucket"] = 1.0
            df["distance_log"]    = np.log1p(800)

        # Direction encoding (0=eastbound, 1=westbound, 2=northbound, 3=southbound)
        df["flight_direction"] = [
            _encode_direction(o, d)
            for o, d in zip(origin.str.upper(), dest.str.upper())
        ]

        return df

    def _carrier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode carrier historical delay rate learned at fit() time."""
        if "carrier_code" not in df.columns:
            df["carrier_delay_rate"] = self._global_delay_rate
            df["carrier_mean_delay"] = self._global_mean_delay
            return df

        df["carrier_delay_rate"] = (
            df["carrier_code"]
            .str.upper()
            .map(self._carrier_delay_rates)
            .fillna(self._global_delay_rate)
        )
        df["carrier_mean_delay"] = (
            df["carrier_code"]
            .str.upper()
            .map(self._carrier_mean_delays)
            .fillna(self._global_mean_delay)
        )
        return df

    def _aircraft_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode aircraft body type from aircraft_type column."""
        if "aircraft_type" not in df.columns:
            df["is_narrow_body"] = 1
            df["is_wide_body"]   = 0
            return df

        ac = df["aircraft_type"].str.upper().fillna("")
        df["is_narrow_body"] = ac.isin(NARROW_BODY).astype(int)
        df["is_wide_body"]   = ac.isin(WIDE_BODY).astype(int)
        # Regional jets (neither narrow nor wide)
        df["is_regional"]    = (~ac.isin(NARROW_BODY) & ~ac.isin(WIDE_BODY)).astype(int)
        return df

    def _chaining_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Propagate prior-leg delay as a feature."""
        if "prior_leg_delay_minutes" not in df.columns:
            df["prior_leg_delay_minutes"] = 0.0
            df["has_prior_delay"]         = 0
            df["prior_delay_log"]         = 0.0
            return df

        df["prior_leg_delay_minutes"] = df["prior_leg_delay_minutes"].fillna(0).clip(lower=0)
        df["has_prior_delay"]         = (df["prior_leg_delay_minutes"] > 0).astype(int)
        df["prior_delay_log"]         = np.log1p(df["prior_leg_delay_minutes"])
        return df

    def _weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode weather condition as ordinal severity."""
        if "weather_condition" not in df.columns:
            df["weather_severity"] = 0
            return df
        df["weather_severity"] = (
            df["weather_condition"]
            .str.lower()
            .map(WEATHER_SEVERITY_MAP)
            .fillna(0)
            .astype(int)
        )
        return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_direction(origin: str, destination: str) -> int:
    """0=eastbound, 1=westbound, 2=northbound, 3=southbound."""
    o_lon = _AIRPORT_LON.get(origin, 0.0)
    d_lon = _AIRPORT_LON.get(destination, 0.0)
    o_lat = _AIRPORT_LAT.get(origin, 0.0)
    d_lat = _AIRPORT_LAT.get(destination, 0.0)

    delta_lon = d_lon - o_lon
    delta_lat = d_lat - o_lat

    if abs(delta_lon) >= abs(delta_lat):
        return 0 if delta_lon > 0 else 1  # east or west
    return 2 if delta_lat > 0 else 3      # north or south
