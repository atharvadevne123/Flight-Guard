"""
Carrier and route risk scoring for flight delay prediction.

Provides:
  - CarrierRiskScorer.score_carrier()  → 0-1 risk with breakdown
  - CarrierRiskScorer.score_route()    → route delay profile
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

# Known hub airports (BTS top-20 by O&D volume)
HUB_AIRPORTS = {
    "ATL",
    "LAX",
    "ORD",
    "DFW",
    "DEN",
    "JFK",
    "SFO",
    "SEA",
    "LAS",
    "MCO",
    "CLT",
    "PHX",
    "MIA",
    "EWR",
    "BOS",
    "MSP",
    "DTW",
    "FLL",
    "LGA",
    "BWI",
}

# Historical on-time performance buckets (2019-2023 BTS averages)
CARRIER_BASELINE_DELAY_RATES = {
    "AA": 0.218,  # American Airlines
    "DL": 0.183,  # Delta Air Lines
    "UA": 0.212,  # United Airlines
    "WN": 0.234,  # Southwest Airlines
    "B6": 0.252,  # JetBlue
    "AS": 0.163,  # Alaska Airlines
    "NK": 0.286,  # Spirit Airlines
    "F9": 0.279,  # Frontier Airlines
    "G4": 0.271,  # Allegiant Air
    "SY": 0.262,  # Sun Country Airlines
}

# Weather sensitivity multiplier per carrier (fleet age / hub exposure proxy)
CARRIER_WEATHER_SENSITIVITY = {
    "AA": 1.10,
    "DL": 0.95,
    "UA": 1.05,
    "WN": 1.15,
    "B6": 1.20,
    "AS": 1.00,
    "NK": 1.25,
    "F9": 1.22,
    "G4": 1.18,
    "SY": 1.12,
}

# Peak-hour delay multiplier (based on NAS congestion data)
PEAK_HOURS = set(range(7, 10)) | set(range(16, 20))  # 07:00-09:59, 16:00-19:59
PEAK_MULTIPLIER = 1.30
OFF_PEAK_MULTIPLIER = 1.00


@dataclass
class CarrierRiskProfile:
    carrier_code: str
    risk_score: float  # 0-1 composite
    base_delay_rate: float  # Historical delay rate 0-1
    weather_sensitivity: float  # Multiplier (1.0 = average)
    peak_hour_multiplier: float  # Applied if departure in peak window
    sample_size: int  # Flights in history_df for this carrier
    mean_delay_minutes: float  # Mean delay among delayed flights
    p90_delay_minutes: float  # 90th-pct delay minutes
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RouteRiskProfile:
    origin: str
    destination: str
    risk_score: float
    historical_delay_rate: float
    mean_delay_minutes: float
    p90_delay_minutes: float
    is_hub_to_hub: bool
    is_eastbound: bool
    route_distance_miles: Optional[float]
    sample_size: int
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CarrierRiskScorer:
    """
    Computes carrier-level and route-level delay risk scores.

    Risk scores are composites of:
      1. Historical delay rate (from BTS-style training data)
      2. Weather sensitivity index (fleet/hub exposure proxy)
      3. Peak-hour congestion multiplier
      4. NAS / late-arriving aircraft propagation factor
    """

    def __init__(
        self,
        delay_threshold_minutes: float = 15.0,
        weather_weight: float = 0.25,
        historical_weight: float = 0.50,
        congestion_weight: float = 0.25,
    ):
        """
        Args:
            delay_threshold_minutes: Minutes of delay to count as "delayed".
            weather_weight: Weight of weather sensitivity in composite score.
            historical_weight: Weight of historical delay rate in composite score.
            congestion_weight: Weight of congestion/peak-hour exposure.
        """
        if abs(weather_weight + historical_weight + congestion_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        self.delay_threshold = delay_threshold_minutes
        self.weather_weight = weather_weight
        self.historical_weight = historical_weight
        self.congestion_weight = congestion_weight

    # ------------------------------------------------------------------
    # Carrier risk
    # ------------------------------------------------------------------

    def score_carrier(
        self,
        carrier_code: str,
        history_df: Optional[pd.DataFrame] = None,
        departure_hour: Optional[int] = None,
        weather_condition: Optional[str] = None,
    ) -> CarrierRiskProfile:
        """Compute a composite risk score for a carrier.

        Args:
            carrier_code: IATA 2-letter carrier code (e.g. "AA").
            history_df: DataFrame with columns [carrier_code, dep_delay_minutes].
                        If None, uses BTS baseline estimates.
            departure_hour: Hour of departure (0-23) for peak multiplier.
            weather_condition: One of clear/rain/snow/fog/wind.

        Returns:
            CarrierRiskProfile with risk_score and breakdown.
        """
        code = carrier_code.upper()

        # 1. Historical delay rate
        if history_df is not None and not history_df.empty:
            carrier_hist = history_df[history_df["carrier_code"].str.upper() == code]
            sample_size = len(carrier_hist)
            if sample_size > 0 and "dep_delay_minutes" in carrier_hist.columns:
                delayed = carrier_hist["dep_delay_minutes"] >= self.delay_threshold
                base_rate = float(delayed.mean())
                mean_delay = float(
                    carrier_hist.loc[delayed, "dep_delay_minutes"].mean() if delayed.any() else 0.0
                )
                p90_delay = float(
                    carrier_hist.loc[delayed, "dep_delay_minutes"].quantile(0.90)
                    if delayed.any()
                    else 0.0
                )
            else:
                base_rate = CARRIER_BASELINE_DELAY_RATES.get(code, 0.22)
                mean_delay = 32.0
                p90_delay = 75.0
                sample_size = 0
        else:
            base_rate = CARRIER_BASELINE_DELAY_RATES.get(code, 0.22)
            mean_delay = 32.0
            p90_delay = 75.0
            sample_size = 0

        # 2. Weather sensitivity
        weather_sens = CARRIER_WEATHER_SENSITIVITY.get(code, 1.05)
        weather_factor = _weather_severity(weather_condition)
        effective_weather_score = min(1.0, (weather_sens - 0.9) / 0.4 * weather_factor)

        # 3. Peak-hour congestion
        if departure_hour is not None:
            peak_mult = PEAK_MULTIPLIER if departure_hour in PEAK_HOURS else OFF_PEAK_MULTIPLIER
        else:
            peak_mult = 1.10  # assume mild peak exposure when unknown
        congestion_score = min(1.0, (peak_mult - 1.0) / 0.35)

        # 4. Composite (weighted sum, capped at 1)
        risk_score = min(
            1.0,
            (
                self.historical_weight * base_rate
                + self.weather_weight * effective_weather_score
                + self.congestion_weight * congestion_score
            ),
        )

        profile = CarrierRiskProfile(
            carrier_code=code,
            risk_score=round(risk_score, 4),
            base_delay_rate=round(base_rate, 4),
            weather_sensitivity=round(weather_sens, 4),
            peak_hour_multiplier=round(peak_mult, 4),
            sample_size=sample_size,
            mean_delay_minutes=round(mean_delay, 2),
            p90_delay_minutes=round(p90_delay, 2),
            breakdown={
                "historical_component": round(self.historical_weight * base_rate, 4),
                "weather_component": round(self.weather_weight * effective_weather_score, 4),
                "congestion_component": round(self.congestion_weight * congestion_score, 4),
            },
        )
        logger.debug("Carrier {} risk_score={:.4f}", code, risk_score)
        return profile

    # ------------------------------------------------------------------
    # Route risk
    # ------------------------------------------------------------------

    def score_route(
        self,
        origin: str,
        destination: str,
        history_df: Optional[pd.DataFrame] = None,
    ) -> RouteRiskProfile:
        """Compute delay risk profile for an origin-destination route.

        Args:
            origin: 3-letter IATA origin code (e.g. "ORD").
            destination: 3-letter IATA destination code (e.g. "LAX").
            history_df: DataFrame with columns [origin, destination, dep_delay_minutes].

        Returns:
            RouteRiskProfile with risk_score and structural attributes.
        """
        orig = origin.upper()
        dest = destination.upper()

        is_hub_to_hub = orig in HUB_AIRPORTS and dest in HUB_AIRPORTS
        is_eastbound = _is_eastbound(orig, dest)

        if history_df is not None and not history_df.empty:
            route_hist = history_df[
                (history_df["origin"].str.upper() == orig)
                & (history_df["destination"].str.upper() == dest)
            ]
            sample_size = len(route_hist)
            if sample_size > 0 and "dep_delay_minutes" in route_hist.columns:
                delayed = route_hist["dep_delay_minutes"] >= self.delay_threshold
                hist_rate = float(delayed.mean())
                mean_delay = float(
                    route_hist.loc[delayed, "dep_delay_minutes"].mean() if delayed.any() else 0.0
                )
                p90_delay = float(
                    route_hist.loc[delayed, "dep_delay_minutes"].quantile(0.90)
                    if delayed.any()
                    else 0.0
                )
                dist = (
                    float(route_hist["distance_miles"].median())
                    if "distance_miles" in route_hist.columns
                    else None
                )
            else:
                hist_rate, mean_delay, p90_delay, dist = 0.22, 32.0, 75.0, None
                sample_size = 0
        else:
            hist_rate, mean_delay, p90_delay, dist = 0.22, 32.0, 75.0, None
            sample_size = 0

        # Hub-to-hub routes suffer from NAS congestion but have more frequent service
        congestion_adj = 1.12 if is_hub_to_hub else 0.95
        # Eastbound flights are generally faster (jet stream) — slight risk reduction
        direction_adj = 0.97 if is_eastbound else 1.03

        risk_score = min(1.0, hist_rate * congestion_adj * direction_adj)

        profile = RouteRiskProfile(
            origin=orig,
            destination=dest,
            risk_score=round(risk_score, 4),
            historical_delay_rate=round(hist_rate, 4),
            mean_delay_minutes=round(mean_delay, 2),
            p90_delay_minutes=round(p90_delay, 2),
            is_hub_to_hub=is_hub_to_hub,
            is_eastbound=is_eastbound,
            route_distance_miles=round(dist, 1) if dist is not None else None,
            sample_size=sample_size,
            breakdown={
                "hub_to_hub_multiplier": congestion_adj,
                "direction_multiplier": direction_adj,
                "base_historical_rate": round(hist_rate, 4),
            },
        )
        logger.debug("Route {}-{} risk_score={:.4f}", orig, dest, risk_score)
        return profile


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_WEATHER_SEVERITY: dict[str, float] = {
    "clear": 0.00,
    "rain": 0.35,
    "wind": 0.45,
    "fog": 0.60,
    "snow": 0.85,
}


def _weather_severity(condition: Optional[str]) -> float:
    """Return 0-1 severity factor for a weather condition string."""
    if condition is None:
        return 0.10  # mild unknown exposure
    return _WEATHER_SEVERITY.get(condition.lower(), 0.10)


# Approximate US airport longitudes for eastbound detection
_AIRPORT_LONGITUDE: dict[str, float] = {
    "ATL": -84.4,
    "BOS": -71.0,
    "BWI": -76.7,
    "CLT": -80.9,
    "DCA": -77.0,
    "DEN": -104.7,
    "DFW": -97.0,
    "DTW": -83.4,
    "EWR": -74.2,
    "FLL": -80.2,
    "IAD": -77.5,
    "IAH": -95.3,
    "JFK": -73.8,
    "LAS": -115.2,
    "LAX": -118.4,
    "LGA": -73.9,
    "MCO": -81.3,
    "MDW": -87.7,
    "MIA": -80.3,
    "MSP": -93.2,
    "ORD": -87.9,
    "PDX": -122.6,
    "PHL": -75.2,
    "PHX": -112.0,
    "SEA": -122.3,
    "SFO": -122.4,
    "SLC": -112.0,
    "STL": -90.4,
}


def _is_eastbound(origin: str, destination: str) -> bool:
    """Return True if the flight travels from west to east (increasing longitude)."""
    orig_lon = _AIRPORT_LONGITUDE.get(origin, 0.0)
    dest_lon = _AIRPORT_LONGITUDE.get(destination, 0.0)
    # Eastbound = destination is to the right (higher longitude in US convention)
    return dest_lon > orig_lon
