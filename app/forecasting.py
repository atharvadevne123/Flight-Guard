"""Time-series forecasting of route delay trends."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def simple_moving_average(values: list[float], window: int = 7) -> list[float]:
    """Compute a trailing simple moving average.

    Args:
        values: Ordered series of observed delay minutes or probabilities.
        window: Number of trailing points per average.

    Returns:
        A list the same length as values; positions before a full window
        use the mean of available points.
    """
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def linear_trend_forecast(values: list[float], horizon: int = 7) -> dict[str, Any]:
    """Fit a least-squares linear trend and project it forward.

    Args:
        values: Ordered historical series (oldest first).
        horizon: Number of future steps to forecast.

    Returns:
        Dict with slope, intercept, forecast list, and trend direction.
    """
    n = len(values)
    if n < 3:
        return {
            "slope": 0.0,
            "intercept": float(values[-1]) if values else 0.0,
            "forecast": [float(values[-1])] * horizon if values else [],
            "trend": "flat",
            "reason": "insufficient_history",
        }

    x = np.arange(n, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = (slope * future_x + intercept).tolist()

    if slope > 0.01:
        trend = "worsening"
    elif slope < -0.01:
        trend = "improving"
    else:
        trend = "flat"

    return {
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "forecast": [round(float(v), 4) for v in forecast],
        "trend": trend,
    }


def seasonal_decompose_naive(values: list[float], period: int = 7) -> dict[str, list[float]]:
    """Split a series into trend (SMA) and seasonal residual components.

    Args:
        values: Ordered historical series.
        period: Season length (7 = weekly pattern for daily data).

    Returns:
        Dict with 'trend' and 'seasonal' component lists.
    """
    if len(values) < period:
        return {"trend": list(map(float, values)), "seasonal": [0.0] * len(values)}

    trend = simple_moving_average(values, window=period)
    residual = [v - t for v, t in zip(values, trend)]

    seasonal_pattern = [0.0] * period
    counts = [0] * period
    for i, r in enumerate(residual):
        seasonal_pattern[i % period] += r
        counts[i % period] += 1
    seasonal_pattern = [s / c if c else 0.0 for s, c in zip(seasonal_pattern, counts)]
    seasonal = [seasonal_pattern[i % period] for i in range(len(values))]

    return {
        "trend": [round(t, 4) for t in trend],
        "seasonal": [round(s, 4) for s in seasonal],
    }
