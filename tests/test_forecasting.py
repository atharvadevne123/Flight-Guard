"""Tests for time-series forecasting utilities."""

from __future__ import annotations

import pytest

from app.forecasting import (
    linear_trend_forecast,
    seasonal_decompose_naive,
    simple_moving_average,
)


class TestSimpleMovingAverage:
    def test_empty_series(self):
        assert simple_moving_average([]) == []

    def test_constant_series(self):
        result = simple_moving_average([5.0] * 10, window=3)
        assert all(abs(v - 5.0) < 1e-9 for v in result)

    def test_output_length_matches_input(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(simple_moving_average(values, window=2)) == len(values)

    @pytest.mark.parametrize("window", [1, 3, 7])
    def test_various_windows(self, window):
        values = list(range(20))
        result = simple_moving_average([float(v) for v in values], window=window)
        assert len(result) == 20

    def test_smoothing_reduces_variance(self):
        import numpy as np

        rng = np.random.default_rng(0)
        noisy = (rng.normal(10, 3, 50)).tolist()
        smooth = simple_moving_average(noisy, window=7)
        assert np.var(smooth) < np.var(noisy)


class TestLinearTrendForecast:
    def test_insufficient_history(self):
        result = linear_trend_forecast([1.0, 2.0])
        assert result.get("reason") == "insufficient_history"

    def test_increasing_trend_detected(self):
        values = [float(i) for i in range(20)]
        result = linear_trend_forecast(values)
        assert result["trend"] == "worsening"
        assert result["slope"] > 0

    def test_decreasing_trend_detected(self):
        values = [float(20 - i) for i in range(20)]
        result = linear_trend_forecast(values)
        assert result["trend"] == "improving"

    def test_flat_trend_detected(self):
        result = linear_trend_forecast([5.0] * 20)
        assert result["trend"] == "flat"

    @pytest.mark.parametrize("horizon", [1, 7, 30])
    def test_forecast_length(self, horizon):
        values = [float(i) for i in range(15)]
        result = linear_trend_forecast(values, horizon=horizon)
        assert len(result["forecast"]) == horizon

    def test_forecast_continues_trend(self):
        values = [float(i) for i in range(10)]
        result = linear_trend_forecast(values, horizon=3)
        # Next values should be ~10, 11, 12
        assert result["forecast"][0] > values[-1] - 1


class TestSeasonalDecompose:
    def test_short_series_no_seasonal(self):
        result = seasonal_decompose_naive([1.0, 2.0, 3.0], period=7)
        assert result["seasonal"] == [0.0, 0.0, 0.0]

    def test_components_same_length(self):
        values = [float(i % 7) for i in range(28)]
        result = seasonal_decompose_naive(values, period=7)
        assert len(result["trend"]) == 28
        assert len(result["seasonal"]) == 28

    def test_periodic_signal_captured(self):
        # Strong weekly pattern
        values = [10.0 if i % 7 == 0 else 2.0 for i in range(35)]
        result = seasonal_decompose_naive(values, period=7)
        seasonal = result["seasonal"]
        # Peak position should have the highest seasonal component
        assert seasonal[0] == max(seasonal[:7])
