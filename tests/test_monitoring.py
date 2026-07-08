"""Tests for drift detection and monitoring utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    _prediction_window,
    compute_drift,
    compute_psi,
    get_online_drift,
    set_reference_scores,
    update_prediction_window,
)


class TestComputeDrift:
    def test_no_drift_identical_distributions(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, 200).tolist()
        result = compute_drift(data[:100], data[100:])
        assert result["p_value"] >= 0.05
        assert result["drift_detected"] is False

    def test_drift_detected_different_distributions(self):
        rng = np.random.default_rng(1)
        ref = rng.uniform(0.0, 0.3, 200).tolist()
        cur = rng.uniform(0.7, 1.0, 200).tolist()
        result = compute_drift(ref, cur)
        assert result["drift_detected"] is True
        assert result["ks_statistic"] > 0.5

    def test_insufficient_data_returns_no_drift(self):
        result = compute_drift([0.1, 0.2], [0.3, 0.4])
        assert result["drift_detected"] is False
        assert result.get("reason") == "insufficient_data"

    def test_result_has_required_keys(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(0, 1, 200).tolist()
        result = compute_drift(data[:100], data[100:])
        for key in ("ks_statistic", "p_value", "drift_detected"):
            assert key in result

    def test_ks_statistic_between_0_and_1(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(0, 1, 200).tolist()
        result = compute_drift(data[:100], data[100:])
        assert 0.0 <= result["ks_statistic"] <= 1.0

    @pytest.mark.parametrize("size", [50, 100, 300])
    def test_drift_with_various_sizes(self, size):
        rng = np.random.default_rng(size)
        ref = rng.beta(2, 5, size).tolist()
        cur = rng.beta(5, 2, size).tolist()
        result = compute_drift(ref, cur)
        assert isinstance(result["drift_detected"], bool)


class TestComputePSI:
    def test_psi_same_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, 500).tolist()
        psi = compute_psi(data[:250], data[250:])
        assert psi < 0.1

    def test_psi_different_distribution_elevated(self):
        rng = np.random.default_rng(1)
        ref = rng.uniform(0.0, 0.3, 300).tolist()
        cur = rng.uniform(0.7, 1.0, 300).tolist()
        psi = compute_psi(ref, cur)
        assert psi > 0.5

    def test_psi_returns_float(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(0, 1, 200).tolist()
        psi = compute_psi(data[:100], data[100:])
        assert isinstance(psi, float)


class TestOnlineDrift:
    def test_online_drift_with_reference_set(self):
        rng = np.random.default_rng(42)
        ref = rng.uniform(0.0, 0.4, 300).tolist()
        set_reference_scores(ref)
        _prediction_window.clear()
        for s in rng.uniform(0.0, 0.4, 100).tolist():
            update_prediction_window(s)
        result = get_online_drift()
        assert isinstance(result, dict)
        assert "drift_detected" in result

    def test_update_window_bounded(self):
        _prediction_window.clear()
        for i in range(600):
            update_prediction_window(float(i) / 600)
        assert len(_prediction_window) <= 500
