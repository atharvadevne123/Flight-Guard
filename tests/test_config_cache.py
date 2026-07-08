"""Tests for settings and TTL cache."""

from __future__ import annotations

import time

from app.cache import TTLCache
from app.config import Settings, get_settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.retrain_auc_gate == 0.70
        assert s.rate_limit_requests == 200

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("RETRAIN_AUC_GATE", "0.85")
        s = get_settings()
        assert s.retrain_auc_gate == 0.85

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("RETRAIN_MIN_SAMPLES", "not-a-number")
        s = get_settings()
        assert s.retrain_min_samples == 200

    def test_settings_frozen(self):
        s = Settings()
        try:
            s.log_level = "DEBUG"  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised


class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=10)
        cache.set("k", {"v": 1})
        assert cache.get("k") == {"v": 1}

    def test_missing_key_returns_none(self):
        cache = TTLCache()
        assert cache.get("nope") is None

    def test_expiry(self):
        cache = TTLCache(ttl_seconds=0.05)
        cache.set("k", "v")
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_eviction_at_capacity(self):
        cache = TTLCache(ttl_seconds=60, max_entries=3)
        for i in range(5):
            cache.set(f"k{i}", i)
        assert len(cache) <= 3

    def test_clear(self):
        cache = TTLCache()
        cache.set("a", 1)
        cache.clear()
        assert len(cache) == 0
