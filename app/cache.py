"""In-memory TTL cache for hot endpoint responses."""

from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    """Thread-safe in-memory cache with per-entry time-to-live.

    Args:
        ttl_seconds: Lifetime of each entry before it expires.
        max_entries: Soft cap; oldest entries are evicted when exceeded.
    """

    def __init__(self, ttl_seconds: float = 30.0, max_entries: int = 256) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return a cached value, or None if missing or expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        """Store a value under key with the configured TTL."""
        with self._lock:
            if len(self._store) >= self._max:
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.monotonic() + self._ttl, value)

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Shared cache for metrics/stats endpoints
metrics_cache = TTLCache(ttl_seconds=15.0)
