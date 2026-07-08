"""Tests for route similarity index."""

from __future__ import annotations

import pytest

from app.similarity import RouteSimilarityIndex


@pytest.fixture
def index() -> RouteSimilarityIndex:
    idx = RouteSimilarityIndex(dim=4)
    idx.add("AA:JFK-LAX", [0.88, 0.87, 18.5, 3983.0])
    idx.add("DL:ATL-SEA", [0.92, 0.72, 12.3, 3510.0])
    idx.add("NK:ORD-EWR", [0.89, 0.91, 28.9, 1178.0])
    return idx


class TestRouteSimilarityIndex:
    def test_size(self, index):
        assert index.size == 3

    def test_search_returns_results(self, index):
        results = index.search([0.88, 0.87, 18.0, 3900.0], k=2)
        assert len(results) == 2

    def test_nearest_is_most_similar(self, index):
        results = index.search([0.88, 0.87, 18.5, 3983.0], k=1)
        assert results[0]["label"] == "AA:JFK-LAX"
        assert results[0]["distance"] < 1e-6

    def test_search_empty_index(self):
        idx = RouteSimilarityIndex(dim=4)
        assert idx.search([1.0, 2.0, 3.0, 4.0]) == []

    def test_k_capped_at_size(self, index):
        results = index.search([0.5, 0.5, 15.0, 2000.0], k=10)
        assert len(results) == 3

    def test_wrong_dimension_raises(self, index):
        with pytest.raises(ValueError):
            index.add("bad", [1.0, 2.0])

    def test_distances_sorted_ascending(self, index):
        results = index.search([0.88, 0.87, 18.0, 2000.0], k=3)
        dists = [r["distance"] for r in results]
        assert dists == sorted(dists)
