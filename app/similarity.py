"""Route similarity search with FAISS and brute-force fallback."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore[import-untyped]

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class RouteSimilarityIndex:
    """Nearest-neighbour index over route feature vectors.

    Uses FAISS IndexFlatL2 when available, otherwise a NumPy
    brute-force L2 search with identical results.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: np.ndarray | None = None
        self._labels: list[str] = []
        self._faiss_index = faiss.IndexFlatL2(dim) if _FAISS_AVAILABLE else None

    @property
    def size(self) -> int:
        """Number of indexed routes."""
        return len(self._labels)

    def add(self, label: str, vector: list[float]) -> None:
        """Add a route vector with its label (e.g. 'AA:JFK-LAX')."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self._dim:
            raise ValueError(f"Expected dim {self._dim}, got {vec.shape[1]}")
        self._labels.append(label)
        if self._faiss_index is not None:
            self._faiss_index.add(vec)
        else:
            self._vectors = vec if self._vectors is None else np.vstack([self._vectors, vec])

    def search(self, vector: list[float], k: int = 5) -> list[dict]:
        """Return up to k nearest routes with L2 distances.

        Args:
            vector: Query feature vector of the index dimension.
            k: Maximum neighbours to return.

        Returns:
            List of {'label', 'distance'} dicts sorted nearest-first.
        """
        if self.size == 0:
            return []
        k = min(k, self.size)
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)

        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(query, k)
            dists, idxs = distances[0], indices[0]
        else:
            assert self._vectors is not None
            diffs = self._vectors - query
            all_dists = np.sum(diffs * diffs, axis=1)
            idxs = np.argsort(all_dists)[:k]
            dists = all_dists[idxs]

        return [
            {"label": self._labels[int(i)], "distance": round(float(d), 4)}
            for d, i in zip(dists, idxs)
        ]
