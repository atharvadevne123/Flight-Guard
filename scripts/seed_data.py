"""Seed the database with synthetic flight records for development.

Usage:
    python scripts/seed_data.py [--rows 500]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.database import Flight, SessionLocal, init_db  # noqa: E402
from app.features import generate_synthetic_data  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed")


def main() -> int:
    """Insert synthetic Flight rows; return process exit code."""
    parser = argparse.ArgumentParser(description="Seed development flight data")
    parser.add_argument("--rows", type=int, default=500, help="Rows to insert")
    args = parser.parse_args()

    init_db()
    X, y = generate_synthetic_data(n_samples=args.rows)
    rng = np.random.default_rng(7)
    base = datetime(2026, 1, 1)

    db = SessionLocal()
    try:
        for i in range(len(X)):
            row = X.iloc[i]
            delayed = bool(y.iloc[i])
            delay_minutes = float(rng.uniform(16, 120)) if delayed else float(rng.uniform(0, 14))
            scheduled = base + timedelta(
                days=int(rng.integers(0, 180)), hours=int(row["scheduled_hour"])
            )
            db.add(
                Flight(
                    carrier=row["carrier"],
                    origin=row["origin"],
                    destination=row["destination"],
                    scheduled_departure=scheduled,
                    actual_departure=scheduled + timedelta(minutes=delay_minutes),
                    delay_minutes=delay_minutes,
                    distance_km=float(row["distance_km"]),
                )
            )
        db.commit()
        logger.info("Seeded %d flight rows", len(X))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
