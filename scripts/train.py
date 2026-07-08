"""CLI script to train the Flight-Guard model.

Usage:
    python scripts/train.py [--samples 2000] [--folds 5]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import generate_synthetic_data  # noqa: E402
from app.model import train_model  # noqa: E402
from app.tracking import log_training_run  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")


def main() -> int:
    """Train the model and log the run; return process exit code."""
    parser = argparse.ArgumentParser(description="Train the Flight-Guard delay model")
    parser.add_argument("--samples", type=int, default=2000, help="Synthetic training rows")
    parser.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    logger.info("Generating %d synthetic samples", args.samples)
    X, y = generate_synthetic_data(n_samples=args.samples)

    logger.info("Training with %d-fold CV", args.folds)
    _, metrics = train_model(X, y, cv_folds=args.folds)

    run_id = log_training_run(
        metrics,
        params={"samples": args.samples, "cv_folds": args.folds},
        tags={"trigger": "cli"},
    )
    logger.info("Run %s complete: %s", run_id, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
