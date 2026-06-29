"""
Train Flight-Guard delay predictor from scratch.
Usage: python scripts/train.py [--data-path PATH] [--model-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble.delay_predictor import DelayPredictor
from pipeline.feature_engineering import FlightFeatureEngineer
from loguru import logger


def generate_synthetic_data(n: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    carriers  = ["AA", "DL", "UA", "WN", "B6", "AS", "NK"]
    airports  = ["ORD", "LAX", "JFK", "ATL", "DFW", "DEN", "SFO", "SEA", "MIA", "BOS"]
    weather   = ["clear", "rain", "snow", "fog", "wind"]
    aircraft  = ["B737", "B757", "B777", "A320", "A321", "E175"]

    df = pd.DataFrame({
        "flight_id":               [f"FLT-{i:06d}" for i in range(n)],
        "carrier_code":            rng.choice(carriers, n),
        "origin":                  rng.choice(airports, n),
        "destination":             rng.choice(airports, n),
        "scheduled_departure":     ["2024-06-15T08:30:00"] * n,
        "aircraft_type":           rng.choice(aircraft, n),
        "distance_miles":          rng.uniform(150, 2800, n),
        "departure_hour":          rng.integers(0, 24, n),
        "day_of_week":             rng.integers(0, 7, n),
        "month":                   rng.integers(1, 13, n),
        "is_holiday":              rng.choice([True, False], n, p=[0.1, 0.9]),
        "prior_leg_delay_minutes": rng.uniform(0, 120, n) * rng.choice([0, 1], n, p=[0.7, 0.3]),
        "weather_condition":       rng.choice(weather, n),
    })

    # Synthetic label: delayed if bad weather + peak hour + high prior delay
    df["is_delayed"] = (
        (df["weather_condition"].isin(["snow", "fog"])) |
        (df["prior_leg_delay_minutes"] > 30) |
        (df["departure_hour"].isin([7, 8, 17, 18, 19]))
    ).astype(int)
    df.loc[rng.random(n) < 0.1, "is_delayed"] ^= 1  # add noise
    return df


def main():
    parser = argparse.ArgumentParser(description="Train Flight-Guard delay predictor")
    parser.add_argument("--data-path",    type=str, default=None)
    parser.add_argument("--model-dir",    type=str, default=str(ROOT / "models"))
    parser.add_argument("--n-synthetic",  type=int, default=10000)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.data_path:
        path = Path(args.data_path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        logger.info("Loaded {:,} rows from {}.", len(df), path)
    else:
        logger.info("Generating {:,} synthetic flights.", args.n_synthetic)
        df = generate_synthetic_data(args.n_synthetic)

    label_col = "is_delayed"
    if label_col not in df.columns:
        df[label_col] = 0

    fe = FlightFeatureEngineer()
    df_feat = fe.fit_transform(df)

    exclude = {label_col, "flight_id", "carrier_code", "origin", "destination",
               "scheduled_departure", "aircraft_type", "weather_condition"}
    feat_cols = [c for c in df_feat.select_dtypes(include="number").columns if c not in exclude]
    X = df_feat[feat_cols].fillna(0)
    y = df_feat[label_col].astype(int)

    logger.info("Training delay predictor on {:,} flights with {} features.", len(X), len(feat_cols))

    model   = DelayPredictor()
    metrics = model.train(X, y)

    model.save(model_dir / "delay_predictor.joblib")
    joblib.dump(fe, model_dir / "feature_engineer.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(feat_cols))

    logger.success("Training complete. Saved to {}. Metrics: {}", model_dir, metrics)


if __name__ == "__main__":
    main()
