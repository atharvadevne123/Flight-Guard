"""ML model training, persistence, and prediction."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.features import build_feature_pipeline, generate_synthetic_data

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "metrics.json"))
MODEL_VERSION = "1.0.0"

DELAY_CLASSES = {
    0: "on_time",
    1: "delayed",
}


def build_model() -> VotingClassifier:
    """Build XGBoost + LightGBM + RandomForest soft-voting ensemble."""
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    lgbm = LGBMClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    return VotingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("rf", rf)],
        voting="soft",
        weights=[2, 2, 1],
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train ensemble pipeline with 5-fold CV and return metrics."""
    feature_pipe = build_feature_pipeline()
    X_transformed = feature_pipe.fit_transform(X)

    model = build_model()

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_transformed, y, cv=skf, scoring="roc_auc")
    cv_acc = cross_val_score(model, X_transformed, y, cv=skf, scoring="accuracy")

    model.fit(X_transformed, y)

    metrics: dict[str, Any] = {
        "auc_mean": round(float(cv_auc.mean()), 4),
        "auc_std": round(float(cv_auc.std()), 4),
        "accuracy_mean": round(float(cv_acc.mean()), 4),
        "accuracy_std": round(float(cv_acc.std()), 4),
        "n_features": X_transformed.shape[1],
        "n_samples": len(y),
        "delay_rate": round(float(y.mean()), 4),
        "model_version": MODEL_VERSION,
    }

    # Wrap feature pipeline + model into single Pipeline for persistence
    full_pipeline = Pipeline(
        [
            ("features", feature_pipe),
            ("classifier", model),
        ]
    )
    # Re-fit full pipeline
    full_pipeline.fit(X, y)

    joblib.dump(full_pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    logger.info(
        "Model trained: AUC=%.4f±%.4f, ACC=%.4f, samples=%d",
        metrics["auc_mean"],
        metrics["auc_std"],
        metrics["accuracy_mean"],
        len(y),
    )
    return full_pipeline, metrics


def load_model() -> Pipeline:
    """Load the trained model pipeline from disk."""
    if not MODEL_PATH.exists():
        logger.info("No model found at %s — training on synthetic data", MODEL_PATH)
        X, y = generate_synthetic_data(n_samples=2000)
        pipeline, _ = train_model(X, y)
        return pipeline
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict[str, Any]:
    """Load model metrics from disk."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {"status": "no_metrics", "model_version": MODEL_VERSION}


def predict(pipeline: Pipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Run prediction and return delay probability and class."""
    prob = pipeline.predict_proba(X)[0]
    delay_prob = float(prob[1])
    delay_class = DELAY_CLASSES[int(delay_prob >= 0.5)]

    # Risk tier
    if delay_prob >= 0.75:
        risk_tier = "high"
    elif delay_prob >= 0.45:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    return {
        "delay_probability": round(delay_prob, 4),
        "on_time_probability": round(float(prob[0]), 4),
        "predicted_class": delay_class,
        "risk_tier": risk_tier,
        "model_version": MODEL_VERSION,
    }
