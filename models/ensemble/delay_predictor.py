"""
Flight delay prediction ensemble.
XGBoost + LightGBM + RandomForest soft-voting with calibrated probabilities.
SHAP explanations per prediction, MLflow tracking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Delay tier thresholds (minutes)
THRESHOLD_SEVERE = 0.80  # P(delay) → SEVERE
THRESHOLD_MODERATE = 0.55  # P(delay) → MODERATE
THRESHOLD_MINOR = 0.30  # P(delay) → MINOR
# Below MINOR → ON_TIME

# Expected delay minutes per tier (median BTS statistics)
TIER_EXPECTED_MINUTES = {
    "SEVERE": 95.0,
    "MODERATE": 42.0,
    "MINOR": 18.0,
    "ON_TIME": 0.0,
}


class DelayPredictor:
    """
    Three-model soft-voting ensemble for flight delay prediction.

    Supports:
    - Probability calibration via isotonic regression
    - SHAP TreeExplainer for per-prediction feature attribution
    - MLflow experiment tracking
    - Delay tier classification (ON_TIME / MINOR / MODERATE / SEVERE)
    """

    def __init__(
        self,
        xgb_params: dict | None = None,
        lgb_params: dict | None = None,
        rf_params: dict | None = None,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.feature_names: list[str] = []

        _xgb = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,  # ~25% delay rate in BTS data
            eval_metric="auc",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
        _lgb = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        _rf = dict(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

        self.xgb_model = xgb.XGBClassifier(**(xgb_params or _xgb))
        self.lgb_model = lgb.LGBMClassifier(**(lgb_params or _lgb))
        self.rf_model = RandomForestClassifier(**(rf_params or _rf))

        self.ensemble: CalibratedClassifierCV | None = None
        self.scaler = StandardScaler()
        self._shap_explainer: shap.TreeExplainer | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_X: pd.DataFrame | None = None,
        eval_y: pd.Series | None = None,
        mlflow_run: bool = True,
    ) -> DelayPredictor:
        """Fit the ensemble on labelled flight data.

        Args:
            X: Feature matrix with flight attributes.
            y: Binary label — 1 if flight delayed ≥ 15 minutes, 0 otherwise.
            eval_X: Validation features for held-out metrics.
            eval_y: Validation labels.
            mlflow_run: Whether to log metrics to MLflow.

        Returns:
            self (fitted)
        """
        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.values.astype(float))
        y_arr = y.values

        logger.info(
            "Training DelayPredictor on {:,} samples (delay rate: {:.2%}).",
            len(y_arr),
            y_arr.mean(),
        )

        voter = VotingClassifier(
            estimators=[
                ("xgb", self.xgb_model),
                ("lgb", self.lgb_model),
                ("rf", self.rf_model),
            ],
            voting="soft",
            weights=[0.4, 0.4, 0.2],
        )
        self.ensemble = CalibratedClassifierCV(
            voter,
            method=self.calibration_method,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state),
        )

        if mlflow_run:
            mlflow.set_experiment("flight_guard_delay_prediction")
            with mlflow.start_run(run_name="delay_ensemble"):
                mlflow.log_params(
                    {
                        "calibration": self.calibration_method,
                        "n_train": len(y_arr),
                        "delay_rate": float(y_arr.mean()),
                        "xgb_n_estimators": self.xgb_model.n_estimators,
                        "lgb_n_estimators": self.lgb_model.n_estimators,
                    }
                )
                self.ensemble.fit(X_arr, y_arr)
                self._log_metrics(X_arr, y_arr, eval_X, eval_y)
        else:
            self.ensemble.fit(X_arr, y_arr)

        # Build SHAP explainer on the first XGBoost base estimator
        try:
            base_xgb = self.ensemble.calibrated_classifiers_[0].estimator.named_estimators_["xgb"]
            self._shap_explainer = shap.TreeExplainer(base_xgb)
            logger.success("SHAP TreeExplainer initialised on XGBoost base learner.")
        except Exception as exc:
            logger.warning("Could not build SHAP explainer: {}", exc)

        self._fitted = True
        logger.success("DelayPredictor fitted successfully.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return delay probabilities (shape: [n_samples,])."""
        self._check_fitted()
        return self.ensemble.predict_proba(self.scaler.transform(X.values.astype(float)))[:, 1]

    def predict_with_tier(self, X: pd.DataFrame) -> list[dict]:
        """Return full prediction dicts including tier and expected delay minutes."""
        proba = self.predict(X)
        results = []
        for p in proba:
            tier = self._classify_tier(p)
            results.append(
                {
                    "delay_probability": round(float(p), 6),
                    "delay_tier": tier,
                    "expected_delay_minutes": TIER_EXPECTED_MINUTES[tier],
                }
            )
        return results

    def explain(self, X: pd.DataFrame, max_features: int = 10) -> list[dict]:
        """Compute SHAP values and return top-k feature attributions per row."""
        if self._shap_explainer is None:
            return [{"error": "SHAP explainer not available."}] * len(X)

        X_scaled = self.scaler.transform(X.values.astype(float))
        raw = self._shap_explainer.shap_values(X_scaled)
        # For binary classifiers, shap_values may be a list [neg_class, pos_class]
        sv = raw if not isinstance(raw, list) else raw[1]

        results = []
        for i in range(len(X)):
            top_idx = np.argsort(np.abs(sv[i]))[::-1][:max_features]
            base_val = (
                float(self._shap_explainer.expected_value)
                if not isinstance(self._shap_explainer.expected_value, list)
                else float(self._shap_explainer.expected_value[1])
            )
            results.append(
                {
                    "shap_features": {
                        self.feature_names[j]: round(float(sv[i][j]), 6) for j in top_idx
                    },
                    "base_value": round(base_val, 6),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "delay_predictor.joblib"
        joblib.dump(self, path)
        logger.info("DelayPredictor saved → {}", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> DelayPredictor:
        path = Path(path) if path else ARTIFACT_DIR / "delay_predictor.joblib"
        obj = joblib.load(path)
        logger.info("DelayPredictor loaded ← {}", path)
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_tier(prob: float) -> str:
        if prob >= THRESHOLD_SEVERE:
            return "SEVERE"
        if prob >= THRESHOLD_MODERATE:
            return "MODERATE"
        if prob >= THRESHOLD_MINOR:
            return "MINOR"
        return "ON_TIME"

    def _log_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_X: pd.DataFrame | None,
        eval_y: pd.Series | None,
    ) -> None:
        train_proba = self.ensemble.predict_proba(X_train)[:, 1]
        mlflow.log_metric("train_auc_roc", roc_auc_score(y_train, train_proba))
        mlflow.log_metric("train_avg_precision", average_precision_score(y_train, train_proba))
        mlflow.log_metric("train_brier", brier_score_loss(y_train, train_proba))

        if eval_X is not None and eval_y is not None:
            X_val_scaled = self.scaler.transform(eval_X.values.astype(float))
            val_proba = self.ensemble.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(eval_y, val_proba)
            mlflow.log_metric("val_auc_roc", auc)
            mlflow.log_metric("val_avg_precision", average_precision_score(eval_y, val_proba))
            mlflow.log_metric("val_brier", brier_score_loss(eval_y, val_proba))
            logger.info("Val AUC-ROC: {:.4f}", auc)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call train() before using DelayPredictor for inference.")
