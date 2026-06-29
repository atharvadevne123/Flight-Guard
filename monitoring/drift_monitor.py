"""
KS-drift detection and distribution monitoring for Flight-Guard.

Detects feature drift in flight prediction inputs using:
  1. Kolmogorov-Smirnov two-sample test (primary)
  2. Evidently DatasetDriftMetric (when available)
  3. Score distribution shift monitoring
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


MONITOR_DIR = Path(__file__).parent / "reports"
MONITOR_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_PATH = MONITOR_DIR / "reference_stats.parquet"

# KS p-value threshold: features with p < 0.05 flagged as drifted
KS_P_THRESHOLD = 0.05

# Prediction score shift threshold that triggers retrain alert
SCORE_SHIFT_THRESHOLD = 0.07


class FlightDriftMonitor:
    """
    Compares current flight batch feature distributions against a reference
    baseline captured at training time.

    Usage:
        monitor = FlightDriftMonitor()
        monitor.set_reference(train_df)

        # Later in production:
        report = monitor.run(new_batch_df)
        if report["drift_detected"]:
            trigger_retrain()
    """

    def __init__(
        self,
        ks_threshold: float = KS_P_THRESHOLD,
        score_shift_threshold: float = SCORE_SHIFT_THRESHOLD,
        min_samples: int = 30,
    ):
        self.ks_threshold = ks_threshold
        self.score_shift_threshold = score_shift_threshold
        self.min_samples = min_samples
        self._reference: Optional[pd.DataFrame] = None
        self._load_reference()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, df: pd.DataFrame) -> None:
        """Store training distribution as drift reference."""
        self._reference = df.copy()
        df.to_parquet(REFERENCE_PATH, index=False)
        logger.info("Flight drift reference saved ({:,} samples).", len(df))

    def run(self, current: pd.DataFrame) -> dict:
        """
        Run KS-drift analysis on current batch vs reference.

        Returns:
            dict with keys:
              - drift_detected (bool)
              - drifted_features (list[str])
              - ks_results (dict: feature → {statistic, p_value, drifted})
              - score_shift (float)
              - missing_ratio (float)
              - run_timestamp (str)
              - batch_size (int)
        """
        if self._reference is None:
            logger.warning("No reference set — storing current batch as reference.")
            self.set_reference(current)
            return {
                "drift_detected": False,
                "reason": "Reference just initialised from this batch.",
                "drifted_features": [],
                "ks_results": {},
                "score_shift": 0.0,
                "missing_ratio": 0.0,
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "batch_size": len(current),
            }

        numeric_cols = [
            c for c in current.select_dtypes(include="number").columns
            if c in self._reference.columns
        ]

        ks_results, drifted_features = self._run_ks_tests(current, numeric_cols)
        score_shift = self._check_score_shift(current)
        missing_ratio = self._check_missing(current)

        # Try Evidently for richer diagnostics; fall back gracefully
        evidently_info = self._run_evidently(current, numeric_cols)

        drift_detected = (
            len(drifted_features) > 0
            or abs(score_shift) > self.score_shift_threshold
        )

        summary = {
            "run_timestamp":   datetime.now(timezone.utc).isoformat(),
            "batch_size":      len(current),
            "drift_detected":  drift_detected,
            "drifted_features": drifted_features,
            "ks_results":      ks_results,
            "score_shift":     round(score_shift, 4),
            "missing_ratio":   round(missing_ratio, 4),
            "evidently":       evidently_info,
            "n_numeric_tested": len(numeric_cols),
        }

        self._save_report(summary)

        if drift_detected:
            logger.warning(
                "FLIGHT DATA DRIFT DETECTED — {} features drifted, score shift {:.4f}.",
                len(drifted_features), score_shift,
            )
        else:
            logger.info("No significant drift detected. Flight data distribution stable.")

        return summary

    # ------------------------------------------------------------------
    # KS tests
    # ------------------------------------------------------------------

    def _run_ks_tests(
        self, current: pd.DataFrame, cols: list[str]
    ) -> tuple[dict, list[str]]:
        ks_results: dict = {}
        drifted: list[str] = []

        for col in cols:
            ref_vals = self._reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < self.min_samples or len(cur_vals) < self.min_samples:
                continue

            stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
            is_drifted = p_val < self.ks_threshold

            ks_results[col] = {
                "statistic": round(float(stat), 6),
                "p_value":   round(float(p_val), 6),
                "drifted":   is_drifted,
            }
            if is_drifted:
                drifted.append(col)

        return ks_results, drifted

    # ------------------------------------------------------------------
    # Evidently integration (optional)
    # ------------------------------------------------------------------

    def _run_evidently(self, current: pd.DataFrame, cols: list[str]) -> dict:
        try:
            from evidently import ColumnMapping
            from evidently.metrics import DatasetDriftMetric, DataDriftTable
            from evidently.report import Report

            ref = self._reference[cols].copy()
            cur = current[cols].copy()

            col_mapping = ColumnMapping()
            if "delay_probability" in current.columns:
                col_mapping.prediction = "delay_probability"

            report = Report(metrics=[
                DatasetDriftMetric(drift_share_threshold=0.20),
                DataDriftTable(num_stattest="ks"),
            ])
            report.run(reference_data=ref, current_data=cur, column_mapping=col_mapping)
            result = report.as_dict()
            metrics = result.get("metrics", [])

            ds_metric = next(
                (m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {}
            )
            return {
                "dataset_drift": ds_metric.get("result", {}).get("dataset_drift", False),
                "share_drifted": ds_metric.get("result", {}).get("share_of_drifted_columns", 0.0),
            }
        except ImportError:
            return {"note": "evidently not installed; using KS fallback only."}
        except Exception as exc:
            logger.warning("Evidently run failed: {}", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Score shift
    # ------------------------------------------------------------------

    def _check_score_shift(self, current: pd.DataFrame) -> float:
        col = "delay_probability"
        if col not in current.columns or col not in self._reference.columns:
            return 0.0
        ref_mean = float(self._reference[col].mean())
        cur_mean = float(current[col].mean())
        shift = cur_mean - ref_mean
        logger.debug(
            "Score shift: {:.4f} (ref={:.4f}, cur={:.4f})", shift, ref_mean, cur_mean
        )
        return shift

    def _check_missing(self, current: pd.DataFrame) -> float:
        total = current.size
        if total == 0:
            return 0.0
        return float(current.isnull().sum().sum() / total)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_report(self, summary: dict) -> None:
        ts = summary["run_timestamp"].replace(":", "-")[:19]
        path = MONITOR_DIR / f"drift_report_{ts}.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger.debug("Drift report saved → {}", path.name)

    def _load_reference(self) -> None:
        if REFERENCE_PATH.exists():
            self._reference = pd.read_parquet(REFERENCE_PATH)
            logger.info(
                "Flight drift reference loaded ({:,} rows).", len(self._reference)
            )
