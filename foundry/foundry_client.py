"""
Palantir Foundry integration client for Flight-Guard.

Provides dataset read/write, model registration, and prediction logging
against the Foundry REST Catalog and Datasets APIs.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger


class FoundryClient:
    """
    Thin wrapper around the Palantir Foundry REST APIs.

    Supports:
      - Dataset read / write (via Foundry Catalog + Datasets v2 APIs)
      - Model artifact registration in Foundry Model Health
      - Prediction batch logging for lineage and monitoring

    Authentication uses a bearer token (FOUNDRY_TOKEN env var).

    Example:
        client = FoundryClient(
            host="https://your-tenant.palantirfoundry.com",
            token=os.environ["FOUNDRY_TOKEN"],
        )
        client.upload_dataset(df, dataset_rid="ri.foundry.main.dataset.xxxx")
    """

    def __init__(self, host: str, token: str, timeout: int = 60):
        self.host    = host.rstrip("/")
        self.token   = token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "User-Agent":    "FlightGuard/1.0",
        })

    # ------------------------------------------------------------------
    # Dataset operations
    # ------------------------------------------------------------------

    def upload_dataset(
        self,
        df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
    ) -> dict:
        """Upload a DataFrame to a Foundry dataset branch.

        Uses the Foundry Datasets v2 API:
          1. Start transaction
          2. Upload Parquet file
          3. Commit transaction

        Args:
            df: DataFrame to upload.
            dataset_rid: Resource Identifier of the target dataset.
            branch: Foundry branch name (default "master").

        Returns:
            dict with transaction_rid and committed_at timestamp.
        """
        logger.info(
            "Uploading {:,} rows to Foundry dataset {} (branch={}).",
            len(df), dataset_rid, branch,
        )

        # 1. Open transaction
        txn = self._open_transaction(dataset_rid, branch)
        txn_rid = txn["rid"]

        # 2. Serialize to Parquet in-memory and upload
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        parquet_bytes = buf.read()

        upload_url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
            f"/transactions/{txn_rid}/files:upload"
        )
        resp = self._session.post(
            upload_url,
            headers={"Content-Type": "application/octet-stream"},
            data=parquet_bytes,
            params={"logicalPath": "data.parquet"},
            timeout=self.timeout,
        )
        self._raise_for_status(resp, "upload_file")

        # 3. Commit transaction
        committed = self._commit_transaction(dataset_rid, txn_rid)
        logger.success(
            "Dataset {} updated (txn={}).", dataset_rid, txn_rid
        )
        return {
            "transaction_rid": txn_rid,
            "committed_at": committed.get("closedTime", datetime.now(timezone.utc).isoformat()),
        }

    def read_dataset(
        self,
        dataset_rid: str,
        branch: str = "master",
        row_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Read a Foundry dataset branch into a DataFrame.

        Uses the Foundry Datasets v2 /rows export endpoint.

        Args:
            dataset_rid: Resource Identifier of the source dataset.
            branch: Foundry branch name.
            row_limit: Optional maximum rows to fetch.

        Returns:
            pd.DataFrame with dataset contents.
        """
        logger.info("Reading dataset {} (branch={}).", dataset_rid, branch)
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
            f"/branches/{branch}/view/tables/foundry-table:export"
        )
        params = {"format": "ARROW_V2"}
        if row_limit is not None:
            params["rowCount"] = row_limit

        resp = self._session.get(url, params=params, timeout=self.timeout)
        self._raise_for_status(resp, "read_dataset")

        try:
            import pyarrow as pa
            reader = pa.ipc.open_stream(io.BytesIO(resp.content))
            df = reader.read_pandas()
        except ImportError:
            # Fall back to JSON rows endpoint
            df = pd.DataFrame(resp.json().get("rows", []))

        logger.success("Read {:,} rows from {}.", len(df), dataset_rid)
        return df

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def register_model(self, model_metadata: dict) -> dict:
        """Register a trained model artifact in Foundry Model Health.

        Args:
            model_metadata: Dict containing at minimum:
                {
                    "model_name": str,
                    "version": str,
                    "artifact_path": str,
                    "metrics": {"val_auc_roc": 0.87, ...},
                    "trained_at": ISO8601 str,
                }

        Returns:
            Foundry model registration response dict.
        """
        logger.info(
            "Registering model '{}' v{} in Foundry.",
            model_metadata.get("model_name"), model_metadata.get("version"),
        )
        url = f"{self.host}/model-health/api/registry/models"
        payload = {
            "modelName":   model_metadata.get("model_name", "flight_guard_delay_predictor"),
            "version":     model_metadata.get("version", "1.0.0"),
            "artifactPath": model_metadata.get("artifact_path", ""),
            "metrics":     model_metadata.get("metrics", {}),
            "trainedAt":   model_metadata.get(
                "trained_at", datetime.now(timezone.utc).isoformat()
            ),
            "framework":   "sklearn/xgboost/lightgbm",
            "description": "Flight delay prediction ensemble (XGB+LGB+RF).",
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        self._raise_for_status(resp, "register_model")
        result = resp.json()
        logger.success("Model registered: {}.", result.get("modelRid", "unknown"))
        return result

    # ------------------------------------------------------------------
    # Prediction logging
    # ------------------------------------------------------------------

    def log_predictions(
        self,
        predictions_df: pd.DataFrame,
        dataset_rid: Optional[str] = None,
        branch: str = "master",
    ) -> dict:
        """Append prediction rows to the Foundry predictions dataset.

        Adds a logged_at timestamp column before uploading.

        Args:
            predictions_df: DataFrame of predictions (must include flight_id column).
            dataset_rid: Target predictions dataset RID. If None, method is a no-op.
            branch: Branch to write to.

        Returns:
            Upload result dict or {"skipped": True} if dataset_rid is None.
        """
        if dataset_rid is None:
            logger.warning("predictions dataset_rid not set — skipping log_predictions.")
            return {"skipped": True}

        df = predictions_df.copy()
        df["logged_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Logging {:,} predictions to {}.", len(df), dataset_rid)
        return self.upload_dataset(df, dataset_rid, branch=branch)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_transaction(self, dataset_rid: str, branch: str) -> dict:
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}/transactions"
        )
        resp = self._session.post(
            url,
            json={"branch": branch, "type": "SNAPSHOT"},
            timeout=self.timeout,
        )
        self._raise_for_status(resp, "open_transaction")
        return resp.json()

    def _commit_transaction(self, dataset_rid: str, txn_rid: str) -> dict:
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
            f"/transactions/{txn_rid}/commit"
        )
        resp = self._session.post(url, json={}, timeout=self.timeout)
        self._raise_for_status(resp, "commit_transaction")
        return resp.json()

    @staticmethod
    def _raise_for_status(resp: requests.Response, operation: str) -> None:
        if not resp.ok:
            logger.error(
                "Foundry API error during '{}': HTTP {} — {}",
                operation, resp.status_code, resp.text[:300],
            )
            resp.raise_for_status()
