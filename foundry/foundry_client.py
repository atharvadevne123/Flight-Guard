"""
Palantir Foundry integration client for Flight-Guard.

Provides dataset read/write, model registration, prediction logging,
transaction management, streaming, lineage, and pipeline orchestration
against the Foundry REST Catalog and Datasets APIs.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401  (used for parquet round-trips)
except ImportError:  # pragma: no cover - pyarrow is optional at runtime
    pa = None
    pq = None


class FoundryError(Exception):
    """Raised when a Foundry API call fails."""


class FoundrySchemaError(FoundryError):
    """Raised when a DataFrame does not conform to the expected Foundry schema."""


class FoundryClient:
    """
    Thin wrapper around the Palantir Foundry REST APIs.

    Supports:
      - Dataset read / write (via Foundry Catalog + Datasets v2 APIs)
      - Model artifact registration in Foundry Model Health
      - Prediction batch logging for lineage and monitoring
      - Transactions, streaming reads, lineage, schema enforcement,
        pipeline publishing, and async uploads

    Authentication uses a bearer token (FOUNDRY_TOKEN env var).

    Example:
        client = FoundryClient(
            host="https://your-tenant.palantirfoundry.com",
            token=os.environ["FOUNDRY_TOKEN"],
        )
        client.upload_dataset(df, dataset_rid="ri.foundry.main.dataset.xxxx")
    """

    def __init__(self, host: str, token: str, timeout: int = 60, max_retries: int = 3):
        self.host    = host.rstrip("/")
        self.token   = token
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "User-Agent":    "FlightGuard/1.0",
        })
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST", "PUT", "DELETE"}),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FoundryClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._session.close()

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
        txn = self.create_transaction(dataset_rid, branch)
        txn_rid = txn["rid"]

        try:
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
        except Exception as exc:
            logger.error("Upload failed, aborting transaction {}: {}", txn_rid, exc)
            try:
                self.abort_transaction(dataset_rid, txn_rid)
            except FoundryError:
                logger.warning("Could not abort transaction {}.", txn_rid)
            raise

        # 3. Commit transaction
        committed = self.commit_transaction(dataset_rid, txn_rid)
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
        params: dict[str, Any] = {"format": "ARROW_V2"}
        if row_limit is not None:
            params["rowCount"] = row_limit

        resp = self._session.get(url, params=params, timeout=self.timeout)
        self._raise_for_status(resp, "read_dataset")

        if pa is not None:
            try:
                reader = pa.ipc.open_stream(io.BytesIO(resp.content))
                df = reader.read_pandas()
            except Exception as exc:
                logger.warning("Arrow decode failed ({}); falling back to JSON rows.", exc)
                df = pd.DataFrame(resp.json().get("rows", []))
        else:
            # Fall back to JSON rows endpoint
            df = pd.DataFrame(resp.json().get("rows", []))

        logger.success("Read {:,} rows from {}.", len(df), dataset_rid)
        return df

    def create_dataset(self, name: str, parent_folder_rid: str) -> dict:
        """Create a new Foundry dataset under the given folder.

        Args:
            name: Display name for the new dataset.
            parent_folder_rid: RID of the parent Compass folder.

        Returns:
            dict with the new dataset's rid and metadata.
        """
        logger.info("Creating Foundry dataset '{}'.", name)
        url = f"{self.host}/foundry-catalog/api/catalog/datasets"
        resp = self._session.post(
            url,
            json={"name": name, "parentFolderRid": parent_folder_rid},
            timeout=self.timeout,
        )
        self._raise_for_status(resp, "create_dataset")
        return resp.json()

    def list_branches(self, dataset_rid: str) -> list[dict]:
        """List all branches of a Foundry dataset.

        Args:
            dataset_rid: Resource Identifier of the dataset.

        Returns:
            List of branch metadata dicts.
        """
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}/branches"
        )
        resp = self._session.get(url, timeout=self.timeout)
        self._raise_for_status(resp, "list_branches")
        return resp.json().get("branches", [])

    def get_schema(self, dataset_rid: str, branch: str = "master") -> dict:
        """Fetch the Foundry schema for a dataset branch.

        Args:
            dataset_rid: Resource Identifier of the dataset.
            branch: Foundry branch name.

        Returns:
            Schema dict with a "fieldSchemaList" of column definitions.
        """
        url = (
            f"{self.host}/foundry-metadata/api/schemas/datasets/{dataset_rid}"
            f"/branches/{branch}"
        )
        resp = self._session.get(url, timeout=self.timeout)
        self._raise_for_status(resp, "get_schema")
        return resp.json().get("schema", resp.json())

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_columns: Optional[list[str]] = None,
        max_null_fraction: float = 0.5,
    ) -> dict:
        """Run lightweight data-quality checks on a DataFrame before upload.

        Args:
            df: DataFrame to validate.
            required_columns: Columns that must be present.
            max_null_fraction: Maximum tolerated fraction of nulls per column.

        Returns:
            dict with "passed" bool and a list of "issues".
        """
        issues: list[str] = []
        if df.empty:
            issues.append("DataFrame is empty.")
        for col in (required_columns or []):
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        if len(df) > 0:
            null_fractions = df.isnull().mean()
            for col, frac in null_fractions.items():
                if frac > max_null_fraction:
                    issues.append(
                        f"Column '{col}' has {frac:.0%} nulls (max {max_null_fraction:.0%})."
                    )
        dup_count = int(df.duplicated().sum())
        if dup_count:
            issues.append(f"{dup_count} duplicate rows found.")
        passed = len(issues) == 0
        if passed:
            logger.success("Data-quality checks passed ({} rows).", len(df))
        else:
            logger.warning("Data-quality issues: {}", issues)
        return {"passed": passed, "issues": issues, "row_count": len(df)}

    def publish_to_pipeline(self, pipeline_rid: str, params: Optional[dict] = None) -> dict:
        """Trigger a Foundry pipeline (build) run.

        Args:
            pipeline_rid: RID of the pipeline / build target.
            params: Optional build parameters.

        Returns:
            dict with the triggered build's rid and status.
        """
        logger.info("Triggering Foundry pipeline {}.", pipeline_rid)
        url = f"{self.host}/foundry-build/api/builds"
        resp = self._session.post(
            url,
            json={"pipelineRid": pipeline_rid, "parameters": params or {}},
            timeout=self.timeout,
        )
        self._raise_for_status(resp, "publish_to_pipeline")
        return resp.json()

    def get_build_status(self, build_rid: str) -> dict:
        """Fetch the status of a Foundry build.

        Args:
            build_rid: RID of the build.

        Returns:
            dict with "status" (e.g. RUNNING, SUCCEEDED, FAILED) and metadata.
        """
        url = f"{self.host}/foundry-build/api/builds/{build_rid}"
        resp = self._session.get(url, timeout=self.timeout)
        self._raise_for_status(resp, "get_build_status")
        return resp.json()

    def batch_upload(
        self,
        dfs: list[pd.DataFrame],
        dataset_rid: str,
        branch: str = "master",
        max_workers: int = 4,
    ) -> list[dict]:
        """Upload multiple DataFrames concurrently to the same dataset.

        Args:
            dfs: DataFrames to upload (each becomes its own transaction).
            dataset_rid: Target dataset RID.
            branch: Branch to write to.
            max_workers: Thread-pool size.

        Returns:
            List of upload result dicts, in input order.
        """
        logger.info("Batch-uploading {} frames to {}.", len(dfs), dataset_rid)
        results: list[Optional[dict]] = [None] * len(dfs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self.upload_dataset, df, dataset_rid, branch): i
                for i, df in enumerate(dfs)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def create_transaction(
        self,
        dataset_rid: str,
        branch: str = "master",
        txn_type: str = "SNAPSHOT",
    ) -> dict:
        """Open a new transaction on a dataset branch.

        Args:
            dataset_rid: Target dataset RID.
            branch: Branch to open the transaction on.
            txn_type: SNAPSHOT, APPEND, UPDATE, or DELETE.

        Returns:
            dict with the transaction "rid".
        """
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}/transactions"
        )
        resp = self._session.post(
            url,
            json={"branch": branch, "type": txn_type},
            timeout=self.timeout,
        )
        self._raise_for_status(resp, "create_transaction")
        return resp.json()

    def commit_transaction(self, dataset_rid: str, txn_rid: str) -> dict:
        """Commit an open transaction.

        Args:
            dataset_rid: Dataset RID the transaction belongs to.
            txn_rid: Transaction RID to commit.

        Returns:
            Commit response dict (includes "closedTime").
        """
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
            f"/transactions/{txn_rid}/commit"
        )
        resp = self._session.post(url, json={}, timeout=self.timeout)
        self._raise_for_status(resp, "commit_transaction")
        return resp.json()

    def abort_transaction(self, dataset_rid: str, txn_rid: str) -> dict:
        """Abort an open transaction, discarding staged files.

        Args:
            dataset_rid: Dataset RID the transaction belongs to.
            txn_rid: Transaction RID to abort.

        Returns:
            Abort response dict.
        """
        url = (
            f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
            f"/transactions/{txn_rid}/abort"
        )
        resp = self._session.post(url, json={}, timeout=self.timeout)
        self._raise_for_status(resp, "abort_transaction")
        return resp.json()

    # ------------------------------------------------------------------
    # Streaming / lineage / monitoring
    # ------------------------------------------------------------------

    def stream_records(
        self,
        dataset_rid: str,
        branch: str = "master",
        chunk_size: int = 10_000,
    ) -> Iterator[pd.DataFrame]:
        """Stream a dataset in chunks instead of loading it all at once.

        Args:
            dataset_rid: Source dataset RID.
            branch: Branch to read.
            chunk_size: Rows per yielded chunk.

        Yields:
            pd.DataFrame chunks of up to chunk_size rows.
        """
        offset = 0
        while True:
            url = (
                f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
                f"/branches/{branch}/rows"
            )
            resp = self._session.get(
                url,
                params={"offset": offset, "limit": chunk_size},
                timeout=self.timeout,
            )
            self._raise_for_status(resp, "stream_records")
            rows = resp.json().get("rows", [])
            if not rows:
                break
            yield pd.DataFrame(rows)
            if len(rows) < chunk_size:
                break
            offset += chunk_size

    def get_lineage(self, dataset_rid: str, depth: int = 3) -> dict:
        """Fetch upstream/downstream lineage for a dataset.

        Args:
            dataset_rid: Dataset RID to trace.
            depth: Maximum lineage graph depth.

        Returns:
            dict with "upstream" and "downstream" node lists.
        """
        url = f"{self.host}/foundry-lineage/api/lineage/{dataset_rid}"
        resp = self._session.get(url, params={"depth": depth}, timeout=self.timeout)
        self._raise_for_status(resp, "get_lineage")
        return resp.json()

    def subscribe_to_dataset(
        self,
        dataset_rid: str,
        callback: Callable[[dict], None],
        poll_interval: float = 30.0,
        max_polls: Optional[int] = None,
    ) -> None:
        """Poll a dataset for new committed transactions and invoke a callback.

        Args:
            dataset_rid: Dataset RID to watch.
            callback: Called with the latest transaction dict on change.
            poll_interval: Seconds between polls.
            max_polls: Stop after this many polls (None = poll forever).
        """
        last_txn: Optional[str] = None
        polls = 0
        while max_polls is None or polls < max_polls:
            url = (
                f"{self.host}/foundry-catalog/api/catalog/datasets/{dataset_rid}"
                f"/transactions/latest"
            )
            resp = self._session.get(url, timeout=self.timeout)
            self._raise_for_status(resp, "subscribe_to_dataset")
            txn = resp.json()
            rid = txn.get("rid")
            if rid and rid != last_txn:
                if last_txn is not None:
                    callback(txn)
                last_txn = rid
            polls += 1
            if max_polls is None or polls < max_polls:
                time.sleep(poll_interval)

    def enforce_schema(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """Validate and coerce a DataFrame against a Foundry schema.

        Args:
            df: DataFrame to check.
            schema: Foundry schema dict with "fieldSchemaList".

        Returns:
            DataFrame with columns ordered per the schema.

        Raises:
            FoundrySchemaError: If required columns are missing.
        """
        fields = schema.get("fieldSchemaList", [])
        expected = [f["name"] for f in fields]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise FoundrySchemaError(
                f"DataFrame missing required columns: {missing}"
            )
        extra = [c for c in df.columns if c not in expected]
        if extra:
            logger.warning("Dropping columns not in schema: {}", extra)
        return df[expected]

    def export_snapshot(
        self,
        dataset_rid: str,
        output_path: str,
        branch: str = "master",
        fmt: str = "parquet",
    ) -> str:
        """Export a dataset branch to a local snapshot file.

        Args:
            dataset_rid: Source dataset RID.
            output_path: Local file path to write.
            branch: Branch to export.
            fmt: "parquet" or "csv".

        Returns:
            The output_path written.
        """
        df = self.read_dataset(dataset_rid, branch=branch)
        if fmt == "parquet":
            df.to_parquet(output_path, index=False)
        elif fmt == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise FoundryError(f"Unsupported export format: {fmt}")
        logger.success("Snapshot of {} written to {}.", dataset_rid, output_path)
        return output_path

    def push_metrics(self, model_rid: str, metrics: dict) -> dict:
        """Push evaluation/monitoring metrics to Foundry Model Health.

        Args:
            model_rid: RID of the registered model.
            metrics: Metric name -> value mapping.

        Returns:
            API response dict.
        """
        url = f"{self.host}/model-health/api/registry/models/{model_rid}/metrics"
        payload = {
            "metrics": metrics,
            "recordedAt": datetime.now(timezone.utc).isoformat(),
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        self._raise_for_status(resp, "push_metrics")
        return resp.json()

    def health_check(self) -> dict:
        """Check connectivity and authentication against the Foundry host.

        Returns:
            dict with "healthy" bool and HTTP status code.
        """
        url = f"{self.host}/foundry-catalog/api/status"
        try:
            resp = self._session.get(url, timeout=self.timeout)
            healthy = resp.ok
            status = resp.status_code
        except requests.RequestException as exc:
            logger.error("Foundry health check failed: {}", exc)
            return {"healthy": False, "status_code": None, "error": str(exc)}
        return {"healthy": healthy, "status_code": status}

    def async_upload_dataset(
        self,
        df: pd.DataFrame,
        dataset_rid: str,
        branch: str = "master",
    ) -> "concurrent.futures.Future[dict]":
        """Upload a DataFrame in a background thread.

        Args:
            df: DataFrame to upload.
            dataset_rid: Target dataset RID.
            branch: Branch to write to.

        Returns:
            A Future resolving to the upload result dict.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.upload_dataset, df, dataset_rid, branch)
        executor.shutdown(wait=False)
        return future

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
        # Retained for backward compatibility; delegates to create_transaction.
        return self.create_transaction(dataset_rid, branch)

    def _commit_transaction(self, dataset_rid: str, txn_rid: str) -> dict:
        # Retained for backward compatibility; delegates to commit_transaction.
        return self.commit_transaction(dataset_rid, txn_rid)

    @staticmethod
    def _raise_for_status(resp: requests.Response, operation: str) -> None:
        if not resp.ok:
            logger.error(
                "Foundry API error during '{}': HTTP {} — {}",
                operation, resp.status_code, resp.text[:300],
            )
            raise FoundryError(
                f"Foundry API error during '{operation}': "
                f"HTTP {resp.status_code} — {resp.text[:300]}"
            )
