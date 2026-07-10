"""Smoke tests for the Flight-Guard Foundry client (no network required)."""

from __future__ import annotations

import inspect
from unittest import mock

import pandas as pd
import pytest

from foundry.foundry_client import FoundryClient, FoundryError, FoundrySchemaError

EXPECTED_METHODS = [
    "upload_dataset",
    "read_dataset",
    "register_model",
    "log_predictions",
    "create_dataset",
    "list_branches",
    "get_schema",
    "validate_data_quality",
    "publish_to_pipeline",
    "get_build_status",
    "batch_upload",
    "create_transaction",
    "commit_transaction",
    "abort_transaction",
    "stream_records",
    "get_lineage",
    "subscribe_to_dataset",
    "enforce_schema",
    "export_snapshot",
    "push_metrics",
    "health_check",
    "__enter__",
    "__exit__",
    "async_upload_dataset",
]


@pytest.fixture
def client() -> FoundryClient:
    return FoundryClient(host="https://tenant.palantirfoundry.com/", token="test-token")


def _mock_response(json_data=None, ok=True, status_code=200, content=b""):
    resp = mock.Mock()
    resp.ok = ok
    resp.status_code = status_code
    resp.content = content
    resp.text = ""
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def test_all_methods_present(client):
    for name in EXPECTED_METHODS:
        assert hasattr(client, name), f"Missing method: {name}"
        assert callable(getattr(client, name))


def test_init_strips_trailing_slash_and_sets_auth(client):
    assert client.host == "https://tenant.palantirfoundry.com"
    assert client._session.headers["Authorization"] == "Bearer test-token"


def test_retry_adapter_mounted(client):
    adapter = client._session.get_adapter("https://tenant.palantirfoundry.com")
    assert adapter.max_retries.total >= 1


def test_context_manager_closes_session():
    with FoundryClient(host="https://h", token="t") as c:
        assert isinstance(c, FoundryClient)
    # session.close() is idempotent; just verify __exit__ ran without error


def test_raise_for_status_raises_foundry_error(client):
    resp = _mock_response(ok=False, status_code=500)
    with pytest.raises(FoundryError):
        client._raise_for_status(resp, "unit_test")


def test_upload_dataset_commits_transaction(client):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with (
        mock.patch.object(client, "create_transaction", return_value={"rid": "txn-1"}),
        mock.patch.object(client._session, "post", return_value=_mock_response()) as post,
        mock.patch.object(
            client, "commit_transaction", return_value={"closedTime": "2026-01-01T00:00:00Z"}
        ) as commit,
    ):
        result = client.upload_dataset(df, "ri.foundry.main.dataset.abc")
    assert result["transaction_rid"] == "txn-1"
    assert result["committed_at"] == "2026-01-01T00:00:00Z"
    commit.assert_called_once_with("ri.foundry.main.dataset.abc", "txn-1")
    assert post.called


def test_upload_dataset_aborts_on_failure(client):
    df = pd.DataFrame({"a": [1]})
    bad = _mock_response(ok=False, status_code=503)
    with (
        mock.patch.object(client, "create_transaction", return_value={"rid": "txn-2"}),
        mock.patch.object(client._session, "post", return_value=bad),
        mock.patch.object(client, "abort_transaction", return_value={}) as abort,
    ):
        with pytest.raises(FoundryError):
            client.upload_dataset(df, "ri.x")
    abort.assert_called_once_with("ri.x", "txn-2")


def test_validate_data_quality_flags_issues(client):
    df = pd.DataFrame({"a": [1, 1, None, None], "b": [1, 1, 2, 3]})
    report = client.validate_data_quality(df, required_columns=["a", "missing_col"])
    assert report["passed"] is False
    assert any("missing_col" in i for i in report["issues"])


def test_validate_data_quality_passes_clean_frame(client):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    report = client.validate_data_quality(df, required_columns=["a", "b"])
    assert report["passed"] is True
    assert report["issues"] == []


def test_enforce_schema_raises_on_missing_column(client):
    df = pd.DataFrame({"a": [1]})
    schema = {"fieldSchemaList": [{"name": "a"}, {"name": "b"}]}
    with pytest.raises(FoundrySchemaError):
        client.enforce_schema(df, schema)


def test_enforce_schema_orders_and_drops_extras(client):
    df = pd.DataFrame({"extra": [0], "b": [2], "a": [1]})
    schema = {"fieldSchemaList": [{"name": "a"}, {"name": "b"}]}
    out = client.enforce_schema(df, schema)
    assert list(out.columns) == ["a", "b"]


def test_log_predictions_skips_without_rid(client):
    df = pd.DataFrame({"flight_id": [1]})
    assert client.log_predictions(df, dataset_rid=None) == {"skipped": True}


def test_stream_records_paginates(client):
    pages = [
        _mock_response({"rows": [{"a": i} for i in range(2)]}),
        _mock_response({"rows": [{"a": 9}]}),
    ]
    with mock.patch.object(client._session, "get", side_effect=pages):
        chunks = list(client.stream_records("ri.x", chunk_size=2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2 and len(chunks[1]) == 1


def test_health_check_handles_connection_error(client):
    import requests as _requests

    with mock.patch.object(client._session, "get", side_effect=_requests.ConnectionError("down")):
        result = client.health_check()
    assert result["healthy"] is False


def test_async_upload_returns_future(client):
    df = pd.DataFrame({"a": [1]})
    with mock.patch.object(client, "upload_dataset", return_value={"transaction_rid": "t"}):
        fut = client.async_upload_dataset(df, "ri.x")
        assert fut.result(timeout=10) == {"transaction_rid": "t"}


def test_batch_upload_preserves_order(client):
    frames = [pd.DataFrame({"a": [i]}) for i in range(3)]
    with mock.patch.object(
        client,
        "upload_dataset",
        side_effect=lambda df, rid, branch: {"n": int(df["a"].iloc[0])},
    ):
        results = client.batch_upload(frames, "ri.x")
    assert [r["n"] for r in results] == [0, 1, 2]


def test_signatures_have_annotations():
    for name in EXPECTED_METHODS:
        if name.startswith("__"):
            continue
        fn = getattr(FoundryClient, name)
        sig = inspect.signature(fn)
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"{name} missing return annotation"
        )
