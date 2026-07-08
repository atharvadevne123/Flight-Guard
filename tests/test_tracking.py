"""Tests for experiment tracking fallback."""

from __future__ import annotations

import json

import pytest

import app.tracking as tracking


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    log_path = tmp_path / "experiments.jsonl"
    monkeypatch.setattr(tracking, "FALLBACK_LOG", log_path)
    monkeypatch.setattr(tracking, "_MLFLOW_AVAILABLE", False)
    return log_path


class TestLogTrainingRun:
    def test_returns_run_id(self):
        run_id = tracking.log_training_run({"auc_mean": 0.82})
        assert run_id.startswith("local-")

    def test_writes_jsonl_record(self, isolated_log):
        tracking.log_training_run({"auc_mean": 0.82}, params={"cv": 5})
        lines = isolated_log.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["metrics"]["auc_mean"] == 0.82
        assert record["params"]["cv"] == 5

    def test_appends_multiple_runs(self, isolated_log):
        tracking.log_training_run({"auc_mean": 0.80})
        tracking.log_training_run({"auc_mean": 0.85})
        assert len(isolated_log.read_text().strip().splitlines()) == 2

    def test_tags_recorded(self, isolated_log):
        tracking.log_training_run({"auc_mean": 0.8}, tags={"trigger": "dag"})
        record = json.loads(isolated_log.read_text().strip())
        assert record["tags"]["trigger"] == "dag"


class TestLoadRunHistory:
    def test_empty_history(self):
        assert tracking.load_run_history() == []

    def test_returns_recent_runs(self):
        for i in range(5):
            tracking.log_training_run({"auc_mean": 0.8 + i * 0.01})
        history = tracking.load_run_history(limit=3)
        assert len(history) == 3
        assert history[-1]["metrics"]["auc_mean"] == pytest.approx(0.84)
