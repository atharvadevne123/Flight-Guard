"""Tests for SQLAlchemy models."""

from __future__ import annotations

from datetime import datetime

from app.database import DriftReport, Flight, PredictionLog, RetrainLog


class TestFlightModel:
    def test_create_flight(self, db_session):
        flight = Flight(
            carrier="AA",
            origin="JFK",
            destination="LAX",
            scheduled_departure=datetime(2026, 7, 8, 8, 0),
            delay_minutes=22.5,
            distance_km=3983.0,
        )
        db_session.add(flight)
        db_session.commit()
        assert flight.id is not None
        assert flight.created_at is not None


class TestPredictionLogModel:
    def test_create_prediction_log(self, db_session):
        log = PredictionLog(
            carrier="DL",
            origin="ATL",
            destination="SEA",
            scheduled_hour=14,
            day_of_week=2,
            month=7,
            delay_probability=0.31,
            predicted_class_name="on_time",
            model_version="1.0.0",
        )
        db_session.add(log)
        db_session.commit()
        assert log.id is not None
        assert log.predicted_class_name == "on_time"

    def test_correlation_id_nullable(self, db_session):
        log = PredictionLog(
            carrier="UA",
            origin="ORD",
            destination="DEN",
            scheduled_hour=9,
            day_of_week=0,
            month=3,
            delay_probability=0.5,
            predicted_class_name="delayed",
        )
        db_session.add(log)
        db_session.commit()
        assert log.correlation_id is None


class TestDriftReportModel:
    def test_create_drift_report(self, db_session):
        report = DriftReport(
            feature_name="delay_probability",
            ks_statistic=0.12,
            p_value=0.34,
            drift_detected=0,
        )
        db_session.add(report)
        db_session.commit()
        assert report.id is not None


class TestRetrainLogModel:
    def test_create_retrain_log(self, db_session):
        log = RetrainLog(trigger="manual", auc_after=0.81, status="success")
        db_session.add(log)
        db_session.commit()
        assert log.id is not None
        assert log.status == "success"
