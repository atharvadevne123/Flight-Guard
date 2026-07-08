"""Initial schema: flights, prediction_logs, drift_reports, retrain_logs.

Revision ID: 001
Revises:
Create Date: 2026-07-08
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "flights",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("carrier", sa.String(10), nullable=False, index=True),
        sa.Column("origin", sa.String(5), nullable=False, index=True),
        sa.Column("destination", sa.String(5), nullable=False, index=True),
        sa.Column("scheduled_departure", sa.DateTime(), nullable=False),
        sa.Column("actual_departure", sa.DateTime(), nullable=True),
        sa.Column("delay_minutes", sa.Float(), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("aircraft_type", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("correlation_id", sa.String(36), nullable=True, index=True),
        sa.Column("carrier", sa.String(10), nullable=False),
        sa.Column("origin", sa.String(5), nullable=False),
        sa.Column("destination", sa.String(5), nullable=False),
        sa.Column("scheduled_hour", sa.Integer(), nullable=False),
        sa.Column("day_of_week", sa.Integer(), nullable=False),
        sa.Column("month", sa.Integer(), nullable=False),
        sa.Column("delay_probability", sa.Float(), nullable=False),
        sa.Column("predicted_class_name", sa.String(20), nullable=False),
        sa.Column("model_version", sa.String(20), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("feature_name", sa.String(50), nullable=False),
        sa.Column("ks_statistic", sa.Float(), nullable=False),
        sa.Column("p_value", sa.Float(), nullable=False),
        sa.Column("drift_detected", sa.Integer(), nullable=False),
        sa.Column("reference_size", sa.Integer(), nullable=True),
        sa.Column("current_size", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "retrain_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("trigger", sa.String(50), nullable=False),
        sa.Column("auc_before", sa.Float(), nullable=True),
        sa.Column("auc_after", sa.Float(), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("retrain_logs")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
    op.drop_table("flights")
