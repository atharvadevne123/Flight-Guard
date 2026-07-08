"""Pytest fixtures and shared test configuration."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.features import generate_synthetic_data
from app.main import app

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def synthetic_data():
    X, y = generate_synthetic_data(n_samples=500)
    return X, y


@pytest.fixture
def sample_flight_request() -> dict:
    return {
        "carrier": "AA",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_hour": 8,
        "day_of_week": 1,
        "month": 7,
        "distance_km": 3983.0,
    }


@pytest.fixture
def peak_flight_request() -> dict:
    """Flight at peak hour and high season for edge case testing."""
    return {
        "carrier": "NK",
        "origin": "ORD",
        "destination": "ATL",
        "scheduled_hour": 17,
        "day_of_week": 4,
        "month": 12,
        "distance_km": 1139.0,
    }


@pytest.fixture
def unknown_carrier_request() -> dict:
    """Flight with unknown carrier to test fallback encoding."""
    return {
        "carrier": "ZZ",
        "origin": "BOS",
        "destination": "DEN",
        "scheduled_hour": 10,
        "day_of_week": 2,
        "month": 9,
        "distance_km": 2840.0,
    }
