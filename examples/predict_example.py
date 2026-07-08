"""Example client showing single and batch prediction calls.

Run the API first (make run), then:
    python examples/predict_example.py
"""

from __future__ import annotations

import json
import urllib.request

BASE_URL = "http://localhost:8000/api/v1"


def post(path: str, payload: dict) -> dict:
    """POST JSON to the API and return the parsed response."""
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Correlation-ID": "example-001"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main() -> None:
    single = post(
        "/predict",
        {
            "carrier": "AA",
            "origin": "JFK",
            "destination": "LAX",
            "scheduled_hour": 8,
            "day_of_week": 1,
            "month": 7,
            "distance_km": 3983.0,
        },
    )
    print("Single prediction:")
    print(json.dumps(single, indent=2))

    batch = post(
        "/predict/batch",
        {
            "flights": [
                {
                    "carrier": "NK",
                    "origin": "ORD",
                    "destination": "EWR",
                    "scheduled_hour": 17,
                    "day_of_week": 4,
                    "month": 12,
                    "distance_km": 1178.0,
                },
                {
                    "carrier": "HA",
                    "origin": "SEA",
                    "destination": "LAS",
                    "scheduled_hour": 11,
                    "day_of_week": 2,
                    "month": 9,
                    "distance_km": 1448.0,
                },
            ]
        },
    )
    print("\nBatch prediction:")
    print(json.dumps(batch, indent=2))


if __name__ == "__main__":
    main()
