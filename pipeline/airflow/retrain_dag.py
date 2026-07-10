"""
Airflow DAG: flight_guard_retrain
Scheduled daily. Pulls flight data from Palantir Foundry, computes carrier risk scores,
checks for drift, retrains the delay predictor, evaluates, and pushes to Foundry.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "flight-guard",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="flight_guard_retrain",
    description="Daily Flight-Guard model retraining with Palantir Foundry sync",
    schedule_interval="0 3 * * *",  # 03:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["flight-guard", "ml", "foundry"],
)


def fetch_flight_data_from_foundry(**ctx):
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from foundry.foundry_client import FoundryClient

    client = FoundryClient()
    dataset_rid = os.getenv("FLIGHTS_DATASET_RID", "")
    if not dataset_rid:
        raise ValueError("FLIGHTS_DATASET_RID env var not set.")

    df = client.read_dataset(dataset_rid)
    if df.empty:
        raise ValueError("No flight data returned from Foundry.")

    tmp_path = "/tmp/flight_training_data.parquet"
    df.to_parquet(tmp_path, index=False)
    ctx["ti"].xcom_push(key="training_data_path", value=tmp_path)
    print(f"Fetched {len(df):,} flight records from Foundry.")


def compute_carrier_risk_scores(**ctx):
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.risk.carrier_risk_scorer import CarrierRiskScorer

    training_path = ctx["ti"].xcom_pull(key="training_data_path")
    df = pd.read_parquet(training_path)

    scorer = CarrierRiskScorer()
    carriers = df["carrier_code"].unique() if "carrier_code" in df.columns else []
    risk_scores = {}
    for carrier in carriers:
        carrier_df = df[df["carrier_code"] == carrier]
        profile = scorer.score_carrier(carrier, carrier_df)
        risk_scores[carrier] = profile.get("risk_score", 0.5)

    print(f"Computed risk scores for {len(risk_scores)} carriers.")
    ctx["ti"].xcom_push(key="carrier_risk_scores", value=risk_scores)


def check_drift(**ctx):
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from monitoring.drift_monitor import DriftMonitor

    training_path = ctx["ti"].xcom_pull(key="training_data_path")
    df = pd.read_parquet(training_path)

    mid = len(df) // 2
    monitor = DriftMonitor(threshold=0.05)
    report = monitor.detect_drift(df.iloc[:mid], df.iloc[mid:])
    drifted = [f for f, r in report.items() if r.get("drift_detected")]
    print(f"Drift check complete: {len(drifted)} features drifted.")


def retrain_model(**ctx):
    import json
    import sys
    from pathlib import Path

    import joblib
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.ensemble.delay_predictor import DelayPredictor
    from pipeline.feature_engineering import FlightFeatureEngineer

    training_path = ctx["ti"].xcom_pull(key="training_data_path")
    df = pd.read_parquet(training_path)

    label_col = "is_delayed"
    if label_col not in df.columns:
        df[label_col] = (df.get("actual_delay_minutes", 0) >= 15).astype(int)

    fe = FlightFeatureEngineer()
    df_feat = fe.fit_transform(df)

    exclude = {
        label_col,
        "flight_id",
        "carrier_code",
        "origin",
        "destination",
        "scheduled_departure",
        "aircraft_type",
        "weather_condition",
    }
    feat_cols = [c for c in df_feat.select_dtypes(include="number").columns if c not in exclude]
    X = df_feat[feat_cols].fillna(0)
    y = df_feat[label_col].astype(int)

    model = DelayPredictor()
    model.train(X, y)

    model_dir = Path(__file__).parent.parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "delay_predictor.joblib")
    joblib.dump(fe, model_dir / "feature_engineer.joblib")
    (model_dir / "feature_cols.json").write_text(json.dumps(feat_cols))

    print(f"Delay predictor retrained on {len(df):,} flights.")
    ctx["ti"].xcom_push(key="model_dir", value=str(model_dir))


def evaluate_model(**ctx):
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from models.ensemble.delay_predictor import DelayPredictor

    model_dir = Path(ctx["ti"].xcom_pull(key="model_dir"))
    training_path = ctx["ti"].xcom_pull(key="training_data_path")

    model = DelayPredictor.load(model_dir / "delay_predictor.joblib")
    df = pd.read_parquet(training_path)
    feat_cols = json.loads((model_dir / "feature_cols.json").read_text())
    X = df.reindex(columns=feat_cols, fill_value=0).fillna(0)

    probs = model.predict_proba(X)
    avg_prob = float(sum(probs) / len(probs)) if probs else 0.5
    print(f"Evaluation — mean delay probability on training set: {avg_prob:.4f}")
    ctx["ti"].xcom_push(key="avg_delay_prob", value=avg_prob)


def push_model_to_foundry(**ctx):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from foundry.foundry_client import FoundryClient

    client = FoundryClient()
    avg_prob = ctx["ti"].xcom_pull(key="avg_delay_prob")

    client.register_model(
        {
            "name": "flight-guard-delay-predictor",
            "version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "framework": "xgboost+lightgbm+rf",
            "metrics": {"avg_delay_probability": avg_prob},
        }
    )
    print("Flight-Guard model registered in Foundry catalog.")


fetch_task = PythonOperator(
    task_id="fetch_flight_data_from_foundry",
    python_callable=fetch_flight_data_from_foundry,
    dag=dag,
)
risk_task = PythonOperator(
    task_id="compute_carrier_risk_scores", python_callable=compute_carrier_risk_scores, dag=dag
)
drift_task = PythonOperator(task_id="check_drift", python_callable=check_drift, dag=dag)
retrain_task = PythonOperator(task_id="retrain_model", python_callable=retrain_model, dag=dag)
eval_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model, dag=dag)
push_task = PythonOperator(
    task_id="push_model_to_foundry", python_callable=push_model_to_foundry, dag=dag
)

fetch_task >> risk_task >> drift_task >> retrain_task >> eval_task >> push_task
