"""Daily BTC pipeline: fetch -> features -> train/evaluate -> quality gate -> reload API."""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "btc_usd.csv")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "15"))
API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://api:5000/admin/reload")


def fetch_data():
    from src.etl_fetch import fetch_and_save

    return fetch_and_save(RAW_PATH)


def preprocess_features():
    import pandas as pd

    from src.features import build_feature_table

    table = build_feature_table(pd.read_csv(RAW_PATH, parse_dates=["date"]))
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)
    print(f"wrote {len(table)} feature rows -> {FEATURES_PATH}")
    return FEATURES_PATH


def train_and_evaluate():
    from src.train_forecast import train_candidate

    return train_candidate(FEATURES_PATH, MODEL_DIR)  # metrics dict -> pushed to XCom


def validate_and_register(ti):
    """Return True (continue) only if the candidate beats the threshold; then promote it."""
    from src.train_forecast import passes_threshold, promote_candidate

    metrics = ti.xcom_pull(task_ids="train_evaluate")
    if not passes_threshold(metrics, MAPE_THRESHOLD):
        print(f"REJECTED: MAPE {metrics['mape']:.2f}% > threshold {MAPE_THRESHOLD}%")
        return False  # downstream tasks are skipped; the previous model stays live
    promote_candidate(MODEL_DIR)
    print(f"REGISTERED {metrics['version']} (MAPE {metrics['mape']:.2f}%)")
    return True


def reload_api():
    import requests

    resp = requests.post(
        API_RELOAD_URL,
        headers={"X-Admin-Token": os.getenv("ADMIN_TOKEN", "")},
        timeout=30,
    )
    resp.raise_for_status()
    print("API says:", resp.json())


default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ETL, retraining, evaluation and API model reload",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["btc", "mlops"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_data", python_callable=fetch_data)
    t_features = PythonOperator(task_id="preprocess_features", python_callable=preprocess_features)
    t_train = PythonOperator(task_id="train_evaluate", python_callable=train_and_evaluate)
    t_gate = ShortCircuitOperator(
        task_id="validate_and_register", python_callable=validate_and_register
    )
    t_reload = PythonOperator(task_id="reload_api", python_callable=reload_api)

    t_fetch >> t_features >> t_train >> t_gate >> t_reload
