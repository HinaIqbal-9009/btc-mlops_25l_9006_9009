"""Train, evaluate on the last 7 days, and register the model only if it is good enough."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.features import FEATURE_COLS
from src.forecast import forecast_closes

HORIZON = 7
HISTORY_KEEP = 120
ARTIFACTS = ["model.joblib", "metrics.json", "meta.json"]  # meta.json LAST = reload signal


def new_model():
    return GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
    )


def fit(table):
    usable = table.dropna(subset=FEATURE_COLS + ["target"])
    model = new_model()
    model.fit(usable[FEATURE_COLS].to_numpy(), usable["target"].to_numpy())
    return model


def evaluate_holdout(table, horizon=HORIZON):
    """Train WITHOUT the last 7 days, forecast them, compare with what really happened."""
    n = len(table)
    model = fit(table.iloc[: n - horizon - 1])
    origin = table["close"].iloc[: n - horizon].to_numpy()
    actual = table["close"].iloc[n - horizon :].to_numpy()
    pred = forecast_closes(model, origin, horizon)
    naive = np.full(horizon, origin[-1])  # baseline: "tomorrow = today"

    def mape(p):
        return float(np.mean(np.abs((actual - p) / actual)) * 100)

    return {
        "horizon_days": horizon,
        "mae": float(np.mean(np.abs(actual - pred))),
        "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
        "mape": mape(pred),
        "naive_baseline_mape": mape(naive),
        "test_window_start": str(table["date"].iloc[n - horizon])[:10],
        "test_window_end": str(table["date"].iloc[-1])[:10],
    }


def train_candidate(features_path, out_dir, horizon=HORIZON):
    table = pd.read_csv(features_path, parse_dates=["date"])
    metrics = evaluate_holdout(table, horizon)
    model = fit(table)  # final model sees ALL the data

    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    tail = table.tail(HISTORY_KEEP)
    bundle = {
        "model": model,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": "GradientBoostingRegressor on next-day log-return, recursive 7-day forecast",
        "feature_cols": FEATURE_COLS,
        "history_close": [float(x) for x in tail["close"]],
        "history_dates": [str(d)[:10] for d in tail["date"]],
    }
    meta = {k: bundle[k] for k in ("version", "trained_at", "model_type")}
    meta.update(rows=int(len(table)), data_end=bundle["history_dates"][-1])
    metrics = dict(metrics, version=version)

    cand = os.path.join(out_dir, "candidate")
    os.makedirs(cand, exist_ok=True)
    joblib.dump(bundle, os.path.join(cand, "model.joblib"))
    with open(os.path.join(cand, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(cand, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[train] candidate {version}: {json.dumps(metrics)}")
    return metrics


def passes_threshold(metrics, mape_threshold):
    return metrics["mape"] <= mape_threshold


def promote_candidate(out_dir):
    cand = os.path.join(out_dir, "candidate")
    for name in ARTIFACTS:
        tmp = os.path.join(out_dir, name + ".tmp")
        shutil.copyfile(os.path.join(cand, name), tmp)
        os.replace(tmp, os.path.join(out_dir, name))
    print(f"[train] promoted candidate -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the BTC 7-day forecaster.")
    parser.add_argument("--features", default="data/processed/features.csv")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--mape-threshold", type=float, default=15.0)
    args = parser.parse_args()

    metrics = train_candidate(args.features, args.out_dir)
    if not passes_threshold(metrics, args.mape_threshold):
        print(f"[train] REJECTED: MAPE {metrics['mape']:.2f}% > {args.mape_threshold}%")
        return 3
    promote_candidate(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
