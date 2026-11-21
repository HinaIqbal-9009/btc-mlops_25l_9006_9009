"""Feature engineering shared by training, serving and the pipelines."""

import argparse
import os

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "ret_7d",
    "ma7_gap",
    "ma30_gap",
    "vol_7",
    "vol_30",
]
MIN_HISTORY = 40  # vol_30 needs 31 closes; 40 leaves head-room


def make_features(close):
    """One feature row per close price (early rows are NaN while windows fill up)."""
    close = pd.Series(close, dtype="float64").reset_index(drop=True)
    logret = np.log(close).diff()
    feats = pd.DataFrame(index=close.index)
    for lag in range(1, 6):
        feats[f"ret_lag_{lag}"] = logret.shift(lag - 1)
    feats["ret_7d"] = np.log(close / close.shift(7))
    feats["ma7_gap"] = close / close.rolling(7).mean() - 1.0
    feats["ma30_gap"] = close / close.rolling(30).mean() - 1.0
    feats["vol_7"] = logret.rolling(7).std()
    feats["vol_30"] = logret.rolling(30).std()
    return feats[FEATURE_COLS]


def build_feature_table(raw):
    """Raw OHLCV -> date, close, features, target (next-day log return)."""
    raw = raw.sort_values("date").reset_index(drop=True)
    table = pd.DataFrame({"date": raw["date"], "close": raw["close"].astype("float64")})
    table = pd.concat([table, make_features(table["close"])], axis=1)
    table["target"] = np.log(table["close"]).diff().shift(-1)
    return table


def main():
    parser = argparse.ArgumentParser(description="Build the BTC feature table.")
    parser.add_argument("--raw", default="data/raw/btc_usd.csv")
    parser.add_argument("--out", default="data/processed/features.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, parse_dates=["date"])
    table = build_feature_table(raw)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"[features] wrote {len(table)} rows ({len(table.dropna())} usable) -> {args.out}")


if __name__ == "__main__":
    main()
