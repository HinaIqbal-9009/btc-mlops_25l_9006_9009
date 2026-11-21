"""Extract step: pull daily BTC-USD OHLCV data and save it as a CSV."""

import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd

REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]
MIN_ROWS = 100


def fetch_yfinance(start):
    import yfinance as yf

    df = yf.Ticker("BTC-USD").history(start=start, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data")
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[REQUIRED_COLS]


def fetch_binance(start):
    import requests

    url = "https://api.binance.com/api/v3/klines"
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    rows = []
    while True:
        params = {"symbol": "BTCUSDT", "interval": "1d", "startTime": start_ms, "limit": 1000}
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 24 * 3600 * 1000
        if len(batch) < 1000:
            break
    if not rows:
        raise RuntimeError("binance returned no data")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = REQUIRED_COLS
    df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
    return df.astype({c: "float64" for c in REQUIRED_COLS[1:]})


def clean(df):
    """Validate and tidy: sorted, unique dates, positive prices, completed days only."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().drop_duplicates(subset="date").sort_values("date")
    df = df[df["close"] > 0]
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    df = df[df["date"] < today].reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise ValueError(f"only {len(df)} usable rows (need >= {MIN_ROWS})")
    return df


def fetch_btc(start="2018-01-01"):
    errors = []
    for name, fn in (("yfinance", fetch_yfinance), ("binance", fetch_binance)):
        for attempt in range(1, 4):
            try:
                df = fn(start)
                print(f"[etl] source={name} rows={len(df)}")
                return clean(df)
            except Exception as exc:
                errors.append(f"{name} attempt {attempt}: {exc}")
                time.sleep(2 * attempt)
    raise RuntimeError("all data sources failed:\n  " + "\n  ".join(errors))


def fetch_and_save(out, start="2018-01-01"):
    df = fetch_btc(start)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    summary = {
        "path": out,
        "rows": int(len(df)),
        "first_date": df["date"].iloc[0].strftime("%Y-%m-%d"),
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
    }
    print(f"[etl] saved {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fetch daily BTC-USD data to a CSV file.")
    parser.add_argument("--out", default="data/raw/btc_usd.csv")
    parser.add_argument("--start", default="2018-01-01")
    args = parser.parse_args()
    fetch_and_save(args.out, args.start)


if __name__ == "__main__":
    main()
