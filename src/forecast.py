"""Recursive multi-step forecasting, used by training and by the API."""

import numpy as np
import pandas as pd

from src.features import FEATURE_COLS, MIN_HISTORY, make_features

WINDOW = 60
MAX_DAILY_LOG_RETURN = 0.20


def forecast_closes(model, closes, horizon=7):
    history = [float(x) for x in closes][-WINDOW:]
    if len(history) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} closing prices, got {len(history)}")

    predictions = []
    for _ in range(horizon):
        row = make_features(pd.Series(history)).iloc[-1][FEATURE_COLS]
        if row.isna().any():
            raise ValueError("history produced NaN features")
        log_return = float(model.predict(row.to_numpy(dtype="float64").reshape(1, -1))[0])
        log_return = float(np.clip(log_return, -MAX_DAILY_LOG_RETURN, MAX_DAILY_LOG_RETURN))
        next_close = history[-1] * float(np.exp(log_return))
        history.append(next_close)  # feed the prediction back in as if it were real
        predictions.append(next_close)
    return np.array(predictions)
