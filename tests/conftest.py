import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))  # so `from app import create_app` finds app/app.py

from app import create_app  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.train_forecast import promote_candidate, train_candidate  # noqa: E402


def synthetic_raw(n_days=900, seed=7):
    """Fake random-walk prices, for tests only."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    close = 10000 * np.exp(np.cumsum(rng.normal(0.0008, 0.03, n_days)))
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture(scope="session")
def features_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "features.csv"
    build_feature_table(synthetic_raw()).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, features_csv):
    out = str(tmp_path_factory.mktemp("models"))
    train_candidate(features_csv, out)
    promote_candidate(out)
    return out


@pytest.fixture()
def client(model_dir):
    return create_app(model_dir=model_dir, admin_token="secret").test_client()


@pytest.fixture()
def empty_client(tmp_path):
    return create_app(model_dir=str(tmp_path), admin_token="secret").test_client()
