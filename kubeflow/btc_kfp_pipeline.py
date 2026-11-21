"""Kubeflow Pipelines (KFP v2 SDK) definition: ingest -> features -> train.

Each step is a *container component*: KFP runs our own Docker image and passes artifact paths
as command-line arguments (the same `python -m src.<module>` CLIs the Airflow DAG uses).

Compile:   python kubeflow/btc_kfp_pipeline.py
Output:    kubeflow/btc_kfp_pipeline.yaml  (upload it in the Kubeflow Pipelines UI)
"""

import os

from kfp import compiler, dsl

# Replace with the image your GitHub Actions workflow pushes (DOCKER_USERNAME/btc-forecast).
IMAGE = os.getenv("BTC_IMAGE", "docker.io/YOUR_DOCKERHUB_USER/btc-forecast:latest")


@dsl.container_component
def ingest_data(raw_data: dsl.Output[dsl.Dataset], start_date: str = "2018-01-01"):
    """Pull daily BTC-USD OHLCV data."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--out", raw_data.path, "--start", start_date],
    )


@dsl.container_component
def build_features(raw_data: dsl.Input[dsl.Dataset], features: dsl.Output[dsl.Dataset]):
    """Technical indicators (moving averages, volatility, lagged returns) + target."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.features"],
        args=["--raw", raw_data.path, "--out", features.path],
    )


@dsl.container_component
def train_model(
    features: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    mape_threshold: float = 15.0,
):
    """Train, evaluate on the last 7 days and register the model if MAPE is low enough."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=[
            "--features",
            features.path,
            "--out-dir",
            model.path,
            "--mape-threshold",
            mape_threshold,
        ],
    )


@dsl.pipeline(
    name="btc-forecast-pipeline",
    description="Ingest BTC-USD data, engineer features, train the 7-day forecaster.",
)
def btc_pipeline(start_date: str = "2018-01-01", mape_threshold: float = 15.0):
    ingest = ingest_data(start_date=start_date)
    ingest.set_caching_options(False)  # always pull fresh market data
    feats = build_features(raw_data=ingest.outputs["raw_data"])
    train_model(features=feats.outputs["features"], mape_threshold=mape_threshold)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_kfp_pipeline.yaml")
    compiler.Compiler().compile(pipeline_func=btc_pipeline, package_path=out)
    print(f"compiled -> {out}")
