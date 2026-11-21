"""Flask REST API serving the 7-day Bitcoin close-price forecast.

    GET  /health        status, uptime, loaded model version
    POST /predict       7-day (or 1..30 day) forecast, optional custom history
    GET  /metrics       validation metrics of the loaded model
    POST /admin/reload  force a model reload (needs X-Admin-Token)

Production:  gunicorn --bind 0.0.0.0:5000 app:app
"""

import hmac
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone

import joblib
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

# Container layout: /app/app.py + /app/src/...   Repo layout: repo/app/app.py + repo/src/...
HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (HERE, os.path.dirname(HERE)):
    if _path not in sys.path:
        sys.path.append(_path)

from src.features import MIN_HISTORY  # noqa: E402
from src.forecast import forecast_closes  # noqa: E402

DEFAULT_DAYS = 7
MAX_DAYS = 30
MAX_HISTORY = 5000

log = logging.getLogger("btc-api")


class ModelStore:
    """Holds the current model and reloads it automatically when meta.json changes."""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self._lock = threading.Lock()
        self._stamp = None
        self.bundle = None
        self.metrics = {}

    def _current_stamp(self):
        try:
            return os.stat(os.path.join(self.model_dir, "meta.json")).st_mtime_ns
        except OSError:
            return None

    def refresh(self, force=False):
        stamp = self._current_stamp()
        if stamp is None or (not force and stamp == self._stamp):
            return
        with self._lock:
            try:
                bundle = joblib.load(os.path.join(self.model_dir, "model.joblib"))
                with open(os.path.join(self.model_dir, "metrics.json")) as fh:
                    metrics = json.load(fh)
            except Exception:
                log.exception("model reload failed; keeping previous model")
                return
            self.bundle, self.metrics, self._stamp = bundle, metrics, stamp
            log.info("loaded model %s", bundle["version"])

    def get(self):
        self.refresh()
        return self.bundle


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _error(status, code, message):
    return jsonify(error=code, message=message), status


def create_app(model_dir=None, admin_token=None):
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = Flask(__name__)
    store = ModelStore(model_dir or os.getenv("MODEL_DIR", "/app/models"))
    token = admin_token if admin_token is not None else os.getenv("ADMIN_TOKEN", "")
    stage = os.getenv("MODEL_STAGE", "development")
    started = time.time()
    store.refresh()

    @app.before_request
    def _start_timer():
        request.environ["t0"] = time.perf_counter()

    @app.after_request
    def _log_request(response):
        ms = (time.perf_counter() - request.environ.get("t0", time.perf_counter())) * 1000
        log.info("%s %s -> %s (%.1f ms)", request.method, request.path, response.status_code, ms)
        return response

    @app.get("/health")
    def health():
        bundle = store.get()
        return jsonify(
            status="ok",
            uptime_seconds=round(time.time() - started, 1),
            model_loaded=bundle is not None,
            model_version=bundle["version"] if bundle else None,
            model_stage=stage,
        )

    @app.post("/predict")
    def predict():
        # ---- parse and validate ----
        payload = {}
        if request.get_data().strip():
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return _error(400, "bad_request", "body must be a valid JSON object")

        days = payload.get("forecast_days", DEFAULT_DAYS)
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= MAX_DAYS:
            return _error(400, "bad_request", f"forecast_days must be an integer in 1..{MAX_DAYS}")

        history = payload.get("history")
        if history is not None:
            if not isinstance(history, list) or not MIN_HISTORY <= len(history) <= MAX_HISTORY:
                return _error(
                    400,
                    "bad_request",
                    f"history must be a list of {MIN_HISTORY}..{MAX_HISTORY} closing prices",
                )
            if not all(_is_number(x) and math.isfinite(x) and x > 0 for x in history):
                return _error(400, "bad_request", "history must contain positive finite numbers")

        last_date = payload.get("last_date")
        if last_date is not None:
            try:
                last_date = date.fromisoformat(last_date)
            except (TypeError, ValueError):
                return _error(400, "bad_request", "last_date must be an ISO date (YYYY-MM-DD)")

        # ---- predict ----
        bundle = store.get()
        if bundle is None:
            return _error(503, "model_not_loaded", "no trained model available yet")
        try:
            if history is None:
                closes = bundle["history_close"]
                last_date = last_date or date.fromisoformat(bundle["history_dates"][-1])
            else:
                closes = history
                last_date = last_date or datetime.now(timezone.utc).date()
            preds = forecast_closes(bundle["model"], closes, days)
        except ValueError as exc:
            return _error(400, "bad_request", str(exc))

        return jsonify(
            model_version=bundle["version"],
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            currency="USD",
            forecast_days=days,
            last_observed_date=last_date.isoformat(),
            last_observed_close=round(float(closes[-1]), 2),
            predictions=[
                {
                    "date": (last_date + timedelta(days=i + 1)).isoformat(),
                    "predicted_close": round(float(p), 2),
                }
                for i, p in enumerate(preds)
            ],
        )

    @app.get("/metrics")
    def metrics():
        bundle = store.get()
        if bundle is None:
            return _error(503, "model_not_loaded", "no trained model available yet")
        return jsonify(
            model_version=bundle["version"],
            trained_at=bundle["trained_at"],
            model_type=bundle["model_type"],
            metrics=store.metrics,
        )

    @app.post("/admin/reload")
    def reload_model():
        if not token:
            return _error(503, "reload_disabled", "ADMIN_TOKEN is not configured")
        supplied = request.headers.get("X-Admin-Token", "")
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            return _error(401, "unauthorized", "missing or invalid X-Admin-Token")
        store.refresh(force=True)
        if store.bundle is None:
            return _error(503, "model_not_loaded", "no model found in MODEL_DIR")
        return jsonify(status="reloaded", model_version=store.bundle["version"])

    @app.errorhandler(HTTPException)
    def _http_error(exc):
        return _error(exc.code, exc.name.lower().replace(" ", "_"), exc.description)

    @app.errorhandler(Exception)
    def _unhandled(exc):
        log.exception("unhandled error")
        return _error(500, "internal_error", "unexpected server error")

    return app


app = create_app()

if __name__ == "__main__":  # local dev only; production uses gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
