from datetime import date, timedelta


def test_health_ok(client):
    body = client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"].startswith("v")


def test_health_without_model_still_ok(empty_client):
    body = empty_client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_predict_defaults_to_seven_days(client):
    resp = client.post("/predict")
    assert resp.status_code == 200
    body = resp.get_json()
    preds = body["predictions"]
    assert len(preds) == 7
    assert all(p["predicted_close"] > 0 for p in preds)
    first = date.fromisoformat(preds[0]["date"])
    assert first == date.fromisoformat(body["last_observed_date"]) + timedelta(days=1)


def test_predict_custom_days(client):
    resp = client.post("/predict", json={"forecast_days": 3})
    assert resp.status_code == 200
    assert len(resp.get_json()["predictions"]) == 3


def test_predict_custom_history(client):
    history = [30000 + 10 * i for i in range(60)]
    resp = client.post("/predict", json={"history": history, "last_date": "2026-01-01"})
    assert resp.status_code == 200
    assert resp.get_json()["predictions"][0]["date"] == "2026-01-02"


def test_predict_rejects_bad_input(client):
    bad_payloads = [
        {"forecast_days": 0},
        {"forecast_days": 100},
        {"forecast_days": "seven"},
        {"forecast_days": True},
        {"history": [1, 2, 3]},
        {"history": [-5] * 60},
        {"history": ["a"] * 60},
        {"last_date": "yesterday"},
    ]
    for payload in bad_payloads:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400, payload


def test_predict_malformed_json_is_400(client):
    resp = client.post("/predict", data="{not json", content_type="application/json")
    assert resp.status_code == 400


def test_predict_without_model_is_503(empty_client):
    assert empty_client.post("/predict").status_code == 503


def test_metrics(client):
    body = client.get("/metrics").get_json()
    for key in ("mae", "rmse", "mape", "naive_baseline_mape"):
        assert key in body["metrics"]


def test_reload_requires_token(client):
    assert client.post("/admin/reload").status_code == 401
    ok = client.post("/admin/reload", headers={"X-Admin-Token": "secret"})
    assert ok.status_code == 200


def test_errors_are_json(client):
    assert client.get("/nope").get_json()["error"] == "not_found"
    assert client.get("/predict").status_code == 405
