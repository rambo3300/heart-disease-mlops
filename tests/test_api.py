import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(trained, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(trained["dir"] / "model.joblib"))
    from api.app import app

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in ("heart disease", "no heart disease")


def test_predict_rejects_missing_field(client, sample_patient):
    sample_patient.pop("chol")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_out_of_range_value(client, sample_patient):
    sample_patient["age"] = 500
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch_predict(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient, sample_patient]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_metrics_endpoint_exposes_custom_metrics(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_request_latency_seconds_bucket" in text
    assert "model_predictions_total" in text


def test_model_info(client):
    assert "features" in client.get("/model-info").json()


def test_service_unavailable_without_model(monkeypatch, tmp_path, sample_patient):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    from api.app import app

    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=sample_patient).status_code == 503
