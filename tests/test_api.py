import json

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Credit Card Fraud Detection API"
    assert data["model"] == "XGBoost"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_invalid_feature_count():
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]}
    )

    assert response.status_code == 400

    assert (
        "Exactly 30 feature values are required"
        in response.json()["detail"]
    )


def test_legitimate_prediction():

    with open(
        "tests/legitimate_sample.json",
        "r"
    ) as f:
        sample = json.load(f)

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["class_name"] == "legitimate"

    assert 0 <= data["fraud_probability"] <= 1


def test_fraud_prediction():

    with open(
        "tests/fraud_sample.json",
        "r"
    ) as f:
        sample = json.load(f)

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["class_name"] == "fraud"

    assert 0 <= data["fraud_probability"] <= 1
