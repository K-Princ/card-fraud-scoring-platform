from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Card Fraud Scoring API is running!"

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["service"] == "fraud-scoring-api"

def test_model_status():
    response = client.get("/model-status")

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["encoder_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["medians_loaded"] is True

def test_predict_transaction():
    payload = {
        "TransactionAmt": 150.0,
        "dist2": 38.0,
        "D7": 0.0,
        "D13": 0.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["threshold"] == 0.10

def test_predict_invalid_amount():
    payload = {
        "TransactionAmt": -10.0,
        "ProductCD": "W",
        "card4": "visa",
        "card6": "debit"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_invalid_category():
    payload = {
        "TransactionAmt": 150.0,
        "ProductCD": "INVALID",
        "card4": "visa",
        "card6": "debit"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_missing_categories():
    payload = {
        "TransactionAmt": 150.0,
        "ProductCD": "W"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0 <= data["fraud_probability"] <= 1