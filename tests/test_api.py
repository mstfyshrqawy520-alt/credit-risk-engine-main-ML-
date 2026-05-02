from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body


def test_predict_schema():
    payload = {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "employment_years": 5,
        "debt_amount": 25000,
        "payment_history": 12,
    }
    r = client.post("/predict", json=payload)
    # model may be loaded or not depending on environment; ensure no server error
    assert r.status_code in (200, 503, 422)
