from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE = {
    "Product_Category": "Electronics", "Product_Price_INR": 22055.97,
    "Discount_Percent": 21.6, "Expected_Delivery_Days": 9, "Actual_Delivery_Days": 12,
    "Customer_Rating_PreOrder": 4.4, "Previous_Returns": 3, "Previous_Orders": 33,
    "Payment_Method": "Card", "Damaged_Package_Flag": "Yes",
    "Package_Size": "Large", "Promotion_Active": "Yes",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ("Returned", "Not Returned")
    assert 0.0 <= body["return_probability"] <= 1.0


def test_predict_invalid_category():
    r = client.post("/predict", json={**SAMPLE, "Product_Category": "Toys"})
    assert r.status_code == 422


def test_predict_invalid_discount():
    r = client.post("/predict", json={**SAMPLE, "Discount_Percent": 150})
    assert r.status_code == 422
