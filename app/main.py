"""FastAPI service: /health and /predict."""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURES, engineer

ROOT = Path(__file__).resolve().parent.parent
MODEL = joblib.load(ROOT / "models" / "return_model.joblib")
_metrics = ROOT / "models" / "metrics.json"
BEST_MODEL = json.loads(_metrics.read_text())["best_model"] if _metrics.exists() else "unknown"
APP_VERSION = "1.0.0"

app = FastAPI(title="E-Commerce Product Return Prediction", version=APP_VERSION)


class Order(BaseModel):
    Product_Category: Literal["Beauty", "Electronics", "Fashion", "Home", "Sports"]
    Product_Price_INR: float = Field(..., gt=0)
    Discount_Percent: float = Field(..., ge=0, le=100)
    Expected_Delivery_Days: int = Field(..., ge=0)
    Actual_Delivery_Days: int = Field(..., ge=0)
    Customer_Rating_PreOrder: float = Field(..., ge=1, le=5)
    Previous_Returns: int = Field(..., ge=0)
    Previous_Orders: int = Field(..., ge=0)
    Payment_Method: Literal["Card", "COD", "UPI", "Wallet"]
    Damaged_Package_Flag: Literal["Yes", "No"]
    Package_Size: Literal["Small", "Medium", "Large"]
    Promotion_Active: Literal["Yes", "No"]

    model_config = {"json_schema_extra": {"example": {
        "Product_Category": "Electronics", "Product_Price_INR": 22055.97,
        "Discount_Percent": 21.6, "Expected_Delivery_Days": 9, "Actual_Delivery_Days": 12,
        "Customer_Rating_PreOrder": 4.4, "Previous_Returns": 3, "Previous_Orders": 33,
        "Payment_Method": "Card", "Damaged_Package_Flag": "Yes",
        "Package_Size": "Large", "Promotion_Active": "Yes"}}}


@app.get("/health")
def health():
    return {"status": "ok", "model": BEST_MODEL, "version": APP_VERSION}


@app.post("/predict")
def predict(order: Order):
    df = engineer(pd.DataFrame([order.model_dump()]))
    proba = float(MODEL.predict_proba(df[FEATURES])[0, 1])
    return {
        "prediction": "Returned" if proba >= 0.5 else "Not Returned",
        "return_probability": round(proba, 4),
    }
