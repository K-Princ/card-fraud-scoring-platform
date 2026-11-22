from fastapi import FastAPI, HTTPException
import joblib
from pathlib import Path
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal

app = FastAPI(
    title="Card Fraud Scoring API",
    description="Real-time fraud scoring service",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "fraud_calibrated.pkl")
encoder = joblib.load(MODEL_DIR / "encoder.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
medians = joblib.load(MODEL_DIR / "train_medians.pkl")

FRAUD_THRESHOLD = 0.10

def preprocess_transaction(transaction):
    data = pd.DataFrame([transaction.model_dump()])

    numerical_columns = [
        "TransactionAmt", "dist2", "D7", "D13"
    ]

    categorical_columns = [
        "ProductCD", "card4", "card6"
    ]

    for col in numerical_columns:
        data[col] = data[col].fillna(medians[col])

    scaled_numeric = scaler.transform(data[numerical_columns])

    data[categorical_columns] = data[categorical_columns].fillna("Unknown")

    encoded_categorical = encoder.transform(data[categorical_columns])

    features = pd.DataFrame(
        scaled_numeric,
        columns=numerical_columns
    )

    encoded_features = pd.DataFrame(
        encoded_categorical,
        columns=encoder.get_feature_names_out(categorical_columns)
    )

    return pd.concat(
        [features, encoded_features],
        axis=1
    )

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(gt = 0)
    dist2: float | None = None
    D7: float | None = None
    D13: float | None = None
    ProductCD: Literal["C", "H", "R", "S", "W"]
    card4: Literal[
    "Unknown", "american express", "discover",
    "mastercard", "visa"
] = "Unknown"

    card6: Literal[
    "Unknown", "charge card", "credit",
    "debit", "debit or credit"
] = "Unknown"

@app.get("/")
def home():
    return {"message": "Card Fraud Scoring API is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "fraud-scoring-api"
    }

@app.get("/model-status")
def model_status():
    return {
        "model_loaded": model is not None,
        "encoder_loaded": encoder is not None,
        "scaler_loaded": scaler is not None,
        "medians_loaded": medians is not None
    }

@app.post("/predict")
def predict_transaction(transaction: TransactionInput):
    try:
        features = preprocess_transaction(transaction)

        probability = float(model.predict_proba(features)[0][1])
        prediction = int(probability >= FRAUD_THRESHOLD)

        if probability < 0.05:
            risk_level = "LOW"
        elif probability < 0.10:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "prediction": prediction,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "threshold": FRAUD_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while scoring the transaction."
        ) from e

