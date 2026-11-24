from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.connection import get_db
from src.database.models import Prediction, Transaction

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
LOW_RISK_THRESHOLD = 0.05
MEDIUM_RISK_THRESHOLD = 0.10


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
def predict_transaction(
    transaction: TransactionInput,
    db: Session = Depends(get_db),  # noqa: B008
):
    try:
        features = preprocess_transaction(transaction)

        probability = float(model.predict_proba(features)[0][1])
        prediction = int(probability >= FRAUD_THRESHOLD)

        if probability < LOW_RISK_THRESHOLD:
            risk_level = "LOW"
        elif probability < MEDIUM_RISK_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        # Save transaction
        db_transaction = Transaction(
            transaction_amt=transaction.TransactionAmt,
            dist2=transaction.dist2,
            d7=transaction.D7,
            d13=transaction.D13,
            product_cd=transaction.ProductCD,
            card4=transaction.card4,
            card6=transaction.card6
        )

        db.add(db_transaction)
        db.flush()

        # Save prediction
        db_prediction = Prediction(
            transaction_id=db_transaction.transaction_id,
            fraud_probability=probability,
            prediction=prediction,
            risk_level=risk_level,
            model_version="1.0.0",
            threshold=FRAUD_THRESHOLD
        )

        db.add(db_prediction)
        db.commit()

        return {
            "transaction_id": db_transaction.transaction_id,
            "prediction": prediction,
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "threshold": FRAUD_THRESHOLD
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while scoring the transaction."
        ) from e


@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)): # noqa: B008
    query = (
        select(Prediction, Transaction)
        .join(
            Transaction,
            Prediction.transaction_id == Transaction.transaction_id
        )
        .order_by(Prediction.prediction_id.desc())
    )

    results = db.execute(query).all()

    return [
        {
            "prediction_id": prediction.prediction_id,
            "transaction_id": transaction.transaction_id,
            "transaction_amt": float(transaction.transaction_amt),
            "fraud_probability": prediction.fraud_probability,
            "prediction": prediction.prediction,
            "risk_level": prediction.risk_level,
            "model_version": prediction.model_version
        }
        for prediction, transaction in results
    ]
