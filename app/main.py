from datetime import datetime, timezone
from pathlib import Path
import csv

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model_tuned.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

LOG_DIR = BASE_DIR / "monitoring" / "logs"
LOG_FILE = LOG_DIR / "prediction_log.csv"

FEATURE_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26",
    "V27", "V28",
    "Amount"
]

# Load production artifacts once when the API starts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Create monitoring directory if it does not exist
LOG_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "XGBoost REST API for predicting potentially "
        "fraudulent credit card transactions."
    ),
    version="1.0.0"
)


class Transaction(BaseModel):
    features: list[float] = Field(
        ...,
        description="30 values ordered as Time, V1-V28, Amount"
    )


def log_prediction(prediction: int, probability: float):
    """Write prediction information to the monitoring log."""

    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "class_name",
                "fraud_probability"
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            prediction,
            "fraud" if prediction == 1 else "legitimate",
            round(probability, 6)
        ])


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API",
        "model": "XGBoost",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "prediction": "/predict",
            "documentation": "/docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "xgboost_fraud_model_tuned.pkl"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    if len(transaction.features) != 30:
        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 30 feature values are required: "
                "Time, V1-V28, Amount."
            )
        )

    try:
        values = np.asarray(
            transaction.features,
            dtype=float
        )
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="All feature values must be numeric."
        )

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=400,
            detail="Feature values must be finite numbers."
        )

    # Build dataframe in the exact training order
    data = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    # Apply the same preprocessing used during training
    data.loc[:, ["Time", "Amount"]] = scaler.transform(
        data[["Time", "Amount"]]
    )

    prediction = int(model.predict(data)[0])

    probability = float(
        model.predict_proba(data)[0][1]
    )

    # Store prediction metadata for monitoring
    log_prediction(prediction, probability)

    return {
        "prediction": prediction,
        "class_name": (
            "fraud" if prediction == 1 else "legitimate"
        ),
        "fraud_probability": round(probability, 6)
    }
