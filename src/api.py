from pathlib import Path
from typing import List

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "cnn_model.h5"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"


# --------------------------------------------------
# Application
# --------------------------------------------------

app = FastAPI(
    title="Intrusion Detection API",
    description="CNN-based network traffic classification API",
    version="1.0.0"
)


# --------------------------------------------------
# Load model and preprocessing
# --------------------------------------------------

try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

except FileNotFoundError:
    cnn_model = None
    scaler = None


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class TrafficData(BaseModel):
    features: List[float]


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Intrusion Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "model_loaded": cnn_model is not None,
        "scaler_loaded": scaler is not None
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
def predict(data: TrafficData):

    if cnn_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifacts not found. "
                "Run src/train_and_evaluate.py first."
            )
        )

    features = np.asarray(
        data.features,
        dtype=np.float32
    ).reshape(1, -1)

    expected_features = scaler.n_features_in_

    if features.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {expected_features} features, "
                f"but received {features.shape[1]}."
            )
        )

    # Apply the same preprocessing used during training
    features = scaler.transform(features)

    # CNN input shape: (batch, features, channels)
    features = features.reshape(
        1,
        features.shape[1],
        1
    )

    probability = float(
        cnn_model.predict(
            features,
            verbose=0
        )[0][0]
    )

    prediction = (
        "Attack"
        if probability >= 0.5
        else "Normal"
    )

    return {
        "prediction": prediction,
        "attack_probability": round(
            probability,
            4
        )
    }