from typing import List
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load trained model
cnn_model = tf.keras.models.load_model("cnn_model.h5")

class TrafficData(BaseModel):
    features: List[float]

@app.post("/predict/")
async def predict(data: TrafficData):
    X = np.array(data.features).reshape(1, -1)
    prediction = cnn_model.predict(X)
    return {"attack_probability": float(prediction[0][0])}
