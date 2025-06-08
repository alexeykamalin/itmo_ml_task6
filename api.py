import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import subprocess
from pathlib import Path
import logging
import sys

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI()

model = None
MODEL_PATH = Path('model.pkl')

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

def load_model():
    global model
    try:
        if not MODEL_PATH.exists():
            logger.warning("Model file not found, pulling from DVC...")
            result = subprocess.run(
                ['dvc', 'pull', str(MODEL_PATH)],
                capture_output=True,
                text=True
            )
            logger.info(f"DVC pull stdout: {result.stdout}")
            logger.info(f"DVC pull stderr: {result.stderr}")

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found after DVC pull at {MODEL_PATH.absolute()}")

        logger.info(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        
        # Тестовое предсказание для проверки модели
        test_data = [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
        test_pred = model.predict(test_data)
        logger.info(f"Test prediction: {test_pred[0]} (should be ~5.02)")
        
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        logger.error("Failed to load model on startup")

@app.post("/predict")
async def predict(wine: WineFeatures):
    if model is None:
        logger.error("Prediction attempt with unloaded model")
        raise HTTPException(
            status_code=503,
            detail="Service Unavailable: Model not loaded"
        )
    
    try:
        # Логируем полученные данные
        logger.info(f"Received prediction request: {wine.dict()}")
        
        # Преобразуем в список списков (как ожидает sklearn)
        input_data = [list(wine.dict().values())]
        
        # Делаем предсказание
        prediction = model.predict(input_data)[0]
        logger.info(f"Successful prediction: {prediction}")
        
        return {"predicted_quality": float(prediction)}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction Error: {str(e)}"
        )

@app.get("/healthcheck")
async def healthcheck():
    if model is None:
        return {"status": "error", "reason": "model"}
    return {"status": "ok"}