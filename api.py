import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import subprocess
from pathlib import Path

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
            print("Модель не найдена, пытаемся загрузить через DVC...")
            result = subprocess.run(['dvc', 'pull', str(MODEL_PATH)], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"DVC pull failed: {result.stderr}")
            
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Файл модели {MODEL_PATH} не найден после DVC pull")
        
        print(f"Загружаем модель из {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        print("Модель успешно загружена")
        return True
    except Exception as e:
        print(f"Ошибка загрузки модели: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    if not load_model():
        print("⚠️ Внимание: модель не загружена! Сервис будет работать с ограниченной функциональностью")

@app.post("/predict")
async def predict(wine: WineFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис временно недоступен: модель не загружена"
        )
    
    try:
        input_data = pd.DataFrame([wine.dict()])
        prediction = model.predict(input_data)[0]
        return {"predicted_quality": float(prediction)}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка предсказания: {str(e)}"
        )

@app.get("/healthcheck")
async def healthcheck():
    try:
        if model is None:
            return {"status": "error", "reason": "model"}
        
        # Тестовая предсказание
        test_data = {
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.7,
            "citric_acid": 0.0,
            "residual_sugar": 1.9,
            "chlorides": 0.076,
            "free_sulfur_dioxide": 11.0,
            "total_sulfur_dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }
        test_df = pd.DataFrame([test_data])
        model.predict(test_df)
        
        return {"status": "ok"}
    except Exception as e:
        print(f"Healthcheck failed: {str(e)}")
        return {"status": "error", "reason": "unknown"}