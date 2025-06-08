import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import subprocess

app = FastAPI()

# Global variable to store the model
model = None

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
        # Pull model from DVC
        if not os.path.exists('model.joblib'):
            subprocess.run(['dvc', 'pull'], check=True)
        
        # Load the model
        model = joblib.load('model.joblib')
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict")
async def predict(wine: WineFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([wine.dict()])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return {"predicted_quality": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/healthcheck")
async def healthcheck():
    try:
        if model is None:
            return {"status": "error", "reason": "model"}
        
        # Test prediction with sample data
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
        print(f"Healthcheck failed: {e}")
        return {"status": "error", "reason": "unknown"}
