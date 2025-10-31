from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import logging
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "models",
    "best_rf_model.joblib"
)
MODEL_PATH = os.path.abspath(MODEL_PATH)
MODEL_VERSION = "1.0.0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("steel_energy_api")


try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    logger.error(f"Could not load model: {e}")


app = FastAPI(
    title="Steel Energy ML API",
    description="API for steel energy RandomForest ML model",
    version=MODEL_VERSION,
)


class PredictRequest(BaseModel):
    Usage_kWh: float
    Lagging_Current_Reactive_Power_kVarh: float = Field(
        ...,
        alias="Lagging_Current_Reactive.Power_kVarh"
    )
    Leading_Current_Reactive_Power_kVarh: float
    CO2_tCO2: float = Field(..., alias="CO2(tCO2)")
    Lagging_Current_Power_Factor: float
    Leading_Current_Power_Factor: float
    NSM: float
    mixed_type_col: float
    WeekStatus: str
    Day_of_week: str


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest]


class ClassProbability(BaseModel):
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    prediction: str
    class_probabilities: List[ClassProbability]


@app.get("/health")
def health():
    status = "ok" if model is not None else "error"
    return {"status": status, "model_loaded": model is not None}


@app.get("/version")
def version():
    return {"version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    input_dict = request.dict(by_alias=True)
    df = pd.DataFrame([input_dict])
    try:
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0].tolist() \
            if hasattr(model, "predict_proba") \
            else None
        classes = list(model.classes_)
        class_probs = [
            ClassProbability(class_name=c, probability=p)
            for c, p in zip(classes, probabilities)
        ]
        return PredictResponse(
            prediction=str(prediction),
            class_probabilities=class_probs
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([r.dict(by_alias=True) for r in request.records])
    try:
        predictions = model.predict(df).tolist()
        probabilities = model.predict_proba(df).tolist() \
            if hasattr(model, "predict_proba") \
            else [None] * len(predictions)
        classes = list(model.classes_)
        result = [
            PredictResponse(
                prediction=str(pred),
                class_probabilities=[
                    ClassProbability(class_name=c, probability=p)
                    for c, p in zip(classes, prob)
                ]
            )
            for pred, prob in zip(predictions, probabilities)
        ]
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    return {"metrics": {
        "requests": "not implemented",
        "accuracy": "not implemented",
    }}


@app.get("/classes")
def get_classes():
    return {"classes": list(model.classes_)}


@app.post("/retrain")
def retrain():
    return {"status": "not implemented"}


@app.get("/")
def root():
    return {"message": "Welcome to the Steel Energy ML API!"}
