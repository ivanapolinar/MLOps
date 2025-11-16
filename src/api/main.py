from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import logging
import os

MODEL_VERSION = "1.0.0"

# Ruta por defecto del modelo
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "models",
        "best_rf_model.joblib"
    )
)

# Pytest puede sobrescribirla
MODEL_PATH = os.getenv("MLOPS_MODEL_PATH", DEFAULT_MODEL_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("steel_energy_api")


# --------------------------------------------------
# APP
# --------------------------------------------------
app = FastAPI(
    title="Steel Energy ML API",
    description="API exposing RandomForest predictions",
    version=MODEL_VERSION,
)

# MODELO GLOBAL
model = None


def load_model():
    """Carga el modelo una vez, usado tanto por FastAPI como por pytest."""
    global model
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        model = None


# Ejecutar carga de modelo incluso antes del startup-event (para pytest)
load_model()


@app.on_event("startup")
def startup_event():
    """Garantiza carga del modelo también en servidor real."""
    load_model()


# --------------------------------------------------
# MODELOS DE REQUEST / RESPONSE
# --------------------------------------------------
class PredictRequest(BaseModel):
    Usage_kWh: float
    Lagging_Current_Reactive_Power_kVarh: float = Field(
        ..., alias="Lagging_Current_Reactive.Power_kVarh"
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


# --------------------------------------------------
# ENDPOINTS
# --------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
    }


@app.get("/version")
def version():
    return {"version": MODEL_VERSION, "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([request.dict(by_alias=True)])

    try:
        pred = model.predict(df)[0]
        prob = (
            model.predict_proba(df)[0].tolist()
            if hasattr(model, "predict_proba")
            else []
        )
        classes = list(model.classes_)

        class_probs = [
            ClassProbability(class_name=c, probability=p)
            for c, p in zip(classes, prob)
        ]

        return PredictResponse(
            prediction=str(pred),
            class_probabilities=class_probs
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([r.dict(by_alias=True) for r in request.records])

    try:
        preds = model.predict(df).tolist()
        probas = (
            model.predict_proba(df).tolist()
            if hasattr(model, "predict_proba")
            else [[] for _ in preds]
        )

        classes = list(model.classes_)

        results = []
        for pred, prob in zip(preds, probas):
            class_probs = [
                ClassProbability(class_name=c, probability=p)
                for c, p in zip(classes, prob)
            ]
            results.append(
                PredictResponse(
                    prediction=str(pred),
                    class_probabilities=class_probs
                )
            )

        return results

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    return {"metrics": {"requests": "not implemented", "accuracy": "not implemented"}}


@app.get("/classes")
def get_classes():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"classes": list(model.classes_)}


@app.post("/retrain")
def retrain():
    return {"status": "not implemented"}


@app.get("/")
def root():
    return {"message": "Welcome to the Steel Energy ML API!"}
