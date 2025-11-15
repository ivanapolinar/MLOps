from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import logging
import os

logger = logging.getLogger("steel_energy_api")
logging.basicConfig(level=logging.INFO)

# Default path relative to repo (if no MODEL_URI provided)
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "best_rf_model.joblib")
)

MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_PATH)
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# Globals set at startup
model = None
model_uri_loaded: Optional[str] = None
model_feature_names: Optional[List[str]] = None
model_classes: Optional[List[str]] = None

# Detect mlflow availability
USE_MLFLOW = False
try:
    import mlflow.pyfunc  # type: ignore
    USE_MLFLOW = True
except Exception:
    USE_MLFLOW = False

# Known alias mapping for incoming JSON keys
ALIAS_TO_COL = {
    "Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh",
    "CO2(tCO2)": "CO2_tCO2",
}

app = FastAPI(
    title="Steel Energy ML API",
    description="API for steel energy RandomForest ML model",
    version=MODEL_VERSION,
)


class PredictRequest(BaseModel):
    Usage_kWh: float
    Lagging_Current_Reactive_Power_kVarh: float = Field(..., alias="Lagging_Current_Reactive.Power_kVarh")
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
    probability: Optional[float]


class PredictResponse(BaseModel):
    prediction: str
    class_probabilities: List[ClassProbability]


def _align_input_df(input_dict: Dict[str, Any]) -> pd.DataFrame:
    global model_feature_names
    input_renamed = {}
    for k, v in input_dict.items():
        if k in ALIAS_TO_COL:
            input_renamed[ALIAS_TO_COL[k]] = v
        else:
            safe_k = k.replace('.', '_').replace('(', '_').replace(')', '_')
            input_renamed[safe_k] = v

    df = pd.DataFrame([input_renamed])

    if model_feature_names:
        for c in model_feature_names:
            if c not in df.columns:
                df[c] = pd.NA
        df = df.loc[:, model_feature_names]
    return df


# TODO: Migrate to @app.lifespan when upgrading FastAPI beyond 0.95.2
@app.on_event("startup")
def load_model():
    global model, model_uri_loaded, model_feature_names, model_classes

    logger.info("Startup: attempting to load model from %s", MODEL_URI)
    try:
        if isinstance(MODEL_URI, str) and MODEL_URI.startswith("models:") and USE_MLFLOW:
            model = mlflow.pyfunc.load_model(MODEL_URI)
            model_uri_loaded = MODEL_URI
            logger.info("Loaded model from MLflow registry: %s", MODEL_URI)
        else:
            if not os.path.exists(MODEL_URI):
                raise FileNotFoundError(f"Model file not found at {MODEL_URI}")
            model = joblib.load(MODEL_URI)
            model_uri_loaded = MODEL_URI
            logger.info("Loaded model from local path: %s", MODEL_URI)

        if hasattr(model, "feature_names_in_"):
            model_feature_names = list(getattr(model, "feature_names_in_"))
            logger.info("Detected feature_names_in_ (%d): %s", len(model_feature_names), model_feature_names[:10])
        else:
            try:
                ct = model.named_steps.get("columntransformer") if hasattr(model, "named_steps") else None
                if ct is not None and hasattr(ct, "get_feature_names_out"):
                    if hasattr(ct, "feature_names_in_"):
                        model_feature_names = list(getattr(ct, "feature_names_in_"))
                        logger.info("Feature names obtained from ColumnTransformer.feature_names_in_")
            except Exception:
                pass

        if hasattr(model, "classes_"):
            model_classes = list(getattr(model, "classes_"))
            logger.info("Model classes: %s", model_classes)
    except Exception as e:
        model = None
        model_uri_loaded = None
        model_feature_names = None
        model_classes = None
        logger.exception("Failed loading model: %s", e)


@app.get("/health")
def health():
    status = "ok" if model is not None else "error"
    return {"status": status, "model_loaded": model is not None}


@app.get("/version")
def version():
    return {"version": MODEL_VERSION, "model_path": model_uri_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    input_dict = request.dict(by_alias=True)
    df_in = _align_input_df(input_dict)
    try:
        preds = model.predict(df_in)
        if preds is None:
            raise RuntimeError("Loaded model returned no predictions")
        prediction_raw = preds[0] if hasattr(preds, "__len__") else preds
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df_in)
                probabilities = proba[0].tolist()
            except Exception:
                probabilities = None
        classes = model_classes or list(getattr(model, "classes_", [])) or []
        class_probs = []
        if probabilities is not None and classes:
            class_probs = [ClassProbability(class_name=c, probability=float(p)) for c, p in zip(classes, probabilities)]
        return PredictResponse(prediction=str(prediction_raw), class_probabilities=class_probs)
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict", response_model=List[PredictResponse])
def batch_predict(request: BatchPredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    records = [r.dict(by_alias=True) for r in request.records]
    dfs = [_align_input_df(rec) for rec in records]
    df_in = pd.concat(dfs, ignore_index=True)
    try:
        preds = model.predict(df_in)
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(df_in)
            except Exception:
                probabilities = None
        classes = model_classes or list(getattr(model, "classes_", [])) or []
        results = []
        for i, pred in enumerate(preds):
            probs = probabilities[i].tolist() if (probabilities is not None) else None
            if probs is not None and classes:
                class_probs = [
                    ClassProbability(class_name=c, probability=float(p))
                    for c, p in zip(classes, probs)
                ]
            else:
                class_probs = []
            results.append(PredictResponse(prediction=str(pred), class_probabilities=class_probs))
        return results
    except Exception as e:
        logger.exception("Batch prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics")
def metrics():
    return {"metrics": {"requests": "not implemented", "accuracy": "not implemented"}}


@app.get("/classes")
def get_classes():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"classes": model_classes or list(getattr(model, "classes_", []))}


@app.post("/retrain")
def retrain():
    return {"status": "not implemented"}


@app.get("/")
def root():
    return {"message": "Welcome to the Steel Energy ML API!"}
