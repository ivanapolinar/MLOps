import os
import json
import pandas as pd

from src.models import predict_model

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib


def _make_dummy_df():
    """Crea un DataFrame pequeño y reproducible para pruebas."""
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6, 7, 8],
        "f2": [2, 3, 3, 4, 2, 1, 2, 4],
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "Load_Type": [0, 1, 1, 0, 1, 0, 1, 0],
        "date": pd.date_range("2022-01-01", periods=8, freq="D"),
    })
    return df


def _train_and_dump_model(df: pd.DataFrame, model_path: str):
    """Entrena un pipeline minimal y lo guarda a disco para las pruebas."""
    y = df["Load_Type"]
    X = df.drop(columns=["Load_Type", "date"])

    num_cols = list(X.select_dtypes("number").columns)
    cat_cols = list(X.select_dtypes("object").columns)

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    rf = RandomForestClassifier(n_estimators=50, random_state=42)

    pipe = make_pipeline(pre, rf)
    pipe.fit(X, y)

    joblib.dump(pipe, model_path)


def test_load_dataset_parses_date(tmp_path):
    """load_dataset debe leer CSV y parsear 'date' a datetime si existe."""
    df = _make_dummy_df()
    csv_path = tmp_path / "input.csv"
    df.to_csv(csv_path, index=False)

    df_loaded = predict_model.load_dataset(str(csv_path))

    assert pd.api.types.is_datetime64_any_dtype(df_loaded["date"])


def test_prepare_features_drops_target_and_date():
    """prepare_features elimina 'Load_Type' y 'date' si están presentes."""
    df = _make_dummy_df()
    X = predict_model.prepare_features(df)

    assert "Load_Type" not in X.columns
    assert "date" not in X.columns


def test_end_to_end_callback_creates_outputs(tmp_path):
    """La callback debe generar predicciones y métricas/figuras."""
    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "input.csv"
    predictions_path = tmp_path / "preds.csv"
    metrics_path = tmp_path / "metrics.json"
    figures_dir = tmp_path / "figs"

    df = _make_dummy_df()
    df.to_csv(input_path, index=False)
    _train_and_dump_model(df, str(model_path))

    predict_model.main.callback(
        str(input_path),
        str(model_path),
        str(predictions_path),
        str(metrics_path),
        str(figures_dir),
    )

    assert os.path.exists(predictions_path)
    assert os.path.exists(metrics_path)
    assert os.path.exists(
        os.path.join(str(figures_dir), "confusion_matrix_predict.png")
    )

    preds_df = pd.read_csv(predictions_path)
    assert "Prediction" in preds_df.columns
    assert any(col.startswith("Prob_") for col in preds_df.columns)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics and "report" in metrics
