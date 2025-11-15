"""
CLI para generar predicciones y métricas a partir de un modelo .joblib.
Incluye la clase `BatchPredictor` que encapsula el flujo de inferencia por lote.
"""

import json
import os
from typing import Optional

import click
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FUNCIONES AUXILIARES (deben ir ARRIBA para evitar NameError)
# ============================================================

def ensure_dirs(predictions_path: str,
                metrics_path: Optional[str] = None,
                figures_dir: Optional[str] = None):
    """Crea carpetas de salida necesarias."""
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)


def attach_probabilities(model, X, out):
    """Agrega columnas Prob_* si el modelo soporta predict_proba."""
    if not hasattr(model, 'predict_proba'):
        return

    try:
        proba = model.predict_proba(X)

        if proba.shape[1] == 2:
            out['Prob_1'] = proba[:, 1]
            return

        classes_ = getattr(model, 'classes_', [])
        for idx, cls in enumerate(classes_):
            out[f'Prob_{cls}'] = proba[:, idx]

    except Exception:
        pass


def save_confusion_matrix(y_true, y_pred, figures_dir: str, name: str = 'predict') -> str:
    os.makedirs(figures_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues',
    )
    plt.title(f"Matriz de confusión ({name})")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()

    fig_path = os.path.join(figures_dir, f"confusion_matrix_{name}.png")
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def maybe_metrics_and_figures(y_true, y_pred, metrics_path: Optional[str], figures_dir: Optional[str]):
    if y_true is None:
        return

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    }

    if metrics_path:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if figures_dir:
        save_confusion_matrix(y_true, y_pred, figures_dir, name="predict")


# ============================================================
# LÓGICA PRINCIPAL
# ============================================================

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df


def prepare_features(df: pd.DataFrame, target: str = 'Load_Type') -> pd.DataFrame:
    return df.drop(
        columns=[c for c in [target, 'date'] if c in df.columns],
        errors='ignore',
    )


class BatchPredictor:
    """Orquesta la inferencia por lote y exportación de artefactos."""

    def predict_file(
        self,
        input_path: str,
        model_path: str,
        predictions_path: str,
        metrics_path: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ) -> None:

        ensure_dirs(predictions_path, metrics_path, figures_dir)

        df = load_dataset(input_path)
        model = joblib.load(model_path)

        X = prepare_features(df)
        y_true = df['Load_Type'] if 'Load_Type' in df.columns else None

        y_pred = model.predict(X)

        out = df.copy()
        out["Prediction"] = y_pred
        attach_probabilities(model, X, out)

        out.to_csv(predictions_path, index=False)

        maybe_metrics_and_figures(y_true, y_pred, metrics_path, figures_dir)


# ============================================================
# CLI
# ============================================================

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("predictions_path", type=click.Path())
@click.argument("metrics_path", required=False, default=None)
@click.argument("figures_dir", required=False, default=None)
def main(input_path: str,
         model_path: str,
         predictions_path: str,
         metrics_path: Optional[str] = None,
         figures_dir: Optional[str] = None):

    BatchPredictor().predict_file(
        input_path,
        model_path,
        predictions_path,
        metrics_path,
        figures_dir,
    )


if __name__ == "__main__":
    main()

