import os
import json
from datetime import datetime
from typing import List, Dict

import click
import numpy as np
import pandas as pd

from src.models.train_model import (
    load_data,
    split_data,
    build_preprocessing,
    train_base_model,
    evaluate_model,
    save_feature_importance,
    save_model,
)

import mlflow
import mlflow.sklearn as mlflow_sklearn


def ensure_mlflow_from_env():
    """Configura MLflow (tracking y experimento) desde variables de entorno."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "steel_energy")
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        pass


def param_grid_default() -> Dict[str, List]:
    """Devuelve una grilla de hiperparámetros por defecto para RandomForest."""
    return {
        "n_estimators": [100, 200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }


def expand_grid(grid: Dict[str, List]) -> List[Dict]:
    """Expande un diccionario de listas a una lista de combinaciones (productos).

    Nota: Para evitar explosión combinatoria, esta función recorta el total
    a las primeras N combinaciones si la grilla es muy grande.
    """
    from itertools import product
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    # Límite de seguridad: máximo 60 combinaciones por corrida
    return combos[:60]


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('model_out', type=click.Path())
@click.argument('figures_dir', type=click.Path())
def main(input_path: str, model_out: str, figures_dir: str):
    """Barrido de hiperparámetros de RandomForest con logging en MLflow.

    - Lee el dataset, separa train/test, construye el preprocesamiento.
    - Itera sobre una grilla de hiperparámetros y registra runs en MLflow.
    - Selecciona la mejor combinación por accuracy en test y guarda el modelo.
    - Exporta importancia de variables del mejor modelo.
    """
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    ensure_mlflow_from_env()

    df = load_data(input_path)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessing, num_cols, cat_cols = build_preprocessing(X_train)

    grid = param_grid_default()
    candidates = expand_grid(grid)

    best_acc = -np.inf
    best_model = None
    best_params = None
    best_fig = None

    for i, params in enumerate(candidates, start=1):
        run_name = f"sweep_rf_{i}_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            # Evitar duplicar random_state (ya se fija en train_base_model)
            safe_params = {k: v for k, v in params.items() if k != "random_state"}
            mlflow.log_params(safe_params)
            model = train_base_model(X_train, y_train, preprocessing, safe_params)
            acc, fig_path, report = evaluate_model(
                model, X_test, y_test, figures_dir, name=run_name
            )
            mlflow.log_metric("accuracy", float(acc))
            if fig_path:
                mlflow.log_artifact(fig_path)

            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_params = params
                best_fig = fig_path

    if best_model is None:
        raise RuntimeError("No se pudo entrenar ningún modelo en el barrido.")

    # Log final del mejor
    with mlflow.start_run(run_name="sweep_rf_best"):
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_accuracy", float(best_acc))
        if best_fig:
            mlflow.log_artifact(best_fig)

        # Importancia de variables y guardado del modelo
        fi_csv, top_png = save_feature_importance(best_model, num_cols, cat_cols, figures_dir)
        mlflow.log_artifact(fi_csv)
        mlflow.log_artifact(top_png)
        save_model(best_model, model_out)

        # Registro de modelo en MLflow Model Registry (opcional)
        try:
            # Evitar warning de esquema de MLflow con columnas enteras sin NAs
            example = X_test[:2].copy()
            int_cols = list(example.select_dtypes(include="integer").columns)
            if int_cols:
                example[int_cols] = example[int_cols].astype("float64")
            register_flag = os.getenv("MLFLOW_REGISTER_IN_REGISTRY", "false").lower() == "true"
            tracking_uri = mlflow.get_tracking_uri() or ""
            if register_flag and tracking_uri.startswith("http"):
                mlflow_sklearn.log_model(
                    best_model,
                    artifact_path="model",
                    input_example=example,
                    signature=mlflow.models.infer_signature(example, best_model.predict(example)),
                    registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "SteelEnergyRF"),
                )
            else:
                mlflow_sklearn.log_model(
                    best_model,
                    artifact_path="model",
                    input_example=example,
                    signature=mlflow.models.infer_signature(example, best_model.predict(example)),
                )
        except Exception:
            pass


if __name__ == "__main__":
    main()
