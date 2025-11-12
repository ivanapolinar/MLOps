"""CLI para ejecutar el flujo end-to-end usando MLOpsPipeline.

Pasos:
- Procesar datos crudos → limpios.
- Entrenar (base + tuning), evaluar y exportar artefactos.
- Generar predicciones por lote y métricas.
"""

from __future__ import annotations

import os
import click

from src.pipeline.mlops_pipeline import MLOpsPipeline


@click.command()
@click.option(
    "--raw-input",
    type=click.Path(exists=True),
    default="data/raw/steel_energy_modified.csv",
    show_default=True,
    help="CSV crudo de entrada",
)
@click.option(
    "--clean-output",
    type=click.Path(),
    default="data/clean/steel_energy_clean.csv",
    show_default=True,
    help="CSV limpio de salida",
)
@click.option(
    "--model-path",
    type=click.Path(),
    default="models/final_model.joblib",
    show_default=True,
    help="Ruta del modelo final (.joblib)",
)
@click.option(
    "--predictions",
    type=click.Path(),
    default="reports/predictions.csv",
    show_default=True,
    help="Ruta del CSV de predicciones",
)
@click.option(
    "--metrics",
    type=click.Path(),
    default="reports/metrics.json",
    show_default=True,
    help="Ruta del JSON de métricas",
)
@click.option(
    "--figures",
    type=click.Path(),
    default="reports/figures",
    show_default=True,
    help="Directorio para figuras",
)
@click.option(
    "--mlflow-uri",
    default=None,
    help=(
        "MLFLOW_TRACKING_URI (si se omite, se usa el valor de entorno y "
        "configuración por defecto)"
    ),
)
@click.option(
    "--mlflow-experiment",
    default=None,
    help=(
        "Nombre del experimento de MLflow (si se omite, usa 'steel_energy' "
        "o valor de entorno)"
    ),
)
@click.option(
    "--register/--no-register",
    default=False,
    show_default=True,
    help=(
        "Registrar modelo en Model Registry (requiere MLflow server con "
        "Registry y MLFLOW_TRACKING_URI http)"
    ),
)
@click.option(
    "--registered-name",
    default=lambda: os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "SteelEnergyRF"),
    show_default=True,
    help="Nombre del modelo registrado en MLflow Registry",
)
def main(
    raw_input: str,
    clean_output: str,
    model_path: str,
    predictions: str,
    metrics: str,
    figures: str,
    mlflow_uri: str | None,
    mlflow_experiment: str | None,
    register: bool,
    registered_name: str,
):
    # Instanciar pipeline con configuración de MLflow
    pipe = MLOpsPipeline(
        mlflow_tracking_uri=mlflow_uri or os.getenv("MLFLOW_TRACKING_URI"),
        mlflow_experiment=mlflow_experiment or os.getenv(
            "MLFLOW_EXPERIMENT", "steel_energy"
        ),
        register_in_registry=register,
        registered_model_name=registered_name,
    )

    # 1) Procesamiento crudo → limpio
    pipe.process_raw_to_clean(raw_input, clean_output)

    # 2) Entrenamiento y exportación de artefactos
    pipe.run_training_pipeline(clean_output, model_path, figures)

    # 3) Predicción por lote
    pipe.batch_predict(clean_output, model_path, predictions, metrics, figures)


if __name__ == "__main__":
    main()

